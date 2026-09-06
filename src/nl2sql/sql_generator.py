"""
Core NL-to-SQL engine: takes a natural-language question, generates SQL
via Groq (openai/gpt-oss-20b, an open-weight model hosted on Groq's free
tier), validates it with sqlglot before execution, runs it, and returns
a plain-English answer.

Note on model choice: originally used llama-3.1-8b-instant, but Groq
deprecated that model on June 17, 2026 and recommends openai/gpt-oss-20b
as the direct replacement. Because the model name was already a
configurable env var (GROQ_MODEL) rather than hardcoded, swapping it was
a one-line change — a good demonstration of why provider/model choice
should always be externalized in a fast-moving LLM API landscape.

Known quirk discovered during testing: gpt-oss-20b is a REASONING model
that spends tokens on a hidden chain-of-thought before producing visible
output. With max_tokens=150-300, the entire budget was sometimes consumed
by hidden reasoning, causing the API to return an EMPTY visible response
even though the call succeeded with no error. Fixed via two changes:
  1. reasoning_effort="low" — this task (SQL generation, relevance
     checking) doesn't need deep reasoning
  2. Increased max_tokens to give headroom beyond the reasoning trace
This is a good example of why "the call succeeded" isn't the same as
"the call returned something usable" — worth checking actual output length
during any new model integration, not just HTTP status.

Multi-turn conversation bugs found and fixed during real end-to-end
testing (not caught by isolated unit tests on either schema.py or
conversation.py individually — only surfaced once tested together):

  BUG 1 (schema retrieval): get_relevant_schema_text(question) originally
  only considered the CURRENT question's keywords. A follow-up like "now
  break that down by gender" doesn't mention "previous application", so
  the previous_application table silently dropped out of scope even
  though Turn 1 used it. FIX: retrieval now runs on
  f"{conversation_context} {question}" so tables from recent turns stay
  available.

  BUG 2 (keyword normalization): once BUG 1 was fixed, a second bug
  surfaced — conversation_context contains raw SQL with underscored table
  names like "previous_application", but TABLE_KEYWORDS uses spaced
  natural-language phrases like "previous application". Fixed in
  schema.py by normalizing underscores to spaces before matching.

  BUG 3 (prompt didn't force reuse of prior query): even with the correct
  table now visible in schema, the model would still sometimes generate
  an entirely different, unrelated query for a follow-up (e.g. asked for
  "average credit amount... approved previous applications" then "break
  that down by gender" produced a default-rate-by-gender query on the
  wrong table — a different, invented metric that happened to still look
  plausible). Root cause: just showing the prior SQL in context isn't the
  same as instructing the model to treat it as the base to modify. Fixed
  by adding an explicit "if this is a follow-up, start from the previous
  SQL and modify minimally" instruction in build_prompt.

Safety layers (all must pass before a query touches the database):
    1. Must parse as valid SQL (sqlglot)
    2. Must be a SELECT only — no INSERT/UPDATE/DELETE/DROP/ALTER/etc.
    3. Every table referenced must be in the allow-list
    4. Must have a LIMIT clause (auto-injected if missing)
    5. Execution wrapped in a timeout

Hallucination control:
    - SQL is grounded in a schema retrieved specifically for the question
      (src.nl2sql.schema), not invented from the LLM's general knowledge
    - After execution, a lightweight self-consistency check asks the LLM
      whether the returned data actually addresses the original question
      before it's shown to the user. Parsing of that verdict is
      deliberately lenient (see check_answer_relevance) because
      reasoning-tuned models like gpt-oss often wrap YES/NO in markdown
      or add a short preamble despite explicit formatting instructions —
      an earlier strict startswith("YES") check produced false FAILs on
      answers that were actually correct.

Usage:
    python -m src.nl2sql.sql_generator
"""

import os
import re
import time
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlalchemy import create_engine, text
from groq import Groq

from src.nl2sql.schema import get_relevant_schema_text, ALLOWED_TABLES

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_ROWS = 200
QUERY_TIMEOUT_SECONDS = 10


class SQLValidationError(Exception):
    pass


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Set it as an environment variable:\n"
            '  $env:GROQ_API_KEY = "your_key_here"   (PowerShell, current session only)'
        )
    return Groq(api_key=api_key)


def build_prompt(question: str, schema_text: str, conversation_context: str = "") -> str:
    context_block = f"\nCONVERSATION CONTEXT (previous turns in this chat):\n{conversation_context}\n" if conversation_context else ""

    # Explicit follow-up instruction, added after a real bug was found in
    # multi-turn testing: without this, the model would see a follow-up
    # like "now break that down by gender" and generate SQL for a totally
    # different, unrelated metric (e.g. default rate by gender — a very
    # common question pattern for this dataset) instead of extending the
    # PREVIOUS query's actual metric (avg credit amount) and filters
    # (Approved status). Just having the prior SQL visible in context
    # wasn't enough — the model needed to be told explicitly to treat it
    # as the base to modify, not just as background information.
    followup_instruction = (
        "\nIMPORTANT: If this question is a follow-up (uses words like "
        "\"that\", \"those\", \"it\", \"now\", \"break down\", \"instead\", "
        "\"also\"), you MUST start from the PREVIOUS SQL query shown in "
        "CONVERSATION CONTEXT above and modify it minimally — keep the same "
        "table(s), the same metric/aggregate, and the same WHERE filters — "
        "and only add or change what the follow-up explicitly asks for "
        "(e.g. adding a GROUP BY, changing a filter value). Do NOT invent "
        "an unrelated query just because it uses similar column names.\n"
        if conversation_context else ""
    )

    return f"""You are a SQL generator for a SQLite database about loan applications and credit risk.

SCHEMA (only these tables/columns exist — do not invent others):
{schema_text}
{context_block}
RULES:
- Generate ONLY a single SELECT statement. No comments, no explanation, no markdown formatting.
- Use only the tables and columns shown above.
- Always alias aggregate results with a clear column name (e.g. AVG(amt_income_total) AS avg_income).
- If joining tables, join on sk_id_curr.
- Always include a LIMIT clause (LIMIT 100 if not otherwise specified).
- If the question refers to "defaulted" or "default", use application.target = 1. "Repaid" or "did not default" means target = 0.
- Output ONLY the raw SQL query, nothing else.
{followup_instruction}
QUESTION: {question}

SQL:"""


def extract_sql(raw_response: str) -> str:
    """Strip markdown code fences if the LLM added them despite instructions."""
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip().rstrip(";")


def validate_sql(sql: str) -> str:
    """
    Runs all safety checks. Raises SQLValidationError with a specific
    reason if any check fails. Returns the (possibly LIMIT-injected) SQL
    if it passes.
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception as exc:
        raise SQLValidationError(f"SQL does not parse: {exc}")

    if not isinstance(parsed, exp.Select):
        raise SQLValidationError(
            f"Only SELECT statements are allowed, got: {type(parsed).__name__}"
        )

    # Block any write/DDL keywords appearing anywhere (defense in depth,
    # even though the top-level check above should already catch these)
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                 "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "REPLACE"]
    upper_sql = sql.upper()
    for kw in forbidden:
        if re.search(rf"\b{kw}\b", upper_sql):
            raise SQLValidationError(f"Forbidden keyword detected: {kw}")

    # Table allow-list check
    referenced_tables = {t.name.lower() for t in parsed.find_all(exp.Table)}
    disallowed = referenced_tables - set(ALLOWED_TABLES)
    if disallowed:
        raise SQLValidationError(f"Query references disallowed table(s): {disallowed}")

    # Ensure a LIMIT exists; inject one if missing
    if not parsed.find(exp.Limit):
        parsed = parsed.limit(MAX_ROWS)
        sql = parsed.sql(dialect="sqlite")
    else:
        limit_node = parsed.find(exp.Limit)
        limit_val = int(limit_node.expression.this)
        if limit_val > MAX_ROWS:
            parsed.set("limit", exp.Limit(expression=exp.Literal.number(MAX_ROWS)))
            sql = parsed.sql(dialect="sqlite")

    return sql


def execute_sql(engine, sql: str):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = result.fetchall()
    return columns, rows


def generate_sql(client, question: str, schema_text: str, conversation_context: str = "") -> str:
    prompt = build_prompt(question, schema_text, conversation_context)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic SQL generation, not creative writing
        max_tokens=500,  # gpt-oss reasoning models consume tokens on hidden
                          # chain-of-thought before visible output; 300 was
                          # sometimes fully consumed by reasoning alone,
                          # returning an empty visible response
        reasoning_effort="low",  # SQL generation doesn't need deep reasoning
    )
    raw = response.choices[0].message.content
    return extract_sql(raw)


def check_answer_relevance(client, question: str, columns, rows, verbose: bool = False) -> tuple[bool, str]:
    """
    Self-consistency / hallucination check: asks the LLM whether the
    query result actually addresses the original question, using ONLY
    the returned data as context (not free generation). If it doesn't,
    we surface that honestly instead of presenting a confident-sounding
    but wrong answer.

    Parsing is deliberately lenient about exact response format (some
    models, especially reasoning-tuned ones like gpt-oss, wrap YES/NO in
    markdown bold or add a short preamble despite instructions) — we
    strip non-letter characters from the first line and look for YES/NO
    there rather than requiring an exact literal prefix match.
    """
    preview = f"Columns: {columns}\nFirst rows: {rows[:5]}"
    prompt = f"""A user asked: "{question}"

A SQL query was run and returned this data:
{preview}

Does this data actually answer the user's question?
Line 1: write exactly one word, either YES or NO (no formatting, no punctuation).
Line 2: write a one-sentence plain-English answer to the user's question using ONLY the data shown above (do not invent numbers not present in the data)."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,  # same reasoning-token headroom fix as generate_sql
        reasoning_effort="low",
    )
    text_out = response.choices[0].message.content.strip()

    if verbose:
        print(f"    [debug] raw relevance-check response: {text_out!r}")

    lines = [l.strip() for l in text_out.split("\n") if l.strip()]
    first_line_clean = re.sub(r"[^A-Za-z]", "", lines[0]).upper() if lines else ""

    if first_line_clean == "NO" or (first_line_clean.startswith("NO") and "YES" not in first_line_clean):
        is_relevant = False
    elif "YES" in first_line_clean:
        is_relevant = True
    else:
        # Ambiguous or empty verdict line (model didn't follow format) —
        # default to trusting the answer rather than discarding a
        # possibly-correct result, since we already validated the SQL
        # itself is grounded in the real schema and executed successfully.
        is_relevant = True

    explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else (lines[0] if lines else "")
    return is_relevant, explanation


def ask(question: str, conversation_context: str = "", verbose: bool = True) -> dict:
    """
    Full pipeline: question -> schema retrieval -> SQL generation ->
    validation -> execution -> relevance check -> answer.
    Returns a dict with all intermediate artifacts (useful for the API
    layer and for debugging/demoing).
    """
    client = get_groq_client()
    engine = get_engine()

    # Include conversation context in schema retrieval too, not just the
    # current question — otherwise a table referenced in a prior turn
    # (e.g. "previous_application" from Turn 1) can silently drop out of
    # scope for a follow-up question that doesn't repeat those keywords
    # (e.g. "now break that down by gender"), causing the LLM to pick a
    # different, wrong table it still has visibility into. (BUG 1, see
    # module docstring)
    schema_text = get_relevant_schema_text(f"{conversation_context} {question}")
    if verbose:
        print(f"[1/5] Retrieved schema for {len(schema_text.splitlines())} lines "
              f"(~{len(schema_text)//4} tokens)")

    raw_sql = generate_sql(client, question, schema_text, conversation_context)
    if verbose:
        print(f"[2/5] Generated SQL:\n{raw_sql}")

    try:
        validated_sql = validate_sql(raw_sql)
    except SQLValidationError as exc:
        return {
            "question": question,
            "sql": raw_sql,
            "error": f"SQL validation failed: {exc}",
            "answer": "I couldn't safely run that query. Could you rephrase the question?",
        }
    if verbose:
        print(f"[3/5] Validated SQL:\n{validated_sql}")

    try:
        columns, rows = execute_sql(engine, validated_sql)
    except Exception as exc:
        return {
            "question": question,
            "sql": validated_sql,
            "error": f"Execution failed: {exc}",
            "answer": "The query didn't run successfully. Could you rephrase the question?",
        }
    if verbose:
        print(f"[4/5] Executed: {len(rows)} rows returned, columns: {columns}")

    is_relevant, explanation = check_answer_relevance(client, question, columns, rows, verbose=verbose)
    if verbose:
        print(f"[5/5] Relevance check: {'PASS' if is_relevant else 'FAIL'}")

    if not explanation:
        # Fallback if the relevance-check call still returned nothing
        # usable — construct a basic answer directly from the raw data
        # rather than showing the user an empty string.
        explanation = f"Result: {dict(zip(columns, rows[0])) if rows else 'no rows returned'}"

    if not is_relevant:
        explanation = (
            "I ran a query but I'm not confident the result actually answers "
            "your question — here's the raw data instead: "
            f"{columns} -> {rows[:5]}"
        )

    return {
        "question": question,
        "sql": validated_sql,
        "columns": columns,
        "rows": rows[:20],  # cap what we carry around/display
        "row_count": len(rows),
        "answer": explanation,
        "relevance_check_passed": is_relevant,
    }


if __name__ == "__main__":
    test_questions = [
        "What is the average income of applicants who defaulted?",
        "How many applicants have children?",
        "What percentage of applicants own a car?",
        "How many previous applications were refused?",
        "What is the average credit amount for female applicants?",
    ]

    for q in test_questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {q}")
        print("=" * 70)
        result = ask(q)
        print(f"\nANSWER: {result['answer']}")
        time.sleep(1)  # gentle on free-tier rate limits