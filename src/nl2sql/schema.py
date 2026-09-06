"""
Schema-grounding for NL-to-SQL.

Instead of dumping the full application table (129+ columns) into every
prompt, we maintain a compact, curated description of each table — just
the columns a business analyst would actually query on, with plain-English
meaning. This is retrieved and injected into the prompt per-question,
keeping token usage low and predictable regardless of how wide the
underlying tables are.

This is deliberately hand-curated rather than auto-generated from
HomeCredit_columns_description.csv, because that file describes all 200+
raw columns (useful for humans reading docs, not for keeping LLM prompts
small) — we only expose the subset that's actually queryable/meaningful
for the chatbot's allowed tables.

Retrieval uses curated per-table KEYWORD lists (see TABLE_KEYWORDS below),
not raw description-word overlap. An earlier version matched against
generic description text and got confused by words like "credit" that
appear in nearly every table's description (e.g. it picked
previous_application for an income/default question, and ranked
credit_card_balance above bureau for an "overdue bureau credits"
question). Curated keywords fix this by only matching terms that are
actually distinctive per table.

BUG FOUND AND FIXED during multi-turn conversation testing: when
sql_generator.py was updated to include conversation_context in the
retrieval query (so a table referenced in a prior turn stays in scope
for follow-ups like "now break that down by gender"), the context string
contains raw SQL/table names with underscores, e.g. "previous_application".
But TABLE_KEYWORDS uses natural-language phrases with spaces, e.g.
"previous application". A plain substring check never matched the two,
so a table from prior-turn SQL silently failed to score any points and
got dropped from the follow-up question's schema — even though the
context-inclusion fix was itself correct. Fixed by normalizing
underscores to spaces before scoring (see get_relevant_schema_text).
"""

# One entry per allowed table. Only columns worth exposing to the LLM are
# listed — this is the "schema RAG" corpus we'll retrieve from per question.
SCHEMA = {
    "application": {
        "description": "One row per loan application. Main table with applicant demographics, income, and the loan outcome (target).",
        "columns": {
            "sk_id_curr": "Unique applicant/loan ID (join key to all other tables)",
            "target": "1 = applicant defaulted, 0 = repaid on time",
            "name_contract_type": "Cash loans or Revolving loans",
            "code_gender": "M or F",
            "cnt_children": "Number of children",
            "amt_income_total": "Annual income of the applicant",
            "amt_credit": "Total credit amount of the loan",
            "amt_annuity": "Loan annuity (periodic payment amount)",
            "amt_goods_price": "Price of the goods the loan is for",
            "name_income_type": "Income source, e.g. Working, Pensioner, Commercial associate, Maternity leave",
            "name_education_type": "Education level, e.g. Secondary, Higher education, Academic degree",
            "name_family_status": "Marital status",
            "name_housing_type": "Housing situation, e.g. House / apartment, Rented apartment",
            "days_birth": "Applicant's age in negative days from application date (divide by -365 for years)",
            "days_employed": "Days employed in negative days (365243 is a sentinel meaning not currently employed)",
            "own_car_age": "Age of owned car in years (null if no car)",
            "flag_own_car": "Y/N owns a car",
            "flag_own_realty": "Y/N owns real estate",
            "occupation_type": "Applicant's occupation",
            "organization_type": "Type of employer organization",
            "region_rating_client": "Rating of the applicant's region (1=best, 3=worst)",
        },
    },
    "bureau": {
        "description": "Applicant's previous credits reported to the Credit Bureau by OTHER financial institutions (not Home Credit itself). Many rows per applicant.",
        "columns": {
            "sk_id_curr": "Applicant ID (join key to application)",
            "sk_id_bureau": "Bureau credit ID",
            "credit_active": "Status: Active, Closed, Sold, Bad debt",
            "credit_day_overdue": "Number of days the credit is/was overdue",
            "amt_credit_sum": "Total credit amount of this bureau record",
            "amt_credit_sum_debt": "Current outstanding debt on this bureau record",
            "credit_type": "Type of credit, e.g. Consumer credit, Credit card, Car loan",
            "days_credit": "Days before application when this bureau credit was opened",
        },
    },
    "previous_application": {
        "description": "Applicant's PREVIOUS applications specifically to Home Credit (not other institutions). Many rows per applicant.",
        "columns": {
            "sk_id_curr": "Applicant ID (join key to application)",
            "sk_id_prev": "Previous application ID",
            "name_contract_status": "Approved, Refused, Canceled, or Unused offer",
            "amt_application": "Amount requested by the applicant",
            "amt_credit": "Amount actually credited (may differ from requested)",
            "amt_annuity": "Annuity of the previous application",
            "name_contract_type": "Cash loans, Consumer loans, or Revolving loans",
            "days_decision": "Days before current application when this decision was made",
        },
    },
    "installments_payments": {
        "description": "Repayment history: one row per installment payment (or missed payment) for the applicant's previous Home Credit loans.",
        "columns": {
            "sk_id_curr": "Applicant ID (join key to application)",
            "sk_id_prev": "Previous application ID this payment belongs to",
            "days_instalment": "Days before application when this installment was due",
            "days_entry_payment": "Days before application when payment was actually made",
            "amt_instalment": "Amount that was due",
            "amt_payment": "Amount actually paid",
        },
    },
    "pos_cash_balance": {
        "description": "Monthly snapshots of the applicant's previous POS (point of sale) and cash loan balances with Home Credit.",
        "columns": {
            "sk_id_curr": "Applicant ID (join key to application)",
            "sk_id_prev": "Previous application ID",
            "months_balance": "Month relative to application (0 = most recent, negative = further in past)",
            "cnt_instalment": "Total number of installments in this loan",
            "sk_dpd": "Days past due at this snapshot",
            "name_contract_status": "Active, Completed, Signed, etc.",
        },
    },
    "credit_card_balance": {
        "description": "Monthly snapshots of the applicant's previous credit card balances with Home Credit.",
        "columns": {
            "sk_id_curr": "Applicant ID (join key to application)",
            "sk_id_prev": "Previous application ID",
            "months_balance": "Month relative to application",
            "amt_balance": "Balance on the credit card at this snapshot",
            "amt_credit_limit_actual": "Credit limit at this snapshot",
            "sk_dpd": "Days past due at this snapshot",
        },
    },
}

ALLOWED_TABLES = list(SCHEMA.keys())

# Curated keywords per table, used ONLY for retrieval scoring (not shown to
# the LLM). Hand-picked terms a business analyst would actually use when
# asking about that table — far more precise than matching generic
# description words, which caused false matches (e.g. "credit" appearing
# in nearly every table's description).
TABLE_KEYWORDS = {
    "application": [
        "applicant", "applicants", "income", "default", "defaulted", "defaulters",
        "gender", "male", "female", "children", "education", "married", "family",
        "housing", "car", "realty", "age", "employed", "employment", "occupation",
        "loan amount", "annuity", "region", "contract type",
    ],
    "bureau": [
        "bureau", "other bank", "other institution", "external credit",
        "overdue", "active credit", "credit active", "closed credit",
    ],
    "previous_application": [
        "previous application", "prior application", "past application",
        "approved", "refused", "rejection", "refusal", "cancelled", "canceled",
        "previous loan", "prior loan",
    ],
    "installments_payments": [
        "installment", "installments", "payment", "payments", "late payment",
        "missed payment", "repayment", "paid late", "underpaid",
    ],
    "pos_cash_balance": [
        "pos", "point of sale", "cash loan", "cash balance", "days past due",
        "dpd", "monthly balance",
    ],
    "credit_card_balance": [
        "credit card", "card balance", "credit limit", "drawings",
        "card utilization", "utilization",
    ],
}


def get_full_schema_text() -> str:
    """Every table's schema — used as a fallback or for a 'what tables exist' question."""
    lines = []
    for table, info in SCHEMA.items():
        lines.append(f"TABLE: {table} — {info['description']}")
        for col, desc in info["columns"].items():
            lines.append(f"  - {col}: {desc}")
    return "\n".join(lines)


def get_relevant_schema_text(question: str, max_tables: int = 3) -> str:
    """
    Keyword-based schema retrieval using curated per-table keyword lists
    (not raw description-word overlap, which was too noisy — generic words
    like "credit" appeared in nearly every table's description and caused
    false matches). Returns only the top N tables' schema text instead of
    dumping the whole schema every time.

    Deliberately simple (no embeddings) — for a schema this small (6
    tables), curated keyword matching is fast, free, transparent, and
    good enough; embeddings would be the natural upgrade if this grew to
    dozens of tables.

    Underscores are normalized to spaces before matching. This matters
    when `question` is actually `f"{conversation_context} {question}"`
    (as called from sql_generator.py for multi-turn memory) — prior-turn
    SQL contains raw table names like "previous_application" (underscore),
    which would never match the "previous application" (space) keyword
    phrase without this normalization, silently dropping that table from
    a follow-up question's retrieved schema.
    """
    q_lower = question.lower().replace("_", " ")
    scores = {}
    for table, keywords in TABLE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in q_lower:
                score += len(kw.split())  # multi-word matches count more (more specific)
        scores[table] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Always include 'application' as the anchor table (almost every
    # business question touches applicant-level attributes), then add
    # the highest-scoring tables that had any real match.
    selected = ["application"]
    for table, score in ranked:
        if len(selected) >= max_tables:
            break
        if score > 0 and table not in selected:
            selected.append(table)

    lines = []
    for table in selected:
        info = SCHEMA[table]
        lines.append(f"TABLE: {table} — {info['description']}")
        for col, desc in info["columns"].items():
            lines.append(f"  - {col}: {desc}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("FULL SCHEMA (baseline token cost)")
    print("=" * 70)
    full = get_full_schema_text()
    print(full)
    print(f"\nFull schema length: {len(full)} chars (~{len(full)//4} tokens)")

    test_questions = [
        "What is the average income of applicants who defaulted?",
        "How many applicants have overdue bureau credits?",
        "Show me applicants with late installment payments",
        "How many previous applications were refused?",
        "What is the average credit card utilization?",
    ]
    for q in test_questions:
        print(f"\n{'-' * 70}")
        print(f"Q: {q}")
        relevant = get_relevant_schema_text(q)
        selected_tables = [line.split(" — ")[0].replace("TABLE: ", "")
                            for line in relevant.split("\n") if line.startswith("TABLE:")]
        print(f"Selected tables: {selected_tables}")
        print(f"Relevant schema length: {len(relevant)} chars (~{len(relevant)//4} tokens) "
              f"vs full schema {len(full)} chars (~{len(full)//4} tokens) "
              f"= {(1 - len(relevant)/len(full))*100:.0f}% reduction")

    # Regression test for the underscore-normalization bug found during
    # multi-turn conversation testing
    print(f"\n{'-' * 70}")
    print("Regression test: table name with underscore (as it appears in raw SQL)")
    context_style_query = "SELECT AVG(amt_credit) FROM previous_application WHERE name_contract_status = 'Approved' now break that down by gender"
    relevant = get_relevant_schema_text(context_style_query)
    selected_tables = [line.split(" — ")[0].replace("TABLE: ", "")
                        for line in relevant.split("\n") if line.startswith("TABLE:")]
    print(f"Selected tables: {selected_tables}")
    assert "previous_application" in selected_tables, "BUG: previous_application not retrieved!"
    print("PASS: previous_application correctly retrieved despite underscore in input")