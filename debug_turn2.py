"""
Diagnostic script: manually reconstructs what Turn 2 sees, so we can
inspect the exact schema_text and prompt sent to the LLM, rather than
guessing from the final output alone.
"""

from src.nl2sql.conversation import conversation_manager
from src.nl2sql.sql_generator import ask, get_groq_client, build_prompt, generate_sql
from src.nl2sql.schema import get_relevant_schema_text

SESSION_ID = "debug-session"

# Simulate Turn 1 exactly as the real test does
result1 = ask(
    "What is the average credit amount for approved previous applications?",
    conversation_context="",
    verbose=True,
)
print("\n" + "=" * 70)
print("TURN 1 RESULT")
print("=" * 70)
print(result1)

conversation_manager.add_turn(
    SESSION_ID,
    question=result1["question"],
    sql=result1.get("sql", ""),
    answer=result1["answer"],
)

context = conversation_manager.get_context_string(SESSION_ID)
print("\n" + "=" * 70)
print("CONTEXT STRING PASSED TO TURN 2")
print("=" * 70)
print(repr(context))

question2 = "Now break that down by gender"
schema_text = get_relevant_schema_text(f"{context} {question2}")
print("\n" + "=" * 70)
print("SCHEMA TEXT FOR TURN 2")
print("=" * 70)
print(schema_text)

full_prompt = build_prompt(question2, schema_text, context)
print("\n" + "=" * 70)
print("FULL PROMPT SENT TO LLM FOR TURN 2")
print("=" * 70)
print(full_prompt)

print("\n" + "=" * 70)
print("TURN 2 GENERATED SQL (with fixed prompt)")
print("=" * 70)
client = get_groq_client()
turn2_sql = generate_sql(client, question2, schema_text, context)
print(turn2_sql)