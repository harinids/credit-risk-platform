"""
Real end-to-end multi-turn test through the actual FastAPI /chat endpoint —
uses the real LLM (Groq), not fake hand-written data. Requires the API
to be running (uvicorn src.api.main:app --reload) in another terminal.
"""

import requests

BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = "e2e-test-session"

def ask(question):
    resp = requests.post(f"{BASE_URL}/chat", json={"question": question, "session_id": SESSION_ID})
    if resp.status_code != 200:
        print(f"\nQ: {question}")
        print(f"ERROR {resp.status_code}: {resp.text}")
        return None
    data = resp.json()
    print(f"\nQ: {question}")
    print(f"SQL: {data.get('sql')}")
    print(f"A: {data['answer']}")
    return data

print("=" * 70)
print("TURN 1")
print("=" * 70)
ask("What is the average credit amount for approved previous applications?")

print("\n" + "=" * 70)
print("TURN 2 (should use conversation memory to resolve 'that')")
print("=" * 70)
ask("Now break that down by gender")

# Clean up the session afterward
requests.delete(f"{BASE_URL}/chat/{SESSION_ID}")
print("\nSession cleared.")