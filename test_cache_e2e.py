"""
Demonstrates the real optimization effect of semantic caching: the same
question asked twice through the live /chat endpoint should be much
faster the second time (cache hit, no LLM calls) than the first
(cache miss, 2 LLM calls: generation + relevance check).
"""

import time
import requests

BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = "cache-test-session"

def ask(question, label):
    start = time.time()
    resp = requests.post(f"{BASE_URL}/chat", json={"question": question, "session_id": SESSION_ID})
    elapsed = time.time() - start
    data = resp.json()
    print(f"\n{label}")
    print(f"  Q: {question}")
    print(f"  A: {data.get('answer')}")
    print(f"  Time: {elapsed:.2f}s")
    return elapsed

t1 = ask("How many applicants own a car?", "FIRST ASK (expect cache MISS, full LLM pipeline)")
t2 = ask("How many applicants own a car?", "SECOND ASK, identical (expect cache HIT)")
t3 = ask("How many applicants own a vehicle?", "THIRD ASK, reworded (expect cache HIT via fuzzy match)")

print(f"\n{'='*70}")
if t2 > 0:
    print(f"Speedup from caching: {t1/t2:.1f}x faster on exact repeat")
print(f"{'='*70}")

resp = requests.get(f"{BASE_URL}/chat/cache/stats")
print(f"\nCache stats: {resp.json()}")

requests.delete(f"{BASE_URL}/chat/{SESSION_ID}")
