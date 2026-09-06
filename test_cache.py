"""
Standalone unit test for SemanticCache - no LLM calls, tests exact and
fuzzy matching logic in isolation using fake cached results.
"""

from src.nl2sql.cache import SemanticCache

cache = SemanticCache(similarity_threshold=0.90)

fake_result = {
    "question": "How many applicants have children?",
    "sql": "SELECT COUNT(*) FROM application WHERE cnt_children > 0",
    "answer": "92140 applicants have children.",
}

print("1. Miss on empty cache:")
print("   ", cache.get("How many applicants have children?"))

cache.set("How many applicants have children?", conversation_context="", result=fake_result)

print("\n2. Exact match hit:")
print("   ", cache.get("How many applicants have children?"))

print("\n3. Near-duplicate (extra whitespace/case) - should still hit:")
print("   ", cache.get("  how many APPLICANTS have children?  "))

print("\n4. Near-duplicate (minor rewording) - should hit if similar enough:")
result = cache.get("How many applicants have any children?")
print("   ", result)

print("\n5. Genuinely different question - should miss:")
print("   ", cache.get("What is the average income of applicants?"))

print("\n6. Same question text but DIFFERENT conversation context - should miss:")
print("   ", cache.get("How many applicants have children?", conversation_context="Q1: something else entirely"))

print("\n7. Stats after all above:")
print("   ", cache.stats())

print("\n8. Clear and confirm empty:")
cache.clear()
print("   ", cache.stats())
