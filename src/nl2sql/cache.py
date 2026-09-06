"""
Semantic caching for NL-to-SQL - avoids re-generating SQL (and re-running
2 LLM calls: generation + relevance check) for questions that are the same
or near-identical to one already answered.

Design choice: uses difflib.SequenceMatcher (Python stdlib) for near-duplicate
detection rather than embedding-based similarity. Consistent with the
schema-retrieval design decision (src/nl2sql/schema.py) - for this scale
of usage, a lightweight string-similarity approach is fast, free, and
avoids adding an extra embedding API call that would itself consume
tokens/latency, working against the exact optimization goal caching is
meant to achieve.

BUG FOUND AND FIXED during real end-to-end testing: initial version always
included conversation_context in the cache key. But conversation_manager
accumulates turns within a session, so by the time a user asks the exact
same standalone question a second time, get_context_string() already
returns the FIRST turn as context - making the second lookup's key
different from the first one's stored key, guaranteeing a miss even for
word-for-word repeats. Verified via test_cache_e2e.py: 3 identical/near-
identical requests produced 0 cache hits and 3 separate LLM round-trips.

Fixed by only folding conversation_context into the key when the question
looks like an actual follow-up (contains a pronoun/continuation marker
like "that", "it", "now", "instead"). A standalone question ("how many
applicants own a car?") caches on its own text regardless of what came
before it in the conversation; only genuine follow-ups ("now break that
down by gender") are context-sensitive in their cache key, since those
are the only cases where the same surface text means different things
depending on history.

KNOWN LIMITATION (documented, not fixed): character-level similarity via
difflib catches whitespace/case/minor-rewording duplicates (verified:
"How many applicants own a car?" vs "  how many APPLICANTS own a car?  "
scores ~1.0), but misses true synonym paraphrases with different word
lengths (e.g. "car" vs "vehicle" scored ~0.85, just under the 0.90
threshold, verified via test_cache_e2e.py).

This was a deliberate decision, not an oversight: lowering the threshold
to catch more paraphrases increases the risk of merging two genuinely
different questions and serving a wrong cached answer - a worse failure
mode for a credit-risk tool than an occasional cache miss. An
embedding-based similarity approach would catch semantic paraphrases
correctly, at the cost of an additional API call per cache lookup, which
would work against the token/latency optimization caching exists to
provide in the first place. Kept difflib for this project's scale;
embeddings would be the natural upgrade path if paraphrase coverage
became a priority over exact/near-exact-repeat optimization.
"""

import difflib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field


SIMILARITY_THRESHOLD = 0.90
MAX_CACHE_SIZE = 200
CACHE_TTL_SECONDS = 3600

FOLLOWUP_MARKERS = [
    r"\bthat\b", r"\bthose\b", r"\bit\b", r"\bthem\b", r"\bthese\b",
    r"\bnow\b", r"\binstead\b", r"\balso\b", r"\btoo\b",
    r"\bbreak (that |it |this )?down\b", r"\bfilter (that|it|this)\b",
    r"\bwhat about\b", r"\band\b.{0,15}\?$",
]


@dataclass
class CacheEntry:
    key_text: str
    result: dict
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS,
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0
        self.exact_hits = 0
        self.fuzzy_hits = 0

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _is_followup(question: str) -> bool:
        q_lower = question.lower()
        return any(re.search(pattern, q_lower) for pattern in FOLLOWUP_MARKERS)

    def _build_key(self, question: str, conversation_context: str) -> str:
        if conversation_context and self._is_followup(question):
            return self._normalize(f"{conversation_context}||{question}")
        return self._normalize(question)

    def _evict_stale(self):
        now = time.time()
        stale_keys = [k for k, v in self._cache.items() if now - v.timestamp > self.ttl_seconds]
        for k in stale_keys:
            del self._cache[k]

    def _evict_lru_if_full(self):
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

    def get(self, question: str, conversation_context: str = "") -> dict | None:
        self._evict_stale()
        key_text = self._build_key(question, conversation_context)

        if key_text in self._cache:
            self._cache.move_to_end(key_text)
            self.hits += 1
            self.exact_hits += 1
            return self._cache[key_text].result

        best_ratio = 0.0
        best_key = None
        for cached_key in self._cache:
            ratio = difflib.SequenceMatcher(None, key_text, cached_key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = cached_key

        if best_key is not None and best_ratio >= self.similarity_threshold:
            self._cache.move_to_end(best_key)
            self.hits += 1
            self.fuzzy_hits += 1
            return self._cache[best_key].result

        self.misses += 1
        return None

    def set(self, question: str, conversation_context: str, result: dict):
        self._evict_stale()
        self._evict_lru_if_full()
        key_text = self._build_key(question, conversation_context)
        self._cache[key_text] = CacheEntry(key_text=key_text, result=result)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "cache_size": len(self._cache),
            "total_requests": total,
            "hits": self.hits,
            "misses": self.misses,
            "exact_hits": self.exact_hits,
            "fuzzy_hits": self.fuzzy_hits,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0.0,
        }

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.exact_hits = 0
        self.fuzzy_hits = 0


semantic_cache = SemanticCache()