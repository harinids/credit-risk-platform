from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional


@dataclass
class Turn:
    question: str
    sql: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConversationManager:
    def __init__(self, max_turns: int = 3, session_ttl_minutes: int = 30):
        self.max_turns = max_turns
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        self._sessions: dict[str, deque[Turn]] = {}
        self._last_seen: dict[str, datetime] = {}
        self._lock = Lock()

    def _evict_stale(self) -> None:
        now = datetime.utcnow()
        stale = [sid for sid, ts in self._last_seen.items()
                 if now - ts > self.session_ttl]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._last_seen.pop(sid, None)

    def add_turn(self, session_id: str, question: str, sql: str, answer: str) -> None:
        with self._lock:
            self._evict_stale()
            if session_id not in self._sessions:
                self._sessions[session_id] = deque(maxlen=self.max_turns)
            self._sessions[session_id].append(Turn(question, sql, answer))
            self._last_seen[session_id] = datetime.utcnow()

    def get_context_string(self, session_id: str) -> str:
        with self._lock:
            turns = self._sessions.get(session_id)
            if not turns:
                return ""

            lines = ["Previous conversation (for resolving references like 'that', 'those', 'it'):"]
            for i, t in enumerate(turns, 1):
                lines.append(f"Q{i}: {t.question}")
                lines.append(f"SQL{i}: {t.sql}")
                lines.append(f"A{i}: {t.answer}")
            return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_seen.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            self._evict_stale()
            return session_id in self._sessions


conversation_manager = ConversationManager()