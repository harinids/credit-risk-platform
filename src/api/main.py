"""
FastAPI entrypoint for the Credit Risk Intelligence Platform backend.

Status:
  - /health          -> fully functional
  - /chat            -> fully functional (NL-to-SQL + conversation memory)
  - /predict, /explain, /rules, /eda/summary -> still 501 stubs, next phase

Conversation memory design note: wired at THIS layer (the /chat route)
rather than inside src.nl2sql.sql_generator.ask(), which already accepts
a conversation_context string parameter. This route fetches prior turns
via conversation_manager.get_context_string(session_id) before calling
ask(), then saves the new turn via conversation_manager.add_turn(...)
after ask() returns. This achieves multi-turn memory with zero changes
to the already-tested sql_generator.py.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # so GROQ_API_KEY / GROQ_MODEL from .env reach os.getenv()
                # calls in sql_generator.py even when running via uvicorn
                # rather than a shell that already has them exported

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from src.api.config import get_settings
from src.nl2sql.sql_generator import ask as nl2sql_ask
from src.nl2sql.conversation import conversation_manager
from src.nl2sql.cache import semantic_cache

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_engine = create_engine(settings.database_url, pool_pre_ping=True)
    yield
    app.state.db_engine.dispose()


app = FastAPI(
    title="Credit Risk Intelligence Platform API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    db_status = "unknown"
    try:
        with app.state.db_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "db": db_status,
        "llm_provider": settings.llm_provider,
    }


@app.get("/")
def root():
    return {"message": "Credit Risk Intelligence Platform API", "docs": "/docs"}


# --- Chat (NL-to-SQL + conversation memory) ---

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    question: str
    answer: str
    sql: str | None = None
    columns: list | None = None
    row_count: int | None = None
    session_id: str
    error: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    prior_context = conversation_manager.get_context_string(request.session_id)

    # Check semantic cache first - skips both LLM calls (SQL generation +
    # relevance check) entirely on a hit. Cache key includes conversation
    # context, not just the question text, since the same follow-up
    # phrasing means something different depending on prior turns.
    cached_result = semantic_cache.get(request.question, prior_context)
    if cached_result is not None:
        if "error" not in cached_result:
            conversation_manager.add_turn(
                request.session_id,
                question=cached_result["question"],
                sql=cached_result.get("sql", ""),
                answer=cached_result["answer"],
            )
        return ChatResponse(
            question=cached_result["question"],
            answer=cached_result["answer"],
            sql=cached_result.get("sql"),
            columns=cached_result.get("columns"),
            row_count=cached_result.get("row_count"),
            session_id=request.session_id,
            error=cached_result.get("error"),
        )

    result = nl2sql_ask(
        question=request.question,
        conversation_context=prior_context,
        verbose=False,
    )

    # Only cache and persist the turn if we actually got a usable result —
    # a validation/execution failure shouldn't pollute the cache or future
    # conversation context with a broken query.
    if "error" not in result:
        semantic_cache.set(request.question, prior_context, result)
        conversation_manager.add_turn(
            request.session_id,
            question=result["question"],
            sql=result.get("sql", ""),
            answer=result["answer"],
        )

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        sql=result.get("sql"),
        columns=result.get("columns"),
        row_count=result.get("row_count"),
        session_id=request.session_id,
        error=result.get("error"),
    )


@app.get("/chat/cache/stats")
def cache_stats():
    """Exposes cache hit-rate for the README's token-optimization documentation."""
    return semantic_cache.stats()


@app.delete("/chat/{session_id}")
def clear_chat_session(session_id: str):
    conversation_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# --- Stub routers, implemented in later phases ---

@app.post("/predict")
def predict_stub():
    return {"status": "not_implemented", "detail": "ML risk scoring lands in Phase 3."}, 501


@app.post("/explain")
def explain_stub():
    return {"status": "not_implemented", "detail": "Explainability lands in Phase 4."}, 501


@app.get("/rules")
def rules_stub():
    return {"status": "not_implemented", "detail": "Rule derivation lands in Phase 4."}, 501


@app.get("/eda/summary")
def eda_summary_stub():
    return {"status": "not_implemented", "detail": "EDA endpoints land in Phase 1."}, 501