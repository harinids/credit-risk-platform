"""
FastAPI entrypoint for the Credit Risk Intelligence Platform backend.

All endpoints now wired to real logic:
  /health        -> DB connectivity check
  /chat          -> NL-to-SQL + conversation memory + semantic caching
  /predict       -> risk score + band for an existing applicant (by sk_id_curr)
  /explain       -> SHAP-based explanation for an existing applicant
  /rules         -> derived business rules (surrogate tree distillation)
  /eda/summary   -> dataset summary, insights, data quality flags

/predict and /explain look up an existing applicant's precomputed feature
row by sk_id_curr rather than accepting raw form fields, since features
are derived from relational joins across 6 tables - this matches a
realistic bank workflow (scoring an application already on file). Use
GET /predict/sample-ids to get valid IDs to try.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from src.api.config import get_settings
from src.nl2sql.sql_generator import ask as nl2sql_ask
from src.nl2sql.conversation import conversation_manager
from src.nl2sql.cache import semantic_cache
from src.ml.predict import predict_for_id, explain_for_id, get_sample_ids
from src.rules.serve import load_rules_text, load_depth_sensitivity
from src.eda.eda import get_eda_summary_dict

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_engine = create_engine(settings.database_url, pool_pre_ping=True)
    yield
    app.state.db_engine.dispose()


app = FastAPI(
    title="Credit Risk Intelligence Platform API",
    version="0.2.0",
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


# --- Chat (NL-to-SQL + conversation memory + semantic caching) ---

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
    return semantic_cache.stats()


@app.delete("/chat/{session_id}")
def clear_chat_session(session_id: str):
    conversation_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# --- Prediction + Explainability ---

@app.get("/predict/sample-ids")
def predict_sample_ids(limit: int = 20):
    """Returns valid sk_id_curr values from the dataset to try with /predict and /explain."""
    return {"sample_ids": get_sample_ids(limit)}


@app.get("/predict/{sk_id_curr}")
def predict(sk_id_curr: int):
    try:
        return predict_for_id(sk_id_curr)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/explain/{sk_id_curr}")
def explain(sk_id_curr: int, top_n: int = 5):
    try:
        return explain_for_id(sk_id_curr, top_n=top_n)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# --- Business Rules ---

@app.get("/rules")
def rules():
    return {
        "rules_text": load_rules_text(),
        "depth_sensitivity": load_depth_sensitivity(),
    }


# --- EDA ---

@app.get("/eda/summary")
def eda_summary():
    return get_eda_summary_dict()
