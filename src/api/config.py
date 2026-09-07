"""
Centralized configuration for the API service.
Reads from environment variables (populated via docker-compose env_file: .env,
or via python-dotenv's load_dotenv() call in main.py for local/uvicorn runs).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Postgres
    postgres_user: str = "credit_admin"
    postgres_password: str = "change_me"
    postgres_db: str = "credit_risk"
    postgres_host: str = "db"
    postgres_port: int = 5432

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_artifact_path: str = "/app/models/risk_model.joblib"
    model_metadata_path: str = "/app/models/model_metadata.json"

    # LLM
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"

    # NL-to-SQL safety
    nl2sql_allowed_tables: str = (
        "application,bureau,previous_application,"
        "installments_payments,pos_cash_balance,credit_card_balance"
    )
    nl2sql_max_rows: int = 500
    nl2sql_query_timeout_seconds: int = 10

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def allowed_tables_list(self) -> list[str]:
        return [t.strip() for t in self.nl2sql_allowed_tables.split(",") if t.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
