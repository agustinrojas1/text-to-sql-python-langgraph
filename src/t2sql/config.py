"""Application settings, read from environment variables / .env."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

CHINOOK_TABLES: tuple[str, ...] = (
    "album",
    "artist",
    "customer",
    "employee",
    "genre",
    "invoice",
    "invoice_line",
    "media_type",
    "playlist",
    "playlist_track",
    "track",
)


def default_embedding_cache_dir() -> str:
    """Persistent per-user fastembed cache (fastembed's own default lives in /tmp and is wiped).

    Precedence: FASTEMBED_CACHE_PATH, then $XDG_CACHE_HOME/t2sql/fastembed, then
    ~/.cache/t2sql/fastembed.
    """
    if explicit := os.environ.get("FASTEMBED_CACHE_PATH"):
        return explicit
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return str(Path(base) / "t2sql" / "fastembed")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="T2SQL_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Admin: schema introspection / indexing only. Never used to run generated SQL.
    admin_database_url: str = "postgresql://t2sql_admin:t2sql_admin@localhost:5433/chinook"
    # Read-only role: the only connection that executes generated SQL.
    readonly_database_url: str = "postgresql://t2sql_reader:t2sql_reader@localhost:5433/chinook"

    statement_timeout_ms: int = 5000
    max_rows: int = 100
    allowed_tables: list[str] = list(CHINOOK_TABLES)

    # Schema RAG. "auto": pgvector if the index is built, else all tables (with a warning).
    retriever: Literal["auto", "pgvector", "all"] = "auto"
    # Multilingual (EN/ES), 384-dim, ~220 MB ONNX download on first use (fastembed).
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Where fastembed stores the downloaded model (default: a persistent per-user cache).
    embedding_cache_dir: str = Field(default_factory=default_embedding_cache_dir)
    retrieval_top_k: int = 5
    # Weight of the keyword (lexical) score added to the embedding score; 0 = embeddings only.
    retrieval_lexical_weight: float = 0.5

    # Repair loop: retries after the first generation (max 1 + max_repairs generations).
    max_repairs: int = 2

    # Tracing. Extra/override prices: {"model": [usd_per_1m_input, usd_per_1m_output]}.
    model_prices: dict[str, tuple[float, float]] = {}
    trace_file: str | None = None  # append one JSON line per question

    # LLM providers, tried in order; those without an API key are skipped.
    # T2SQL_LLM_PROVIDERS accepts a comma-separated list ("gemini,groq").
    llm_providers: Annotated[list[str], NoDecode] = ["gemini", "groq"]
    # gemini-3.6-flash (and the other 3.x flash models) allow only 20 free-tier requests per day
    # (checked 2026-10-19), not enough for even one eval run; flash-lite does the full dataset.
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_model: str = "openai/gpt-oss-120b"
    anthropic_model: str = "claude-haiku-5-5"
    llm_timeout_s: float = 30
    llm_max_retries: int = 3  # per provider (429/503 backoff), before falling over to the next one
    # None = provider default (nothing is sent). Gemini models with fixed sampling ignore it.
    llm_temperature: float | None = 0
    # Minimum seconds between two calls to the same provider (0 = no throttle). For free tiers:
    # 5 requests/minute needs 12.
    llm_min_interval_s: float = 0
    # Keys use the providers' standard variable names (no T2SQL_ prefix needed).
    google_api_key: str | None = Field(
        None, validation_alias=AliasChoices("T2SQL_GOOGLE_API_KEY", "GOOGLE_API_KEY")
    )
    groq_api_key: str | None = Field(
        None, validation_alias=AliasChoices("T2SQL_GROQ_API_KEY", "GROQ_API_KEY")
    )
    anthropic_api_key: str | None = Field(
        None, validation_alias=AliasChoices("T2SQL_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    )

    @field_validator("llm_providers", mode="before")
    @classmethod
    def _split_providers(cls, v: object) -> object:
        if isinstance(v, str):
            return [p.strip().lower() for p in v.split(",") if p.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
