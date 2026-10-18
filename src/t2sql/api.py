"""HTTP API: `POST /ask` and `GET /health`.

Run with `uv run uvicorn t2sql.api:app --reload`.

Status codes:
- 200: the agent finished, including out of scope (`in_scope=false`) and failures after the
  repair budget is spent (`error` is set, `answer` is the failure message). Both are valid
  agent outcomes for a valid request, so the client always reads one response shape and
  decides from `in_scope` / `error`. 422 is reserved for invalid requests (FastAPI default).
- 503: no LLM provider is configured, or every provider failed on this call.
- 500: anything unexpected. The detail is logged; the client gets a generic message.

JSON values: `Decimal` becomes a float (Chinook money is numeric(10,2), so nothing is lost
in practice; very large or high-precision values could be), dates and times become ISO 8601
strings, UUIDs and anything else unknown become strings, bytes become hex.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from t2sql.config import Settings, get_settings
from t2sql.graph import build_graph, run_question
from t2sql.llm import PROVIDERS, LLMUnavailableError, build_llm
from t2sql.schema_index import index_exists

log = logging.getLogger("t2sql.api")


# ---------------------------------------------------------------- models


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Summary(BaseModel):
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None if some model is not in the price table
    llm_calls: int
    attempts: int
    repaired: bool
    first_try_valid: bool


class AskResponse(BaseModel):
    answer: str
    sql: str | None
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    in_scope: bool
    language: str | None
    error: str | None
    candidate_tables: list[str]
    attempts: list[dict[str, Any]]  # failed attempts: {attempt, sql, kind, error}
    summary: Summary
    trace: list[dict[str, Any]] | None = None  # only with ?include_trace=true


class Health(BaseModel):
    status: str  # "ok" | "degraded"
    database: bool
    schema_index: bool
    llm_providers: list[str]  # names with an API key configured, never the keys


# ---------------------------------------------------------------- runtime


@dataclass
class Runtime:
    """What the endpoints need. Built once at startup; tests build their own."""

    graph: CompiledStateGraph | None
    settings: Settings
    llm_error: str | None = None
    llm_providers: list[str] = field(default_factory=list)
    db_check: Callable[[], bool] = lambda: True
    index_check: Callable[[], bool] = lambda: True


def _db_reachable(settings: Settings) -> bool:
    try:
        with psycopg.connect(settings.readonly_database_url, connect_timeout=3) as conn:
            conn.read_only = True
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False


def build_runtime() -> Runtime:
    """Default factory. Never raises for a missing LLM: the app starts and reports it."""
    settings = get_settings()
    providers = [
        p for p in settings.llm_providers if p in PROVIDERS and getattr(settings, PROVIDERS[p][0])
    ]
    runtime = Runtime(
        graph=None,
        settings=settings,
        llm_providers=providers,
        db_check=lambda: _db_reachable(settings),
        index_check=lambda: index_exists(settings.embedding_model, settings),
    )
    try:
        llm = build_llm(settings)
        runtime.graph = build_graph(llm, settings=settings)
    except LLMUnavailableError as e:
        runtime.llm_error = str(e)
        log.warning("Starting without an LLM: %s", e)
    return runtime


def get_runtime(request: Request) -> Runtime:
    """Dependency; override it (or pass a factory to `create_app`) in tests."""
    return request.app.state.runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]


# ---------------------------------------------------------------- serialization


def json_safe(v: Any) -> Any:
    if v is None or isinstance(v, bool | int | float | str):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, datetime | date | time):
        return v.isoformat()
    if isinstance(v, timedelta):
        return v.total_seconds()
    if isinstance(v, bytes | bytearray | memoryview):
        return bytes(v).hex()
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, list | tuple | set):
        return [json_safe(x) for x in v]
    if isinstance(v, UUID):
        return str(v)
    return str(v)


def to_response(state: dict[str, Any], include_trace: bool) -> AskResponse:
    rows = json_safe(state.get("rows") or [])
    return AskResponse(
        answer=state.get("answer", ""),
        sql=state.get("sql"),
        columns=state.get("columns") or [],
        rows=rows,
        row_count=len(rows),
        in_scope=state.get("in_scope", True),
        language=state.get("language"),
        error=state.get("error"),
        candidate_tables=state.get("candidate_tables") or [],
        attempts=json_safe(state.get("attempt_history") or []),
        summary=Summary(**state["summary"]),
        trace=json_safe(state["trace"]) if include_trace else None,
    )


# ---------------------------------------------------------------- app


def create_app(factory: Callable[[], Runtime] = build_runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = factory()
        yield

    app = FastAPI(title="t2sql", description="Text-to-SQL agent over PostgreSQL", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    # Plain `def`: FastAPI runs it in a threadpool, so the synchronous graph (LLM and
    # database calls) does not block the event loop.
    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, runtime: RuntimeDep, include_trace: bool = False) -> AskResponse:
        if runtime.graph is None:
            raise HTTPException(503, runtime.llm_error or "No LLM provider is available")
        try:
            state = run_question(runtime.graph, body.question, runtime.settings)
        except LLMUnavailableError as e:
            log.error("All LLM providers failed: %s", e)
            raise HTTPException(503, "All LLM providers failed; try again later") from e
        return to_response(state, include_trace)

    @app.get("/health", response_model=Health)
    def health(runtime: RuntimeDep) -> Health:
        db = runtime.db_check()
        index = db and runtime.index_check()
        ok = db and index and runtime.graph is not None
        return Health(
            status="ok" if ok else "degraded",
            database=db,
            schema_index=index,
            llm_providers=runtime.llm_providers,
        )

    return app


app = create_app()
