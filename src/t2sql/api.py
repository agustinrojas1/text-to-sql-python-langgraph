"""HTTP API: `POST /ask`, `POST /ask/stream` (NDJSON, event types in its docstring), `GET /health`
and the web UI at `GET /`.

Run with `uv run uvicorn t2sql.api:app --reload`.

Status codes:
- 200: the agent finished, including out of scope (`in_scope=false`) and failures after the
  repair budget is spent (`error` is set, `answer` is the failure message). Both are valid
  agent outcomes for a valid request, so the client always reads one response shape and
  decides from `in_scope` / `error`. 422 is reserved for invalid requests (FastAPI default).
- 503: no LLM provider is configured, every provider failed on this call (a quota or rate
  limit gets a specific safe message), or the database is unreachable (an outage is never sent to the repair loop).
- 500: anything unexpected. The detail is logged; the client gets a generic message.

JSON values: `Decimal` becomes a float (Chinook money is numeric(10,2), so nothing is lost
in practice; very large or high-precision values could be), dates and times become ISO 8601
strings, UUIDs and anything else unknown become strings, bytes become hex.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import UUID

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from t2sql.config import Settings, get_settings
from t2sql.db import DatabaseUnavailableError
from t2sql.graph import build_graph, run_question, stream_question
from t2sql.llm import PROVIDERS, LLMUnavailableError, build_llm
from t2sql.local_classifier import ScorerPool, build_scorer
from t2sql.replay import load_recording, record_lines, replay_lines
from t2sql.schema_index import index_exists

log = logging.getLogger("t2sql.api")

INDEX_HTML = Path(__file__).parent / "web" / "index.html"


# ---------------------------------------------------------------- models


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    # Per-request override of T2SQL_CLASSIFIER / T2SQL_LOCAL_CLASSIFIER_MODEL (None = server settings).
    classifier: Literal["llm", "local", "cascade"] | None = None
    local_model: Literal["tev1", "d1"] | None = None


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
    classifier: str  # "llm" | "local" | "cascade"
    # Is the local scorer's server up? None when the classifier is "llm". Not part of `status`:
    # without it the cloud LLM classifies, so the service still works.
    local_classifier: bool | None
    local_model: str  # the server's default local model (T2SQL_LOCAL_CLASSIFIER_MODEL)
    # Reachability of each model's llama-server, e.g. {"tev1": true, "d1": false}; {} if not checked.
    # Probed in parallel with a short timeout. A down model never degrades `status`: requests
    # that pick it fall back to the cloud LLM.
    local_models: dict[str, bool] = {}
    replay: bool = False  # T2SQL_REPLAY_DIR is set: recorded questions are replayed, no network needed


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
    local_check: Callable[[], bool] | None = None  # None: no local classifier configured
    local_models_check: Callable[[], dict[str, bool]] | None = None  # per-model reachability


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
    scorer = build_scorer(settings)
    pool = ScorerPool(settings, preset={settings.local_classifier_model: scorer} if scorer else None)
    runtime = Runtime(
        graph=None,
        settings=settings,
        llm_providers=providers,
        db_check=lambda: _db_reachable(settings),
        index_check=lambda: index_exists(settings.embedding_model, settings),
        local_check=scorer.available if scorer else None,
        local_models_check=pool.availability,
    )
    try:
        llm = build_llm(settings)
        runtime.graph = build_graph(llm, settings=settings, scorer=scorer)
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


def _error_line(status: int, detail: str) -> str:
    return json.dumps({"type": "error", "status": status, "detail": detail}) + "\n"


def event_lines(
    runtime: "Runtime", question: str, classifier: str | None = None, local_model: str | None = None
) -> Iterator[str]:
    """The NDJSON lines of one live run of the graph. Failures become a final `error` line."""
    try:
        if runtime.graph is None:
            raise HTTPException(503, runtime.llm_error or "No LLM provider is available")
        for ev in stream_question(runtime.graph, question, runtime.settings, classifier, local_model):
            if ev["type"] == "final":
                ev = {"type": "final", **to_response(ev["state"], False).model_dump()}
            yield json.dumps(json_safe(ev), ensure_ascii=False) + "\n"
    except HTTPException as e:
        yield _error_line(e.status_code, e.detail)
    except LLMUnavailableError as e:
        log.error("All LLM providers failed: %s", e)
        yield _error_line(503, e.public_detail)
    except DatabaseUnavailableError as e:
        log.error("Database unavailable: %s", e)
        yield _error_line(503, "Database unavailable; try again later")
    except Exception:
        log.exception("Unhandled error in /ask/stream")
        yield _error_line(500, "Internal server error")


def stream_lines(
    runtime: "Runtime", question: str, classifier: str | None = None, local_model: str | None = None
) -> Iterator[str]:
    """What `/ask/stream` sends: a recording if replay is on and one matches (same question AND
    same classifier mode / local model), else a live run (saved to T2SQL_RECORD_DIR if set).
    `classifier` / `local_model` None mean the server settings. With neither replay nor record
    configured this is exactly `event_lines`."""
    s = runtime.settings
    mode, model = classifier or s.classifier, local_model or s.local_classifier_model
    if s.replay_dir and (
        events := load_recording(s.replay_dir, question, mode, model, (s.classifier, s.local_classifier_model))
    ):
        return replay_lines(events, s.replay_max_gap_s)
    lines = event_lines(runtime, question, classifier, local_model)
    return record_lines(lines, s.record_dir, question, mode, model) if s.record_dir else lines


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
            state = run_question(runtime.graph, body.question, runtime.settings, body.classifier, body.local_model)
        except LLMUnavailableError as e:
            log.error("All LLM providers failed: %s", e)
            raise HTTPException(503, e.public_detail) from e
        except DatabaseUnavailableError as e:
            log.error("Database unavailable: %s", e)
            raise HTTPException(503, "Database unavailable; try again later") from e
        return to_response(state, include_trace)

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, runtime: RuntimeDep) -> StreamingResponse:
        """NDJSON, one JSON object per line, in the order things happen:

        - `start`: a node begins (`node`, `attempt`).
        - `prompt`: the exact messages about to go to the LLM (classify, generate_sql, respond).
        - `token`: best-effort live LLM output for generate_sql (decoded `sql` so far) and
          respond (`text` so far); may never arrive (fake LLM, provider that cannot stream).
        - `node`: a node finished, with latency, LLM metadata, cost, error and a node-specific
          `data` payload (retrieval ranking, guard checks, execution sample...).
        - `final` (the AskResponse) or `error`, always last.

        The body may carry `classifier` ("llm" | "local" | "cascade") and `local_model`
        ("tev1" | "d1") to override the server's settings for this question; the classify
        `node` event reports what was used (`classifier`, `local_model`, `decided_by`).

        Clients must ignore event types and keys they do not know. Errors after the response has started cannot change the HTTP status, so they are
        sent as an `error` event carrying the status /ask would have used.
        """

        # A sync generator: Starlette iterates it in a threadpool, so the blocking graph
        # calls do not stall the event loop.
        return StreamingResponse(
            stream_lines(runtime, body.question, body.classifier, body.local_model), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_HTML, media_type="text/html")

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
            classifier=runtime.settings.classifier,
            local_classifier=runtime.local_check() if runtime.local_check else None,
            local_model=runtime.settings.local_classifier_model,
            local_models=runtime.local_models_check() if runtime.local_models_check else {},
            replay=bool(runtime.settings.replay_dir),
        )

    return app


app = create_app()
