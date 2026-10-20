"""Graph state and the dependencies injected into the nodes."""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from t2sql.config import Settings
from t2sql.db import QueryResult
from t2sql.llm import LLM
from t2sql.local_classifier import ScorerPool, ScopeScorer
from t2sql.retrieval import SchemaRetriever


ClassifierMode = Literal["llm", "local", "cascade"]
LocalModel = Literal["tev1"]


class AgentState(TypedDict, total=False):
    question: str
    # Per-run overrides of T2SQL_CLASSIFIER / T2SQL_LOCAL_CLASSIFIER_MODEL; absent = server settings.
    classifier: ClassifierMode
    local_model: LocalModel
    language: str  # "en" | "es", detected by classify
    in_scope: bool
    scope_reason: str
    candidate_tables: list[str]
    schema_context: str  # DDL-like text handed to generate_sql
    schema_tables: list[dict[str, Any]]  # the same schema as data (retrieval.schema_summary)
    sql: str  # raw from the LLM, then replaced by the validated/regenerated SQL
    tables_used: list[str]
    assumptions: list[str]
    error: str | None
    error_kind: str | None  # "guard" | "execution" when error is set
    attempts: int  # generation attempts so far (the repair loop caps this)
    # One {attempt, sql, kind, error} per failed attempt (kind: guard|execution|empty).
    attempt_history: Annotated[list[dict[str, Any]], operator.add]
    summary: dict[str, Any]  # summarize_trace(trace), added when the run finishes
    columns: list[str] | None
    rows: list[dict[str, Any]] | None
    answer: str
    # One entry per executed node; the reducer appends instead of overwriting.
    trace: Annotated[list[dict[str, Any]], operator.add]


Executor = Callable[[str], QueryResult]


@dataclass(frozen=True)
class Deps:
    llm: LLM
    retriever: SchemaRetriever
    settings: Settings
    executor: Executor
    scorer: ScopeScorer | None = None  # local scope classifier; None = the cloud LLM decides
    # Scorers by model, for per-run overrides; None = `scorer` is used whatever the model asked for.
    scorers: ScorerPool | None = None
