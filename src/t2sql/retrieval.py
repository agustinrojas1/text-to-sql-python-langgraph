"""Schema retrieval: which tables (and columns) the SQL generator gets to see."""

from functools import lru_cache
from typing import Any, Protocol

from t2sql.config import Settings
from t2sql.db import TableInfo, fetch_schema_metadata


class SchemaRetriever(Protocol):
    """question -> tables for the prompt. May also expose `retrieve(question)` returning
    an object with `.tables` and `.details` (scores, expansion) for the trace."""

    def __call__(self, question: str) -> list[TableInfo]: ...


def retrieve_with_details(
    retriever: SchemaRetriever, question: str
) -> tuple[list[TableInfo], dict[str, Any]]:
    if hasattr(retriever, "retrieve"):
        r = retriever.retrieve(question)
        return r.tables, r.details
    tables = retriever(question)
    return tables, {"retriever": type(retriever).__name__}


class AllTablesRetriever:
    """Returns every allowed table; metadata is fetched once per process."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._load = lru_cache(maxsize=1)(lambda: fetch_schema_metadata(settings))

    def __call__(self, question: str) -> list[TableInfo]:
        return self._load()


def format_schema(tables: list[TableInfo]) -> str:
    """Compact DDL-like text: columns with types, PK marker and FK targets."""
    blocks = []
    for t in tables:
        fks = {fk.column: fk for fk in t.foreign_keys}
        lines = []
        for c in t.columns:
            line = f"  {c.name} {c.data_type}"
            if c.is_primary_key:
                line += " PRIMARY KEY"
            if c.name in fks:
                line += f" REFERENCES {fks[c.name].ref_table}({fks[c.name].ref_column})"
            lines.append(line)
        blocks.append(f"TABLE {t.name} (\n" + ",\n".join(lines) + "\n)")
    return "\n\n".join(blocks)


def schema_summary(tables: list[TableInfo]) -> list[dict[str, Any]]:
    """Structured twin of `format_schema`: what the SQL prompt shows, as data.

    `[{"name", "columns": [{"name", "type", "pk", "fk": "table.column" | None}]}]`, in the
    order of `tables`. The UI draws it and the validate node resolves column references
    against it, so both see exactly what the model saw.
    """
    out = []
    for t in tables:
        fks = {fk.column: f"{fk.ref_table}.{fk.ref_column}" for fk in t.foreign_keys}
        columns = [
            {"name": c.name, "type": c.data_type, "pk": c.is_primary_key, "fk": fks.get(c.name)}
            for c in t.columns
        ]
        out.append({"name": t.name, "columns": columns})
    return out
