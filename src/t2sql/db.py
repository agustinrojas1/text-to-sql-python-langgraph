"""Postgres access: read-only query execution and schema introspection."""

from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import sql as pgsql
from psycopg.rows import dict_row

from t2sql.config import Settings, get_settings


class QueryError(Exception):
    """A query failed in Postgres. `message` is what the repair loop shows the LLM."""

    def __init__(self, message: str, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.sqlstate = sqlstate

    @property
    def is_timeout(self) -> bool:
        """statement_timeout fired. Repairable: the repair prompt can ask for a cheaper query."""
        return self.sqlstate == "57014"  # query_canceled


class DatabaseUnavailableError(Exception):
    """The database itself is unreachable or going away (not a problem with the SQL).

    Never repairable: asking the LLM to rewrite valid SQL cannot fix an outage, so this
    propagates out of the graph and the CLI / API report it (the API answers 503).
    """


def _is_infrastructure_error(e: psycopg.Error) -> bool:
    """True for connection / server-availability failures, False for errors in the query.

    Decided by sqlstate, not by exception class: `QueryCanceled` (statement timeout, 57014)
    is also an `OperationalError`, and it is a query problem. No sqlstate means the error
    never came from the server (cannot connect, connection dropped mid-query).
    """
    state = e.sqlstate
    if state is None:
        return isinstance(e, psycopg.OperationalError | psycopg.InterfaceError)
    # 08 connection exception, 53 insufficient resources (e.g. too many connections),
    # 57P0x admin / crash shutdown, cannot connect now, 58 system errors.
    return state[:2] in ("08", "53", "58") or state.startswith("57P")


def _raise_for(e: psycopg.Error) -> None:
    message = e.diag.message_primary or str(e)
    if _is_infrastructure_error(e):
        raise DatabaseUnavailableError(message) from e
    raise QueryError(message, e.sqlstate) from e


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]

    @property
    def row_count(self) -> int:
        return len(self.rows)


def execute_readonly(sql: str, settings: Settings | None = None) -> QueryResult:
    """Run an already validated query with the read-only role.

    Defense in depth on top of sql_guard and the role's own settings: the
    transaction is READ ONLY and statement_timeout is set again here.
    """
    settings = settings or get_settings()
    try:
        with psycopg.connect(settings.readonly_database_url, row_factory=dict_row) as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                # SET LOCAL cannot take bind parameters; the value is an int, so format it.
                cur.execute(
                    pgsql.SQL("SET LOCAL statement_timeout = {}").format(
                        pgsql.Literal(int(settings.statement_timeout_ms))
                    )
                )
                cur.execute(sql)  # type: ignore[arg-type]  # validated upstream
                columns = [c.name for c in cur.description or []]
                rows = cur.fetchall()
    except psycopg.Error as e:
        _raise_for(e)
    return QueryResult(columns=columns, rows=rows)


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool = False


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)


_COLUMNS_SQL = """
SELECT table_name, column_name, data_type, is_nullable = 'YES' AS nullable
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = ANY(%(tables)s)
ORDER BY table_name, ordinal_position
"""

_PKS_SQL = """
SELECT kcu.table_name, kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = 'public'
"""

# information_schema's constraint_column_usage is ambiguous for composite keys,
# so foreign keys come from pg_catalog, unnesting both column arrays in lockstep.
_FKS_SQL = """
SELECT c.conrelid::regclass::text AS table_name,
       a.attname AS column_name,
       c.confrelid::regclass::text AS ref_table,
       fa.attname AS ref_column
FROM pg_constraint c
CROSS JOIN LATERAL unnest(c.conkey, c.confkey) AS k(attnum, fattnum)
JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
JOIN pg_attribute fa ON fa.attrelid = c.confrelid AND fa.attnum = k.fattnum
WHERE c.contype = 'f' AND c.connamespace = 'public'::regnamespace
ORDER BY table_name, column_name
"""


def fetch_schema_metadata(settings: Settings | None = None) -> list[TableInfo]:
    """Tables, columns, types and FKs of the allowed tables (admin connection).

    Feeds the schema index (RAG) in a later phase. Never used to run generated SQL.
    """
    settings = settings or get_settings()
    tables = sorted(settings.allowed_tables)
    try:
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            columns = conn.execute(_COLUMNS_SQL, {"tables": tables}).fetchall()
            pks = {(r["table_name"], r["column_name"]) for r in conn.execute(_PKS_SQL)}
            fks = conn.execute(_FKS_SQL).fetchall()
    except psycopg.Error as e:
        _raise_for(e)

    info = {t: TableInfo(name=t) for t in tables}
    for r in columns:
        info[r["table_name"]].columns.append(
            ColumnInfo(
                name=r["column_name"],
                data_type=r["data_type"],
                nullable=r["nullable"],
                is_primary_key=(r["table_name"], r["column_name"]) in pks,
            )
        )
    for r in fks:
        if r["table_name"] in info:
            info[r["table_name"]].foreign_keys.append(
                ForeignKey(r["column_name"], r["ref_table"], r["ref_column"])
            )
    return [t for t in info.values() if t.columns]
