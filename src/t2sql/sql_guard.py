"""Static validation of LLM-generated SQL (pure, no DB access).

The model may ask for anything; this module decides what reaches the database.
Flow: parse -> exactly one statement -> read-only query -> no write/DDL nodes
anywhere -> no dangerous functions -> only allow-listed tables -> forced LIMIT.
The SQL that is returned is regenerated from the AST (comments stripped), so
what Postgres executes is what we validated, not the original text.
"""

from dataclasses import dataclass
from enum import StrEnum

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

DIALECT = "postgres"


class GuardCode(StrEnum):
    PARSE_ERROR = "parse_error"
    EMPTY_QUERY = "empty_query"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_SELECT = "not_select"
    FORBIDDEN_CLAUSE = "forbidden_clause"  # INTO, FOR UPDATE/SHARE, data-modifying CTE...
    FORBIDDEN_FUNCTION = "forbidden_function"
    FORBIDDEN_TABLE = "forbidden_table"


class SqlGuardError(ValueError):
    """Raised when a query is rejected. `code` is stable and machine-readable."""

    def __init__(self, code: GuardCode, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ValidatedQuery:
    sql: str  # regenerated from the AST, with the forced LIMIT
    tables: frozenset[str]  # real tables referenced (CTE names excluded)


# Node types that must not appear anywhere in the tree. Checking the whole tree
# (not only the root) is what catches `WITH x AS (DELETE ... RETURNING *) SELECT ...`.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,
    exp.Create, exp.Drop, exp.Alter, exp.TruncateTable,
    exp.Copy, exp.Command, exp.Set, exp.Grant, exp.Transaction,
    exp.Commit, exp.Rollback, exp.Use, exp.Show, exp.Describe,
    exp.Execute, exp.Declare, exp.Pragma,
    exp.Into,  # SELECT ... INTO creates a table
    exp.Lock,  # FOR UPDATE / FOR SHARE take row locks
)  # fmt: skip

# Functions blocked by exact name. `current_setting` is blocked too: it leaks
# server configuration and an analytics question never needs it.
_FORBIDDEN_FUNCTIONS = frozenset({
    "set_config", "current_setting",
    "nextval", "setval", "currval", "lastval",  # sequences: writes / state
    "cursor_to_xml", "query_to_xml", "query_to_xml_and_xmlschema",  # run SQL from a string
    "query_to_xmlschema", "table_to_xml", "table_to_xml_and_xmlschema",
    "table_to_xmlschema", "schema_to_xml", "schema_to_xmlschema",
    "schema_to_xml_and_xmlschema", "database_to_xml", "database_to_xmlschema",
    "database_to_xml_and_xmlschema",
})  # fmt: skip
# Blocked by prefix: every pg_* function (pg_sleep, pg_read_file, pg_ls_dir,
# pg_terminate_backend, pg_cancel_backend, pg_advisory_lock...), large objects
# (lo_import/lo_export...), dblink*, and txid_*.
_FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "txid_")

# Schemas that refer to the Chinook tables (`public.track` == `track`).
_DEFAULT_SCHEMAS = frozenset({"public"})


def validate_sql(sql: str, allowed_tables: set[str], max_limit: int) -> ValidatedQuery:
    """Return the safe, LIMIT-bounded version of `sql` or raise `SqlGuardError`."""
    query = _parse_single_query(sql)
    _check_forbidden_nodes(query)
    _check_functions(query)
    tables = _check_tables(query, {t.lower() for t in allowed_tables})
    query = _force_limit(query, max_limit)
    return ValidatedQuery(sql=query.sql(dialect=DIALECT, comments=False), tables=frozenset(tables))


def _parse_single_query(sql: str) -> exp.Expression:
    try:
        # A trailing `;` or `; -- comment` parses as an empty / Semicolon node, not a statement.
        statements = [
            s for s in sqlglot.parse(sql, read=DIALECT) if s is not None and not isinstance(s, exp.Semicolon)
        ]
    except SqlglotError as e:
        raise SqlGuardError(GuardCode.PARSE_ERROR, str(e).splitlines()[0]) from e
    if not statements:
        raise SqlGuardError(GuardCode.EMPTY_QUERY, "no SQL statement found")
    if len(statements) > 1:
        raise SqlGuardError(GuardCode.MULTIPLE_STATEMENTS, "only one statement is allowed")

    stmt = statements[0]
    if isinstance(stmt, exp.Subquery):  # `(SELECT ...)` -> SELECT ...
        stmt = stmt.unnest()
    if not isinstance(stmt, exp.Select | exp.SetOperation):
        raise SqlGuardError(
            GuardCode.NOT_SELECT, f"only SELECT queries are allowed, got {type(stmt).__name__}"
        )
    return stmt


def _check_forbidden_nodes(query: exp.Expression) -> None:
    bad = query.find(*_FORBIDDEN_NODES)
    if bad is not None:
        raise SqlGuardError(
            GuardCode.FORBIDDEN_CLAUSE, f"{type(bad).__name__} is not allowed in a read-only query"
        )


def _check_functions(query: exp.Expression) -> None:
    for func in query.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_FUNCTION_PREFIXES):
            raise SqlGuardError(GuardCode.FORBIDDEN_FUNCTION, f"function {name}() is not allowed")


def _ident(node: exp.Expression | None) -> str:
    """Name as Postgres sees it: unquoted identifiers fold to lowercase, quoted ones don't."""
    if node is None:
        return ""
    return node.name if node.args.get("quoted") else node.name.lower()


def _check_tables(query: exp.Expression, allowed: set[str]) -> set[str]:
    cte_refs = _cte_references(query)
    found: set[str] = set()
    for table in query.find_all(exp.Table):
        if id(table) in cte_refs:
            continue
        if not isinstance(table.this, exp.Identifier):
            if isinstance(table.this, exp.Func):
                continue  # table function (generate_series...): vetted by _check_functions
            raise SqlGuardError(GuardCode.FORBIDDEN_TABLE, f"unsupported table: {table.sql()}")

        name = _ident(table.this)
        schema = _ident(table.args.get("db"))
        if table.args.get("catalog") or (schema and schema not in _DEFAULT_SCHEMAS):
            raise SqlGuardError(GuardCode.FORBIDDEN_TABLE, f"table not allowed: {table.sql()}")
        if name not in allowed:
            raise SqlGuardError(GuardCode.FORBIDDEN_TABLE, f"table not allowed: {name}")
        found.add(name)
    return found


def _cte_references(query: exp.Expression) -> set[int]:
    """ids of Table nodes that refer to a CTE visible at that point.

    Uses sqlglot's scope analysis instead of "any CTE name in the query", so a CTE
    defined in one subquery cannot hide a same-named real table elsewhere
    (`SELECT * FROM pg_user, (WITH pg_user AS (SELECT 1) SELECT * FROM pg_user) x`).
    Tables not proven to be CTE references are treated as real tables.
    """
    try:
        scopes = traverse_scope(query)
    except SqlglotError as e:
        raise SqlGuardError(GuardCode.PARSE_ERROR, f"cannot analyse query: {e}") from e
    return {
        id(table)
        for scope in scopes
        for table in scope.tables
        if not table.args.get("db") and table.name in scope.cte_sources
    }


def _force_limit(query: exp.Expression, max_limit: int) -> exp.Expression:
    """Make the outermost query return at most `max_limit` rows.

    For UNION/INTERSECT/EXCEPT the LIMIT lives on the set operation node itself,
    so it bounds the combined result. Anything that is not a plain integer
    literal (LIMIT ALL, expressions, FETCH ... PERCENT/WITH TIES) is replaced.
    """
    current = query.args.get("limit")
    value = _literal_limit(current)
    effective = max_limit if value is None else min(value, max_limit)
    query.set("limit", exp.Limit(expression=exp.Literal.number(effective)))
    return query


def _literal_limit(node: exp.Expression | None) -> int | None:
    if isinstance(node, exp.Limit):
        count = node.expression
    elif isinstance(node, exp.Fetch):
        options = node.args.get("limit_options")
        if options is not None and (options.args.get("percent") or options.args.get("with_ties")):
            return None
        count = node.args.get("count") or exp.Literal.number(1)  # FETCH FIRST ROW ONLY
    else:
        return None
    if isinstance(count, exp.Literal) and not count.is_string and count.name.isdigit():
        return int(count.name)
    return None
