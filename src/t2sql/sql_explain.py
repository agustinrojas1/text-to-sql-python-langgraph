"""Explain what the SQL guard checked and changed (pure, best effort, never raises).

`sql_guard.validate_sql` answers "is this safe?" with a SQL string or an exception. A demo UI
wants the story behind that answer: which of the seven guard checks passed, which tables and
columns the query touches, how the query is built (CTEs, joins, filters...) and whether the
guard had to add or lower the LIMIT. This module derives all of it from the input SQL, the
outcome of the guard and the retrieved schema, without touching the guard itself.

Every piece is best effort: sqlglot may fail to analyse an odd query, and the explanation
must never turn a valid run into an error, so each step falls back to an empty value.
"""

from collections.abc import Iterable
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from t2sql.sql_guard import DIALECT, GuardCode

# The guard's checks in the order it runs them, and the check each failure code belongs to.
CHECKS = (
    "parse",
    "single_statement",
    "select_only",
    "no_forbidden_clauses",
    "no_forbidden_functions",
    "tables_allowed",
    "limit",
)
_FAILING_CHECK = {
    GuardCode.PARSE_ERROR: "parse",
    GuardCode.EMPTY_QUERY: "parse",
    GuardCode.MULTIPLE_STATEMENTS: "single_statement",
    GuardCode.NOT_SELECT: "select_only",
    GuardCode.FORBIDDEN_CLAUSE: "no_forbidden_clauses",
    GuardCode.FORBIDDEN_FUNCTION: "no_forbidden_functions",
    GuardCode.FORBIDDEN_TABLE: "tables_allowed",
}
_PASSED_DETAIL = {
    "parse": "parsed as PostgreSQL",
    "single_statement": "one statement",
    "select_only": "read-only query (SELECT)",
    "no_forbidden_clauses": "no writes, DDL, INTO or row locks",
    "no_forbidden_functions": "no pg_*, sequence, file or large-object functions",
}


def explain_validation(
    input_sql: str,
    validated_sql: str | None,
    error_code: str | None,
    allowed_tables: Iterable[str],
    max_rows: int,
    schema_tables: list[dict[str, Any]] | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """The `checks`, `tables`, `columns`, `clauses`... of a validate_sql stream event.

    `validated_sql` is the guard's output (None when it rejected the query, in which case
    `error_code` is the GuardCode value). `schema_tables` is `retrieval.schema_summary`.
    """
    del allowed_tables  # the guard already enforced it; kept so callers pass the full context
    if validated_sql is None:
        return _failure(error_code, error_message)
    tree = _parse(validated_sql)
    original = _outer_limit(_parse(input_sql))
    added = original is None or original > max_rows
    enforced = _outer_limit(tree)
    if original is None:
        limit_detail = f"LIMIT {enforced} added"
    elif original > max_rows:
        limit_detail = f"LIMIT {original} lowered to {enforced}"
    else:
        limit_detail = f"LIMIT {enforced} kept (max {max_rows})"
    tables = sorted(_real_tables(tree))
    columns, unresolved = _resolve_columns(tree, schema_tables)
    checks = [{"name": n, "ok": True, "detail": _PASSED_DETAIL.get(n, "")} for n in CHECKS]
    checks[CHECKS.index("tables_allowed")]["detail"] = ", ".join(tables) or "no tables"
    checks[CHECKS.index("limit")]["detail"] = limit_detail
    return {
        "checks": checks,
        "tables": tables,
        "columns": columns,
        "unresolved_columns": unresolved,
        "clauses": _clauses(tree),
        "limit_enforced": enforced,
        "limit_added": added,
    }


def _failure(code: str | None, message: str | None) -> dict[str, Any]:
    """Checks before the failing one passed, it failed, the rest were never reached (ok=None)."""
    failing = _FAILING_CHECK.get(GuardCode(code), "parse") if _is_code(code) else "parse"
    reached = True
    checks = []
    for name in CHECKS:
        if name == failing:
            checks.append({"name": name, "ok": False, "detail": message or str(code or "rejected")})
            reached = False
        elif reached:
            checks.append({"name": name, "ok": True, "detail": _PASSED_DETAIL.get(name, "")})
        else:
            checks.append({"name": name, "ok": None, "detail": ""})
    return {
        "checks": checks,
        "tables": [],
        "columns": [],
        "unresolved_columns": [],
        "clauses": None,
        "limit_enforced": None,
        "limit_added": False,
    }


def _is_code(code: str | None) -> bool:
    return code in {c.value for c in GuardCode}


# ---------------------------------------------------------------- parsing helpers


def _parse(sql: str) -> exp.Expression | None:
    """The single outermost query of `sql`, or None if sqlglot cannot make sense of it."""
    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except Exception:  # noqa: BLE001 - best effort by design
        return None
    if isinstance(tree, exp.Subquery):  # `(SELECT ...)` -> SELECT ...
        tree = tree.unnest()
    return tree


def _outer_limit(tree: exp.Expression | None) -> int | None:
    """Literal LIMIT of the outermost query (of the set operation itself for UNION...)."""
    limit = tree.args.get("limit") if tree is not None else None
    count = limit.expression if isinstance(limit, exp.Limit) else None
    if isinstance(count, exp.Literal) and not count.is_string and count.name.isdigit():
        return int(count.name)
    return None


def _first_select(tree: exp.Expression | None) -> exp.Select | None:
    """The SELECT whose clauses describe the query: the leftmost one of a set operation."""
    while isinstance(tree, exp.SetOperation):
        tree = tree.this
    if isinstance(tree, exp.Subquery):
        tree = tree.unnest()
    return tree if isinstance(tree, exp.Select) else None


def _real_tables(tree: exp.Expression | None) -> set[str]:
    """Names of the real tables referenced, CTE references excluded (lowercase, no schema)."""
    if tree is None:
        return set()
    try:
        scopes = traverse_scope(tree)
    except Exception:  # noqa: BLE001
        return set()
    return {
        table.name.lower()
        for scope in scopes
        for table in scope.tables
        if isinstance(table.this, exp.Identifier) and not (not table.args.get("db") and table.name in scope.cte_sources)
    }


# ---------------------------------------------------------------- clauses


def _sql(node: exp.Expression | None) -> str | None:
    return node.sql(dialect=DIALECT, comments=False) if node is not None else None


def _clauses(tree: exp.Expression | None) -> dict[str, Any] | None:
    """The outermost query split into readable clauses (the LIMIT is the query's own)."""
    select = _first_select(tree)
    if tree is None or select is None:
        return None
    try:
        with_ = tree.args.get("with_") or select.args.get("with_")
        from_ = select.args.get("from_")
        where, having = select.args.get("where"), select.args.get("having")
        group, order = select.args.get("group"), tree.args.get("order") or select.args.get("order")
        return {
            "ctes": [cte.alias_or_name for cte in with_.expressions] if with_ else [],
            "select": [_sql(e) for e in select.expressions],
            "from": _sql(from_.this) if from_ is not None else None,
            "joins": [_sql(j) for j in select.args.get("joins") or []],
            "where": _sql(where.this) if where is not None else None,
            "group_by": [_sql(e) for e in group.expressions] if group is not None else [],
            "having": _sql(having.this) if having is not None else None,
            "order_by": [_sql(e) for e in order.expressions] if order is not None else [],
            "limit": _outer_limit(tree),
        }
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- column resolution


def _known_columns(schema_tables: list[dict[str, Any]] | None) -> dict[str, set[str]]:
    return {t["name"]: {c["name"] for c in t["columns"]} for t in schema_tables or []}


def _lookup_source(scope: Scope, alias: str) -> tuple[bool, exp.Table | None]:
    """(found, real table) for `alias`, looking outwards for correlated subqueries.

    The table is None for a CTE / derived table: its own columns are resolved in its own scope.
    """
    current: Scope | None = scope
    while current is not None:
        if alias in current.sources:
            source = current.sources[alias]
            return True, source if isinstance(source, exp.Table) else None
        current = current.parent
    return False, None


def _owner_of(scope: Scope, name: str, known: dict[str, set[str]]) -> tuple[str, exp.Table | None]:
    """Which source an unqualified column belongs to: ("table", t) | ("inner", None) | ("none", None).

    "inner" means a CTE / derived table or a select alias of this query provides it (nothing to
    report); "none" means no source has it or several do (Postgres would call it ambiguous).
    """
    current: Scope | None = scope
    while current is not None:
        tables = [
            src
            for src in current.sources.values()
            if isinstance(src, exp.Table) and name in known.get(src.name, ())
        ]
        inner = [
            src
            for src in current.sources.values()
            if isinstance(src, Scope) and name in src.expression.named_selects
        ]
        if len(tables) + len(inner) == 1:
            return ("table", tables[0]) if tables else ("inner", None)
        if len(tables) + len(inner) > 1:
            return "none", None
        if name in current.expression.named_selects:  # ORDER BY total, HAVING n > 1...
            return "inner", None
        current = current.parent
    return "none", None


def _resolve_columns(
    tree: exp.Expression | None, schema_tables: list[dict[str, Any]] | None
) -> tuple[list[str], list[str]]:
    """("table.column" refs resolved against the schema, names that could not be attributed).

    Done on sqlglot's scopes rather than `optimizer.qualify`: qualify raises on the first
    unknown column ("Unknown column: nope") and returns nothing, while here every reference is
    judged on its own, so one hallucinated column does not hide the rest.
    """
    if tree is None:
        return [], []
    columns = list(tree.find_all(exp.Column))
    try:
        known = _known_columns(schema_tables)
        resolved: set[str] = set()
        unresolved: set[str] = set()
        for scope in traverse_scope(tree):
            for col in scope.columns:
                if isinstance(col.this, exp.Star):
                    continue
                # sqlglot also lists the columns of subqueries nested in this scope's
                # expressions; they are judged in their own scope, against their own FROM.
                if col.find_ancestor(exp.Select, exp.SetOperation) is not scope.expression:
                    continue
                if col.table:
                    found, table = _lookup_source(scope, col.table)
                    if table is None:
                        if not found:  # a qualifier that names no source
                            unresolved.add(col.name)
                    elif table.name not in known or col.name in known[table.name]:
                        # A table missing from the schema cannot vouch for its columns.
                        resolved.add(f"{table.name}.{col.name}")
                    else:
                        unresolved.add(col.name)
                else:
                    kind, table = _owner_of(scope, col.name, known)
                    if kind == "table" and table is not None:
                        resolved.add(f"{table.name}.{col.name}")
                    elif kind == "none":
                        unresolved.add(col.name)
        return sorted(resolved), sorted(unresolved)
    except Exception:  # noqa: BLE001 - unusual queries must not break the run
        return [], sorted({c.name for c in columns if not isinstance(c.this, exp.Star)})
