from t2sql.nodes.common import Node, traced
from t2sql.sql_guard import SqlGuardError, validate_sql
from t2sql.state import AgentState, Deps


def make_validate_sql(deps: Deps) -> Node:
    allowed = set(deps.settings.allowed_tables)

    @traced("validate_sql", deps)
    def validate(state: AgentState) -> dict:
        try:
            q = validate_sql(state["sql"], allowed, deps.settings.max_rows)
        except SqlGuardError as e:
            return {"error": f"SQL rejected ({e.code}): {e.message}", "error_kind": "guard"}
        return {"sql": q.sql, "error": None, "error_kind": None}

    return validate
