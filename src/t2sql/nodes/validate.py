from t2sql.nodes.common import Node, traced
from t2sql.sql_explain import explain_validation
from t2sql.sql_guard import SqlGuardError, validate_sql
from t2sql.state import AgentState, Deps


def make_validate_sql(deps: Deps) -> Node:
    allowed = set(deps.settings.allowed_tables)
    max_rows = deps.settings.max_rows

    @traced("validate_sql", deps)
    def validate(state: AgentState) -> dict:
        input_sql = state["sql"]
        schema_tables = state.get("schema_tables")
        try:
            q = validate_sql(input_sql, allowed, max_rows)
        except SqlGuardError as e:
            # The explanation (which checks passed, ...) goes to the trace / stream only.
            details = {"code": str(e.code), "input_sql": input_sql} | explain_validation(
                input_sql, None, str(e.code), allowed, max_rows, schema_tables, e.message
            )
            return {
                "error": f"SQL rejected ({e.code}): {e.message}",
                "error_kind": "guard",
                "_details": details,
            }
        details = {"code": None, "input_sql": input_sql} | explain_validation(
            input_sql, q.sql, None, allowed, max_rows, schema_tables
        )
        return {"sql": q.sql, "error": None, "error_kind": None, "_details": details}

    return validate
