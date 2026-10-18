from t2sql.db import QueryError
from t2sql.nodes.common import Node, traced
from t2sql.state import AgentState, Deps


def make_execute(deps: Deps) -> Node:
    @traced("execute", deps)
    def execute(state: AgentState) -> dict:
        try:
            result = deps.executor(state["sql"])
        except QueryError as e:
            return {"error": f"Query failed: {e.message}", "error_kind": "execution"}
        return {
            "columns": result.columns,
            "rows": result.rows,
            "error": None,
            "error_kind": None,
            "_details": {"row_count": result.row_count},
        }

    return execute
