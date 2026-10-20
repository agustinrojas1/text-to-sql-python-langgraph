from t2sql.db import QueryError
from t2sql.nodes.common import Node, traced
from t2sql.state import AgentState, Deps


def make_execute(deps: Deps) -> Node:
    @traced("execute", deps)
    def execute(state: AgentState) -> dict:
        try:
            result = deps.executor(state["sql"])
        except QueryError as e:
            message = f"Query failed: {e.message}"
            if e.is_timeout:
                # Repairable on purpose: a cheaper query (fewer joins, a filter, a smaller
                # aggregation) can fit in the timeout. The hint steers the repair prompt.
                message += " (the query took too long; write a cheaper one)"
            return {"error": message, "error_kind": "execution"}
        return {
            "columns": result.columns,
            "rows": result.rows,
            "error": None,
            "error_kind": None,
            "_details": {"row_count": result.row_count},
        }

    return execute
