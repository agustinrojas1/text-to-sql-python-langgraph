from t2sql.nodes.common import Node, traced
from t2sql.retrieval import format_schema, retrieve_with_details, schema_summary
from t2sql.state import AgentState, Deps


def make_retrieve_schema(deps: Deps) -> Node:
    @traced("retrieve_schema", deps)
    def retrieve_schema(state: AgentState) -> dict:
        tables, details = retrieve_with_details(deps.retriever, state["question"])
        return {
            "candidate_tables": [t.name for t in tables],
            "schema_context": format_schema(tables),
            "schema_tables": schema_summary(tables),
            "_details": details,
        }

    return retrieve_schema
