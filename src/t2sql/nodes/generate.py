from pydantic import BaseModel, Field

from t2sql import prompts
from t2sql.nodes.common import Node, traced
from t2sql.state import AgentState, Deps


class GeneratedSql(BaseModel):
    sql: str = Field(description="A single PostgreSQL SELECT statement")
    tables_used: list[str] = Field(default_factory=list, description="Tables the query reads")
    assumptions: list[str] = Field(default_factory=list, description="Interpretations made")


def make_generate_sql(deps: Deps) -> Node:
    @traced("generate_sql", deps)
    def generate_sql(state: AgentState) -> dict:
        history = state.get("attempt_history")
        if history:
            msgs = prompts.repair_messages(state["question"], state["schema_context"], history)
        else:
            msgs = prompts.generate_messages(state["question"], state["schema_context"])
        res = deps.llm.structured(msgs, GeneratedSql)
        return {
            "sql": res.output.sql,
            "tables_used": res.output.tables_used,
            "assumptions": res.output.assumptions,
            "attempts": state.get("attempts", 0) + 1,
            "_llm": res,
        }

    return generate_sql
