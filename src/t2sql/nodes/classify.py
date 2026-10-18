from typing import Literal

from pydantic import BaseModel, Field

from t2sql import prompts
from t2sql.nodes.common import Node, traced
from t2sql.state import AgentState, Deps


class ScopeDecision(BaseModel):
    in_scope: bool = Field(description="True if the database can answer the question")
    reason: str = Field(description="One short sentence explaining the decision")
    language: Literal["en", "es"] = Field(description="Language of the question")


def make_classify(deps: Deps) -> Node:
    @traced("classify", deps)
    def classify(state: AgentState) -> dict:
        msgs = prompts.classify_messages(state["question"], sorted(deps.settings.allowed_tables))
        res = deps.llm.structured(msgs, ScopeDecision)
        d = res.output
        return {
            "in_scope": d.in_scope,
            "scope_reason": d.reason,
            "language": d.language,
            "_llm": res,
        }

    return classify


def make_respond_out_of_scope() -> Node:
    @traced("respond_out_of_scope")
    def respond_out_of_scope(state: AgentState) -> dict:
        return {"answer": prompts.OUT_OF_SCOPE[state.get("language", "en")]}

    return respond_out_of_scope


def route_after_classify(state: AgentState) -> str:
    return "retrieve_schema" if state["in_scope"] else "respond_out_of_scope"
