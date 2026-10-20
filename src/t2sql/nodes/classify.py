import logging
from typing import Literal

from pydantic import BaseModel, Field

from t2sql import prompts
from t2sql.local_classifier import LocalClassifierUnavailable, ScopeScore, detect_language
from t2sql.nodes.common import Node, emit_prompt, traced
from t2sql.state import AgentState, Deps

log = logging.getLogger(__name__)


class ScopeDecision(BaseModel):
    in_scope: bool = Field(description="True if the database can answer the question")
    reason: str = Field(description="One short sentence explaining the decision")
    language: Literal["en", "es"] = Field(description="Language of the question")


def make_classify(deps: Deps) -> Node:
    """classify: the cloud LLM ("llm"), a local scorer ("local") or both ("cascade").

    cascade: P < cascade_low is out of scope and P > cascade_high is in scope, decided locally
    (no tokens, no quota); in between, or if the local server is unavailable, the cloud LLM
    decides. "local" applies local_threshold and also falls back to the LLM when unavailable:
    a down local model never fails a request.
    """
    s = deps.settings
    scorer = deps.scorer if s.classifier != "llm" else None
    warned = False

    def ask_local(question: str) -> ScopeScore | None:
        nonlocal warned
        try:
            score = scorer.score(question)
        except LocalClassifierUnavailable as e:
            if not warned:  # one warning per outage, not one per request
                log.warning("Local classifier unavailable, using the cloud LLM: %s", e)
                warned = True
            return None
        warned = False
        return score

    def local_verdict(p: float) -> bool | None:
        """in_scope, or None when the cascade is not confident enough."""
        if s.classifier == "local":
            return p >= s.local_threshold
        if p < s.cascade_low:
            return False
        return True if p > s.cascade_high else None

    @traced("classify", deps)
    def classify(state: AgentState) -> dict:
        question = state["question"]
        score = ask_local(question) if scorer else None
        details: dict = {"decided_by": "llm"}
        if scorer and not score:
            details["local_unavailable"] = True
        if score:
            details |= {"p_in": round(score.p_in, 4), "local_ms": score.latency_ms, "local_model": score.model}
            in_scope = local_verdict(score.p_in)
            if in_scope is not None:
                details["decided_by"] = "local"
                return {
                    "in_scope": in_scope,
                    "scope_reason": f"local classifier {score.model}: P(in scope)={score.p_in:.2f}",
                    "language": detect_language(question),
                    "_details": details,
                }
        msgs = prompts.classify_messages(question, sorted(s.allowed_tables))
        emit_prompt("classify", state.get("attempts", 0), msgs)
        res = deps.llm.structured(msgs, ScopeDecision)
        d = res.output
        return {
            "in_scope": d.in_scope,
            "scope_reason": d.reason,
            "language": d.language,
            "_llm": res,
            "_details": details,
        }

    return classify


def make_respond_out_of_scope() -> Node:
    @traced("respond_out_of_scope")
    def respond_out_of_scope(state: AgentState) -> dict:
        return {"answer": prompts.OUT_OF_SCOPE[state.get("language", "en")]}

    return respond_out_of_scope


def route_after_classify(state: AgentState) -> str:
    return "retrieve_schema" if state["in_scope"] else "respond_out_of_scope"
