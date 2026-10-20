from t2sql import prompts
from t2sql.nodes.common import Node, traced
from t2sql.state import AgentState, Deps

# The user sees the full table; the LLM only needs a sample plus the total count to summarize.
MAX_ROWS_FOR_LLM = 10


def make_respond(deps: Deps) -> Node:
    @traced("respond", deps)
    def respond(state: AgentState) -> dict:
        rows = state["rows"]
        msgs = prompts.respond_messages(
            state["question"], state["sql"], state["columns"], rows[:MAX_ROWS_FOR_LLM], len(rows)
        )
        res = deps.llm.text(msgs)
        return {"answer": res.output, "_llm": res}

    return respond


def make_respond_error() -> Node:
    """Deterministic failure answer, reached when the repair budget is exhausted."""

    @traced("respond_error")
    def respond_error(state: AgentState) -> dict:
        template = prompts.FAILURE[state.get("language", "en")]
        return {"answer": template.format(error=state["error"])}

    return respond_error
