"""Repair loop: record why the last attempt failed, then go back to generate_sql."""

from t2sql.nodes.common import Node, traced
from t2sql.state import AgentState, Deps

EMPTY_NOTICE = "The query ran but returned 0 rows."


def _budget_left(state: AgentState, max_repairs: int) -> bool:
    # attempts = generations so far; max 1 + max_repairs generations in total.
    return state.get("attempts", 0) <= max_repairs


def make_repair(deps: Deps) -> Node:
    @traced("repair", deps)
    def repair(state: AgentState) -> dict:
        """Append the failed attempt to the history and clear the per-attempt result.

        generate_sql sees the history and switches to the repair prompt.
        """
        if state.get("error"):
            kind, error = state["error_kind"] or "execution", state["error"]
        else:
            kind, error = "empty", EMPTY_NOTICE
        entry = {"attempt": state["attempts"], "sql": state["sql"], "kind": kind, "error": error}
        return {
            "attempt_history": [entry],
            "error": None,
            "error_kind": None,
            "rows": None,
            "columns": None,
            "_details": {
                "kind": kind,
                "failed_attempt": entry["attempt"],
                # Generations still allowed once this repair is done (max 1 + max_repairs).
                "attempts_left": deps.settings.max_repairs + 1 - state["attempts"],
            },
        }

    return repair


def make_route_after_validate(deps: Deps):
    def route(state: AgentState) -> str:
        if not state.get("error"):
            return "execute"
        return "repair" if _budget_left(state, deps.settings.max_repairs) else "respond_error"

    return route


def make_route_after_execute(deps: Deps):
    def route(state: AgentState) -> str:
        budget = _budget_left(state, deps.settings.max_repairs)
        if state.get("error"):
            return "repair" if budget else "respond_error"
        # An empty result is suspicious once; after that it is a legitimate answer.
        empty_retried = any(h["kind"] == "empty" for h in state.get("attempt_history", []))
        if not state["rows"] and budget and not empty_retried:
            return "repair"
        return "respond"

    return route
