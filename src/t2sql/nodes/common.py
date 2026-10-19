"""Shared node plumbing: the tracing wrapper."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from t2sql.state import AgentState, Deps
from t2sql.tracing import compute_cost

Node = Callable[[AgentState], dict[str, Any]]


def traced(name: str, deps: Deps | None = None) -> Callable[[Node], Node]:
    """Append a trace entry per run: node, attempt, latency, LLM metadata, cost, error, details.

    A node may return `_llm` (an LLMResult) and/or `_details` (a dict) in its update;
    both are consumed here. `attempt` is the generation attempt the node worked on
    (0 before the first generation). `deps` supplies price overrides for the cost.
    """
    overrides = deps.settings.model_prices if deps else None

    def deco(fn: Node) -> Node:
        @wraps(fn)
        def wrapper(state: AgentState) -> dict[str, Any]:
            start = time.perf_counter()
            update = fn(state)
            llm = update.pop("_llm", None)
            details = update.pop("_details", None)
            meta = llm.meta if llm else None
            entry: dict[str, Any] = {
                "node": name,
                "attempt": update.get("attempts", state.get("attempts", 0)),
                "latency_ms": round((time.perf_counter() - start) * 1000 - (llm.waited_ms if llm else 0), 1),
                "llm": meta,
                "cost_usd": compute_cost(
                    meta["model"], meta["input_tokens"], meta["output_tokens"], overrides
                )
                if meta
                else None,
                "error": update.get("error"),
                "details": details,
            }
            return {**update, "trace": [entry]}

        return wrapper

    return deco
