"""Shared node plumbing: the tracing wrapper and the live-event helpers."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from langgraph.config import get_stream_writer

from t2sql.llm import Message
from t2sql.state import AgentState, Deps
from t2sql.tracing import compute_cost

Node = Callable[[AgentState], dict[str, Any]]


def emit(event: dict[str, Any]) -> None:
    """Send a live event on LangGraph's "custom" stream; a no-op anywhere else.

    Nodes are also called directly in unit tests and the graph is also run with `invoke`
    (CLI, `/ask`): there is either no runnable context (get_stream_writer raises
    RuntimeError) or nobody listening (the writer is a no-op). Either way nothing happens,
    so emitting costs nothing when the run is not streamed.
    """
    try:
        writer = get_stream_writer()
    except (RuntimeError, KeyError):  # no runnable context / a context without a runtime
        return
    writer(event)


def emit_prompt(node: str, attempt: int, messages: list[Message]) -> None:
    """Emit a `prompt` event: the exact messages about to be sent to the LLM."""
    emit({
        "type": "prompt",
        "node": node,
        "attempt": attempt,
        "data": {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "chars": sum(len(m.content) for m in messages),
        },
    })  # fmt: skip


def traced(name: str, deps: Deps | None = None, new_attempt: bool = False) -> Callable[[Node], Node]:
    """Append a trace entry per run: node, attempt, latency, LLM metadata, cost, error, details.

    A node may return `_llm` (an LLMResult) and/or `_details` (a dict) in its update;
    both are consumed here. `attempt` is the generation attempt the node worked on
    (0 before the first generation). `deps` supplies price overrides for the cost.

    Before the body runs it emits a live `start` event (see `emit`). `new_attempt` marks the
    node that increments `attempts` (generate_sql): its start event already carries the number
    of the attempt it is about to produce, so start / prompt / token / node events of one
    generation all agree.
    """
    overrides = deps.settings.model_prices if deps else None

    def deco(fn: Node) -> Node:
        @wraps(fn)
        def wrapper(state: AgentState) -> dict[str, Any]:
            emit({"type": "start", "node": name, "attempt": state.get("attempts", 0) + new_attempt})
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
