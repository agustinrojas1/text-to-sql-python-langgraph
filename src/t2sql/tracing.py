"""Cost computation and trace summaries. Trace entries are produced by `nodes.common.traced`."""

import json
from pathlib import Path
from typing import Any

# USD per 1M tokens (input, output). Standard paid-tier list prices, checked 2026-10-17:
# - gemini-3.5-flash-lite: $0.30 / $2.50 (same page, checked 2026-10-19).
# - gemini-3.6-flash: https://ai.google.dev/gemini-api/docs/pricing  ($0.75 / $3.75 until
#   2026-12-31, then $1.50 / $7.50). Used even when running on the free tier ("would cost").
# - openai/gpt-oss-20b on Groq: https://console.groq.com/docs/model/openai/gpt-oss-20b ($0.075 / $0.30, checked 2026-10-19).
# - openai/gpt-oss-120b on Groq: https://console.groq.com/docs/model/openai/gpt-oss-120b
# - claude-haiku-5-5: https://platform.claude.com/docs/en/about-claude/pricing
#   (prompts up to 100k tokens; above that it is $0.50 / $2.50, not modelled).
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "claude-haiku-5-5": (0.10, 0.50),
}


def compute_cost(
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    overrides: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """USD cost of one call, or None if the model is unpriced or tokens are unknown."""
    prices = {**MODEL_PRICES, **(overrides or {})}
    if model is None or model not in prices or input_tokens is None or output_tokens is None:
        return None
    p_in, p_out = prices[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def summarize_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over a trace. `cost_usd` is None if any LLM call could not be priced."""
    llm = [e["llm"] for e in trace if e.get("llm")]
    costs = [e.get("cost_usd") for e in trace if e.get("llm")]
    attempts = sum(e["node"] == "generate_sql" for e in trace)
    repairs = [e for e in trace if e["node"] == "repair"]
    ended_ok = bool(trace) and trace[-1]["node"] == "respond"
    executes = [e for e in trace if e["node"] == "execute"]
    final_empty = bool(executes) and (executes[-1].get("details") or {}).get("row_count") == 0
    # A repair only counts as a recovery if it fixed an error or produced rows.
    recovered = any(e["details"]["kind"] != "empty" for e in repairs) or not final_empty
    return {
        "latency_ms": round(sum(e["latency_ms"] for e in trace), 1),
        "input_tokens": sum(m["input_tokens"] or 0 for m in llm),
        "output_tokens": sum(m["output_tokens"] or 0 for m in llm),
        "cost_usd": None if any(c is None for c in costs) else sum(costs),
        "llm_calls": len(llm),
        "attempts": attempts,
        "repaired": bool(repairs) and ended_ok and recovered,
        # No guard rejection or execution error on the first generated query.
        "first_try_valid": not any(e.get("attempt") == 1 and e.get("error") for e in trace),
    }


def append_trace_file(path: str, question: str, state: dict[str, Any]) -> None:
    """Append one JSON line (question, sql, answer, summary, trace)."""
    record = {k: state.get(k) for k in ("sql", "answer", "error", "summary", "trace")}
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"question": question, **record}, default=str, ensure_ascii=False) + "\n")
