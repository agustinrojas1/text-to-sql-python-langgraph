import json

import pytest

from t2sql.config import Settings
from t2sql.tracing import MODEL_PRICES, append_trace_file, compute_cost, summarize_trace


def test_cost_known_model():
    # 1M in + 1M out = list price in + out
    p_in, p_out = MODEL_PRICES["openai/gpt-oss-120b"]
    assert compute_cost("openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(p_in + p_out)
    assert compute_cost("gemini-3.6-flash", 2000, 500) == pytest.approx((2000 * 0.75 + 500 * 3.75) / 1e6)


def test_cost_unknown_model_or_tokens_is_none():
    assert compute_cost("mystery-model", 10, 10) is None
    assert compute_cost("gemini-3.6-flash", None, 10) is None
    assert compute_cost(None, 10, 10) is None


def test_cost_overrides():
    assert compute_cost("mystery", 1_000_000, 0, {"mystery": (2.0, 4.0)}) == pytest.approx(2.0)
    assert compute_cost("gemini-3.6-flash", 1_000_000, 0, {"gemini-3.6-flash": (1.0, 1.0)}) == pytest.approx(1.0)


def test_price_override_from_env(monkeypatch):
    monkeypatch.setenv("T2SQL_MODEL_PRICES", '{"x": [1, 2]}')
    assert Settings(_env_file=None).model_prices == {"x": (1, 2)}


def entry(node, attempt, latency=10.0, tin=None, tout=None, cost=None, error=None, details=None):
    llm = {"provider": "p", "model": "m", "input_tokens": tin, "output_tokens": tout, "latency_ms": latency}
    return {
        "node": node, "attempt": attempt, "latency_ms": latency, "llm": llm if tin is not None else None,
        "cost_usd": cost, "error": error, "details": details,
    }  # fmt: skip


def test_summary_totals_with_repair():
    trace = [
        entry("classify", 0, tin=100, tout=10, cost=0.001),
        entry("retrieve_schema", 0),
        entry("generate_sql", 1, tin=500, tout=50, cost=0.002),
        entry("validate_sql", 1, error="SQL rejected"),
        entry("repair", 1, details={"kind": "guard", "failed_attempt": 1}),
        entry("generate_sql", 2, tin=700, tout=60, cost=0.003),
        entry("validate_sql", 2),
        entry("execute", 2, details={"row_count": 3}),
        entry("respond", 2, tin=80, tout=20, cost=0.0005),
    ]
    s = summarize_trace(trace)
    assert s["latency_ms"] == 90.0
    assert (s["input_tokens"], s["output_tokens"]) == (1380, 140)
    assert s["cost_usd"] == pytest.approx(0.0065)
    assert s["llm_calls"] == 4 and s["attempts"] == 2
    assert s["repaired"] is True and s["first_try_valid"] is False


def test_summary_clean_run_and_unpriced_cost():
    trace = [entry("classify", 0, tin=1, tout=1), entry("generate_sql", 1, tin=1, tout=1, cost=0.1),
             entry("validate_sql", 1), entry("execute", 1, details={"row_count": 2}), entry("respond", 1)]  # fmt: skip
    s = summarize_trace(trace)
    assert s["cost_usd"] is None  # classify had no price
    assert s["repaired"] is False and s["first_try_valid"] is True and s["attempts"] == 1


def test_trace_file_appends_json_lines(tmp_path):
    f = tmp_path / "t.jsonl"
    state = {"sql": "SELECT 1", "answer": "a", "error": None, "summary": {"attempts": 1}, "trace": []}
    append_trace_file(str(f), "q1", state)
    append_trace_file(str(f), "q2", state)
    lines = [json.loads(line) for line in f.read_text().splitlines()]
    assert [r["question"] for r in lines] == ["q1", "q2"] and lines[0]["summary"] == {"attempts": 1}


def test_traced_latency_excludes_throttle_wait():
    import time

    from t2sql.llm import LLMResult
    from t2sql.nodes.common import traced

    @traced("slow")
    def slow(state):
        time.sleep(0.06)  # 60 ms in the node, 50 of them waiting for the client-side throttle
        return {"_llm": LLMResult("x", "p", "m", 1, 1, 5.0, waited_ms=50.0)}

    assert 0 <= slow({})["trace"][0]["latency_ms"] < 40
