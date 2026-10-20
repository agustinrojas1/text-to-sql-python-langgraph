"""classify node in llm / local / cascade mode, its trace, stream event, /health, CLI and evals plumbing."""

import json
import logging

import pytest
from rich.console import Console
from fastapi.testclient import TestClient

from evals import run as ev
from t2sql import cli
from t2sql.api import Runtime, create_app
from t2sql.config import Settings
from t2sql.db import ColumnInfo, QueryResult, TableInfo
from t2sql.graph import build_graph, run_question, stream_question
from t2sql.local_classifier import LocalClassifierUnavailable, ScopeScore
from t2sql.nodes.classify import ScopeDecision
from t2sql.nodes.generate import GeneratedSql
from t2sql.testing import FakeLLM

TABLES = [TableInfo("customer", [ColumnInfo("customer_id", "integer", False, True)])]
RESULT = QueryResult(["n"], [{"n": 59}])


class FakeScorer:
    """Scripted P(in scope) per call (or an exception)."""

    model = "tev1"

    def __init__(self, *outcomes, up=True):
        self.outcomes, self.questions, self.up = list(outcomes), [], up

    def score(self, question):
        self.questions.append(question)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return ScopeScore(p_in=o, latency_ms=180.0, model=self.model, detail=self.detail)

    detail = None

    def available(self):
        return self.up


def cfg(mode, **kw):
    return Settings(_env_file=None, classifier=mode, cascade_low=0.2, cascade_high=0.8, **kw)


def llm_scope(ok=True, lang="en"):
    return ScopeDecision(in_scope=ok, reason="cloud says so", language=lang)


def sql_steps():
    return [GeneratedSql(sql="SELECT count(*) AS n FROM customer", tables_used=["customer"]), "59 customers."]


def run(q, mode, scorer, llm_responses, **kw):
    llm = FakeLLM(llm_responses)
    graph = build_graph(llm, lambda _: TABLES, cfg(mode, **kw), lambda sql: RESULT, scorer=scorer)
    return run_question(graph, q), llm


def classify_entry(state):
    return state["trace"][0]


# ---------------------------------------------------------------- settings


def test_defaults_keep_the_cloud_llm_only():
    s = Settings(_env_file=None)
    assert s.classifier == "llm" and s.local_classifier_url == "http://127.0.0.1:8090"
    assert s.local_classifier_timeout_s == 2.0 and s.local_threshold == 0.5
    assert (s.cascade_low, s.cascade_high) == (0.10, 0.93) and s.local_classifier_model == "tev1"


def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("T2SQL_CLASSIFIER", "cascade")
    monkeypatch.setenv("T2SQL_LOCAL_CLASSIFIER_MODEL", "tev1")
    monkeypatch.setenv("T2SQL_CASCADE_LOW", "0.1")
    monkeypatch.setenv("T2SQL_CASCADE_HIGH", "0.9")
    s = Settings(_env_file=None)
    assert (s.classifier, s.local_classifier_model, s.cascade_low, s.cascade_high) == ("cascade", "tev1", 0.1, 0.9)


@pytest.mark.parametrize("kw", [{"classifier": "magic"}, {"local_classifier_model": "x"}, {"cascade_low": 0.9, "cascade_high": 0.1}, {"cascade_high": 1.5}])
def test_invalid_classifier_settings_are_rejected(kw):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **kw)


# ---------------------------------------------------------------- llm mode (unchanged behaviour)


def test_llm_mode_ignores_a_scorer_and_matches_the_old_behaviour():
    scorer = FakeScorer()  # would raise IndexError if asked
    s, llm = run("How many customers?", "llm", scorer, [llm_scope(), *sql_steps()])
    assert scorer.questions == [] and len(llm.calls) == 3
    assert classify_entry(s)["details"] == {"classifier": "llm", "decided_by": "llm"} and classify_entry(s)["llm"] is not None
    assert s["scope_reason"] == "cloud says so"


# ---------------------------------------------------------------- local mode


@pytest.mark.parametrize("p, in_scope", [(0.97, True), (0.5, True), (0.49, False), (0.01, False)])
def test_local_mode_applies_the_threshold_without_calling_the_llm(p, in_scope):
    s, llm = run("How many customers?", "local", FakeScorer(p), sql_steps() if in_scope else [])
    e = classify_entry(s)
    assert s["in_scope"] is in_scope
    assert e["llm"] is None and e["cost_usd"] is None
    assert e["details"] == {"classifier": "local", "decided_by": "local", "p_in": round(p, 4), "local_ms": 180.0, "local_model": "tev1"}
    assert s["scope_reason"].startswith("local classifier tev1")
    assert s["summary"]["llm_calls"] == (2 if in_scope else 0)  # classify costs nothing
    assert s["trace"][-1]["node"] == ("respond" if in_scope else "respond_out_of_scope")


def test_local_mode_uses_the_configured_threshold():
    s, _ = run("q", "local", FakeScorer(0.6), [], local_threshold=0.7)
    assert s["in_scope"] is False


def test_local_decision_gets_the_language_from_the_heuristic():
    s, _ = run("¿Cuál es el clima en Madrid?", "local", FakeScorer(0.02), [])
    assert s["language"] == "es" and "No puedo responder" in s["answer"]
    s, _ = run("What is the weather in Madrid?", "local", FakeScorer(0.02), [])
    assert s["language"] == "en" and s["answer"].startswith("I can't answer")
    s, _ = run("Mostrame los clientes de Brasil", "local", FakeScorer(0.99), sql_steps())
    assert s["language"] == "es"


# ---------------------------------------------------------------- cascade


def test_cascade_routing_bands():
    # confident out of scope, confident in scope: local, no LLM call at classify
    s, llm = run("capital of France?", "cascade", FakeScorer(0.05), [])
    assert not s["in_scope"] and classify_entry(s)["details"]["decided_by"] == "local" and llm.calls == []
    s, llm = run("How many customers?", "cascade", FakeScorer(0.95), sql_steps())
    assert s["in_scope"] and classify_entry(s)["details"]["decided_by"] == "local" and len(llm.calls) == 2
    # unsure: the cloud LLM decides (even against the local lean), and p_in is still recorded
    s, llm = run("Who is the best artist?", "cascade", FakeScorer(0.5), [llm_scope(False)])
    e = classify_entry(s)
    assert not s["in_scope"] and e["details"] == {"classifier": "cascade", "decided_by": "llm", "p_in": 0.5, "local_ms": 180.0, "local_model": "tev1"}
    assert e["llm"]["model"] == "fake-model" and s["scope_reason"] == "cloud says so"
    assert len(llm.calls) == 1 and llm.calls[0].schema is ScopeDecision


@pytest.mark.parametrize("p, decided", [(0.1999, "local"), (0.2, "llm"), (0.5, "llm"), (0.8, "llm"), (0.8001, "local")])
def test_cascade_band_edges(p, decided):
    local_in = p > 0.8
    llm_answers = sql_steps() if local_in else [llm_scope(False)]  # decided locally: only generate + respond call it
    s, _ = run("q", "cascade", FakeScorer(p), llm_answers)
    assert classify_entry(s)["details"]["decided_by"] == decided


@pytest.mark.parametrize("mode", ["local", "cascade"])
def test_unavailable_scorer_falls_back_to_the_llm(mode):
    scorer = FakeScorer(LocalClassifierUnavailable("refused"))
    s, llm = run("How many customers?", mode, scorer, [llm_scope(), *sql_steps()])
    assert s["in_scope"] and s["answer"] == "59 customers."
    assert classify_entry(s)["details"] == {"classifier": mode, "decided_by": "llm", "local_unavailable": True, "local_model": "tev1"}
    assert len(llm.calls) == 3


def test_unavailable_warns_once_per_outage(caplog):
    err = LocalClassifierUnavailable("refused")
    scorer = FakeScorer(err, err, 0.01, err)
    graph = build_graph(FakeLLM([llm_scope(False)] * 3), lambda _: TABLES, cfg("cascade"), lambda s: RESULT, scorer=scorer)
    with caplog.at_level(logging.WARNING, logger="t2sql.nodes.classify"):
        for _ in range(4):
            run_question(graph, "q")
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 2 and "refused" in msgs[0]  # outage, recovery, outage again


def test_cascade_without_a_scorer_object_just_uses_the_llm():
    s, _ = run("q", "cascade", None, [llm_scope(False)])
    assert classify_entry(s)["details"]["decided_by"] == "llm"


def test_build_graph_creates_the_scorer_from_settings():
    graph = build_graph(FakeLLM([]), lambda _: TABLES, cfg("cascade", local_classifier_url="http://127.0.0.1:1"), lambda s: RESULT)
    # Nothing listens on port 1: the real scorer fails and the node falls back to the LLM.
    graph.get_graph()  # compiles fine
    llm = FakeLLM([llm_scope(False)])
    g = build_graph(llm, lambda _: TABLES, cfg("cascade", local_classifier_url="http://127.0.0.1:1"), lambda s: RESULT)
    s = run_question(g, "q")
    assert classify_entry(s)["details"]["local_unavailable"] is True and len(llm.calls) == 1


# ---------------------------------------------------------------- stream event, CLI, API


def stream_classify(mode, scorer, llm_responses):
    g = build_graph(FakeLLM(llm_responses), lambda _: TABLES, cfg(mode), lambda s: RESULT, scorer=scorer)
    return next(e for e in stream_question(g, "How many customers?") if e["type"] == "node" and e["node"] == "classify")


def test_stream_event_carries_the_local_decision():
    ev_ = stream_classify("cascade", FakeScorer(0.97), sql_steps())
    assert ev_["data"] == {
        "in_scope": True, "language": "en", "reason": "local classifier tev1: P(in scope)=0.97",
        "classifier": "cascade", "decided_by": "local", "p_in": 0.97, "local_ms": 180.0, "local_model": "tev1",
        "local_unavailable": False,
    }  # fmt: skip
    assert ev_["llm"] is None and ev_["cost_usd"] is None
    ev_ = stream_classify("cascade", FakeScorer(0.5), [llm_scope(), *sql_steps()])
    assert ev_["data"]["decided_by"] == "llm" and ev_["data"]["p_in"] == 0.5 and ev_["llm"] is not None
    ev_ = stream_classify("local", FakeScorer(LocalClassifierUnavailable("x")), [llm_scope(), *sql_steps()])
    assert ev_["data"]["local_unavailable"] is True and ev_["data"]["p_in"] is None


def test_cli_detail_and_trace_table(capsys, monkeypatch):
    monkeypatch.setattr(cli, "console", Console(width=250))
    local = {"decided_by": "local", "p_in": 0.97, "local_ms": 180.4, "local_model": "tev1"}
    assert cli._detail({"node": "classify", "details": local}) == "decided locally (local tev1 p=0.97, 180 ms)"
    unsure = {**local, "decided_by": "llm", "p_in": 0.5}
    assert cli._detail({"node": "classify", "details": unsure}) == "decided by the cloud LLM (local tev1 p=0.50, 180 ms)"
    down = {"decided_by": "llm", "local_unavailable": True}
    assert "local classifier unavailable" in cli._detail({"node": "classify", "details": down})
    assert cli._detail({"node": "classify", "details": {"decided_by": "llm"}}) == "decided by the cloud LLM"
    s, _ = run("How many customers?", "cascade", FakeScorer(0.97), sql_steps())
    cli._print_trace(s["trace"], s["summary"])
    out = capsys.readouterr().out
    assert "local/tev1" in out and "decided locally (local tev1 p=0.97, 180 ms)" in out and "2 LLM call(s)" in out


def client(mode, scorer, llm_responses=(), **kw):
    s = cfg(mode)
    graph = build_graph(FakeLLM(list(llm_responses)), lambda _: TABLES, s, lambda sql: RESULT, scorer=scorer)
    rt = Runtime(graph=graph, settings=s, local_check=scorer.available if scorer else None, **kw)
    return TestClient(create_app(lambda: rt), raise_server_exceptions=False)


def test_ask_trace_shows_who_decided():
    with client("cascade", FakeScorer(0.01)) as c:
        b = c.post("/ask?include_trace=true", json={"question": "Capital of France?"}).json()
    assert b["in_scope"] is False and b["summary"]["llm_calls"] == 0 and b["summary"]["cost_usd"] == 0
    assert b["trace"][0]["details"]["decided_by"] == "local" and b["trace"][0]["details"]["p_in"] == 0.01


def test_health_reports_classifier_mode_and_reachability():
    with client("llm", None) as c:
        h = c.get("/health").json()
    assert h["classifier"] == "llm" and h["local_classifier"] is None
    with client("cascade", FakeScorer(up=True)) as c:
        h = c.get("/health").json()
    assert h["classifier"] == "cascade" and h["local_classifier"] is True and h["status"] == "ok"
    with client("local", FakeScorer(up=False)) as c:
        h = c.get("/health").json()
    # A down local model does not degrade the service: the cloud LLM takes over.
    assert h["classifier"] == "local" and h["local_classifier"] is False and h["status"] == "ok"


# ---------------------------------------------------------------- evals


def rec(by, p=None, calls=2):
    return {
        "id": "x", "level": "easy", "classified_by": by, "local_p_in": p, "exec_match": True, "exec_strict": True,
        "rejected": False, "summary": {"latency_ms": 1.0, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0,
                                       "attempts": 1, "first_try_valid": True, "llm_calls": calls},
    }  # fmt: skip


def test_aggregate_and_markdown_report_local_vs_cloud():
    recs = [rec("local", 0.95, 2), rec("local", 0.01, 0), rec("llm", 0.5, 3), rec("llm", None, 3)]
    a = ev.aggregate(recs)
    assert a["n_decided_locally"] == 2 and a["decided_locally"] == 0.5
    assert a["cloud_calls_per_question"] == pytest.approx(2.0)
    md = ev.agent_markdown({"fake/oracle": recs})
    assert "Scope decided locally" in md and "| fake/oracle | 2/4 (50.0%) | 2/4 | 2.00 |" in md
    assert "Scope decided locally" not in ev.agent_markdown({"m": [rec("llm", None)]})  # cloud-only runs: no extra table


def test_classifier_flag_reaches_settings_and_meta(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(ev, "build_retriever", lambda settings, index_table=None: lambda q: TABLES)
    monkeypatch.setattr(ev, "make_runs", lambda args, settings, items, retriever: [])

    def fake_run_agent(runs, items, settings, sleep_s):
        seen["settings"] = settings
        return {}

    monkeypatch.setattr(ev, "run_agent", fake_run_agent)
    out = tmp_path / "r.json"
    assert ev.main(["--fake", "--classifier", "cascade", "--limit", "1", "--out", str(out)]) == 0
    assert seen["settings"].classifier == "cascade"
    meta = json.loads(out.read_text())["meta"]
    assert meta["classifier"] == "cascade" and meta["local_classifier"][0] == seen["settings"].local_classifier_model
    with pytest.raises(SystemExit):
        ev.main(["--fake", "--classifier", "nope"])


# ---------------------------------------------------------------- local prompt details


DETAIL = {"endpoint": "http://x/completion", "params": {"n_predict": 1}, "requests": [{"order": "in_first", "prompt": "P"}]}


def with_detail(scorer):
    scorer.detail = DETAIL
    return scorer


def test_local_details_present_when_decided_locally_and_when_escalated():
    for mode, p, steps, by in [("cascade", 0.97, sql_steps(), "local"), ("cascade", 0.5, [llm_scope(), *sql_steps()], "llm"),
                               ("local", 0.9, sql_steps(), "local")]:
        ev_ = stream_classify(mode, with_detail(FakeScorer(p)), steps)
        loc = ev_["data"]["local"]
        assert ev_["data"]["decided_by"] == by and loc["requests"][0]["prompt"] == "P" and loc["mode"] == mode
        assert (loc["threshold"] if mode == "local" else (loc["low"], loc["high"])) == (0.5 if mode == "local" else (0.2, 0.8))
    s, _ = run("How many customers?", "cascade", with_detail(FakeScorer(0.5)), [llm_scope(), *sql_steps()])
    assert classify_entry(s)["details"]["local"]["endpoint"] == "http://x/completion"


def test_local_details_absent_in_llm_mode_and_when_unavailable():
    ev_ = stream_classify("llm", with_detail(FakeScorer()), [llm_scope(), *sql_steps()])
    assert "local" not in ev_["data"]
    ev_ = stream_classify("cascade", FakeScorer(LocalClassifierUnavailable("x")), [llm_scope(), *sql_steps()])
    assert "local" not in ev_["data"]
