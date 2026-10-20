"""Per-request classifier mode / local model: validation, routing, URLs, /health, CLI, replay keys."""

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from t2sql import cli
from t2sql.api import Runtime, create_app, stream_lines
from t2sql.config import Settings
from t2sql.db import ColumnInfo, QueryResult, TableInfo
from t2sql.graph import build_graph, run_question, stream_question
from t2sql.local_classifier import D1Scorer, LocalClassifierUnavailable, ScorerPool, Tev1Scorer, build_scorer
from t2sql.nodes.classify import ScopeDecision
from t2sql.nodes.generate import GeneratedSql
from t2sql.replay import load_recording, record_lines, recording_path, variant
from t2sql.testing import FakeLLM
from tests.test_classifier_modes import FakeScorer

TABLES = [TableInfo("customer", [ColumnInfo("customer_id", "integer", False, True)])]
RESULT = QueryResult(["n"], [{"n": 59}])
runner = CliRunner()


class Named(FakeScorer):
    def __init__(self, model, *outcomes, up=True):
        super().__init__(*outcomes, up=up)
        self.model = model


def cfg(mode="llm", **kw):
    return Settings(_env_file=None, classifier=mode, cascade_low=0.2, cascade_high=0.8, **kw)


def cloud(ok=True):
    return ScopeDecision(in_scope=ok, reason="cloud says so", language="en")


def sql_steps():
    return [GeneratedSql(sql="SELECT count(*) AS n FROM customer", tables_used=["customer"]), "59 customers."]


def graph(llm, settings, **scorers):
    return build_graph(llm, lambda _: TABLES, settings, lambda sql: RESULT, scorers=scorers)


def details(state):
    return state["trace"][0]["details"]


# ---------------------------------------------------------------- request validation


def client(g, settings, **kw):
    return TestClient(create_app(lambda: Runtime(graph=g, settings=settings, **kw)), raise_server_exceptions=False)


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
@pytest.mark.parametrize("body", [{"classifier": "magic"}, {"local_model": "gpt"}, {"classifier": 3}])
def test_invalid_overrides_are_422(path, body):
    with client(graph(FakeLLM([]), cfg()), cfg()) as c:
        assert c.post(path, json={"question": "q", **body}).status_code == 422


def test_overrides_are_optional_and_null_means_server_default():
    t, d = Named("tev1", 0.01), Named("d1")
    s = cfg("cascade")
    with client(graph(FakeLLM([]), s, tev1=t, d1=d), s) as c:
        r = c.post("/ask?include_trace=true", json={"question": "q", "classifier": None, "local_model": None})
    assert r.status_code == 200 and r.json()["trace"][0]["details"]["local_model"] == "tev1" and t.questions == ["q"]


# ---------------------------------------------------------------- routing per request


def test_llm_override_on_a_cascade_server_never_asks_the_local_model():
    t = Named("tev1")  # would raise IndexError if asked
    s = cfg("cascade")
    state = run_question(graph(FakeLLM([cloud(), *sql_steps()]), s, tev1=t), "q", s, classifier="llm")
    assert t.questions == [] and details(state) == {"classifier": "llm", "decided_by": "llm"}


def test_local_override_on_an_llm_server_decides_locally_with_the_requested_model():
    t, d = Named("tev1"), Named("d1", 0.01)
    s = cfg("llm")
    llm = FakeLLM([])
    state = run_question(graph(llm, s, tev1=t, d1=d), "capital of France?", s, classifier="local", local_model="d1")
    assert d.questions == ["capital of France?"] and t.questions == [] and llm.calls == []
    assert details(state)["decided_by"] == "local" and details(state)["local_model"] == "d1"
    assert not state["in_scope"] and details(state)["classifier"] == "local"


@pytest.mark.parametrize(
    "p, decided",
    [(0.05, "local"), (0.5, "llm"), (0.95, "local")],
)
def test_cascade_override_uses_the_band(p, decided):
    t = Named("tev1", p)
    s = cfg("llm")
    answers = {0.05: [], 0.5: [cloud(False)], 0.95: sql_steps()}[p]
    state = run_question(graph(FakeLLM(answers), s, tev1=t), "q", s, classifier="cascade")
    assert details(state)["decided_by"] == decided and details(state)["classifier"] == "cascade"


def test_local_model_override_picks_the_other_scorer_in_the_same_graph():
    t, d = Named("tev1", 0.9), Named("d1", 0.9)
    s = cfg("local")
    g = graph(FakeLLM(sql_steps() + sql_steps()), s, tev1=t, d1=d)  # one compiled graph, two runs
    run_question(g, "first")
    run_question(g, "second", s, local_model="d1")
    assert t.questions == ["first"] and d.questions == ["second"]


def test_unavailable_requested_model_falls_back_to_the_cloud_and_says_which():
    d = Named("d1", LocalClassifierUnavailable("refused"))
    s = cfg("llm")
    llm = FakeLLM([cloud(), *sql_steps()])
    state = run_question(graph(llm, s, d1=d), "q", s, classifier="cascade", local_model="d1")
    assert state["in_scope"] and len(llm.calls) == 3
    assert details(state) == {"classifier": "cascade", "decided_by": "llm", "local_model": "d1", "local_unavailable": True}


def test_stream_event_reports_mode_and_model():
    d = Named("d1", 0.99)
    s = cfg("llm")
    g = graph(FakeLLM(sql_steps()), s, d1=d)
    ev = next(e for e in stream_question(g, "q", s, "local", "d1") if e["type"] == "node" and e["node"] == "classify")
    assert ev["data"]["classifier"] == "local" and ev["data"]["local_model"] == "d1" and ev["data"]["decided_by"] == "local"


def test_endpoints_thread_the_override_through():
    d = Named("d1", 0.01, 0.01)
    s = cfg("llm")
    body = {"question": "capital of France?", "classifier": "local", "local_model": "d1"}
    with client(graph(FakeLLM([]), s, d1=d), s) as c:
        r = c.post("/ask", json=body)
        assert r.status_code == 200 and r.json()["in_scope"] is False and r.json()["summary"]["llm_calls"] == 0
        lines = [json.loads(x) for x in c.post("/ask/stream", json=body).text.splitlines()]
    cls = next(e for e in lines if e["type"] == "node" and e["node"] == "classify")
    assert cls["data"]["decided_by"] == "local" and cls["data"]["local_model"] == "d1"
    assert d.questions == ["capital of France?"] * 2


# ---------------------------------------------------------------- one URL per model


def test_default_urls_and_the_legacy_override():
    s = Settings(_env_file=None)
    assert (s.local_url("tev1"), s.local_url("d1")) == ("http://127.0.0.1:8090", "http://127.0.0.1:8091")
    s = Settings(_env_file=None, tev1_url="http://a:1", d1_url="http://b:2")
    assert (s.local_url("tev1"), s.local_url("d1")) == ("http://a:1", "http://b:2")
    # T2SQL_LOCAL_CLASSIFIER_URL overrides the configured default model only.
    s = Settings(_env_file=None, local_classifier_model="d1", local_classifier_url="http://c:3", tev1_url="http://a:1")
    assert (s.local_url("d1"), s.local_url("tev1")) == ("http://c:3", "http://a:1")


def test_legacy_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("T2SQL_LOCAL_CLASSIFIER_URL", "http://legacy:9")
    s = Settings(_env_file=None)
    assert s.local_url("tev1") == "http://legacy:9" and s.local_url("d1") == "http://127.0.0.1:8091"


def test_scorer_pool_builds_one_scorer_per_model_with_its_own_url():
    calls = []

    def http(url, payload, timeout):
        calls.append(url)
        return {"status": "ok"}

    pool = ScorerPool(Settings(_env_file=None, tev1_url="http://t:1", d1_url="http://d:2/"), http)
    t, d = pool.get("tev1"), pool.get("d1")
    assert isinstance(t, Tev1Scorer) and isinstance(d, D1Scorer) and pool.get("tev1") is t  # cached
    assert t.available() and d.available() and calls == ["http://t:1/health", "http://d:2/health"]
    assert build_scorer(Settings(_env_file=None, classifier="cascade", local_classifier_model="d1")).model == "d1"


def test_availability_is_per_model():
    def http(url, payload, timeout):
        if ":8091" in url:
            raise LocalClassifierUnavailable("refused")
        return {"status": "ok"}

    assert ScorerPool(Settings(_env_file=None), http).availability() == {"tev1": True, "d1": False}


def test_health_reports_local_models():
    s = cfg("llm")
    with client(None, s, local_models_check=lambda: {"tev1": True, "d1": False}) as c:
        h = c.get("/health").json()
    assert h["local_models"] == {"tev1": True, "d1": False} and h["local_model"] == "tev1"
    assert h["status"] == "degraded"  # no graph here; the local models alone never change the status
    with client(graph(FakeLLM([]), s), s, local_models_check=lambda: {"tev1": False, "d1": False}) as c:
        assert c.get("/health").json()["status"] == "ok"


# ---------------------------------------------------------------- CLI


@pytest.fixture
def seen(monkeypatch):
    box = {}
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "build_llm", lambda settings: object())
    monkeypatch.setattr(cli, "build_graph", lambda llm, settings=None: object())

    def fake_run(graph, question, settings=None, **kw):
        box["kw"] = kw
        return {"answer": "x", "error": None, "trace": [], "summary": {}}

    monkeypatch.setattr(cli, "run_question", fake_run)
    return box


def test_ask_passes_the_flags(seen):
    r = runner.invoke(cli.app, ["ask", "q", "--classifier", "cascade", "--local-model", "d1"])
    assert r.exit_code == 0, r.output
    assert seen["kw"] == {"classifier": "cascade", "local_model": "d1"}
    runner.invoke(cli.app, ["ask", "q"])
    assert seen["kw"] == {}  # nothing given: server settings


@pytest.mark.parametrize("args", [["--classifier", "magic"], ["--local-model", "gpt"]])
def test_ask_rejects_unknown_values(seen, args):
    r = runner.invoke(cli.app, ["ask", "q", *args])
    assert r.exit_code == 2 and "kw" not in seen


def test_record_passes_the_flags_and_names_the_variant_file(monkeypatch, tmp_path):
    got = []

    def fake_lines(runtime, q, **kw):
        got.append(kw)
        yield '{"type": "final", "in_scope": true, "error": null, "row_count": 3}\n'

    class RT:
        settings = Settings(_env_file=None)

    monkeypatch.setattr(cli, "build_runtime", lambda: RT())
    monkeypatch.setattr(cli, "stream_lines", fake_lines)
    r = runner.invoke(cli.app, ["record", "one", "--dir", str(tmp_path), "--classifier", "cascade", "--local-model", "d1"])
    assert r.exit_code == 0, r.output
    assert got == [{"classifier": "cascade", "local_model": "d1"}] and "cascade-d1.ndjson" in r.output.replace("\n", "")


# ---------------------------------------------------------------- replay keys

FINAL = json.dumps({"type": "final", "answer": "ok"}) + "\n"


def recorded(directory, question, mode, model, answer):
    line = json.dumps({"type": "final", "answer": answer}) + "\n"
    list(record_lines(iter([line]), directory, question, mode, model))


def test_variant_keys():
    assert variant("llm", "tev1") == variant("llm", "d1") == "llm"  # the model is irrelevant without a local classifier
    assert variant("cascade", "tev1") == "cascade-tev1" != variant("cascade", "d1") != variant("local", "d1")


def test_recordings_are_keyed_by_mode_and_model(tmp_path):
    recorded(tmp_path, "Q", "cascade", "tev1", "A")
    assert load_recording(tmp_path, "q", "cascade", "tev1")[-1][1].count("A") == 1
    for mode, model in [("llm", "tev1"), ("local", "tev1"), ("cascade", "d1")]:
        assert load_recording(tmp_path, "q", mode, model) is None
    header = json.loads(recording_path(tmp_path, "q", "cascade-tev1").read_text().splitlines()[0])
    assert header["classifier"] == "cascade" and header["local_model"] == "tev1"


def test_a_header_that_disagrees_with_its_file_name_is_ignored(tmp_path):
    recorded(tmp_path, "q", "local", "d1", "A")
    recording_path(tmp_path, "q", "local-d1").rename(recording_path(tmp_path, "q", "cascade-d1"))
    assert load_recording(tmp_path, "q", "cascade", "d1") is None


def test_legacy_recordings_replay_only_for_the_server_default(tmp_path):
    legacy = recording_path(tmp_path, "q")
    legacy.write_text(json.dumps({"question": "q", "format": 1}) + "\n" + json.dumps({"t_ms": 0, "line": FINAL}) + "\n")
    default = ("cascade", "tev1")
    assert load_recording(tmp_path, "q", "cascade", "tev1", default) is not None
    assert load_recording(tmp_path, "q", "llm", "tev1", default) is None
    assert load_recording(tmp_path, "q", "cascade", "d1", default) is None
    assert load_recording(tmp_path, "q", "cascade", "tev1") is None  # no default given: legacy file ignored


def test_stream_lines_replays_the_matching_mode_only(tmp_path):
    recorded(tmp_path, "q", "cascade", "tev1", "from cascade")
    s = cfg("llm", replay_dir=str(tmp_path), replay_max_gap_s=0)
    llm = FakeLLM([cloud(False)])
    rt = Runtime(graph=graph(llm, s), settings=s)
    # llm mode (server default) has no recording: it runs live; cascade does: it replays.
    live = list(stream_lines(rt, "q"))
    assert llm.calls and json.loads(live[-1])["type"] == "final"
    replayed = list(stream_lines(rt, "q", "cascade", "tev1"))
    assert replayed == [json.dumps({"type": "final", "answer": "from cascade"}) + "\n"]
    assert len(llm.calls) == 1  # the replay did not touch the model


def test_stream_lines_records_under_the_effective_variant(tmp_path):
    s = cfg("cascade", record_dir=str(tmp_path))
    d = Named("d1", 0.01)
    rt = Runtime(graph=graph(FakeLLM([]), s, d1=d), settings=s)
    list(stream_lines(rt, "capital of France?", None, "d1"))  # server mode (cascade), request model d1
    assert recording_path(tmp_path, "capital of France?", "cascade-d1").is_file()
