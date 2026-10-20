import json
from datetime import date, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from t2sql.api import Runtime, _db_reachable, build_runtime, create_app, json_safe
from t2sql.config import Settings, get_settings
from t2sql.db import ColumnInfo, QueryResult, TableInfo
from t2sql.graph import build_graph
from t2sql.llm import ChatModelLLM, FallbackLLM
from t2sql.nodes.classify import ScopeDecision
from t2sql.nodes.generate import GeneratedSql
from t2sql.schema_index import index_exists
from t2sql.testing import FakeLLM

SETTINGS = Settings(_env_file=None, max_rows=100)
ONE_REPAIR = Settings(_env_file=None, max_rows=100, max_repairs=1)
TABLES = [TableInfo("customer", [ColumnInfo("customer_id", "integer", False, True)])]


def make_client(llm=None, result=None, error=None, settings=SETTINGS, **runtime_kw):
    """TestClient (lifespan on) whose runtime uses a FakeLLM, a fake retriever and executor."""

    def executor(sql):
        if error:
            raise error
        return result

    def factory():
        graph = build_graph(llm, lambda q: TABLES, settings, executor) if llm else None
        return Runtime(graph=graph, settings=settings, **runtime_kw)

    return TestClient(create_app(factory), raise_server_exceptions=False)


def scope(ok=True, lang="en"):
    return ScopeDecision(in_scope=ok, reason="r", language=lang)


def happy_llm():
    return FakeLLM([
        scope(),
        GeneratedSql(sql="SELECT count(*) AS n FROM customer", tables_used=["customer"]),
        "There are 59 customers.",
    ])


def test_ask_happy_path():
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}])) as c:
        r = c.post("/ask", json={"question": "How many customers?"})
    assert r.status_code == 200
    b = r.json()
    assert b["answer"] == "There are 59 customers."
    assert "LIMIT 100" in b["sql"].upper()
    assert b["columns"] == ["n"] and b["rows"] == [{"n": 59}] and b["row_count"] == 1
    assert b["in_scope"] is True and b["language"] == "en" and b["error"] is None
    assert b["attempts"] == []
    s = b["summary"]
    assert s["llm_calls"] == 3 and s["attempts"] == 1 and s["first_try_valid"] and not s["repaired"]
    assert s["input_tokens"] == 30 and s["output_tokens"] == 15 and s["latency_ms"] >= 0
    assert "trace" in b and b["trace"] is None


def test_out_of_scope_is_200():
    with make_client(FakeLLM([scope(False, "es")])) as c:
        r = c.post("/ask", json={"question": "¿Qué clima hace?"})
    assert r.status_code == 200
    b = r.json()
    assert b["in_scope"] is False and b["sql"] is None and b["rows"] == [] and b["row_count"] == 0
    assert b["language"] == "es" and b["answer"].startswith("No puedo responder eso")


def test_error_after_repairs_is_200_with_error():
    llm = FakeLLM([scope(), GeneratedSql(sql="DELETE FROM track"), GeneratedSql(sql="DROP TABLE track")])
    with make_client(llm, settings=ONE_REPAIR) as c:
        r = c.post("/ask", json={"question": "Delete tracks"})
    assert r.status_code == 200
    b = r.json()
    assert "not_select" in b["error"] and "rejected" in b["answer"]
    assert b["rows"] == [] and b["in_scope"] is True
    assert b["attempts"][0]["kind"] == "guard" and b["summary"]["attempts"] == 2


@pytest.mark.parametrize("question", ["", "x" * 501])
def test_invalid_question_is_422(question):
    with make_client(happy_llm()) as c:
        assert c.post("/ask", json={"question": question}).status_code == 422


def test_missing_question_is_422():
    with make_client(happy_llm()) as c:
        assert c.post("/ask", json={}).status_code == 422


def test_max_length_question_is_accepted():
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 1}])) as c:
        assert c.post("/ask", json={"question": "x" * 500}).status_code == 200


def test_no_llm_provider_503_and_health_still_works():
    msg = "No LLM provider is available: missing API key for gemini (GOOGLE_API_KEY)"
    with make_client(None, llm_error=msg) as c:
        r = c.post("/ask", json={"question": "How many customers?"})
        assert r.status_code == 503 and "GOOGLE_API_KEY" in r.json()["detail"]
        h = c.get("/health").json()
    assert h["status"] == "degraded" and h["llm_providers"] == []


def test_build_runtime_without_keys_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        "t2sql.api.get_settings", lambda: Settings(_env_file=None, google_api_key=None, groq_api_key=None)
    )
    rt = build_runtime()
    assert rt.graph is None and "No LLM provider" in rt.llm_error and rt.llm_providers == []


class _ErrChat:
    """Chat model whose every call raises `error` (as the SDK would after its own retries)."""

    def __init__(self, error):
        self.error = error

    def invoke(self, messages):
        raise self.error

    def with_structured_output(self, schema, **kw):
        return self


def failing_llm(error, provider="gemini"):
    # One provider, as built by build_llm: it must still be wrapped in a FallbackLLM.
    return FallbackLLM([ChatModelLLM(provider, "m", _ErrChat(error), parse_retries=0)])


def google_429():
    from google.genai.errors import ClientError
    from langchain_google_genai.chat_models import GoogleRateLimitError

    err = GoogleRateLimitError("429 RESOURCE_EXHAUSTED: quota key=AIza-secret")
    err.__cause__ = ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
    return err


QUOTA_MSG = (
    "LLM quota or rate limit exceeded for gemini; try again later or configure another "
    "provider (T2SQL_LLM_PROVIDERS)"
)


def test_all_providers_failing_at_call_time_is_503():
    with make_client(failing_llm(TimeoutError("gemini: secret-ish detail"))) as c:
        r = c.post("/ask", json={"question": "How many customers?"})
    assert r.status_code == 503 and "secret-ish" not in r.text
    assert r.json() == {"detail": "All LLM providers failed; try again later"}


def test_single_provider_quota_is_503_with_specific_safe_message():
    with make_client(failing_llm(google_429())) as c:
        r = c.post("/ask", json={"question": "How many customers?"})
    assert r.status_code == 503 and r.json() == {"detail": QUOTA_MSG}
    assert "AIza" not in r.text and "RESOURCE_EXHAUSTED" not in r.text


def test_stream_quota_is_error_event_with_same_status_and_detail():
    with make_client(failing_llm(google_429())) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "q"}))
    assert evs[-1] == {"type": "error", "status": 503, "detail": QUOTA_MSG}
    assert [e["type"] for e in evs[:-1]] == ["start", "prompt"]  # classify announced itself first


def test_database_outage_is_503_and_not_repaired():
    from t2sql.db import DatabaseUnavailableError

    llm = FakeLLM([scope(), GeneratedSql(sql="SELECT 1 FROM customer")])
    with make_client(llm, error=DatabaseUnavailableError("connection refused to 10.0.0.5")) as c:
        r = c.post("/ask", json={"question": "How many customers?"})
    assert r.status_code == 503 and r.json() == {"detail": "Database unavailable; try again later"}
    assert "10.0.0.5" not in r.text
    assert llm.responses == []  # exactly classify + one generation: no repair call was attempted


def test_unexpected_exception_is_generic_500(caplog):
    llm = FakeLLM([scope(), GeneratedSql(sql="SELECT 1 FROM customer")])
    with make_client(llm, error=RuntimeError("boom: password=hunter2")) as c:
        r = c.post("/ask", json={"question": "How many customers?"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}
    assert "hunter2" not in r.text and "Traceback" not in r.text
    assert "boom" in caplog.text  # details are logged, not returned


def test_include_trace_flag():
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}])) as c:
        b = c.post("/ask?include_trace=true", json={"question": "How many customers?"}).json()
    nodes = [e["node"] for e in b["trace"]]
    assert nodes == ["classify", "retrieve_schema", "generate_sql", "validate_sql", "execute", "respond"]
    assert b["trace"][0]["llm"]["provider"] == "fake"


def test_values_are_json_safe():
    rows = [{"total": Decimal("12.50"), "d": date(2024, 3, 1), "ts": datetime(2024, 3, 1, 10, 30), "n": None}]
    llm = FakeLLM([scope(), GeneratedSql(sql="SELECT 1 FROM customer"), "ok"])
    with make_client(llm, QueryResult(["total", "d", "ts", "n"], rows)) as c:
        r = c.post("/ask", json={"question": "Totals?"})
    assert r.json()["rows"] == [{"total": 12.5, "d": "2024-03-01", "ts": "2024-03-01T10:30:00", "n": None}]


def test_json_safe_fallbacks():
    out = json_safe({"b": b"\x01\xff", "s": {1}, "x": object.__name__})
    json.dumps(out)
    assert out["b"] == "01ff" and out["s"] == [1]


def test_health_with_fakes():
    ok = dict(db_check=lambda: True, index_check=lambda: True, llm_providers=["gemini"])
    with make_client(happy_llm(), **ok) as c:
        assert c.get("/health").json() == {
            "status": "ok", "database": True, "schema_index": True, "llm_providers": ["gemini"],
            "classifier": "llm", "local_classifier": None, "replay": False,
        }
    down = dict(db_check=lambda: False, index_check=lambda: pytest.fail("not asked when DB is down"))
    with make_client(happy_llm(), **down) as c:
        h = c.get("/health").json()
    assert h["status"] == "degraded" and h["database"] is False and h["schema_index"] is False


@pytest.mark.integration
def test_health_against_real_database():
    s = get_settings()
    rt = Runtime(
        graph=object(),  # not used by /health
        settings=s,
        llm_providers=[],
        db_check=lambda: _db_reachable(s),
        index_check=lambda: index_exists(s.embedding_model, s),
    )
    with TestClient(create_app(lambda: rt)) as c:
        h = c.get("/health").json()
    assert h["database"] is True and h["schema_index"] is True and h["status"] == "ok"


# ---------------------------------------------------------------- streaming and UI


def ndjson(r):
    assert r.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in r.text.splitlines()]


def nodes_of(evs):
    """The finished-node events only (start / prompt / token events filtered out)."""
    return [e for e in evs if e["type"] == "node"]


def test_stream_happy_path_events_in_order():
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}])) as c:
        r = c.post("/ask/stream", json={"question": "How many customers?"})
    evs = ndjson(r)
    # Exact order: every node is announced (start) before it finishes (node); the LLM nodes
    # also show their prompt in between. A FakeLLM does not stream, so there are no tokens.
    assert [(e["type"], e.get("node")) for e in evs] == [
        ("start", "classify"), ("prompt", "classify"), ("node", "classify"),
        ("start", "retrieve_schema"), ("node", "retrieve_schema"),
        ("start", "generate_sql"), ("prompt", "generate_sql"), ("node", "generate_sql"),
        ("start", "validate_sql"), ("node", "validate_sql"),
        ("start", "execute"), ("node", "execute"),
        ("start", "respond"), ("prompt", "respond"), ("node", "respond"),
        ("final", None),
    ]  # fmt: skip
    by = {e["node"]: e for e in nodes_of(evs)}
    assert by["classify"]["data"] == {
        "in_scope": True, "language": "en", "reason": "r", "decided_by": "llm", "p_in": None,
        "local_ms": None, "local_model": None, "local_unavailable": False,
    }  # fmt: skip
    assert by["classify"]["llm"]["model"] == "fake-model" and by["classify"]["cost_usd"] is None
    assert by["retrieve_schema"]["data"]["tables"] == ["customer"]
    assert by["generate_sql"]["data"] == {
        "sql": "SELECT count(*) AS n FROM customer", "tables_used": ["customer"], "assumptions": [],
    }
    v = by["validate_sql"]["data"]
    assert v["ok"] and "LIMIT 100" in v["sql"].upper() and v["code"] is None
    assert v["input_sql"] == "SELECT count(*) AS n FROM customer"
    assert [c["ok"] for c in v["checks"]] == [True] * 7 and v["tables"] == ["customer"]
    assert v["limit_added"] is True and v["limit_enforced"] == 100
    assert by["execute"]["data"] == {
        "ok": True, "error": None, "row_count": 1, "columns": ["n"], "sample": [{"n": 59}],
    }
    assert by["respond"]["data"] == {"answer": "There are 59 customers."}
    assert all(e["attempt"] >= 0 and e["latency_ms"] >= 0 for e in nodes_of(evs))
    f = evs[-1]
    assert f["answer"] == "There are 59 customers." and f["rows"] == [{"n": 59}] and f["row_count"] == 1
    assert f["columns"] == ["n"] and f["summary"]["llm_calls"] == 3


def test_stream_start_and_prompt_events():
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}])) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "How many customers?"}))
    starts = [e for e in evs if e["type"] == "start"]
    assert [(e["node"], e["attempt"]) for e in starts] == [
        ("classify", 0), ("retrieve_schema", 0), ("generate_sql", 1), ("validate_sql", 1),
        ("execute", 1), ("respond", 1),
    ]  # fmt: skip
    assert all(set(e) == {"type", "node", "attempt"} for e in starts)
    prompts = {e["node"]: e for e in evs if e["type"] == "prompt"}
    assert set(prompts) == {"classify", "generate_sql", "respond"}
    gen = prompts["generate_sql"]
    assert gen["attempt"] == 1
    msgs = gen["data"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[1]["content"] == "How many customers?"
    assert "TABLE customer" in msgs[0]["content"]  # the retrieved schema is in what the model sees
    assert gen["data"]["chars"] == sum(len(m["content"]) for m in msgs)


def test_stream_retrieve_payload_for_a_retriever_without_scores():
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}])) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "How many customers?"}))
    d = next(e for e in nodes_of(evs) if e["node"] == "retrieve_schema")["data"]
    assert d["tables"] == d["selected"] == ["customer"] and d["bridges"] == [] and d["bridge_paths"] == []
    assert d["top_k"] is None and d["column_hits"] == {} and "top_scores" not in d
    assert d["ranking"] == [
        {"table": "customer", "score": None, "dense": None, "lexical": None, "selected": True, "bridge": False}
    ]
    assert d["schema"] == [
        {"name": "customer", "columns": [{"name": "customer_id", "type": "integer", "pk": True, "fk": None}]}
    ]
    assert d["schema_context"].startswith("TABLE customer (") and "customer_id integer PRIMARY KEY" in d["schema_context"]


def test_stream_repair_path():
    llm = FakeLLM([
        scope(),
        GeneratedSql(sql="DELETE FROM customer"),
        GeneratedSql(sql="SELECT count(*) AS n FROM customer"),
        "59.",
    ])
    with make_client(llm, QueryResult(["n"], [{"n": 59}])) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "How many customers?"}))
    done = nodes_of(evs)
    assert [e["node"] for e in done] == [
        "classify", "retrieve_schema", "generate_sql", "validate_sql", "repair",
        "generate_sql", "validate_sql", "execute", "respond",
    ]
    bad = done[3]
    assert bad["error"] and bad["data"]["ok"] is False and bad["data"]["sql"] is None
    # The guard explanation: select_only is where it stopped, so the later checks were never reached.
    v = bad["data"]
    assert v["code"] == "not_select" and v["input_sql"] == "DELETE FROM customer"
    assert [(c["name"], c["ok"]) for c in v["checks"]] == [
        ("parse", True), ("single_statement", True), ("select_only", False),
        ("no_forbidden_clauses", None), ("no_forbidden_functions", None),
        ("tables_allowed", None), ("limit", None),
    ]  # fmt: skip
    assert v["tables"] == v["columns"] == [] and v["clauses"] is None and v["limit_enforced"] is None
    assert done[4]["data"] == {
        "kind": "guard", "error": bad["error"], "failed_sql": "DELETE FROM customer",
        "failed_attempt": 1, "attempts_left": 2,  # max_repairs (2) + 1 - attempts (1)
    }  # fmt: skip
    assert [e["attempt"] for e in done if e["node"] == "generate_sql"] == [1, 2]
    # start / prompt of the second generation already carry attempt 2.
    gen_events = [e for e in evs if e["type"] in ("start", "prompt") and e["node"] == "generate_sql"]
    assert [(e["type"], e["attempt"]) for e in gen_events] == [
        ("start", 1), ("prompt", 1), ("start", 2), ("prompt", 2),
    ]  # fmt: skip
    assert evs[-1]["type"] == "final" and evs[-1]["summary"]["repaired"] is True


def test_stream_out_of_scope():
    with make_client(FakeLLM([scope(False, "es")])) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "¿Qué clima hace?"}))
    assert [(e["type"], e.get("node")) for e in evs] == [
        ("start", "classify"), ("prompt", "classify"), ("node", "classify"),
        ("start", "respond_out_of_scope"), ("node", "respond_out_of_scope"), ("final", None),
    ]  # fmt: skip
    done = nodes_of(evs)
    assert done[0]["data"]["in_scope"] is False and done[0]["data"]["language"] == "es"
    assert done[1]["data"]["answer"].startswith("No puedo responder eso")
    assert evs[-1]["in_scope"] is False and evs[-1]["sql"] is None


def test_stream_exhausted_repairs_ends_with_final_error():
    llm = FakeLLM([scope(), GeneratedSql(sql="DELETE FROM track"), GeneratedSql(sql="DROP TABLE track")])
    with make_client(llm, settings=ONE_REPAIR) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "Delete tracks"}))
    assert evs[-2]["node"] == "respond_error" and evs[-1]["type"] == "final" and evs[-1]["error"]


def test_stream_no_llm_is_error_event():
    with make_client(None, llm_error="No LLM provider is available: missing key") as c:
        r = c.post("/ask/stream", json={"question": "How many customers?"})
    assert ndjson(r) == [{"type": "error", "status": 503, "detail": "No LLM provider is available: missing key"}]


def test_stream_errors_become_error_events():
    from t2sql.db import DatabaseUnavailableError

    with make_client(failing_llm(TimeoutError("gemini: secret-ish"))) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "q"}))
    assert evs[-1] == {"type": "error", "status": 503, "detail": "All LLM providers failed; try again later"}
    assert [e["type"] for e in evs[:-1]] == ["start", "prompt"]  # the failure came during classify

    llm = FakeLLM([scope(), GeneratedSql(sql="SELECT 1 FROM customer")])
    with make_client(llm, error=DatabaseUnavailableError("refused 10.0.0.5")) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "q"}))
    assert evs[-1] == {"type": "error", "status": 503, "detail": "Database unavailable; try again later"}
    assert [e["node"] for e in nodes_of(evs)] == ["classify", "retrieve_schema", "generate_sql", "validate_sql"]

    llm = FakeLLM([scope(), GeneratedSql(sql="SELECT 1 FROM customer")])
    with make_client(llm, error=RuntimeError("boom password=hunter2")) as c:
        r = c.post("/ask/stream", json={"question": "q"})
    assert ndjson(r)[-1] == {"type": "error", "status": 500, "detail": "Internal server error"}
    assert "hunter2" not in r.text


def test_stream_invalid_question_is_422():
    with make_client(happy_llm()) as c:
        assert c.post("/ask/stream", json={"question": ""}).status_code == 422


def test_root_serves_the_ui():
    with make_client(None) as c:
        r = c.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "/ask/stream" in r.text and "<title>t2sql</title>" in r.text


# ---------------------------------------------------------------- record and replay


def recording_settings(tmp_path, **kw):
    return Settings(_env_file=None, max_rows=100, **kw)


def test_record_then_replay_gives_the_same_bytes(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("t2sql.replay.time.sleep", sleeps.append)
    rec = recording_settings(tmp_path, record_dir=str(tmp_path))
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}]), settings=rec) as c:
        live = c.post("/ask/stream", json={"question": "How many customers?"})
    assert len(list(tmp_path.glob("*.ndjson"))) == 1 and not list(tmp_path.glob("*.part"))

    # Replay needs no LLM at all (graph is None) and matches the question after normalising.
    rep = recording_settings(tmp_path, replay_dir=str(tmp_path))
    with make_client(None, settings=rep) as c:
        replayed = c.post("/ask/stream", json={"question": "  how MANY   customers? "})
    assert replayed.status_code == 200 and replayed.text == live.text
    evs = ndjson(replayed)
    assert evs[-1]["type"] == "final" and evs[-1]["answer"] == "There are 59 customers."
    assert [e["type"] for e in evs].count("node") == 6
    assert all(0 < g <= rep.replay_max_gap_s for g in sleeps)


def test_replay_missing_recording_falls_through_to_the_graph(tmp_path):
    rep = recording_settings(tmp_path, replay_dir=str(tmp_path))
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}]), settings=rep) as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "How many customers?"}))
    assert evs[-1]["type"] == "final" and evs[-1]["answer"] == "There are 59 customers."
    assert list(tmp_path.iterdir()) == []  # replay alone never records


def test_replay_miss_without_llm_is_still_the_usual_error(tmp_path):
    rep = recording_settings(tmp_path, replay_dir=str(tmp_path))
    with make_client(None, settings=rep, llm_error="no key") as c:
        evs = ndjson(c.post("/ask/stream", json={"question": "never recorded"}))
    assert evs == [{"type": "error", "status": 503, "detail": "no key"}]


def test_failed_run_does_not_replace_a_good_recording(tmp_path):
    rec = recording_settings(tmp_path, record_dir=str(tmp_path))
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}]), settings=rec) as c:
        good = c.post("/ask/stream", json={"question": "How many customers?"}).text
    with make_client(None, settings=rec, llm_error="no key") as c:
        c.post("/ask/stream", json={"question": "How many customers?"})
    rep = recording_settings(tmp_path, replay_dir=str(tmp_path))
    with make_client(None, settings=rep) as c:
        assert c.post("/ask/stream", json={"question": "How many customers?"}).text == good
    assert not list(tmp_path.glob("*.part"))


def test_neither_setting_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with make_client(happy_llm(), QueryResult(["n"], [{"n": 59}])) as c:
        assert ndjson(c.post("/ask/stream", json={"question": "How many customers?"}))[-1]["type"] == "final"
    assert list(tmp_path.iterdir()) == []


def test_health_reports_replay(tmp_path):
    with make_client(None, settings=recording_settings(tmp_path, replay_dir=str(tmp_path))) as c:
        assert c.get("/health").json()["replay"] is True
    with make_client(None, settings=recording_settings(tmp_path, record_dir=str(tmp_path))) as c:
        assert c.get("/health").json()["replay"] is False  # recording alone is not replay
