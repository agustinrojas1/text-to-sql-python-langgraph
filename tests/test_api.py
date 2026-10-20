import json
from datetime import date, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from t2sql.api import Runtime, _db_reachable, build_runtime, create_app, json_safe
from t2sql.config import Settings, get_settings
from t2sql.db import ColumnInfo, QueryResult, TableInfo
from t2sql.graph import build_graph
from t2sql.llm import LLMUnavailableError
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


def test_all_providers_failing_at_call_time_is_503():
    llm = FakeLLM([LLMUnavailableError("All LLM providers failed:\ngemini: secret-ish detail")])
    with make_client(llm) as c:
        r = c.post("/ask", json={"question": "How many customers?"})
    assert r.status_code == 503 and "secret-ish" not in r.text


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
