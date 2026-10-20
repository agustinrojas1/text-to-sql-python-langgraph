"""CLI tests with typer's CliRunner. No network, no LLM, no database: everything is patched."""

import pytest
from typer.testing import CliRunner

from t2sql import cli
from t2sql.config import Settings
from t2sql.db import DatabaseUnavailableError
from t2sql.llm import LLMUnavailableError

runner = CliRunner(env={"COLUMNS": "200"})  # keep rich from wrapping table cells


def trace_entry(node, **kw):
    return {"node": node, "attempt": 1, "latency_ms": 1.0, "llm": None, "cost_usd": None,
            "error": None, "details": None, **kw}  # fmt: skip


SUMMARY = {"attempts": 1, "llm_calls": 3, "input_tokens": 30, "output_tokens": 15, "latency_ms": 12.0,
           "cost_usd": 0.0, "repaired": False, "first_try_valid": True}  # fmt: skip


def ok_state(**kw):
    return {
        "answer": "There are 59 customers.",
        "sql": "SELECT count(*) AS n FROM customer LIMIT 100",
        "columns": ["n"],
        "rows": [{"n": 59}],
        "trace": [
            trace_entry("classify"),
            trace_entry("execute", details={"row_count": 1}),
            trace_entry("respond"),
        ],
        "summary": SUMMARY,
        "error": None,
        **kw,
    }


@pytest.fixture
def patched(monkeypatch):
    """Patch the CLI's collaborators; the returned dict lets a test choose the final state."""
    box = {"state": ok_state(), "raises": None}
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "build_llm", lambda settings: object())
    monkeypatch.setattr(cli, "build_graph", lambda llm, settings=None: object())

    def fake_run(graph, question, settings=None):
        if box["raises"]:
            raise box["raises"]
        return box["state"]

    monkeypatch.setattr(cli, "run_question", fake_run)
    return box


def test_ask_happy_path(patched):
    r = runner.invoke(cli.app, ["ask", "How many customers?"])
    assert r.exit_code == 0
    assert "There are 59 customers." in r.output
    assert "SELECT" in r.output and "customer" in r.output
    assert "│ 59 │" in r.output and "row(s)" in r.output  # caption wraps to the narrow table
    assert "Path:" not in r.output


def test_ask_verbose_shows_path_and_trace_table(patched):
    r = runner.invoke(cli.app, ["ask", "How many customers?", "--verbose"])
    assert r.exit_code == 0
    assert "Path: classify -> execute -> respond" in r.output
    assert "3 LLM call(s)" in r.output and "30/15" in r.output
    assert "row(s)" in r.output


def test_ask_out_of_scope(patched):
    patched["state"] = {
        "answer": "I can't answer that: it is outside the database.",
        "in_scope": False,
        "trace": [trace_entry("classify"), trace_entry("respond_out_of_scope")],
        "summary": SUMMARY,
    }
    r = runner.invoke(cli.app, ["ask", "What's the weather?"])
    assert r.exit_code == 0
    assert "I can't answer that" in r.output and "SELECT" not in r.output


def test_ask_error_after_repairs_exits_1(patched):
    patched["state"] = ok_state(
        answer="I could not produce a valid query.", error="Query failed: boom", rows=None, columns=None
    )
    r = runner.invoke(cli.app, ["ask", "q"])
    assert r.exit_code == 1
    assert "I could not produce a valid query." in r.output


def test_ask_without_provider_exits_2_with_env_example_hint(monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, google_api_key=None, groq_api_key=None))
    r = runner.invoke(cli.app, ["ask", "q"])
    assert r.exit_code == 2
    assert "No LLM provider is available" in r.output and ".env.example" in r.output


def test_ask_database_outage_exits_1_with_clear_message(patched):
    patched["raises"] = DatabaseUnavailableError("connection refused")
    r = runner.invoke(cli.app, ["ask", "q"])
    assert r.exit_code == 1
    assert "Database unavailable: connection refused" in r.output


def test_ask_other_exception_exits_1(patched):
    patched["raises"] = LLMUnavailableError("All LLM providers failed")
    r = runner.invoke(cli.app, ["ask", "q"])
    assert r.exit_code == 1 and "LLMUnavailableError: All LLM providers failed" in r.output


def test_error_messages_are_not_interpreted_as_rich_markup(patched):
    # Postgres can quote things like [bold] or [/red]; unescaped, rich would swallow or crash on them.
    patched["raises"] = RuntimeError('syntax error at or near "[/red]" in [bold]x[/bold]')
    r = runner.invoke(cli.app, ["ask", "q"])
    assert r.exit_code == 1
    assert '"[/red]"' in r.output and "[bold]x[/bold]" in r.output


def test_answer_and_trace_errors_are_escaped(patched):
    patched["state"] = ok_state(
        answer="Result for [genre] and [/bold]",
        trace=[trace_entry("execute", error="Query failed: column [x] missing"), trace_entry("respond")],
    )
    r = runner.invoke(cli.app, ["ask", "q", "-v"])
    assert r.exit_code == 0
    assert "[genre]" in r.output and "[/bold]" in r.output and "[x]" in r.output


def test_index_wiring(monkeypatch):
    seen = {}

    class FakeEmbedder:
        name = "fake-embedder"

        def __init__(self, model, cache_dir=None):
            seen["model"] = model

    def fake_build_index(embedder, settings, table, keywords):
        seen.update(embedder=embedder, table=table, keywords=keywords)
        return 75

    settings = Settings(_env_file=None, embedding_model="some/model")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "FastEmbedEmbedder", FakeEmbedder)
    monkeypatch.setattr(cli, "build_index", fake_build_index)
    r = runner.invoke(cli.app, ["index", "--table", "my_docs", "--keywords"])
    assert r.exit_code == 0 and "Indexed 75 documents into t2sql_meta.my_docs." in r.output
    assert seen["model"] == "some/model" and isinstance(seen["embedder"], FakeEmbedder)
    assert seen["table"] == "my_docs" and seen["keywords"] is True


def test_index_failure_exits_1_and_escapes(monkeypatch):
    class FakeEmbedder:
        name = "fake"

        def __init__(self, model, cache_dir=None):
            pass

    def boom(*a, **k):
        raise RuntimeError("bad [bold]thing")

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "FastEmbedEmbedder", FakeEmbedder)
    monkeypatch.setattr(cli, "build_index", boom)
    r = runner.invoke(cli.app, ["index"])
    assert r.exit_code == 1 and "RuntimeError: bad [bold]thing" in r.output
