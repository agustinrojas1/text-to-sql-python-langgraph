import pytest

from t2sql.config import Settings
from t2sql.db import ColumnInfo, QueryError, QueryResult, TableInfo
from t2sql.graph import build_graph, run_question
from t2sql.nodes.classify import ScopeDecision
from t2sql.nodes.generate import GeneratedSql
from t2sql.retrieval import format_schema
from t2sql.testing import FakeLLM

SETTINGS = Settings(_env_file=None, max_rows=100)
TABLES = [TableInfo("customer", [ColumnInfo("customer_id", "integer", False, True)])]


def retriever(question):
    return TABLES


class Executor:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def __call__(self, sql):
        self.calls.append(sql)
        if self.error:
            raise self.error
        return self.result


def scope(ok=True, lang="en"):
    return ScopeDecision(in_scope=ok, reason="r", language=lang)


def path(state):
    return [e["node"] for e in state["trace"]]


def run(llm, executor, q="How many customers?", settings=SETTINGS):
    graph = build_graph(llm, retriever, settings, executor)
    return run_question(graph, q)


def test_happy_path():
    llm = FakeLLM([
        scope(),
        GeneratedSql(sql="SELECT count(*) AS n FROM customer", tables_used=["customer"]),
        "There are 59 customers.",
    ])
    ex = Executor(QueryResult(["n"], [{"n": 59}]))
    s = run(llm, ex)
    assert path(s) == ["classify", "retrieve_schema", "generate_sql", "validate_sql", "execute", "respond"]
    assert "LIMIT 100" in s["sql"].upper() and ex.calls == [s["sql"]]
    assert s["answer"] == "There are 59 customers."
    assert s["rows"] == [{"n": 59}] and s["columns"] == ["n"] and s["attempts"] == 1
    assert s["error"] is None
    # LLM metadata lands in the trace; non-LLM nodes have none.
    assert s["trace"][0]["llm"]["provider"] == "fake" and s["trace"][1]["llm"] is None
    # The generator saw the retrieved schema; the responder saw the rows.
    assert "TABLE customer" in llm.calls[1].prompt
    assert "59" in llm.calls[2].prompt and "SELECT" in llm.calls[2].prompt


def test_respond_truncates_rows_for_llm():
    rows = [{"i": i} for i in range(50)]
    llm = FakeLLM([scope(), GeneratedSql(sql="SELECT i FROM track"), "ok"])
    run(llm, Executor(QueryResult(["i"], rows)))
    prompt = llm.calls[2].prompt
    assert "Total rows: 50" in prompt and "first 10 of them" in prompt
    assert '"i": 9}' in prompt and '"i": 10}' not in prompt


@pytest.mark.parametrize("lang,msg", [("en", "I can't answer that"), ("es", "No puedo responder eso")])
def test_out_of_scope(lang, msg):
    llm = FakeLLM([scope(False, lang)])
    ex = Executor()
    s = run(llm, ex, "What's the weather?")
    assert path(s) == ["classify", "respond_out_of_scope"]
    assert s["answer"].startswith(msg) and not ex.calls and "sql" not in s


NO_REPAIR = Settings(_env_file=None, max_rows=100, max_repairs=0)


def test_guard_rejection_ends_with_error_and_never_executes():
    llm = FakeLLM([scope(), GeneratedSql(sql="DELETE FROM track")])
    ex = Executor()
    s = run(llm, ex, settings=NO_REPAIR)
    assert path(s) == ["classify", "retrieve_schema", "generate_sql", "validate_sql", "respond_error"]
    assert not ex.calls and "not_select" in s["error"] and "rejected" in s["answer"]


def test_execution_error_ends_with_error():
    llm = FakeLLM([scope(lang="es"), GeneratedSql(sql="SELECT nope FROM customer")])
    ex = Executor(error=QueryError('column "nope" does not exist', "42703"))
    s = run(llm, ex, "¿cuántos?", settings=NO_REPAIR)
    assert path(s)[-2:] == ["execute", "respond_error"]
    assert 'column "nope"' in s["answer"] and s["answer"].startswith("No pude")
    assert len(ex.calls) == 1


def test_format_schema_marks_pk_and_fk():
    from t2sql.db import ForeignKey

    t = TableInfo(
        "invoice",
        [ColumnInfo("invoice_id", "integer", False, True), ColumnInfo("customer_id", "integer", False)],
        [ForeignKey("customer_id", "customer", "customer_id")],
    )
    text = format_schema([t])
    assert "invoice_id integer PRIMARY KEY" in text
    assert "customer_id integer REFERENCES customer(customer_id)" in text


def test_respond_prompt_asks_for_short_summary_not_enumeration():
    from t2sql import prompts

    sys_prompt = prompts.RESPOND_SYSTEM
    assert "1 or 2" in sys_prompt and "Never enumerate more than 3" in sys_prompt
    assert "emails" in sys_prompt and "next to your answer" in sys_prompt


def test_respond_prompt_requires_plain_text():
    from t2sql import prompts

    assert "plain text only" in prompts.RESPOND_SYSTEM and "no Markdown" in prompts.RESPOND_SYSTEM
