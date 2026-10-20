from t2sql.config import Settings
from t2sql.db import ColumnInfo, QueryError, QueryResult, TableInfo
from t2sql.graph import build_graph, run_question
from t2sql.nodes.classify import ScopeDecision
from t2sql.nodes.generate import GeneratedSql
from t2sql.testing import FakeLLM

SETTINGS = Settings(_env_file=None, max_rows=100, max_repairs=2)
TABLES = [TableInfo("customer", [ColumnInfo("customer_id", "integer", False, True)])]
GOOD = "SELECT count(*) AS n FROM customer"


class Executor:
    """Pops scripted outcomes: a QueryResult or an Exception to raise."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def __call__(self, sql):
        self.calls.append(sql)
        r = self.outcomes.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def gen(sql):
    return GeneratedSql(sql=sql, tables_used=["customer"])


def run(responses, executor, settings=SETTINGS):
    llm = FakeLLM([ScopeDecision(in_scope=True, reason="r", language="en"), *responses])
    graph = build_graph(llm, lambda q: TABLES, settings, executor)
    return llm, run_question(graph, "How many customers?")


def path(s):
    return [e["node"] for e in s["trace"]]


def test_guard_rejection_then_fix():
    ex = Executor(QueryResult(["n"], [{"n": 59}]))
    llm, s = run([gen("DELETE FROM customer"), gen(GOOD), "59 customers."], ex)
    assert path(s) == [
        "classify", "retrieve_schema", "generate_sql", "validate_sql", "repair",
        "generate_sql", "validate_sql", "execute", "respond",
    ]  # fmt: skip
    assert s["attempts"] == 2 and s["error"] is None and s["answer"] == "59 customers."
    assert s["summary"]["repaired"] and not s["summary"]["first_try_valid"]
    assert s["summary"]["attempts"] == 2
    assert [h["kind"] for h in s["attempt_history"]] == ["guard"]
    # Each trace entry carries the attempt it belongs to.
    assert [e["attempt"] for e in s["trace"] if e["node"] == "generate_sql"] == [1, 2]
    # The repair prompt shows the previous SQL and the guard code + message.
    repair_prompt = llm.calls[2].prompt
    assert "DELETE FROM customer" in repair_prompt and "not_select" in repair_prompt
    assert "Attempt 1" in repair_prompt and "TABLE customer" in repair_prompt


def test_execution_error_then_fix():
    err = QueryError('column "nope" does not exist', "42703")
    ex = Executor(err, QueryResult(["n"], [{"n": 59}]))
    llm, s = run([gen("SELECT nope FROM customer"), gen(GOOD), "ok"], ex)
    assert path(s).count("repair") == 1 and path(s)[-1] == "respond"
    assert s["attempts"] == 2 and len(ex.calls) == 2
    assert 'column "nope" does not exist' in llm.calls[2].prompt
    assert "SELECT nope FROM customer" in llm.calls[2].prompt
    assert s["summary"]["repaired"] and not s["summary"]["first_try_valid"]
    assert s["rows"] == [{"n": 59}]


def test_empty_result_repairs_once_then_responds():
    empty = QueryResult(["n"], [])
    ex = Executor(empty, empty)
    llm, s = run(
        [gen("SELECT n FROM customer WHERE country = 'brazil'"), gen("SELECT n FROM customer WHERE country ILIKE 'brazil'"), "Nothing found."],
        ex,
    )  # fmt: skip
    assert path(s).count("repair") == 1 and path(s)[-1] == "respond"  # not respond_error
    assert s["attempts"] == 2 and s["error"] is None and s["answer"] == "Nothing found."
    assert "0 rows" in llm.calls[2].prompt and "case" in llm.calls[2].prompt
    # First try was valid (empty is not an error), and nothing was actually recovered.
    assert s["summary"]["first_try_valid"] and not s["summary"]["repaired"]


def test_empty_result_then_rows_counts_as_repaired():
    ex = Executor(QueryResult(["n"], []), QueryResult(["n"], [{"n": 5}]))
    _, s = run([gen(GOOD), gen(GOOD + " WHERE 1=1"), "five"], ex)
    assert s["summary"]["repaired"] and s["summary"]["first_try_valid"]


def test_empty_after_error_repair_does_not_loop_more_than_budget():
    # error -> repair -> empty -> repair (empty allowed once) -> error -> budget exhausted
    ex = Executor(QueryError("boom"), QueryResult(["n"], []), QueryError("boom2"))
    llm, s = run([gen(GOOD), gen(GOOD), gen(GOOD)], ex)
    assert path(s)[-1] == "respond_error" and s["attempts"] == 3


def test_exhausted_retries_end_in_respond_error():
    llm, s = run([gen("DROP TABLE customer")] * 3, Executor())
    gens = [c for c in llm.calls if c.kind == "structured" and c.schema is GeneratedSql]
    assert len(gens) == 3 and s["attempts"] == 3
    assert path(s).count("repair") == 2 and path(s)[-1] == "respond_error"
    assert "not_select" in s["error"] and "rejected" in s["answer"]
    assert s["summary"]["attempts"] == 3 and not s["summary"]["repaired"]
    # The last prompt shows every earlier failed attempt.
    assert "Attempt 1" in gens[2].prompt and "Attempt 2" in gens[2].prompt
    assert s["rows"] is None


def test_max_repairs_is_configurable():
    s_one = Settings(_env_file=None, max_repairs=1)
    llm, s = run([gen("DROP TABLE customer")] * 2, Executor(), s_one)
    assert s["attempts"] == 2 and path(s)[-1] == "respond_error"


def test_database_outage_is_not_repaired():
    # An outage must not burn repair attempts: it propagates after one execute, with no repair LLM call.
    import pytest

    from t2sql.db import DatabaseUnavailableError

    ex = Executor(DatabaseUnavailableError("connection refused"))
    llm = FakeLLM([ScopeDecision(in_scope=True, reason="r", language="en"), gen(GOOD)])
    graph = build_graph(llm, lambda q: TABLES, SETTINGS, ex)
    with pytest.raises(DatabaseUnavailableError):
        run_question(graph, "How many customers?")
    assert len(ex.calls) == 1
    assert len([c for c in llm.calls if c.schema is GeneratedSql]) == 1


def test_statement_timeout_is_repairable_and_hints_a_cheaper_query():
    timeout = QueryError("canceling statement due to statement timeout", "57014")
    ex = Executor(timeout, QueryResult(["n"], [{"n": 59}]))
    llm, s = run([gen("SELECT count(*) FROM track a, track b"), gen(GOOD), "ok"], ex)
    assert path(s).count("repair") == 1 and path(s)[-1] == "respond"
    assert "statement timeout" in llm.calls[2].prompt and "cheaper" in llm.calls[2].prompt


def test_recursion_limit_covers_the_longest_legal_path():
    from t2sql.graph import RECURSION_MARGIN, recursion_limit

    # Worst case, max_repairs=2: 2 + 3 generations * 3 nodes + 2 repairs + 1 final node = 14 steps.
    assert recursion_limit(2) == 14 + RECURSION_MARGIN
    assert recursion_limit(0) == 6 + RECURSION_MARGIN
    # The longest real run (every attempt fails at execute) fits under the limit.
    settings = Settings(_env_file=None, max_repairs=2)
    ex = Executor(*[QueryError("boom")] * 3)
    _, s = run([gen(GOOD)] * 3, ex, settings)
    assert len(s["trace"]) == 14 and path(s)[-1] == "respond_error"
    assert build_graph(FakeLLM([]), lambda q: TABLES, settings, ex).config["recursion_limit"] == 17


def test_misbehaving_router_hits_recursion_limit_instead_of_looping():
    import pytest
    from langgraph.errors import GraphRecursionError

    settings = Settings(_env_file=None, max_repairs=1)
    llm = FakeLLM([ScopeDecision(in_scope=True, reason="r", language="en"), *[gen(GOOD)] * 50])
    ex = Executor(*[QueryError("boom")] * 50)
    graph = build_graph(llm, lambda q: TABLES, settings, ex)  # limit is computed here, for 1 repair
    settings.max_repairs = 10_000  # the routers read this live: the attempts guard is now useless
    with pytest.raises(GraphRecursionError):
        run_question(graph, "How many customers?")
    assert len(ex.calls) < 50  # it stopped by itself, long before the scripted responses ran out
