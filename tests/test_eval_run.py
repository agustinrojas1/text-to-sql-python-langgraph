import json

import pytest

from evals import run as ev
from evals.dataset import Item, load_dataset
from t2sql.config import get_settings
from t2sql.db import ColumnInfo, TableInfo
from t2sql.graph import build_graph
from t2sql.retrieval import AllTablesRetriever

ITEMS = load_dataset()


def test_percentile_interpolates():
    assert ev.percentile([1, 2, 3, 4], 50) == 2.5
    assert ev.percentile([10], 95) == 10
    assert ev.percentile([], 50) is None
    assert ev.percentile(list(range(101)), 95) == 95


def test_table_recall_definition():
    assert ev.table_recall(("a", "b", "c"), ["a", "c", "x"]) == pytest.approx(2 / 3)
    assert ev.table_recall(("a",), ["a"]) == 1.0
    assert ev.table_recall((), ["a"]) is None


class StubRetriever:
    def __call__(self, question):
        return [TableInfo(n, [ColumnInfo("id", "integer", False)]) for n in ("customer", "invoice")]


def test_evaluate_retrieval_scores_recall_and_context_size():
    item = Item("x", "medium", "en", "q", "SELECT 1", ("customer", "invoice_line"))
    r = ev.evaluate_retrieval(StubRetriever(), item)
    assert r["table_recall"] == 0.5 and r["n_candidates"] == 2 and r["candidate_tables"] == ["customer", "invoice"]
    md = ev.retrieval_markdown([r], "stub")
    assert "50.0%" in md and "missing ['invoice_line']" in md


def test_aggregate_metric_definitions():
    def rec(id_, level, match, first_valid=True, ended_ok=True, rejected=False, recall=1.0, **kw):
        base = {
            "id": id_, "level": level, "exec_match": match, "exec_strict": match, "rejected": rejected,
            "ended_ok": ended_ok, "table_recall": recall, "n_candidates": 4,
            "summary": {"latency_ms": 1000.0, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01,
                        "attempts": 1 if not rejected else 0, "first_try_valid": first_valid},
        }  # fmt: skip
        return {**base, **kw}

    recs = [
        rec("a", "easy", True),
        rec("b", "easy", False, first_valid=False, ended_ok=True),  # recovered but wrong
        rec("c", "easy", False, first_valid=False, ended_ok=False),  # not recovered
        rec("d", "medium", False, rejected=True, recall=None, n_candidates=None),
        {"id": "o1", "level": "out_of_scope", "rejected": True, "correct_rejection": True, "summary": {"latency_ms": 500.0, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0, "attempts": 0, "first_try_valid": True}},
        {"id": "o2", "level": "out_of_scope", "rejected": False, "correct_rejection": False, "summary": {"latency_ms": 500.0, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0, "attempts": 1, "first_try_valid": True}},
    ]  # fmt: skip
    del recs[3]["n_candidates"], recs[3]["table_recall"]  # a rejected question never reaches retrieval
    a = ev.aggregate(recs)
    assert (a["n_in_scope"], a["n_out_of_scope"]) == (4, 2)
    assert a["exec_accuracy"] == 0.25  # rejections and wrong results count against accuracy
    assert a["first_try_valid"] == pytest.approx(1 / 3)  # only the 3 that reached generation
    assert (a["first_try_failed"], a["recovered"]) == (2, 1)
    assert a["correct_rejections"] == 1 and a["false_rejections"] == 1
    assert a["table_recall"] == 1.0 and a["n_recall"] == 3
    assert a["latency_p50_s"] == 1.0 and a["cost_per_question"] == pytest.approx(0.04 / 6)  # mean over all 6 priced questions


@pytest.fixture
def retriever():
    return AllTablesRetriever(get_settings())


@pytest.mark.integration
def test_fake_mode_end_to_end_is_perfect(retriever, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert ev.main(["--fake", "--retriever", "all", "--out", str(out)]) == 0
    md = capsys.readouterr().out
    assert "| fake/oracle | **all answerable** | 50 | 100% (50/50) |" in md
    assert "5/5" in md and "0/50" in md
    data = json.loads(out.read_text())
    agg = data["aggregates"]["fake/oracle"]
    assert agg["exec_accuracy"] == 1.0 and agg["exec_accuracy_strict"] == 1.0
    assert agg["table_recall"] == 1.0 and agg["avg_candidates"] == 11  # "all tables" retriever
    assert agg["first_try_valid"] == 1.0 and agg["first_try_failed"] == 0
    assert len(data["results"]["fake/oracle"]) == 55


@pytest.mark.integration
def test_wrong_answers_are_scored_as_wrong(retriever):
    s = get_settings()
    items = [i for i in ITEMS if i.id in {"e01", "m09", "h01", "o01"}]
    graph = build_graph(ev.OracleLLM(items, fault="wrong_answer"), retriever, s)
    recs = [ev.evaluate_item(graph, i, ev.reference_result(i, s) if i.sql else None, s) for i in items]
    a = ev.aggregate(recs)
    assert a["exec_accuracy"] == 0.0 and a["first_try_valid"] == 1.0  # valid SQL, wrong result
    assert a["correct_rejections"] == 1


@pytest.mark.integration
def test_repair_loop_is_measured(retriever):
    s = get_settings()
    items = [i for i in ITEMS if i.id in {"e01", "m09", "h01"}]
    graph = build_graph(ev.OracleLLM(items, fault="broken_first"), retriever, s)
    recs = [ev.evaluate_item(graph, i, ev.reference_result(i, s), s) for i in items]
    a = ev.aggregate(recs)
    assert a["first_try_valid"] == 0.0
    assert (a["first_try_failed"], a["recovered"]) == (3, 3)
    assert a["exec_accuracy"] == 1.0  # the repaired query returns the reference result


def test_select_items_filters():
    class A:  # argparse.Namespace stand-in
        ids, level, limit = None, "hard", 3

    assert [i.id for i in ev.select_items(ITEMS, A())] == ["h01", "h02", "h03"]
    A.ids, A.level, A.limit = "e01,o01", None, None
    assert [i.id for i in ev.select_items(ITEMS, A())] == ["e01", "o01"]
