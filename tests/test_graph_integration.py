import pytest

from t2sql.config import get_settings
from t2sql.graph import build_graph, run_question
from t2sql.nodes.classify import ScopeDecision
from t2sql.nodes.generate import GeneratedSql
from t2sql.retrieval import AllTablesRetriever
from t2sql.testing import FakeLLM

pytestmark = pytest.mark.integration


def test_real_graph_against_real_db_with_fake_llm():
    llm = FakeLLM([
        ScopeDecision(in_scope=True, reason="customers", language="en"),
        GeneratedSql(sql="SELECT count(*) AS n FROM customer", tables_used=["customer"]),
        "There are 59 customers.",
    ])
    state = run_question(build_graph(llm, AllTablesRetriever(get_settings()), get_settings()), "How many customers are there?")
    assert state["rows"] == [{"n": 59}]
    assert state["error"] is None and state["answer"] == "There are 59 customers."
    # Real retriever fed the generator the real schema.
    assert "TABLE customer" in llm.calls[1].prompt and "REFERENCES" in llm.calls[1].prompt
