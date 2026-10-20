import psycopg
import pytest

from t2sql.config import Settings, get_settings
from t2sql.db import ColumnInfo, ForeignKey, TableInfo, fetch_schema_metadata
from t2sql.graph import build_graph, run_question
from t2sql.schema_index import (
    META_SCHEMA,
    PgVectorRetriever,
    aggregate_scores,
    build_documents,
    build_index,
    default_retriever,
    expand_with_bridges,
    fk_graph,
)
from t2sql.retrieval import AllTablesRetriever
from t2sql.testing import FakeEmbedder, FakeLLM


def graph_of(edges: list[tuple[str, str]]):
    tables = {n: TableInfo(n) for e in edges for n in e}
    for a, b in edges:
        tables[a].foreign_keys.append(ForeignKey("x", b, "id"))
    return fk_graph(list(tables.values()))


CHAIN = graph_of([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("c", "side")])


def test_expand_adds_only_path_nodes():
    assert expand_with_bridges(["a", "c"], CHAIN) == ["b"]  # not "side", not "d"
    assert expand_with_bridges(["a", "b"], CHAIN) == []  # adjacent: nothing to bridge
    assert expand_with_bridges(["a", "d"], CHAIN) == ["b", "c"]


def test_expand_is_bounded_and_ignores_disconnected():
    assert expand_with_bridges(["a", "f"], CHAIN) == []  # 4 intermediates > MAX_BRIDGES
    assert expand_with_bridges(["a", "f"], CHAIN, max_bridges=4) == ["b", "c", "d", "e"]
    assert expand_with_bridges(["a", "zzz"], CHAIN) == []


def test_expand_ignores_self_reference():
    t = TableInfo("emp", foreign_keys=[ForeignKey("reports_to", "emp", "id")])
    assert fk_graph([t]) == {"emp": set()}


def test_aggregate_scores():
    hits = [("t1", None, 0.5), ("t1", "a", 0.9), ("t1", "b", 0.7), ("t1", "c", 0.1), ("t2", None, 0.6), ("t3", "x", 0.4)]
    s = aggregate_scores(hits)
    assert s["t1"] == pytest.approx(0.5 + 0.5 * 0.8)  # top-2 columns averaged
    assert s["t2"] == pytest.approx(0.6)  # no column evidence
    assert s["t3"] == pytest.approx(0.5 * 0.4)  # column evidence only


def tiny_schema():
    def t(name, *fks):
        return TableInfo(name, [ColumnInfo(f"{name}_id", "integer", False, True)], [ForeignKey(f"{r}_id", r, f"{r}_id") for r in fks])

    return [t("playlist"), t("track"), t("playlist_track", "playlist", "track"), t("genre")]


def test_documents_one_per_table_and_column():
    docs = build_documents(tiny_schema())
    assert sum(d.kind == "table" for d in docs) == 4 and sum(d.kind == "column" for d in docs) == 4
    pt = next(d for d in docs if d.kind == "table" and d.table == "playlist_track")
    assert "playlist_id -> playlist.playlist_id" in pt.text and "link table" in pt.text


def test_fake_embedder_is_deterministic_and_normalised():
    e = FakeEmbedder()
    a, b = e.embed(["track duration"]), e.embed(["track duration"])
    assert a == b and sum(x * x for x in a[0]) == pytest.approx(1.0)


def test_default_retriever_all_setting():
    s = Settings(_env_file=None, retriever="all")
    assert isinstance(default_retriever(s), AllTablesRetriever)


def test_auto_falls_back_to_all_with_warning(caplog):
    s = Settings(_env_file=None, retriever="auto", admin_database_url="postgresql://x:x@localhost:1/x")
    assert isinstance(default_retriever(s, FakeEmbedder()), AllTablesRetriever)
    assert "Schema index not found" in caplog.text
    with pytest.raises(RuntimeError):
        default_retriever(s.model_copy(update={"retriever": "pgvector"}), FakeEmbedder())


# ---- integration: a separate table, the real index is untouched

TEST_TABLE = "schema_docs_test"
integration = pytest.mark.integration


@pytest.fixture
def test_index():
    settings, emb = get_settings(), FakeEmbedder()
    n = build_index(emb, settings, table=TEST_TABLE)
    yield settings, emb, n
    with psycopg.connect(settings.admin_database_url) as c:
        c.execute(f"DROP TABLE IF EXISTS {META_SCHEMA}.{TEST_TABLE}")


@integration
def test_build_is_idempotent_and_retrieves_with_expansion(test_index):
    settings, emb, n = test_index
    assert build_index(emb, settings, table=TEST_TABLE) == n  # rebuild: same count, no duplicates
    with psycopg.connect(settings.admin_database_url) as c:
        assert c.execute(f"SELECT count(*) FROM {META_SCHEMA}.{TEST_TABLE}").fetchone()[0] == n
    r = PgVectorRetriever(emb, settings, table=TEST_TABLE, top_k=3)
    res = r.retrieve("songs in the playlist Music playlists track name")
    names = [t.name for t in res.tables]
    assert names == sorted(names)  # stable order
    assert {"playlist", "track"} <= set(names)
    assert "playlist_track" in names  # bridge: selected or added by FK expansion
    assert res.details["retriever"] == "pgvector" and set(res.details["scores"]) >= set(res.details["selected"])


@integration
def test_expansion_on_chinook_metadata():
    g = fk_graph(fetch_schema_metadata())
    assert expand_with_bridges(["playlist", "track"], g) == ["playlist_track"]
    assert expand_with_bridges(["invoice", "track"], g) == ["invoice_line"]
    assert expand_with_bridges(["customer", "employee"], g) == []  # direct FK


@integration
def test_graph_records_retrieval_details_in_trace(test_index):
    from t2sql.nodes.classify import ScopeDecision
    from t2sql.nodes.generate import GeneratedSql

    settings, emb, _ = test_index
    llm = FakeLLM([ScopeDecision(in_scope=True, reason="r", language="en"), GeneratedSql(sql="SELECT count(*) FROM track"), "ok"])
    retriever = PgVectorRetriever(emb, settings, table=TEST_TABLE)
    state = run_question(build_graph(llm, retriever, settings), "how many tracks are there")
    entry = next(e for e in state["trace"] if e["node"] == "retrieve_schema")
    assert entry["details"]["selected"] and "track" in state["candidate_tables"]


@integration
def test_reader_cannot_read_meta_schema(test_index):
    settings, _, _ = test_index
    with psycopg.connect(settings.readonly_database_url) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(f"SELECT * FROM {META_SCHEMA}.{TEST_TABLE}")
    with psycopg.connect(settings.readonly_database_url) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(f"SELECT * FROM {META_SCHEMA}.schema_docs")


# ---- lexical (keyword) scoring, added after the Phase 4 evals ---------------------------

from t2sql.schema_index import lexical_scores  # noqa: E402

KW = {
    "customer": "customer, buyer; who bought or purchased; cliente, compró",
    "invoice_line": "items sold, bought, purchased; vendidas",
    "track": "song, track; canción, canciones",
    "playlist": "song list; canciones de una lista",
}


def test_lexical_scores_stems_accents_and_stopwords():
    s = lexical_scores("¿QUÉ clientes compraron canciones?", KW)
    assert s["customer"] > 0 and s["track"] > 0  # "client"/"compr"/"canci" stems, accents ignored
    assert lexical_scores("the of what", KW) == {t: 0.0 for t in KW}  # stopwords carry no evidence


def test_lexical_scores_down_weight_words_shared_by_many_tables():
    s = lexical_scores("songs", KW)  # "song" appears in track and playlist: half evidence each
    assert s["track"] == pytest.approx(0.5) and s["playlist"] == pytest.approx(0.5)
    only_customer = lexical_scores("buyer", KW)
    assert only_customer["customer"] == 1.0 and only_customer["track"] == 0.0


def test_lexical_weight_reorders_the_ranking():
    meta = {n: TableInfo(n, [ColumnInfo(f"{n}_id", "integer", False, True)]) for n in KW}
    hits = [("track", None, 0.6), ("playlist", None, 0.5), ("customer", None, 0.3), ("invoice_line", None, 0.2)]

    def run(weight):
        r = PgVectorRetriever(
            FakeEmbedder(), Settings(_env_file=None), metadata=lambda: list(meta.values()),
            top_k=2, lexical_weight=weight, keywords=KW,
        )
        r._hits = lambda q: hits  # no database in this test
        return r.retrieve("who purchased something")

    assert run(0.0).details["selected"] == ["track", "playlist"]
    boosted = run(0.5).details
    assert "customer" in boosted["selected"] and boosted["scores"]["customer"] > 0.3


def test_fastembed_embedder_passes_cache_dir(monkeypatch):
    import sys
    import types

    import numpy as np

    from t2sql.schema_index import FastEmbedEmbedder

    calls = {}

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def embed(self, texts):
            return [np.zeros(3) for _ in texts]

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    FastEmbedEmbedder("some/model", "/persist/cache").embed(["x"])
    assert calls == {"model_name": "some/model", "cache_dir": "/persist/cache"}
