"""Schema RAG: embed table/column documents into pgvector and retrieve the relevant tables.

Storage is `t2sql_meta.schema_docs`, owned by the admin role. The read-only role
has no privileges on that schema, so generated SQL can never read it.
No ANN index on purpose: with 75 rows an exact scan is faster than HNSW/ivfflat
and has perfect recall (an index only pays off beyond thousands of documents).
"""

import logging
import re
import unicodedata
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

import psycopg
from psycopg import sql as pgsql
from psycopg.rows import dict_row

from t2sql.catalog import COLUMN_HINTS, TABLE_DESCRIPTIONS, TABLE_KEYWORDS
from t2sql.config import Settings, get_settings
from t2sql.db import TableInfo, fetch_schema_metadata

log = logging.getLogger(__name__)

META_SCHEMA = "t2sql_meta"
DEFAULT_TABLE = "schema_docs"
COLUMN_WEIGHT = 0.5  # weight of the column evidence when ranking a table
TOP_COLUMNS = 2  # how many best column docs count per table
COLUMN_HITS_SHOWN = 3  # best column docs reported per table in the retrieval details
MAX_BRIDGES = 3  # max intermediate tables when connecting two selected tables
SEARCH_LIMIT = 60  # docs fetched per query (the index has 75)

STEM_LEN = 5  # crude stemmer: drop a plural "s", then keep the first STEM_LEN characters
_STOPWORDS = frozenset(
    "the and for per with that from what which who how many much were was are have has not any "
    "each every all their there show list give get del las los una uno unos unas por con que como "
    "cual cuales cuantos cuantas quien quienes para entre sobre mas menos son fue ese esa".split()
)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (CPU, no API key). Model loads lazily."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.name, cache_dir=self._cache_dir)
        return [v.tolist() for v in self._model.embed(texts)]


# ---------------------------------------------------------------- documents


@dataclass(frozen=True)
class SchemaDoc:
    kind: str  # "table" | "column"
    table: str
    column: str | None
    text: str


def build_documents(tables: list[TableInfo], keywords: bool = False) -> list[SchemaDoc]:
    """One document per table and one per column.

    `keywords` appends the everyday EN/ES vocabulary of `catalog.TABLE_KEYWORDS` to each
    table document. Off by default: measured in docs/eval-findings.md, it did not help the
    dense retriever (the vocabulary is used by the lexical scorer instead).
    """
    docs = []
    for t in tables:
        desc = TABLE_DESCRIPTIONS.get(t.name, "")
        cols = ", ".join(f"{c.name} ({c.data_type})" for c in t.columns)
        text = f"Table {t.name}: {desc}. Columns: {cols}."
        if t.foreign_keys:
            refs = ", ".join(f"{f.column} -> {f.ref_table}.{f.ref_column}" for f in t.foreign_keys)
            text += f" Foreign keys: {refs}."
        if keywords and t.name in TABLE_KEYWORDS:
            text += f" Also: {TABLE_KEYWORDS[t.name]}."
        docs.append(SchemaDoc("table", t.name, None, text))
        for c in t.columns:
            hint = COLUMN_HINTS.get(f"{t.name}.{c.name}", "")
            text = f"Column {t.name}.{c.name} ({c.data_type}) of table {t.name}" + (f": {hint}" if hint else "")
            docs.append(SchemaDoc("column", t.name, c.name, text))
    return docs


# ---------------------------------------------------------------- storage


def _tbl(table: str) -> pgsql.Composed:
    return pgsql.SQL("{}.{}").format(pgsql.Identifier(META_SCHEMA), pgsql.Identifier(table))


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def build_index(
    embedder: Embedder,
    settings: Settings | None = None,
    tables: list[TableInfo] | None = None,
    table: str = DEFAULT_TABLE,
    keywords: bool = False,
) -> int:
    """Drop and recreate the index table, then insert one row per document. Idempotent."""
    settings = settings or get_settings()
    docs = build_documents(tables if tables is not None else fetch_schema_metadata(settings), keywords)
    vectors = embedder.embed([d.text for d in docs])
    dim = len(vectors[0])
    with psycopg.connect(settings.admin_database_url) as conn:  # one transaction
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(pgsql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(pgsql.Identifier(META_SCHEMA)))
        conn.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(_tbl(table)))
        conn.execute(
            pgsql.SQL(
                "CREATE TABLE {} (id serial PRIMARY KEY, kind text NOT NULL, table_name text NOT NULL,"
                " column_name text, content text NOT NULL, model text NOT NULL, embedding vector({}) NOT NULL)"
            ).format(_tbl(table), pgsql.Literal(dim))
        )
        # Defense in depth: the schema is never granted to anyone but its owner.
        conn.execute(pgsql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(pgsql.Identifier(META_SCHEMA)))
        with conn.cursor() as cur:
            cur.executemany(
                pgsql.SQL(
                    "INSERT INTO {} (kind, table_name, column_name, content, model, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s::vector)"
                ).format(_tbl(table)),
                [(d.kind, d.table, d.column, d.text, embedder.name, _vec(v)) for d, v in zip(docs, vectors)],
            )
    return len(docs)


def index_exists(embedder_name: str, settings: Settings, table: str = DEFAULT_TABLE) -> bool:
    """True if the index has documents for this embedding model."""
    try:
        with psycopg.connect(settings.admin_database_url, connect_timeout=3) as conn:
            row = conn.execute(
                pgsql.SQL("SELECT count(*) FROM {} WHERE model = %s").format(_tbl(table)), (embedder_name,)
            ).fetchone()
            return bool(row and row[0])
    except psycopg.Error:
        return False


# ---------------------------------------------------------------- scoring and FK expansion


def column_hits(
    hits: list[tuple[str, str | None, float]], tables: list[str], top: int = COLUMN_HITS_SHOWN
) -> dict[str, list[dict[str, Any]]]:
    """The `top` column docs most similar to the question, per table (for the UI / trace)."""
    per_table: dict[str, list[tuple[float, str]]] = {t: [] for t in tables}
    for t, c, s in hits:
        if c is not None and t in per_table:
            per_table[t].append((s, c))
    return {
        t: [{"column": c, "score": round(s, 4)} for s, c in sorted(found, key=lambda x: (-x[0], x[1]))[:top]]
        for t, found in per_table.items()
    }


def aggregate_scores(hits: list[tuple[str, str | None, float]]) -> dict[str, float]:
    """(table, column|None, cosine similarity) hits -> one score per table.

    score = table-doc similarity + COLUMN_WEIGHT * mean of its TOP_COLUMNS best column docs.
    """
    table_doc: dict[str, float] = {}
    columns: dict[str, list[float]] = {}
    for t, c, s in hits:
        if c is None:
            table_doc[t] = s
        else:
            columns.setdefault(t, []).append(s)
    scores = {}
    for t in table_doc.keys() | columns.keys():
        best = sorted(columns.get(t, []), reverse=True)[:TOP_COLUMNS]
        scores[t] = table_doc.get(t, 0.0) + COLUMN_WEIGHT * (sum(best) / len(best) if best else 0.0)
    return scores


def _stems(text: str) -> set[str]:
    """Lowercase, accent-free, stopword-free word stems (first STEM_LEN characters)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = (w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", text))
    return {w[:STEM_LEN] for w in words if len(w) >= 3 and w not in _STOPWORDS}


def lexical_scores(question: str, keywords: dict[str, str]) -> dict[str, float]:
    """Keyword evidence per table: an IDF-weighted overlap between the question and the
    table's everyday vocabulary (`catalog.TABLE_KEYWORDS`), in 0..1.

    The dense embedding of a whole question is dominated by its nouns ("songs", "Jazz"), so
    transactional intent ("who bought", "revenue", "compraron") hardly moves customer /
    invoice / invoice_line. A word shared by many tables counts less (weight 1/df), so
    "song" says little and "bought" says more. Stemming is deliberately crude (prefix).
    """
    table_stems = {t: _stems(text) for t, text in keywords.items()}
    df: dict[str, int] = {}
    for stems in table_stems.values():
        for w in stems:
            df[w] = df.get(w, 0) + 1
    q = _stems(question)
    return {t: min(1.0, sum(1 / df[w] for w in stems & q)) for t, stems in table_stems.items()}


def fk_graph(tables: list[TableInfo]) -> dict[str, set[str]]:
    """Undirected table adjacency from foreign keys (self-references ignored)."""
    g: dict[str, set[str]] = {t.name: set() for t in tables}
    for t in tables:
        for fk in t.foreign_keys:
            if fk.ref_table != t.name and fk.ref_table in g:
                g[t.name].add(fk.ref_table)
                g[fk.ref_table].add(t.name)
    return g


def _shortest_path(g: dict[str, set[str]], src: str, dst: str) -> list[str] | None:
    prev: dict[str, str | None] = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            path = []
            node: str | None = cur
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        for n in sorted(g.get(cur, ())):  # sorted: deterministic among equal-length paths
            if n not in prev:
                prev[n] = cur
                q.append(n)
    return None


def bridge_paths(
    selected: list[str], g: dict[str, set[str]], max_bridges: int = MAX_BRIDGES
) -> list[list[str]]:
    """Shortest FK paths between pairs of selected tables whose interior adds a table.

    Pairs further apart than `max_bridges` intermediate tables are left alone, so the
    context stays bounded. Each path is returned whole (endpoints included, e.g.
    ["invoice_line", "track", "album", "artist"]), in pair order, so the UI can draw why a
    table was pulled in; a path whose interior is already selected explains nothing and is skipped.
    """
    chosen = set(selected)
    paths = []
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            path = _shortest_path(g, a, b)
            if path and 0 < len(path) - 2 <= max_bridges and not set(path[1:-1]) <= chosen:
                paths.append(path)
    return paths


def expand_with_bridges(
    selected: list[str], g: dict[str, set[str]], max_bridges: int = MAX_BRIDGES
) -> list[str]:
    """Intermediate tables of `bridge_paths`: only path nodes are added, never all neighbours."""
    extra = {t for path in bridge_paths(selected, g, max_bridges) for t in path[1:-1]}
    return sorted(extra - set(selected))


# ---------------------------------------------------------------- retriever


@dataclass
class Retrieval:
    tables: list[TableInfo]
    details: dict[str, Any] = field(default_factory=dict)


class PgVectorRetriever:
    """Embed the question, rank tables by cosine similarity, take top-k, add FK bridges."""

    def __init__(
        self,
        embedder: Embedder,
        settings: Settings,
        metadata: Callable[[], list[TableInfo]] | None = None,
        table: str = DEFAULT_TABLE,
        top_k: int | None = None,
        lexical_weight: float | None = None,
        keywords: dict[str, str] | None = None,
    ) -> None:
        self._embedder, self._settings, self._table = embedder, settings, table
        self._lexical_weight = settings.retrieval_lexical_weight if lexical_weight is None else lexical_weight
        self._keywords = TABLE_KEYWORDS if keywords is None else keywords
        self._top_k = top_k or settings.retrieval_top_k
        self._metadata = lru_cache(maxsize=1)(metadata or (lambda: fetch_schema_metadata(settings)))

    def _hits(self, question: str) -> list[tuple[str, str | None, float]]:
        vec = _vec(self._embedder.embed([question])[0])
        query = pgsql.SQL(
            "SELECT table_name, column_name, 1 - (embedding <=> %(v)s::vector) AS score FROM {}"
            " WHERE model = %(m)s ORDER BY embedding <=> %(v)s::vector LIMIT %(n)s"
        ).format(_tbl(self._table))
        with psycopg.connect(self._settings.admin_database_url, row_factory=dict_row) as conn:
            rows = conn.execute(query, {"v": vec, "m": self._embedder.name, "n": SEARCH_LIMIT}).fetchall()
        return [(r["table_name"], r["column_name"], float(r["score"])) for r in rows]

    def retrieve(self, question: str) -> Retrieval:
        meta = {t.name: t for t in self._metadata()}
        hits = [h for h in self._hits(question) if h[0] in meta]
        dense = aggregate_scores(hits)
        scores = dense
        lex: dict[str, float] = {}
        if self._lexical_weight:
            lex = lexical_scores(question, self._keywords)
            scores = {t: sc + self._lexical_weight * lex.get(t, 0.0) for t, sc in dense.items()}
        ranked = sorted(scores, key=lambda t: (-scores[t], t))
        selected = ranked[: self._top_k]
        g = fk_graph(list(meta.values()))
        paths = bridge_paths(selected, g)
        expanded = expand_with_bridges(selected, g)
        final = sorted(selected + expanded)  # stable order for the prompt
        details: dict[str, Any] = {
            "retriever": "pgvector",
            "top_k": self._top_k,
            "scores": {t: round(scores[t], 4) for t in ranked},
            "dense": {t: round(dense[t], 4) for t in ranked},
            "selected": selected,
            "expanded": expanded,
            "bridge_paths": paths,
            "column_hits": column_hits(hits, final),
        }
        if self._lexical_weight:  # omitted (not zeros) when the lexical scorer is off
            details["lexical"] = {t: round(lex.get(t, 0.0), 4) for t in ranked}
        return Retrieval([meta[t] for t in final], details)

    def __call__(self, question: str) -> list[TableInfo]:
        return self.retrieve(question).tables


def default_retriever(settings: Settings, embedder: Embedder | None = None):
    """Pick the retriever from `settings.retriever` ("auto" falls back to all tables with a warning)."""
    from t2sql.retrieval import AllTablesRetriever

    if settings.retriever == "all":
        return AllTablesRetriever(settings)
    embedder = embedder or FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    if index_exists(embedder.name, settings):
        return PgVectorRetriever(embedder, settings)
    if settings.retriever == "pgvector":
        raise RuntimeError("Schema index not found; run `t2sql index` first.")
    log.warning("Schema index not found; using all tables. Run `t2sql index` to enable schema RAG.")
    return AllTablesRetriever(settings)
