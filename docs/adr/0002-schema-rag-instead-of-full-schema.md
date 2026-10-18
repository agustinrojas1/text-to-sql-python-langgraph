# 0002. Schema RAG instead of sending the whole schema

Status: accepted (retriever revised after the Phase 4 evals, see below)

## Context

Chinook has 11 tables, small enough to paste entirely into the prompt. That is what we
would do for a demo, but the project is meant to show what happens with a real schema
(hundreds of tables): the full schema costs tokens on every call, dilutes the model's
attention and invites invented columns. The cost of the alternative is that the retriever
can drop a table the query needs, and then no model can write the right SQL.

## Decision

Retrieve the relevant tables per question, and measure the retriever separately from the
model.

- **Index.** One document per table (description, columns, foreign keys) and one per
  column (type plus a hint where the name is ambiguous), embedded with a local
  multilingual model (`paraphrase-multilingual-MiniLM-L12-v2` through fastembed; ONNX on
  CPU, no API key, handles English and Spanish questions). Stored in pgvector
  (`t2sql_meta.schema_docs`). No ANN index: with about 75 rows an exact scan is faster
  and has perfect recall.
- **Ranking.** A table's score is its table-document similarity plus half the mean of
  its two best column documents. The first version used only this (top-k = 4).
- **FK bridge expansion.** After taking the top-k tables, the intermediate tables on the
  shortest foreign-key path between each pair of selected tables are added (at most 3
  intermediates), so a join path is never cut in the middle without pulling in every
  neighbour.
- **Keyword score (added after the evals).** A small IDF-weighted overlap between the
  question and an English/Spanish vocabulary per table (`catalog.TABLE_KEYWORDS`) is added
  to the embedding score (weight 0.5), and top-k went from 4 to 5. Motivation and numbers
  are in `docs/eval-findings.md`: with embeddings alone the retriever missed transactional
  intent ("who bought", "revenue") and left out `customer`, `invoice` and `invoice_line`.
- **Metric.** Table recall, `|needed ∩ candidates| / |needed|` where `needed` is derived
  from the reference SQL with sqlglot, plus the share of questions with full recall and the
  average number of tables in the context. `uv run python -m evals.run --retrieval-only`
  needs no LLM key.

## Consequences

- On the 50-question dataset, table recall went from 95.4% (88% of questions with every
  needed table) with embeddings only and k=4, to 100% (100%) with the keyword score and
  k=5, at the price of a larger context (4.7 to 6.0 tables on average, out of 11). On a
  18-question held-out probe set written before the fix: 83.4% (61%) to 95.6% (89%). The
  dataset figure is partly in-sample; the held-out one is the honest estimate. Details,
  failures that remain and the variants that did not help are in `docs/eval-findings.md`.
- With 11 tables the saving is modest (about 45% fewer tables than the full schema), and
  a full-schema baseline (`T2SQL_RETRIEVER=all`) is one setting away for comparison once
  an LLM key is available. The design pays off only on bigger schemas; here it is mostly a
  measured, testable component.
- The keyword vocabulary is hand-written per schema (and in two languages). It is cheap but
  it is maintenance, and it can overfit the questions its author has in mind.
- A model download (about 220 MB) happens on the first `t2sql index` or question.
- If the index does not exist, the `auto` retriever falls back to all tables with a warning
  instead of failing.
