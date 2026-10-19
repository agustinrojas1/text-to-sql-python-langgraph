# text-to-sql-python-langgraph

A LangGraph agent that answers questions, in English or Spanish, over a PostgreSQL database
(Chinook, a music store). It writes the SQL, checks it before running it, executes it with a
read-only role, fixes its own mistakes when the query fails, and has an eval suite with
the numbers to show how well it does. The safety does not depend on the prompt: the model
can ask for anything, and the code and the database refuse what is not a plain `SELECT`.

<!-- TODO: record a CLI demo (asciinema or GIF) and embed it with ![CLI demo](docs/demo.gif) -->
_CLI demo: not recorded yet._

## How it works

```mermaid
flowchart TD
    Q([question]) --> classify
    classify -- out of scope --> respond_out_of_scope --> END1([polite refusal])
    classify -- in scope --> retrieve_schema
    retrieve_schema --> generate_sql
    generate_sql --> validate_sql
    validate_sql -- valid --> execute
    validate_sql -- rejected, retries left --> repair
    validate_sql -- rejected, no retries --> respond_error
    execute -- rows --> respond
    execute -- error or empty result, retries left --> repair
    execute -- error, no retries --> respond_error
    repair --> generate_sql
    respond --> END2([answer + SQL + table])
    respond_error --> END3([failure message])
```

The diagram follows `src/t2sql/graph.py`. `repair` records the failed SQL and the reason,
and `generate_sql` is called again with that history; at most 2 repairs per question. An
empty result gets one retry and is then accepted as a valid answer.

## Features

- **Three layers of safety, in code** ([ADR 0001](docs/adr/0001-security-in-code-not-in-prompt.md)):
  an sqlglot AST guard (one statement, `SELECT` only, no write nodes anywhere in the tree,
  dangerous functions blocked, allow-listed tables, forced `LIMIT`); a `READ ONLY`
  transaction with `statement_timeout`; and a database role that can only `SELECT`. The
  schema index lives in a schema the role cannot read.
- **Schema RAG** ([ADR 0002](docs/adr/0002-schema-rag-instead-of-full-schema.md)): table and
  column documents embedded locally (multilingual MiniLM, no API key) in pgvector, top-k
  tables plus the tables that connect them through foreign keys, plus a small keyword score
  for business words. Its quality is measured as table recall.
- **Capped repair loop** ([ADR 0003](docs/adr/0003-capped-repair-loop.md)): the model sees the
  error (or the empty result) and retries, at most twice.
- **Tracing**: every node records latency, provider/model, tokens, cost and errors;
  `-v` prints it. Provider fallback (Gemini, Groq, Anthropic) behind one small interface.
- **CLI** (`t2sql ask`, `t2sql index`), an **HTTP API** (FastAPI) and an **eval harness** (`python -m evals.run`).

Not built yet: human-in-the-loop with `interrupt`, multi-turn memory
(see `PLAN.md`).

## Quickstart

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Python 3.12, and a free API key
from [Google AI Studio](https://aistudio.google.com/) (Gemini) or [Groq](https://console.groq.com/).
Free-tier limits matter (checked 2026-10-19): `gemini-3.6-flash` and the other 3.x flash models
allow 20 requests per day, so the default is `gemini-3.5-flash-lite`; Groq's
`gpt-oss-120b` allows 8,000 tokens per minute and 200,000 per day. A full eval run is about
170 LLM calls. `T2SQL_LLM_MIN_INTERVAL_S` spaces the calls to a provider (for example 7) and
`T2SQL_LLM_MAX_RETRIES` (default 3) retries 429/503 with backoff.

```bash
docker compose up -d                 # Postgres 16 + pgvector, Chinook loaded, read-only role created
uv sync
cp .env.example .env                 # then set GOOGLE_API_KEY=... and/or GROQ_API_KEY=...
uv run t2sql index                   # builds the schema index (downloads ~220 MB embedding model once)

uv run t2sql ask "¿Qué clientes compraron canciones de Jazz?"
uv run t2sql ask "Which 3 artists generated the most revenue?" -v
```

The answer, the executed SQL and the result table are printed. With `-v` it also prints the
path through the graph, one row per node (provider/model, tokens in/out, latency, cost,
error) with totals, and the retriever's scores, selected tables and FK bridges.

## HTTP API

```bash
uv run uvicorn t2sql.api:app --reload      # http://127.0.0.1:8000, docs at /docs

curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which 3 artists generated the most revenue?"}'
```

```json
{
  "answer": "The top 3 artists by revenue are ...",
  "sql": "SELECT ... LIMIT 100",
  "columns": ["artist", "revenue"],
  "rows": [{"artist": "Iron Maiden", "revenue": 138.6}],
  "row_count": 3,
  "in_scope": true,
  "language": "en",
  "error": null,
  "candidate_tables": ["artist", "album", "track", "invoice_line", "invoice"],
  "attempts": [],
  "summary": {"latency_ms": 2310.4, "input_tokens": 2150, "output_tokens": 180,
              "cost_usd": 0.0023, "llm_calls": 3, "attempts": 1, "repaired": false,
              "first_try_valid": true}
}
```

- `POST /ask` takes `{"question": "..."}` (1 to 500 characters; else 422). Add
  `?include_trace=true` to get the per-node `trace`. Out-of-scope questions and failures
  after the repair budget is spent are `200` (`in_scope: false`, or `error` set); `503` if no
  LLM provider is configured or all of them failed; `500` with a generic message otherwise.
- `GET /health` reports database reachability (read-only role), whether the schema index
  exists, and the names of the LLM providers with a key. The app starts without API keys;
  only `/ask` answers 503.
- `Decimal` values are returned as floats and dates as ISO 8601 strings.

## Evals

`evals/dataset.yaml`: **55 items** over Chinook, each with a reference SQL that is executed
against the real database in the test suite.

| | Easy (1 table) | Medium (joins + aggregations) | Hard (subqueries, window functions, dates) | Out of scope |
|---|---|---|---|---|
| Items | 20 | 20 | 10 | 5 |
| English / Spanish | 10 / 10 | 10 / 10 | 5 / 5 | 2 / 3 |

So 50 answerable questions (25 + 25) and 5 that must be rejected (weather, "delete the
invoices", a football question, a coding request, and "what salary does each employee
earn", which the database does not hold).

### Metrics

- **Execution accuracy**: the result of the generated SQL equals the result of the
  reference SQL, never the SQL text. Both go through the same guard and `LIMIT`. Rows are
  compared as multisets (order only matters when the question implies it, e.g. top-N, and
  then rows tied on the sort key may come in any order);
  column names, aliases and column order are ignored (columns are matched by their
  values); numbers are rounded to 2 decimals; midnight timestamps equal dates. Extra
  columns in the generated result are accepted (headline figure); a stricter variant that
  requires the same number of columns is reported next to it. Rejected or failed
  questions count as wrong. Details in `evals/compare.py`.
- **Table recall** of the RAG: `|needed ∩ candidates| / |needed|` per question, averaged;
  `needed` is derived from the reference SQL with sqlglot. Also the share of questions with
  full recall and the average number of tables in the context (out of 11).
- **First-try valid SQL**: no guard rejection or database error on the first generated
  query. **Recovered by repair**: of the questions whose first attempt failed, how many
  ended with a result.
- **Correct rejection**: out-of-scope questions that were rejected; also the answerable
  ones that were wrongly rejected.
- **Latency p50/p95** (end to end, without the sleep between questions) and **cost per
  question** from the model prices in `tracing.py`.

### Results: schema retrieval (real, no LLM)

`uv run python -m evals.run --retrieval-only`, 50 answerable questions. "Before" is
embeddings only with k=4; "after" is the current default (keyword score + k=5).
The story behind the change is in [docs/eval-findings.md](docs/eval-findings.md).

| Level | N | Table recall before -> after | Questions with all needed tables, before -> after | Avg tables in context, before -> after |
|---|---|---|---|---|
| easy | 20 | 100% -> 100% | 100% -> 100% | 4.8 -> 6.3 |
| medium | 20 | 91.1% -> 100% | 80% -> 100% | 4.7 -> 5.8 |
| hard | 10 | 95.0% -> 100% | 80% -> 100% | 4.2 -> 5.8 |
| **all** | 50 | **95.4% -> 100%** | **88% -> 100%** | 4.7 -> 6.0 |

Caveat: the 100% is partly in-sample (the fix was designed after seeing which tables were
missing). On a separate 18-question probe set written beforehand (`evals/holdout.yaml`) the
same change moved recall from 83.4% to 95.6% (61% -> 89% of questions with every table).

### Results: end to end (real models, 2026-10-19)

Full dataset (50 answerable + 5 out-of-scope), one run per model, default settings (schema
RAG with k=5 and keyword score, at most 2 repairs), temperature 0 (Gemini 3.x ignores it and uses its own sampling defaults).
Models: `gemini-3.5-flash-lite` (Google AI Studio, free tier) and `openai/gpt-oss-20b`
(Groq, free tier). Raw per-question results are in `evals/results/` (gitignored), the table
below is `uv run python -m evals.run --report evals/results/full-gemini.json,evals/results/full-groq-20b.json`.
Cost is the list price of the paid tier applied to the tokens used (the runs themselves were
free). Latency is the end-to-end time of a question without the client-side throttle.

| Model | Level | N | Execution accuracy | Strict* | Table recall | Full recall | Avg tables in context | First-try valid | Recovered by repair | Latency p50 / p95 | Cost / question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini/gemini-3.5-flash-lite | easy | 20 | 100% (20/20) | 100% | 100% | 100% | 6.3 | 100% | 0/0 (n/a) | 6.3s / 61.4s | $0.00061 |
| gemini/gemini-3.5-flash-lite | medium | 20 | 100% (20/20) | 100% | 100% | 100% | 5.8 | 100% | 0/0 (n/a) | 5.0s / 9.9s | $0.00086 |
| gemini/gemini-3.5-flash-lite | hard | 10 | 100% (10/10) | 80.0% | 100% | 100% | 5.8 | 100% | 0/0 (n/a) | 7.4s / 11.6s | $0.00112 |
| gemini/gemini-3.5-flash-lite | **all answerable** | 50 | 100% (50/50) | 96.0% | 100% | 100% | 6.0 | 100% | 0/0 (n/a) | 5.5s / 29.9s | $0.00081 |
| groq/openai/gpt-oss-20b | easy | 20 | 100% (20/20) | 100% | 100% | 100% | 6.3 | 100% | 0/0 (n/a) | 8.1s / 12.0s | $0.00023 |
| groq/openai/gpt-oss-20b | medium | 20 | 90.0% (18/20) | 90.0% | 100% | 100% | 5.8 | 100% | 0/0 (n/a) | 11.6s / 16.0s | $0.00030 |
| groq/openai/gpt-oss-20b | hard | 10 | 80.0% (8/10) | 60.0% | 100% | 100% | 5.8 | 100% | 0/0 (n/a) | 16.1s / 18.9s | $0.00041 |
| groq/openai/gpt-oss-20b | **all answerable** | 50 | 92.0% (46/50) | 88.0% | 100% | 100% | 6.0 | 100% | 0/0 (n/a) | 10.3s / 18.0s | $0.00029 |

| Model | Out-of-scope correctly rejected | Answerable questions falsely rejected | Crashed runs |
|---|---|---|---|
| gemini/gemini-3.5-flash-lite | 5/5 | 0/50 | 0 |
| groq/openai/gpt-oss-20b | 5/5 | 0/50 | 0 |

*Strict = execution accuracy also requiring the same number of columns as the reference (the headline figure tolerates extra columns).

**Read these numbers with care.**

- The dataset is small (50 answerable questions: one question is 2 points) and it is the
  same one the retriever vocabulary was tuned on, so 100% is a development-set figure, not a
  claim about unseen questions.
- The comparator and one question were adjusted after the first real runs (listed in
  [docs/eval-findings.md](docs/eval-findings.md), section 7). Before those two changes the
  same Gemini model scored 96.0% (48/50) on an earlier full run.
- The repair loop is barely exercised: in the two runs above every first attempt was valid.
  In the earlier Gemini run one query came out without its `SELECT` after a CTE, the guard
  rejected it, and the repair fixed it (1/1 recovered); that is the only real recovery seen.
- The Gemini p95 is dominated by three easy questions that took 38 to 71 s. Gemini's API was
  returning `503 high demand` during the session, and the client retries with backoff, so
  this is most likely retry time (the run does not log retries, so I cannot prove it).
- `openai/gpt-oss-120b` (still the Groq default in `config.py`) could not be measured cleanly: the
  free tier allows 200,000 tokens per day and the debugging runs used it up. An earlier,
  partial-fix run gave 94.0% (47/50), but its raw file was overwritten, so it is not in the
  table. Re-run after the quota resets:
  `uv run python -m evals.run --providers groq --sleep 4 --out evals/results/full-groq-120b.json`.

**What fails.** Gemini failed nothing in the final run (two answers carry an extra column,
which is why strict is 96%). `gpt-oss-20b` failed 4 questions, all of them model mistakes
and not retrieval (table recall is 100%, so the right tables were always in the prompt):
"which artist has the most tracks" returned 100 rows instead of the top one (m05); "how many
songs does each playlist have" used an inner join and dropped empty playlists (m06); "who
bought in 2021 but not in 2025" lost a `DISTINCT` and returned duplicates (h06); and "who
bought in all five years" was read as "bought in any of the five years" (h10, also the one
`gpt-oss-120b` missed). Two more from that 120b run: a year-over-year variation that
treated the first year as 0 instead of NULL (h02), and a month shown as a date where the
reference uses the month number (h05, the question was reworded afterwards). Wrong
answers here are silent: the SQL is valid and returns rows, so nothing in the repair loop
can catch them.

**Full schema instead of RAG** ([ADR 0002](docs/adr/0002-schema-rag-instead-of-full-schema.md)),
`gemini-3.5-flash-lite`, `--retriever all` (all 11 tables in every prompt):

| Retriever | Execution accuracy | Avg tables in context | Tokens / question | Cost / question |
|---|---|---|---|---|
| RAG (k=5 + keyword score) | 100% (50/50) | 6.0 | 1,335 | $0.00076 |
| Full schema | 98.0% (49/50), see note | 11.0 | 1,551 | $0.00083 |

Note: the one miss in the full-schema run is a `504 DEADLINE_EXCEEDED` from the API on m13
(a crashed run, not a wrong query); re-running that single question gave a correct answer,
so on completed questions both are 50/50. RAG saves about 14% of the tokens and about 9%
of the cost per question here; on an 11-table schema it does not change accuracy.

### Running the evals

```bash
uv run python -m evals.run --retrieval-only              # RAG only, no API key
uv run python -m evals.run --fake                        # oracle "LLM" that returns the reference SQL: checks the harness (expects 100%)
uv run python -m evals.run --providers gemini,groq       # one separate run per provider, a model column in the table
uv run python -m evals.run --report a.json,b.json        # one combined table from earlier result files
T2SQL_GROQ_MODEL=openai/gpt-oss-20b uv run python -m evals.run --providers groq --level easy --limit 10
```

Other options: `--ids e01,m09`, `--level easy,hard`, `--limit N`, `--sleep SECONDS` (default 2
with a real LLM, for free-tier rate limits), `--out file.json` (raw per-question results; by
default `evals/results/`, which is gitignored), `--retriever all` (full schema as a
baseline), `--top-k`, `--lexical-weight`. The markdown tables go to stdout, progress to
stderr.

## Design decisions

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-security-in-code-not-in-prompt.md) | Security in code: AST guard, read-only transaction with timeout, read-only role |
| [0002](docs/adr/0002-schema-rag-instead-of-full-schema.md) | Retrieve the relevant tables (pgvector, local embeddings, FK bridges), measured by table recall |
| [0003](docs/adr/0003-capped-repair-loop.md) | Repair loop capped at 2, empty-result rule, every attempt traced |
| [0004](docs/adr/0004-langgraph-over-a-handwritten-loop.md) | LangGraph instead of a hand-written loop, and its trade-offs |

## Project structure

```
src/t2sql/
  graph.py          LangGraph definition and run_question()
  nodes/            classify, retrieve, generate, validate, execute, repair, respond
  sql_guard.py      sqlglot validation (pure, no database)
  db.py             read-only execution, schema introspection
  schema_index.py   embeddings in pgvector, ranking, keyword score, FK bridge expansion
  catalog.py        table descriptions, column hints, keyword vocabulary
  llm.py            provider-agnostic interface, Gemini/Groq/Anthropic, fallback
  tracing.py        cost and trace summaries
  cli.py            t2sql ask / t2sql index
  api.py            FastAPI: POST /ask, GET /health
evals/
  dataset.yaml      55 items with reference SQL     holdout.yaml  18-question probe set (retrieval)
  run.py            harness and markdown report     compare.py    result-set comparator
db/                 Chinook, pgvector extension, read-only role (loaded by docker compose)
docs/adr/           decision records                docs/eval-findings.md  the retrieval story
tests/
```

## Testing

```bash
uv run pytest                    # 379 tests
uv run pytest -m "not integration"   # 289 tests, no database needed
```

Tests marked `integration` need `docker compose up -d` and are skipped automatically if
the database is unreachable. They cover the read-only role and timeout, the graph against
the real database with a scripted fake LLM, every reference SQL in the datasets (guard,
non-empty result, below the row limit), and the eval harness end to end with the oracle
LLM. No test needs an API key.
