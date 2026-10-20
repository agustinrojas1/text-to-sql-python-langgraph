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
empty result gets one retry and is then accepted as a valid answer. Only errors in the query
are repaired (guard rejections, SQL errors, statement timeouts, which also get a "write a
cheaper query" hint). A database outage (cannot connect, connection dropped, server
shutting down) is not: it propagates out of the graph, because rewriting valid SQL cannot
fix it and would only spend LLM calls. The `attempts` counter is the loop guard; an explicit
LangGraph `recursion_limit` derived from `T2SQL_MAX_REPAIRS` (17 for 2 repairs) is the
second safety net.

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
  error (or the empty result) and retries, at most twice; database outages are never
  "repaired", and a recursion limit backs up the counter.
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
uv run t2sql index                   # builds the schema index (downloads the ~240 MB embedding model once)

uv run t2sql ask "¿Qué clientes compraron canciones de Jazz?"
uv run t2sql ask "Which 3 artists generated the most revenue?" -v
```

The embedding model is cached in `~/.cache/t2sql/fastembed` (`T2SQL_EMBEDDING_CACHE_DIR`), so it is
downloaded only once, also across reboots. After a reboot only the database container is down:
run `docker compose up -d` again (if it is down, `t2sql ask` prints "Database unavailable" and
the API answers 503).

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
  LLM provider is configured, all of them failed (a quota or rate limit says "LLM quota or
  rate limit exceeded for <provider>; try again later or configure another provider", also
  with a single provider), or the database is unavailable ("Database unavailable"); `500` with a generic message otherwise.
- `GET /health` reports database reachability (read-only role), whether the schema index
  exists, and the names of the LLM providers with a key. The app starts without API keys
  or with the database down; only `/ask` answers 503.
- `POST /ask/stream` takes the same body and answers `application/x-ndjson`: one JSON object
  per line, in the order things happen. Event types: `start` (a node begins), `prompt` (the
  exact messages about to go to the LLM, for classify, generate_sql and respond), `token`
  (best-effort live output of the LLM: the decoded `sql` so far for generate_sql, the answer
  `text` so far for respond; providers that cannot stream, like Groq's structured output,
  send the whole value in one event, or none), and `node` (a node finished: attempt, latency,
  LLM provider/model/tokens/cost, error, and a `data` payload: the retrieval ranking with
  dense and keyword scores, column matches, FK bridge paths and the schema text the model
  sees; the generated SQL; the SQL guard's seven checks with tables, resolved columns,
  clauses and the LIMIT it added; an execution sample; the repair reason). It ends with one
  `final` event (the same fields as `/ask`) or an `error` event (`status` 503/500 with the
  same meaning as above, since the HTTP status is already sent). Clients should ignore event
  types and keys they do not know.
- `Decimal` values are returned as floats and dates as ISO 8601 strings.

## Web UI

```bash
uv run uvicorn t2sql.api:app        # then open http://127.0.0.1:8000
```

A single static page (`src/t2sql/web/index.html`, no build step, no external assets) served
at `GET /`. It sends the question to `/ask/stream` and fills in a step list as the agent
moves through classify, retrieve schema, generate SQL, validate, execute, repair and answer,
with the latency of each step and an expandable detail (SQL, retrieval scores, the prompt,
guard checks, tokens, cost). When it finishes it shows the answer, the result table, the SQL and the run totals.
Each question is independent: there is no conversation memory.

Next to the composer there is a selector for how the scope check runs, per question:
`LLM` (cloud only), `Cascade` (local model first, cloud for the uncertain band) and `Local`
(local model only). With Cascade or Local a second select picks the local model (Tev1 or d1).
The choice is remembered in the browser; options whose local server is not reachable say
"offline" (they still work: the request falls back to the cloud LLM and the classify step says
"local tev1 unavailable → cloud LLM"). Each turn shows which mode and model it used. The same
choice is available over HTTP and on the command line: `/ask` and `/ask/stream` accept
`"classifier": "llm" | "local" | "cascade"` and `"local_model": "tev1" | "d1"` in the body
(default: the server settings), and `t2sql ask --classifier cascade --local-model d1`.

## Demo without network

The demo normally needs the cloud LLM and the internet. To run it offline, record the questions once while online, then replay them:

```bash
# online: saves one file per question in recordings/ (default: the five examples of the web UI)
uv run t2sql record                          # or: uv run t2sql record "Top 3 artists by revenue" --dir demo/

# offline: recorded questions stream back from disk, no LLM and no database call
T2SQL_REPLAY_DIR=recordings uv run uvicorn t2sql.api:app
```

`T2SQL_RECORD_DIR` does the same recording for any run of the server (`/ask/stream` only), overwriting the file of that question. A run is saved only if it ends with a `final` event, so a failed run never replaces a good recording. Questions match after lowercasing, trimming and collapsing whitespace; a question without a recording runs the real agent. Replay sends the recorded lines unchanged, with the original pause between events capped at `T2SQL_REPLAY_MAX_GAP_S` (1.5 s by default). With replay on, `/health` reports `"replay": true` and the web UI shows a small "replay" badge in its header. Without either setting nothing changes.

Recordings are keyed by question **and** classifier mode / local model (`<slug>-<hash>-cascade-tev1.ndjson`,
the header repeats both), so a cascade recording is never replayed for an LLM-mode request; record each
mode you want to show (`uv run t2sql record --classifier cascade --local-model tev1`, and so on). A request
that does not choose a mode gets the server's default one (`T2SQL_CLASSIFIER`, `T2SQL_LOCAL_CLASSIFIER_MODEL`).
Recordings made before the selector existed (no mode in the file name or header) still replay, for
requests that resolve to the server's default mode only. In replay mode, picking a mode with no recording
for that question runs the agent live (which needs the network).

Recordings hold the full prompts and results of the run, so keep them out of the repository unless you want them there.

## Optional local classifier (cascade)

By default the cloud LLM decides whether a question is about the database (`classify`). An
optional small model running locally can take about half of those decisions: it returns
P(in scope) in roughly 300 ms on a consumer GPU (two forward passes of about 150 ms), spends
no tokens or quota, and in the benchmark below the cascade settled 53% of the test questions
on its own with the same scope accuracy as the cloud LLM alone (see
[ADR 0005](docs/adr/0005-local-system1-classifier-cascade.md); the full story of how the model was chosen is in [docs/system1-classifier-study.md](docs/system1-classifier-study.md)). It is off unless you set
`T2SQL_CLASSIFIER`:

| `T2SQL_CLASSIFIER` | Behavior |
|---|---|
| `llm` (default) | The cloud LLM decides, as before. Nothing else is needed. |
| `local` | The local model decides alone: in scope if P >= `T2SQL_LOCAL_THRESHOLD` (0.5). |
| `cascade` | P < `T2SQL_CASCADE_LOW` is out of scope, P > `T2SQL_CASCADE_HIGH` is in scope (both decided locally); in between, the cloud LLM decides. |

If the local server is down or answers badly, `local` and `cascade` fall back to the cloud
LLM (one warning in the log per outage); a request never fails because of it. The language
of a locally decided question (needed for the refusal message) comes from a small
heuristic (`detect_language`: `¿ ¡ ñ` and English/Spanish function words, including
rioplatense forms such as "tenés" or "mostrame").

Two models are supported, both GGUF files served by llama.cpp's `llama-server`
(`T2SQL_LOCAL_CLASSIFIER_MODEL`). The default and the recommended one is `tev1`:

- `tev1`: Tev1-4B-experimental, queried through `/completion`. P(in scope) is the
  probability of the option letter A against B, averaged over two requests with the options
  in both orders (this cancels position bias, and is exactly what was benchmarked).
- `d1`: LiquidAI d1-3B, queried through `/v1/systemone` with a `choice` question. Supported,
  but not recommended: it separates worse on the adversarial set (see the ADR).

Setup for `tev1` (about 3 GB of VRAM; tested with an 8 GB NVIDIA GPU):

```bash
# 1. llama.cpp with CUDA (or download a prebuilt release from github.com/ggml-org/llama.cpp/releases
#    and skip the build; CPU-only works too, only slower). Benchmarked at commit 9df0b73.
git clone https://github.com/ggml-org/llama.cpp ~/.cache/t2sql/llama.cpp
cmake -S ~/.cache/t2sql/llama.cpp -B ~/.cache/t2sql/llama.cpp/build -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86          # 86 = RTX 30xx; use your GPU's compute capability
cmake --build ~/.cache/t2sql/llama.cpp/build --config Release -j --target llama-server

# 2. The model: Q4_K_M quantization, 2.8 GB
mkdir -p ~/.cache/t2sql/models/Tev1-4B-GGUF
curl -L -o ~/.cache/t2sql/models/Tev1-4B-GGUF/togethercomputer_Tev1-4B-experimental-Q4_K_M.gguf \
  https://huggingface.co/bartowski/togethercomputer_Tev1-4B-experimental-GGUF/resolve/main/togethercomputer_Tev1-4B-experimental-Q4_K_M.gguf

# 3. The server (-np 1: one slot, the requests are sequential; -c 2048 is enough for the prompt)
~/.cache/t2sql/llama.cpp/build/bin/llama-server \
  -m ~/.cache/t2sql/models/Tev1-4B-GGUF/togethercomputer_Tev1-4B-experimental-Q4_K_M.gguf \
  -ngl 99 -c 2048 -np 1 --host 127.0.0.1 --port 8090

# 4. The agent
T2SQL_CLASSIFIER=cascade uv run t2sql ask -v "How many customers?"
```

**Both models in the web UI.** One llama-server serves one model, so each model has its own
port: Tev1 on 8090 (`T2SQL_TEV1_URL`) and d1 on 8091 (`T2SQL_D1_URL`). Start whichever you want
to offer (the UI marks the other "offline"):

```bash
LS=~/.cache/t2sql/llama.cpp/build/bin/llama-server
$LS -m ~/.cache/t2sql/models/Tev1-4B-GGUF/togethercomputer_Tev1-4B-experimental-Q4_K_M.gguf \
    -ngl 99 -c 2048 -np 1 --host 127.0.0.1 --port 8090 &      # ~3 GB VRAM
$LS -m ~/.cache/t2sql/models/d1-3B-GGUF/d1-3B-Q4_K_M.gguf \
    -ngl 99 -c 2048 -np 1 --host 127.0.0.1 --port 8091 &      # ~2 GB VRAM
uv run uvicorn t2sql.api:app                                  # pick the mode and model in the page
```

`T2SQL_LOCAL_CLASSIFIER_URL` still works: when set, it overrides the URL of the default model
(`T2SQL_LOCAL_CLASSIFIER_MODEL`) only; the other model keeps its own `T2SQL_TEV1_URL` /
`T2SQL_D1_URL`. `/health` adds `local_models` (`{"tev1": true, "d1": false}`) and `local_model`
(the default).

**License caveat**: the Tev1-4B-experimental model card says the license of the fine-tuned
weights "is being finalized" (the base model, Qwen3.5-4B, is Apache-2.0). Treat it as
experimental until that is settled; the repository does not bundle it. d1-3B is under the
LFM Open License v1.0.

Settings (see `.env.example`): `T2SQL_LOCAL_CLASSIFIER_MODEL` (`tev1`),
`T2SQL_TEV1_URL` (default `http://127.0.0.1:8090`), `T2SQL_D1_URL` (default `http://127.0.0.1:8091`),
`T2SQL_LOCAL_CLASSIFIER_URL` (optional override for the default model's URL),
`T2SQL_LOCAL_CLASSIFIER_TIMEOUT_S` (2, per request), `T2SQL_LOCAL_THRESHOLD` (0.5, `local`
mode), `T2SQL_CASCADE_LOW` (0.10), `T2SQL_CASCADE_HIGH` (0.93). The band was chosen on the dev
split of the scope benchmark; raising LOW or lowering HIGH sends more questions to the local
model and makes confident mistakes possible (sensitivity in the ADR).

Measured with `evals/scope_run.py` on the 136-question test split of
`evals/scope_dataset.yaml` (71 out of scope, 65 answerable; English and Spanish; the cloud
LLM is `gemini-3.5-flash-lite`; 2026-10-20):

| Classifier | Scope accuracy | OOS recall | False rejection | Decided locally | Cloud classify calls | Mean latency |
|---|---|---|---|---|---|---|
| `llm` (cloud only) | 91.2% | 83.1% | 0.0% | 0% | 136 | 18.3 s (Gemini was slow and timed out on some calls that day; p95 56 s) |
| `local` (Tev1 alone, threshold 0.5) | 90.4% | 87.3% | 6.2% | 100% | 0 | 0.33 s |
| `cascade` (0.10 / 0.93) | 91.2% | 83.1% | 0.0% | 52.9% | 64 | 0.33 s locally decided; escalated ones add one cloud call |

The cascade made the same decisions as the cloud LLM on all 136 questions and made no
confident local mistakes, while skipping the cloud call for 72 of them. Its cloud latency
is not shown because the escalated decisions were replayed from the `llm` run's cache.
The cloud LLM's own misses are the prompt-injection-embedded questions and the multi-intent
ones (one answerable part, one not); the SQL guard still protects the database in those
cases, the cost is a wasted generation. Caveats: the set is small and hand-written, dev and
test come from the same pool of questions, and only one cloud model was measured, so read the
differences between rows as indicative (one question is 0.7 points).

Who decided is visible everywhere: the trace entry of `classify` has
`details = {classifier, decided_by: "local" | "llm", p_in, local_ms, local_model}`; `t2sql ask -v` prints
"decided locally (local d1 p=0.97, 180 ms)"; the web UI shows "in scope · es · local d1
p=0.97" or "· cloud LLM"; `/health` adds `classifier` and `local_classifier` (reachable or
not, short timeout; it never makes the status `degraded`). Locally decided classifications
have no LLM metadata, so they add no tokens or cost to the summary.
`uv run python -m evals.run --classifier cascade` runs the full agent evals with it and adds a
table with the share of questions decided locally and the cloud calls per question. To measure
only the scope decision (fast, and the way the numbers above were produced):

```bash
uv run python -m evals.scope_run --classifier local   --split test
uv run python -m evals.scope_run --classifier llm     --split test --providers gemini --sleep 4 --cache /tmp/cloud.json
uv run python -m evals.scope_run --classifier cascade --split test --providers gemini --cache /tmp/cloud.json
```

`--cache` stores each cloud decision, so a run interrupted by a quota limit resumes where it
stopped and `cascade` reuses the answers of the `llm` run instead of calling the cloud again.

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
| [0005](docs/adr/0005-local-system1-classifier-cascade.md) | Optional local "System 1" scope classifier in front of the cloud LLM (cascade, off by default) |

The write-up of how the local classifier was chosen (four rounds of measurements, the bar, what failed) is [docs/system1-classifier-study.md](docs/system1-classifier-study.md); the retrieval story is [docs/eval-findings.md](docs/eval-findings.md).

## Project structure

```
src/t2sql/
  graph.py          LangGraph definition and run_question()
  nodes/            classify, retrieve, generate, validate, execute, repair, respond
  sql_guard.py      sqlglot validation (pure, no database)
  db.py             read-only execution, schema introspection
  schema_index.py   embeddings in pgvector, ranking, keyword score, FK bridge expansion
  local_classifier.py  optional local scope scorers (llama-server: Tev1, d1), language heuristic
  catalog.py        table descriptions, column hints, keyword vocabulary
  llm.py            provider-agnostic interface, Gemini/Groq/Anthropic, fallback
  tracing.py        cost and trace summaries
  cli.py            t2sql ask / index / record
  replay.py         record and replay of /ask/stream runs (offline demo)
  api.py            FastAPI: POST /ask, GET /health
evals/
  dataset.yaml      55 items with reference SQL     holdout.yaml  18-question probe set (retrieval)
  run.py            harness and markdown report     compare.py    result-set comparator
  scope_dataset.yaml  272 labelled questions (EN/ES) for the scope decision
  scope_run.py      classify-only evaluation (llm | local | cascade)
db/                 Chinook, pgvector extension, read-only role (loaded by docker compose)
docs/adr/           decision records                docs/eval-findings.md  the retrieval story
                    docs/system1-classifier-study.md  how the local classifier was chosen
tests/
```

## Testing

```bash
uv run pytest                    # 534 tests
uv run pytest -m "not integration"   # 444 tests, no database needed
```

Tests marked `integration` need `docker compose up -d` and are skipped automatically if
the database is unreachable. They cover the read-only role and timeout, the graph against
the real database with a scripted fake LLM, every reference SQL in the datasets (guard,
non-empty result, below the row limit), and the eval harness end to end with the oracle
LLM. The CLI (`typer` `CliRunner`), settings loading from `.env`, and the infrastructure-vs-query
error split in `db.py` have their own tests. No test needs an API key.
