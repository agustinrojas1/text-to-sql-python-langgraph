# Eval findings: the schema retriever missed the tables a purchase question needs

This is the story of the first thing the evals found, how it was measured, what was tried
and what actually helped. Everything below that mentions retrieval was produced without
any LLM key (`uv run python -m evals.run --retrieval-only`); end-to-end numbers (execution
accuracy, repair, cost) come in section 7, added on 2026-10-19 after the first real runs.

## 1. The symptom

While building Phase 3 I noticed by hand that for "¿Qué clientes compraron canciones de
Jazz?" the schema retriever (top-k = 4 by embedding similarity, plus foreign-key bridges)
returned `album`, `artist`, `genre`, `playlist`, `playlist_track`, `track` and **none of
`customer`, `invoice`, `invoice_line`**. The SQL generator only sees the tables the
retriever hands over, so no model can write that query from that context. The question
was whether this was one bad example or a pattern.

## 2. Measuring it

`evals/dataset.yaml` has 50 answerable questions (20 easy, 20 medium, 10 hard; 25 English,
25 Spanish) plus 5 out-of-scope ones. Each answerable question has a reference SQL, and the
tables it needs are derived from that SQL with sqlglot (`tests/test_eval_dataset.py` checks
the declared `tables` against it). Table recall for a question is
`|needed ∩ candidates| / |needed|`, where `candidates` is what the generator would receive
(top-k plus FK bridges).

Baseline (embeddings only, k = 4; reproduce with
`uv run python -m evals.run --retrieval-only --top-k 4 --lexical-weight 0`):

| Level | N | Mean table recall | Questions with every needed table | Avg tables in context |
|---|---|---|---|---|
| easy | 20 | 100% | 100% | 4.8 |
| medium | 20 | 91.1% | 80.0% | 4.7 |
| hard | 10 | 95.0% | 80.0% | 4.2 |
| **all answerable** | 50 | **95.4%** | **88.0%** | 4.7 |

Six questions lost at least one table, and by language Spanish was worse than English
(93.3% vs 97.6% mean recall, 25 questions each).

## 3. Failure patterns

| Pattern | Questions | What happened |
|---|---|---|
| Transactional intent is invisible: "who **bought** X" | m09, m10 | `customer`, `invoice`, `invoice_line` scored about 0.30 (`customer`) and 0.22 (`invoice_line`) while `album`, `playlist`, `genre` scored about 0.6. Nouns like "Jazz", "Classical", "tracks" dominate the embedding of the question; the verb is the part that points at the sales tables. Playlist names such as "Classical" even attracted `playlist`. |
| Revenue per genre/artist skips the sales line item | m08, m19, h08 | "Ingreso por género" retrieved `genre`, `track`, `album`, `artist`, but not `invoice_line`, the table holding `unit_price * quantity`. Nothing tied "revenue" to that table. |
| Spanish word not mapped to its table | h04 | "Género" did not pull in `genre` although `invoice`, `invoice_line` and `track` were found. |

A fourth effect shows in the other direction: `playlist` and `playlist_track` take a slot
in many questions that only say "songs" or "tracks".

So it was a pattern, not an example, and the pattern is a vocabulary gap: users describe
the business ("bought", "revenue", "compraron") and the schema documents describe the
structure.

## 4. What was tried

Every variant below was scored on two sets: the dataset (50 questions) and a 18-question
held-out probe set, `evals/holdout.yaml`, written before the fix was designed and never used
to tune it. It exists because a fix evaluated on the questions that motivated it proves
little. Columns are mean table recall / questions with full recall / average tables in
context.

| Variant | Dataset (50) | Held-out (18) |
|---|---|---|
| Baseline: embeddings, k=4 | 95.4% / 88% / 4.7 | 83.4% / 61% / 4.8 |
| Embeddings, k=5 | 96.4% / 92% / 6.1 | 89.5% / 72% / 6.2 |
| Embeddings, k=6 | 97.6% / 94% / 6.9 | 93.4% / 83% / 6.9 |
| Add EN/ES business vocabulary to the table documents before embedding, k=4 | 95.1% / 90% / 4.7 | 81.8% / 56% / 4.8 |
| Keyword score (weight 0.5), k=4 | 98.1% / 94% / 4.6 | 93.1% / 78% / 4.8 |
| **Keyword score (weight 0.5), k=5 (adopted)** | **100% / 100% / 6.0** | **95.6% / 89% / 6.2** |
| Keyword score (weight 0.5), k=6 | 100% / 100% / 6.7 | 96.7% / 94% / 7.1 |

What each one says:

- **Putting synonyms into the embedded text did not help** (95.4% to 95.1% on the dataset,
  83.4% to 81.8% on the held-out set). With a small multilingual embedding model, a longer
  table document is a blurrier one; the added words are diluted by the column list. This
  is the first idea that seemed obvious, and it was a null result (slightly negative).
  `t2sql index --keywords` still builds that variant for anyone who wants to re-run it.
- **More k alone helps but is expensive**: k=6 gets to 93% full recall on the held-out
  set, with 6.9 tables of 11 in the prompt.
- **A lexical keyword score fixes most of it at the same k.** `catalog.TABLE_KEYWORDS`
  holds the everyday English/Spanish words per table ("bought", "purchased", "revenue",
  "compraron", "ventas"...). The retriever adds `0.5 * overlap(question, keywords)` to the
  embedding score, where each shared word counts `1/df` (df = number of tables listing
  that word), so "song" (several tables) says little and "bought" (fewer) says more. Words
  are lowercased, accent-stripped, stopword-filtered, de-pluralised and cut to 5
  characters, a crude stemmer that happens to cover "compra/compró/compraron". At k=4 this
  already moves the dataset to 98.1% / 94% with no extra context.
- **Weight sensitivity (k=4, dataset recall / full, held-out recall / full):** 0 -> 95.4% /
  88%, 83.4% / 61%; 0.1 -> 95.9% / 90%, 89.5% / 72%; 0.2 -> 96.9% / 94%, 88.4% / 67%;
  0.3 -> 98.8% / 96%, 91.2% / 72%; 0.5 -> 98.1% / 94%, 93.1% / 78%; 0.8 -> 98.9% / 96%,
  91.9% / 78%; 1.2 -> 98.9% / 96%, 90.8% / 78%. Anything from 0.3 to 1.2 is on the plateau;
  0.5 was chosen as a round number inside it, not as the arg-max.
- **k=5 on top of it** buys the last missing tables on the dataset and the held-out set
  moves from 78% to 89% full recall, at 6.0 tables of 11 on average. k=6 adds a bit more
  recall for another table of context; I stopped at 5.

Not tried: LLM query rewriting (would need a key and adds a call), a bigger embedding model
(heavier local dependency), and an explicit "sales path" rule that adds
`customer-invoice-invoice_line-track` whenever one end is selected. The FK bridges already
do this when both ends are in the top-k; the problem was that the ends were not there.

## 5. Result (default settings now: k=5, keyword weight 0.5)

Same dataset, same command without overrides (`uv run python -m evals.run --retrieval-only`):

| Level | N | Mean table recall (before -> after) | Full recall (before -> after) | Avg tables in context (before -> after) |
|---|---|---|---|---|
| easy | 20 | 100% -> 100% | 100% -> 100% | 4.8 -> 6.3 |
| medium | 20 | 91.1% -> 100% | 80.0% -> 100% | 4.7 -> 5.8 |
| hard | 10 | 95.0% -> 100% | 80.0% -> 100% | 4.2 -> 5.8 |
| **all answerable** | 50 | **95.4% -> 100%** | **88.0% -> 100%** | 4.7 -> 6.0 |

By language, mean recall went from 97.6% (en) and 93.3% (es) to 100% and 100%.

Held-out probe set (18 questions, not used for tuning):

| Level | N | Mean table recall (before -> after) | Full recall (before -> after) |
|---|---|---|---|
| easy | 2 | 100% -> 100% | 100% -> 100% |
| medium | 13 | 83.1% -> 95.4% | 69.2% -> 92.3% |
| hard | 3 | 73.9% -> 93.3% | 0% -> 66.7% |
| **all** | 18 | **83.4% -> 95.6%** | **61.1% -> 88.9%** |

Two held-out questions still lose tables: "Who purchased Metal songs?" (no word says
"customer"; "purchased" is shared by three tables so each gets a third of the evidence) and
"¿Cuántas playlists incluyen canciones de AC/DC?" (the artist name in the question is data,
not vocabulary, so `artist` is not retrieved).

## 6. Caveats, stated plainly

- **The 100% on the dataset is partly in-sample.** I wrote `TABLE_KEYWORDS` after seeing
  which tables were missing, so some entries (revenue per genre/artist/album) describe
  exactly those cases. The held-out figure (83.4% -> 95.6%, 61% -> 89% full recall) is the
  better estimate of generalisation, and with 18 questions it is a rough one. The
  vocabulary is deliberately generic domain language and no dataset question is copied
  into it, but the risk of tuning to the questions you are looking at is real, and the
  held-out set was written by the same person.
- **The weight and k were picked from a sweep that includes the dataset**, although the
  conclusion (plateau between 0.3 and 1.2) does not depend on a fine value.
- **More recall costs context.** Average tables in the prompt went from 4.7 to 6.0 of 11,
  and single-table (easy) questions now carry 6.3 tables. Whether the extra tables hurt SQL
  accuracy, or just cost tokens, is not measured yet; it needs the LLM runs. An adaptive
  cut-off by score instead of a fixed k is the obvious next experiment.
- **Table recall is not execution accuracy.** It is an upper bound on what the generator
  can do with the context it gets, nothing more.
- **The keywords are hand-written for Chinook**, in two languages. A real deployment would
  need them per schema (or generated and then reviewed).
- The dataset has 55 items, the held-out set 18; differences of one question are 2 and 5.6
  percentage points respectively.

## 7. Reproduce

```bash
# before
uv run python -m evals.run --retrieval-only --top-k 4 --lexical-weight 0
uv run python -m evals.run --retrieval-only --top-k 4 --lexical-weight 0 --dataset evals/holdout.yaml
# after (defaults)
uv run python -m evals.run --retrieval-only
uv run python -m evals.run --retrieval-only --dataset evals/holdout.yaml
# the variant that did not help
uv run t2sql index --keywords --table schema_docs_kw
uv run python -m evals.run --retrieval-only --index-table schema_docs_kw --top-k 4 --lexical-weight 0
```
The index (`t2sql_meta.schema_docs`) is the same for before and after: the fix changes
scoring, not what is stored.


## 7. End to end with real models (2026-10-19)

First runs against real APIs. Everything here is from runs I executed that day; the final
tables are in the README. Models: `gemini-3.5-flash-lite` and `openai/gpt-oss-20b` (plus
partial data on `openai/gpt-oss-120b`).

### What broke before any accuracy could be measured

| Problem | Evidence | Fix |
|---|---|---|
| Free-tier quota, Gemini | `gemini-3.6-flash` (the old default): 429 `GenerateRequestsPerDayPerProjectPerModel`, limit 20 per day, hit after about 20 calls; 5 per minute too. Other 3.x flash models list the same 20. | Default changed to `gemini-3.5-flash-lite`, which completed three full runs in a day (about 170 calls each). `T2SQL_LLM_MIN_INTERVAL_S` throttles calls per provider (7 s used). |
| Gemini 503 "high demand" / 504 | crashed questions in the first quick run; one 504 in the full-schema run | `llm_max_retries` default 1 -> 3 (backoff in the client) |
| Groq 429 on tokens per minute (limit 8,000) | 9 of 55 questions crashed in the first full run | same retries (the message says "try again in 764 ms") |
| Groq structured output returns the schema, not an instance | 3 crashes: `Failed to parse GeneratedSql from completion {"description": "", "properties": ...}` | `ChatModelLLM.structured` retries up to 2 more times on unparsed output, summing the tokens of the discarded attempts |
| Groq 200,000 tokens per day | 47 of 55 questions crashed in a third full run in one day | not fixable in code; documented, `gpt-oss-120b` re-run pending |
| Throttle sleep counted as latency | p50 of 31 s for an easy question | `LLMResult.waited_ms`, subtracted from the traced latency |
| Cost `n/a` | new model without a price | prices added to `tracing.py` (checked on the providers' pages) |

The pre-fix run on `gpt-oss-120b` scored 78.0% (39/50), but 9 of the 11 misses were
crashes from the first three rows above, not wrong SQL. Reporting that number as the model's
accuracy would have been wrong in the other direction; the harness separates "crashed" from
"wrong" for that reason.

### Two things that were the harness, not the model

I changed these after seeing results, so they are listed explicitly:

1. **Comparator, ties in ordered results (e16).** "Total billed per country, highest
   first". Seven countries tie at 37.62, and the reference breaks ties by country name.
   The question does not ask for that, and both models returned the right totals in the
   right order with the tied countries in another order. `compare.py` now accepts a
   different order among rows that tie on a sort key of the reference (still requires the
   same key sequence and the same rows inside each tie group; tests added, including that
   a reversed or displaced order still fails). Without the change Gemini and gpt-oss-120b both
   failed e16.
2. **Question wording, h05.** "Cumulative revenue month by month during 2024". The reference
   returns the month number (1 to 12); both models returned the month as a date
   (`2024-01-01`) or text. Neither is wrong. Changed the question to "..., identifying each
   month by its number (1-12)." The SQL reference is unchanged. After the change Gemini passed
   (strict-only: it added a monthly revenue column). This is a dataset edit after seeing the
   result; before it the Gemini full run was 96.0% (48/50) with these two failures and nothing
   else.

No other reference SQL or question was changed. h02 and h10 below look like candidates and
I left them: the reference reading is the natural one.

### Remaining failures

All table recall 100%; none is a retrieval problem. Classification:

| Model | Question | Cause |
|---|---|---|
| gpt-oss-20b | m05 artist with most tracks | wrong interpretation: asked for "the" artist, returned 100 rows (no `LIMIT 1`) |
| gpt-oss-20b | m06 songs per playlist | wrong join: inner join drops empty playlists, the reference counts them as 0 ("each playlist") |
| gpt-oss-20b | h06 bought in 2021, not in 2025 | missing `DISTINCT`: one row per 2021 invoice |
| gpt-oss-20b, gpt-oss-120b | h10 bought in all five years | wrong interpretation: "any invoice in the range" instead of "in each of the five years" |
| gpt-oss-120b (earlier run) | h02 year-over-year variation | `COALESCE(prev, 0)`: the first year shows its whole total as variation instead of NULL |

Gemini flash-lite has no failure in the final run. The h10 error is the same in two
different models, which makes it look like a hard question ("in all five years" needs
`HAVING count(DISTINCT year) = 5`) and not a flaw in the reference.

### The repair loop, in practice

In the final runs every first attempt was valid, so repair was never needed. In an earlier
Gemini run (before the comparator change) the one first-try failure was h09: the model wrote
a CTE and then forgot the `SELECT` keyword; sqlglot rejected it (`parse_error: column does
not support CTE`), the repair prompt returned the error, and the second attempt was correct.
One recovery in one occurrence: the loop works, but this evaluation says almost nothing
about how well. A harder or noisier benchmark would be needed. And the failures that do
happen are the silent kind: valid SQL, rows returned, wrong meaning. The loop only sees
errors and empty results (ADR 0003).

### Full schema versus RAG

Gemini flash-lite with all 11 tables in every prompt: 49/50 (the miss is a 504 timeout; the
same question re-run was correct) against 50/50 with RAG. Tokens per question 1,551 versus
1,335 (RAG 14% fewer), cost per question $0.00083 versus $0.00076. The retrieval work
(section 2 to 5) was needed to make RAG not lose tables, but on this schema it is not
needed to answer correctly. Its benefit is cost and scaling, not accuracy.

### Caveats

- 50 questions, two models, one run each, temperature 0 where honoured. One question is 2
  points; I did not repeat runs, so run-to-run noise is unmeasured.
- The dataset and the keyword vocabulary were written by the same person, and the harness
  fixes above were made after seeing results.
- Free-tier throttling and provider outages affect latency numbers (Gemini p95), and
  `gpt-oss-120b` has no clean full run.
- Cost uses list prices applied to tokens from free-tier runs.
