# 0005. An optional local "System 1" classifier in front of the cloud LLM

Status: accepted as an optional feature, default off (`T2SQL_CLASSIFIER=llm`). Numbers come
from the benchmark described below (2026-10-20); the set is small and hand-written, so they
are indicative.

## Context

Every question starts with `classify`: a cloud LLM call that decides whether the database
can answer it. That call costs a free-tier request (the quota is the scarcest resource: a
single eval run already hits the daily limits) and adds a network round trip of roughly
0.7 to 1 s before anything useful happens. The decision is a simple binary one, and about
a third of the traffic is unambiguous (clearly about customers and sales, or clearly a
greeting or general knowledge). A small local model could answer those in a fraction of the
time, if it errs on the safe side, and defer the rest.

What we need from it: high recall on out-of-scope questions (letting one through only costs
a wasted SQL generation, and the SQL guard still protects the database), a low false-rejection
rate on in-scope questions (a wrong refusal is what the user notices), English and Spanish,
and a score rather than a hard label so that a threshold band can send doubtful cases to
the cloud.

## Options evaluated

The full story, round by round, is in [the study](../system1-classifier-study.md).


Two rounds on labelled questions over Chinook (English and Spanish, rioplatense included).
Round 1: 110 questions (the project's own plus out-of-scope ones written for it). Round 2 added
162 adversarial questions (see `evals/scope_dataset.yaml` for the categories): unanswerable
attributes, polite writes, forecasts, opinions, other companies' data, meta questions,
injection embedded in a legitimate question, multi-intent questions, code requests, and
in-scope hard negatives (SQL verbs used as data words, employee and HR questions, terse and
misspelled questions). Total 272 items, split 136 dev / 136 test; every threshold was chosen
on dev and reported on test. Dev and test come from the same small pool and the author saw
both while writing, so differences of a few questions are noise.

Round 1 (110 items, test split; thresholds tuned on dev):

| Option | Out-of-scope recall | False rejection | Latency | Verdict |
|---|---|---|---|---|
| Sentence embeddings + classifier (MiniLM, mpnet, EmbeddingGemma) | at most 0.57 | n/a | ms | Cannot separate "about music" from "about this database" |
| NLI zero-shot (mDeBERTa-v3 XNLI) | 0.67 | 0.44 | | AUC about 0.48: close to noise here |
| qwen2.5:3b prompted (Ollama, GPU) | 0.857 | 0.118 | 145 ms | Too many misses, hard label |
| LiquidAI d1-3B, bf16 (transformers) | 0.952 (threshold tuned to 0.62; 0.714 at 0.5) | 0.029 | 83 ms | Good on the easy set |
| LiquidAI d1-3B, Q4 GGUF (llama-server) | 0.952 (tuned to 0.66; 0.571 at 0.5) | 0.088 (0.029 at 0.5) | 61 ms | Quantization costs false rejections once tuned |
| Tev1-4B-experimental, Q4 GGUF (llama-server) | 1.00 (at 0.5, untuned) | 0.029 | about 205 ms | Best recall |

Round 2 (all 272 items; the combined TEST half of 136 for the figures below):

| Option | AUC | At P = 0.5: accuracy / OOS recall / false rejection | Verdict |
|---|---|---|---|
| Tev1-4B Q4_K_M, mean of both option orders | 0.957 | 0.904 / 0.873 / 0.062 | Chosen |
| d1-3B Q4_K_M | 0.899 | OOS recall 0.48 | Rejected |
| d1-3B Q8_0 | 0.905 | OOS recall 0.55 | Rejected |

d1 is rejected because it does not separate the adversarial questions (AUC 0.90) and no
threshold band that was safe on dev held on test: the dev-tuned cascade bands produced 2 to 3
confident errors on test (a confident error is a wrong local verdict that the cloud never sees).
Q8 barely helps, so quantization is not the cause. Tev1 is slower (two requests of about
150 to 200 ms against one fast request) but separates much better.

Tev1 weak spots (test split): prompt injection embedded in a legitimate question (7 of 8 items
leak, P(in) 0.73 to 0.92, because the question really is answerable), multi-intent questions
(6 of 8 wrong), forecasts, and some employee/HR questions that it rejects (false rejections).
The cascade handles these by construction only when their P falls inside the band, so the band
is placed above the highest leak.

Tev1 and d1 are decision models: one forward pass yields a probability over the options, so
the answer is a calibrated-enough score, not a sampled label. For Tev1 the question is asked
twice with the options in both orders and P(in scope) is the mean, which removes the position
bias (this is what `Tev1Scorer` does and what was measured).

## Decision

Add a scorer abstraction (`local_classifier.py`: `ScopeScorer.score(question) -> ScopeScore`)
with a Tev1 and a d1 implementation, both talking HTTP to a user-run `llama-server`, and let
`classify` run in one of three modes (`T2SQL_CLASSIFIER`):

- `llm` (default): unchanged, the cloud LLM only.
- `local`: the scorer decides alone (P >= 0.5).
- `cascade`: P < `CASCADE_LOW` is out of scope, P > `CASCADE_HIGH` is in scope, anything in
  between goes to the cloud LLM. Defaults: model `tev1`, `CASCADE_LOW=0.10`,
  `CASCADE_HIGH=0.93`.

**How the band was chosen.** On dev, the widest band that made no confident error (no
answerable question below LOW, no out-of-scope question above HIGH), then moved slightly
for margin above the highest embedded-injection leaks (0.923 to 0.925). The
dev-only optimum was (0.0814, 0.9264); it was rounded to (0.10, 0.93), which lies in the
zero-error zone on test as well. Sensitivity on test (share decided locally / confident
errors): (0.0814, 0.9264) 50.7% / 0; (0.10, 0.93) 53% / 0; (0.15, 0.93) 54% / 0;
(0.20, 0.93) 57% / 1. The gain from loosening is small and the first error appears quickly,
so the conservative band was kept. The band must be re-tuned for any other model, prompt or
quantization.

Design points:

- **Never fail because of the local model.** Connection errors, timeouts (2 s) and malformed
  responses raise `LocalClassifierUnavailable`; the node logs one warning per outage and asks
  the cloud LLM. `/health` reports whether the scorer is reachable but does not degrade.
- **The prompts are the benchmarked ones**, built from `catalog.py` (the same table
  listing as the cloud prompt), so the local models stay in sync with the schema description.
- **The language is a heuristic** (`detect_language`), since neither model outputs it: it is
  only needed to choose the refusal message, and "en" is the default, as in the cloud prompt.
- **Standard-library HTTP** (urllib): one JSON POST per question does not justify a runtime
  dependency; tests inject a fake transport.
- **Visible in the trace.** `details` of the classify entry carry `decided_by`, `p_in`,
  `local_ms` and the model; local decisions have no LLM metadata, so cost and token summaries
  drop them naturally. The web UI, `ask -v`, `/health` and `evals.run --classifier` show who
  decided and the share of questions decided locally.

## Consequences

- With the cascade on, 52.9% of the test questions skipped the cloud classify call (72 of 136):
  less quota and cost, and about 0.33 s instead of a cloud round trip (0.7 to 1 s when the
  provider is healthy; the cloud run itself averaged 18 s that day because of Gemini timeouts).
- A wrong local verdict is not corrected by the cloud: a confident false rejection becomes a
  refusal. The band trades this against savings; widening it (lower LOW, higher HIGH) sends
  more to the cloud. The thresholds must be re-tuned when the model or its quantization
  changes (Q4 vs bf16 moved the false-rejection rate from 0.029 to 0.088 for d1).
- Operations: the user must run `llama-server` with a GGUF file; a 3 to 4B model at Q4 needs
  a few GB of VRAM (or runs slowly on CPU), which competes with whatever else uses the GPU.
  Fallback to the cloud makes this optional rather than a new point of failure.
- Licensing: the Tev1-4B-experimental model card says the license for the fine-tuned weights
  "is being finalized", so it should be treated as experiment-only until that is settled.
  d1-3B is under the `lfm1.0` license (Hugging Face metadata: `other`), whose terms (in
  particular for commercial use) must be read before deploying it. The code does not bundle
  either model; both are user-supplied.
- The prompts and options are tuned to this schema and these two models; another model needs
  its own request builder and a benchmark, which is why scorers are a small protocol.
- Measured end to end on the scope decision (`evals/scope_run.py`, test split, 136 questions,
  cloud = gemini-3.5-flash-lite):

  | Classifier | Accuracy | OOS recall | False rejection | Decided locally | Cloud calls |
  |---|---|---|---|---|---|
  | cloud LLM only | 91.2% | 83.1% | 0.0% | 0% | 136 |
  | Tev1 alone (P >= 0.5) | 90.4% | 87.3% | 6.2% | 100% | 0 |
  | cascade (0.10 / 0.93) | 91.2% | 83.1% | 0.0% | 52.9% | 64 |

  The cascade took exactly the cloud LLM's decisions (same 12 misses, none from the local
  half) while sparing 72 calls. The cloud classifier is itself weak on embedded injection (3 of
  4 leak) and multi-intent (4 of 4), so those categories are not a reason for or against the
  cascade; the SQL guard and the repair loop are what protect the database. Tev1 alone has more
  out-of-scope recall but rejects 4 answerable questions, which is why it is not offered as the
  recommended mode. Limits: one cloud model, one run (cloud calls are not deterministic), a
  small hand-written set shared by tuning and testing; and the agent-level eval
  (`evals.run --classifier cascade`) has not been run.
- The cascade run replayed the escalated decisions from the cloud run's cache, so its latency
  column covers only the local part.

> Update 2026-10-20: after the evaluation, d1 support was removed from the code; Tev1 is the only local model. The history above is kept as written.

## See also

- [docs/system1-classifier-study.md](../system1-classifier-study.md): the chronological study behind this decision (four rounds, the bar, what failed).
