"""Evaluation harness: `uv run python -m evals.run [options]`.

Modes
  (default)         run the real agent once per configured provider/model
  --retrieval-only  only the schema retriever (no LLM, no API key): table recall / context size
  --fake            an oracle "LLM" that returns the reference SQL: proves the harness
                    end to end (expects ~100% execution accuracy), no API key

Metric definitions live in `aggregate()` and are summarised in the README.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from evals.compare import compare_results
from evals.dataset import DATASET_PATH, LEVELS, Item, load_dataset
from t2sql.config import Settings, get_settings
from t2sql.db import QueryError, QueryResult, execute_readonly
from t2sql.graph import build_graph, run_question
from t2sql.llm import PROVIDERS, LLM, LLMResult, LLMUnavailableError, Message, build_llm
from t2sql.nodes.classify import ScopeDecision
from t2sql.nodes.generate import GeneratedSql
from t2sql.retrieval import AllTablesRetriever, SchemaRetriever, retrieve_with_details
from t2sql.schema_index import FastEmbedEmbedder, PgVectorRetriever, default_retriever, index_exists
from t2sql.sql_guard import validate_sql

RESULTS_DIR = Path(__file__).with_name("results")


# ---------------------------------------------------------------- oracle "LLM" (--fake)


class OracleLLM:
    """Answers every call with the dataset's reference, looked up by the question text.

    classify -> in_scope iff the item has reference SQL; generate -> the reference SQL.
    """

    provider = "fake"
    model = "oracle"

    def __init__(self, items: list[Item], fault: str | None = None) -> None:
        """`fault` (tests): "wrong_answer" always generates `SELECT 1`; "broken_first" generates
        invalid SQL once per question so the repair loop has to recover."""
        self._by_question = {i.question: i for i in items}
        self._fault = fault
        self._broken: set[str] = set()

    def _result[T](self, output: T) -> LLMResult[T]:
        return LLMResult(output, self.provider, self.model, 10, 5, 1.0)

    def structured[S: BaseModel](self, messages: list[Message], schema: type[S]) -> LLMResult[S]:
        item = self._by_question[messages[-1].content.split("\n")[0].removeprefix("Question: ")]
        if schema is ScopeDecision:
            out: BaseModel = ScopeDecision(
                in_scope=item.in_scope, reason="oracle", language=item.language  # type: ignore[arg-type]
            )
        elif schema is GeneratedSql:
            if item.sql is None:
                raise AssertionError(f"{item.id}: oracle asked to generate SQL for an out-of-scope item")
            sql = item.sql
            if self._fault == "wrong_answer":
                sql = "SELECT 1 AS wrong"
            elif self._fault == "broken_first" and item.id not in self._broken:
                self._broken.add(item.id)
                sql = "SELEC nothing FROM"
            out = GeneratedSql(sql=sql, tables_used=list(item.tables))
        else:
            raise AssertionError(f"OracleLLM: unexpected schema {schema}")
        return self._result(out)  # type: ignore[return-value]

    def text(self, messages: list[Message]) -> LLMResult[str]:
        return self._result("(oracle answer)")


# ---------------------------------------------------------------- reference results


def reference_result(item: Item, settings: Settings) -> QueryResult:
    """Run the reference SQL exactly as the agent would run generated SQL (guard + LIMIT)."""
    assert item.sql is not None
    return execute_readonly(validate_sql(item.sql, set(settings.allowed_tables), settings.max_rows).sql, settings)


# ---------------------------------------------------------------- per-question evaluation


def table_recall(needed: tuple[str, ...], candidates: list[str]) -> float | None:
    return len(set(needed) & set(candidates)) / len(needed) if needed else None


def evaluate_item(
    graph: CompiledStateGraph, item: Item, ref: QueryResult | None, settings: Settings
) -> dict[str, Any]:
    """Run one question through the agent and score it. Never raises."""
    rec: dict[str, Any] = {
        "id": item.id, "level": item.level, "language": item.language, "question": item.question,
        "needed_tables": list(item.tables),
    }  # fmt: skip
    wall = time.perf_counter()
    try:
        state = run_question(graph, item.question, settings)
    except Exception as e:  # LLM outage after fallbacks, DB down...
        rec.update(crashed=f"{type(e).__name__}: {e}", exec_match=False, exec_strict=False)
        if item.in_scope:
            rec["rejected"] = None
        rec["wall_ms"] = round((time.perf_counter() - wall) * 1000, 1)
        return rec

    summary = state["summary"]
    rejected = not state.get("in_scope", True)
    rec.update(
        rejected=rejected,
        sql=state.get("sql"),
        error=state.get("error"),
        answer=state.get("answer"),
        candidate_tables=state.get("candidate_tables"),
        path=[e["node"] for e in state["trace"]],
        attempt_history=state.get("attempt_history", []),
        summary=summary,
        model=next((e["llm"]["provider"] + "/" + e["llm"]["model"] for e in state["trace"] if e["llm"]), None),
        wall_ms=round((time.perf_counter() - wall) * 1000, 1),
    )
    # The run ended with an answer from executed rows (not a rejection, not respond_error).
    rec["ended_ok"] = not rejected and state.get("rows") is not None and not state.get("error")
    if not item.in_scope:
        rec["correct_rejection"] = rejected
        return rec

    if state.get("candidate_tables") is not None:
        rec["table_recall"] = table_recall(item.tables, state["candidate_tables"])
        rec["n_candidates"] = len(state["candidate_tables"])
    if rejected or state.get("rows") is None or state.get("error"):
        rec.update(exec_match=False, exec_strict=False, mismatch="rejected" if rejected else "no result")
        return rec
    assert ref is not None and item.sql is not None
    cmp = compare_results(ref.columns, ref.rows, state["columns"], state["rows"], item.order_matters)
    rec.update(exec_match=cmp.match, exec_strict=cmp.strict, mismatch=cmp.reason)
    return rec


def evaluate_retrieval(retriever: SchemaRetriever, item: Item) -> dict[str, Any]:
    tables, details = retrieve_with_details(retriever, item.question)
    names = [t.name for t in tables]
    selected = details.get("selected", names)
    return {
        "id": item.id, "level": item.level, "language": item.language, "question": item.question,
        "needed_tables": list(item.tables),
        "candidate_tables": names,
        "selected": selected,
        "expanded": details.get("expanded", []),
        "scores": details.get("scores"),
        "table_recall": table_recall(item.tables, names),
        "table_recall_topk": table_recall(item.tables, selected),
        "n_candidates": len(names),
        "n_selected": len(selected),
    }  # fmt: skip


# ---------------------------------------------------------------- aggregation


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (p in 0..100); None on empty input."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics over a set of per-question records (one model).

    Definitions
      n_in_scope           answerable questions (level != out_of_scope)
      exec_accuracy        in-scope questions whose result matches the reference (lenient: extra
                           columns allowed) / n_in_scope. Rejections and crashes count as wrong.
      exec_accuracy_strict same, but the number of columns must also match
      table_recall         mean over in-scope questions that reached retrieval of
                           |needed ∩ candidate| / |needed|; candidates = top-k + FK bridges
      full_recall          share of those questions with recall == 1
      avg_candidates       mean number of tables handed to the SQL generator
      first_try_valid      among questions that reached generation: no guard rejection or execution
                           error on the first generated query (an empty result is still "valid")
      recovered            among questions whose first attempt failed: share that ended with a result
                           after the repair loop
      correct_rejection    out-of-scope questions that were rejected / n_out_of_scope
      false_rejection      in-scope questions rejected / n_in_scope
      latency p50/p95      end-to-end graph time (sum of node latencies, excluding --sleep), all
                           questions that did not crash
      cost_per_question    mean USD over questions with a priced model (None if unpriced / fake)
    """
    ins = [r for r in records if r["level"] != "out_of_scope"]
    oos = [r for r in records if r["level"] == "out_of_scope"]
    ok = [r for r in records if "summary" in r]
    gen = [r for r in ins if r.get("summary") and r["summary"]["attempts"] > 0]
    failed_first = [r for r in gen if not r["summary"]["first_try_valid"]]
    recovered = [r for r in failed_first if r.get("ended_ok")]
    recall = [r["table_recall"] for r in ins if r.get("table_recall") is not None]
    costs = [r["summary"]["cost_usd"] for r in ok if r["summary"]["cost_usd"] is not None]
    lat = [r["summary"]["latency_ms"] / 1000 for r in ok]
    return {
        "n": len(records), "n_in_scope": len(ins), "n_out_of_scope": len(oos),
        "crashed": sum("crashed" in r for r in records),
        "exec_correct": sum(r["exec_match"] for r in ins),
        "exec_accuracy": _ratio(sum(r["exec_match"] for r in ins), len(ins)),
        "exec_accuracy_strict": _ratio(sum(r["exec_strict"] for r in ins), len(ins)),
        "table_recall": _mean(recall), "n_recall": len(recall),
        "full_recall": _ratio(sum(x == 1 for x in recall), len(recall)),
        "avg_candidates": _mean([r["n_candidates"] for r in ins if "n_candidates" in r]),
        "first_try_valid": _ratio(sum(r["summary"]["first_try_valid"] for r in gen), len(gen)),
        "first_try_failed": len(failed_first), "recovered": len(recovered),
        "recovered_rate": _ratio(len(recovered), len(failed_first)),
        "correct_rejection": _ratio(sum(r["correct_rejection"] for r in oos if "correct_rejection" in r), len(oos)),
        "correct_rejections": sum(r.get("correct_rejection", False) for r in oos),
        "false_rejections": sum(bool(r.get("rejected")) for r in ins),
        "latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
        "cost_per_question": _mean(costs), "n_priced": len(costs),
        "tokens_per_question": _mean([r["summary"]["input_tokens"] + r["summary"]["output_tokens"] for r in ok]),
    }  # fmt: skip


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


# ---------------------------------------------------------------- markdown


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%" if x in (0, 1) else f"{100 * x:.1f}%"


def _num(x: float | None, fmt: str = ".1f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.5f}"


def group_by_level(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups = [(lv, [r for r in records if r["level"] == lv]) for lv in LEVELS if lv != "out_of_scope"]
    groups = [(lv, rs) for lv, rs in groups if rs]
    ins = [r for r in records if r["level"] != "out_of_scope"]
    if ins and len(groups) > 1:
        groups.append(("**all answerable**", ins))
    return groups


def agent_markdown(runs: dict[str, list[dict[str, Any]]]) -> str:
    head = (
        "| Model | Level | N | Execution accuracy | Strict* | Table recall | Full recall | Avg tables in context "
        "| First-try valid | Recovered by repair | Latency p50 / p95 | Cost / question |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for model, recs in runs.items():
        for level, rs in group_by_level(recs):
            a = aggregate(rs)
            rows.append(
                f"| {model} | {level} | {a['n_in_scope']} | {_pct(a['exec_accuracy'])} ({a['exec_correct']}/{a['n_in_scope']}) "
                f"| {_pct(a['exec_accuracy_strict'])} | {_pct(a['table_recall'])} | {_pct(a['full_recall'])} "
                f"| {_num(a['avg_candidates'])} | {_pct(a['first_try_valid'])} "
                f"| {a['recovered']}/{a['first_try_failed']} ({_pct(a['recovered_rate'])}) "
                f"| {_num(a['latency_p50_s'])}s / {_num(a['latency_p95_s'])}s | {_usd(a['cost_per_question'])} |"
            )
    scope = ["| Model | Out-of-scope correctly rejected | Answerable questions falsely rejected | Crashed runs |", "|---|---|---|---|"]
    for model, recs in runs.items():
        a = aggregate(recs)
        scope.append(
            f"| {model} | {a['correct_rejections']}/{a['n_out_of_scope']} | {a['false_rejections']}/{a['n_in_scope']} | {a['crashed']} |"
        )
    note = (
        "\n*Strict = execution accuracy also requiring the same number of columns as the reference "
        "(the headline figure tolerates extra columns)."
    )
    return head + "\n" + "\n".join(rows) + "\n\n" + "\n".join(scope) + note


def retrieval_markdown(records: list[dict[str, Any]], label: str) -> str:
    head = (
        "| Retriever | Level | N | Mean table recall | Full recall (all needed tables) | Recall of top-k only (no FK bridges) "
        "| Avg tables in context |\n|---|---|---|---|---|---|---|"
    )
    rows = []
    for level, rs in group_by_level(records):
        rec = [r["table_recall"] for r in rs]
        topk = [r["table_recall_topk"] for r in rs]
        rows.append(
            f"| {label} | {level} | {len(rs)} | {_pct(_mean(rec))} | {_pct(_ratio(sum(x == 1 for x in rec), len(rec)))} "
            f"| {_pct(_mean(topk))} | {_num(_mean([r['n_candidates'] for r in rs]))} |"
        )
    misses = [r for r in records if r["table_recall"] < 1]
    out = head + "\n" + "\n".join(rows)
    if misses:
        out += "\n\nQuestions with missing tables:\n\n" + "\n".join(
            f"- `{r['id']}` ({r['level']}) missing {sorted(set(r['needed_tables']) - set(r['candidate_tables']))} "
            f"- got {r['candidate_tables']} - {r['question']}"
            for r in misses
        )
    return out


# ---------------------------------------------------------------- driver


@dataclass
class Run:
    label: str
    graph: CompiledStateGraph


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def select_items(items: list[Item], args: argparse.Namespace) -> list[Item]:
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        unknown = wanted - {i.id for i in items}
        if unknown:
            raise SystemExit(f"unknown ids: {sorted(unknown)}")
        items = [i for i in items if i.id in wanted]
    if args.level:
        levels = {x.strip() for x in args.level.split(",")}
        if levels - set(LEVELS):
            raise SystemExit(f"unknown levels: {sorted(levels - set(LEVELS))} (known: {', '.join(LEVELS)})")
        items = [i for i in items if i.level in levels]
    return items[: args.limit] if args.limit else items


def build_retriever(settings: Settings, index_table: str | None = None) -> SchemaRetriever:
    if index_table:  # an alternative pgvector index (e.g. the pre-fix baseline), see docs/eval-findings.md
        embedder = FastEmbedEmbedder(settings.embedding_model)
        if not index_exists(embedder.name, settings, index_table):
            raise RuntimeError(f"index table {index_table!r} not found; build it with `t2sql index --table {index_table}`")
        r: SchemaRetriever = PgVectorRetriever(embedder, settings, table=index_table)
    else:
        r = default_retriever(settings)
    log(f"retriever: {type(r).__name__} (top_k={settings.retrieval_top_k})")
    return r


def make_runs(args: argparse.Namespace, settings: Settings, items: list[Item], retriever: SchemaRetriever) -> list[Run]:
    if args.fake:
        return [Run("fake/oracle", build_graph(OracleLLM(items), retriever, settings))]
    names = [p.strip().lower() for p in args.providers.split(",")] if args.providers else list(settings.llm_providers)
    if args.model and len(names) != 1:
        raise SystemExit("--model needs exactly one provider (--providers NAME)")
    runs = []
    for name in names:
        if name not in PROVIDERS:
            raise SystemExit(f"unknown provider '{name}' (known: {', '.join(PROVIDERS)})")
        s = settings.model_copy(update={"llm_providers": [name], **({f"{name}_model": args.model} if args.model else {})})
        try:
            llm: LLM = build_llm(s)
        except LLMUnavailableError as e:
            if args.providers:
                raise SystemExit(str(e)) from e
            log(f"skipping {name}: no API key")
            continue
        runs.append(Run(f"{name}/{getattr(s, f'{name}_model')}", build_graph(llm, retriever, s)))
    if not runs:
        raise SystemExit(
            "No LLM provider has an API key (see .env.example). "
            "Use --retrieval-only for the RAG metrics or --fake to smoke-test the harness."
        )
    return runs


def run_agent(runs: list[Run], items: list[Item], settings: Settings, sleep_s: float) -> dict[str, list[dict[str, Any]]]:
    refs: dict[str, QueryResult] = {}
    for i in items:
        if i.sql:
            try:
                refs[i.id] = reference_result(i, settings)
            except QueryError as e:
                raise SystemExit(f"reference SQL of {i.id} fails: {e.message}") from e
    out: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        recs = out[run.label] = []
        for n, item in enumerate(items, 1):
            rec = evaluate_item(run.graph, item, refs.get(item.id), settings)
            recs.append(rec)
            verdict = "crashed" if "crashed" in rec else ("ok" if rec.get("exec_match") or rec.get("correct_rejection") else "FAIL")
            note = f": {rec['crashed'][:200]}" if "crashed" in rec else ""
            log(f"[{run.label}] {n}/{len(items)} {item.id} {verdict}{note}")
            if sleep_s and n < len(items):
                time.sleep(sleep_s)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default=str(DATASET_PATH))
    p.add_argument("--ids", help="comma-separated item ids, e.g. e01,m09")
    p.add_argument("--level", help="comma-separated levels: easy,medium,hard,out_of_scope")
    p.add_argument("--limit", type=int, help="only the first N items after filtering (dataset order: easy first)")
    p.add_argument("--providers", help="comma-separated providers, one separate run each (gemini,groq,anthropic)")
    p.add_argument("--model", help="model id override (needs exactly one provider)")
    p.add_argument("--retrieval-only", action="store_true", help="only the retriever; no LLM and no API key needed")
    p.add_argument("--fake", action="store_true", help="oracle LLM that returns the reference SQL (harness smoke test)")
    p.add_argument("--retriever", choices=["auto", "pgvector", "all"], help="override T2SQL_RETRIEVER")
    p.add_argument("--index-table", help="pgvector index table in t2sql_meta (default: schema_docs)")
    p.add_argument("--lexical-weight", type=float, help="override T2SQL_RETRIEVAL_LEXICAL_WEIGHT")
    p.add_argument("--top-k", type=int, help="override T2SQL_RETRIEVAL_TOP_K")
    p.add_argument("--sleep", type=float, help="seconds between questions (default 2 with a real LLM, else 0)")
    p.add_argument("--out", help="raw results JSON (default: evals/results/<timestamp>-<mode>.json)")
    p.add_argument("--report", help="comma-separated result JSONs of earlier agent runs: print one combined table and exit")
    args = p.parse_args(argv)

    if args.report:
        merged: dict[str, list[dict[str, Any]]] = {}
        for f in args.report.split(","):
            merged.update(json.loads(Path(f).read_text(encoding="utf-8"))["results"])
        print(agent_markdown(merged))
        return 0

    settings = get_settings()
    updates: dict[str, Any] = {}
    if args.retriever:
        updates["retriever"] = args.retriever
    if args.lexical_weight is not None:
        updates["retrieval_lexical_weight"] = args.lexical_weight
    if args.top_k:
        updates["retrieval_top_k"] = args.top_k
    if args.retrieval_only and not args.retriever:
        updates["retriever"] = "pgvector"  # an evaluation must not silently fall back to "all tables"
    settings = settings.model_copy(update=updates)

    items = select_items(load_dataset(args.dataset), args)
    if not items:
        raise SystemExit("no items selected")
    mode = "retrieval" if args.retrieval_only else "fake" if args.fake else "agent"
    sleep_s = args.sleep if args.sleep is not None else (2.0 if mode == "agent" else 0.0)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), "mode": mode, "dataset": args.dataset,
        "n_items": len(items), "retriever": settings.retriever, "top_k": settings.retrieval_top_k, "lexical_weight": settings.retrieval_lexical_weight,
        "embedding_model": settings.embedding_model, "index_table": args.index_table or "schema_docs", "max_repairs": settings.max_repairs,
    }  # fmt: skip

    try:
        retriever = build_retriever(settings, args.index_table)
    except Exception as e:
        raise SystemExit(f"cannot build the retriever: {e}") from e

    if mode == "retrieval":
        records = [evaluate_retrieval(retriever, i) for i in items if i.in_scope]
        label = f"{args.index_table or 'schema_docs'}, k={settings.retrieval_top_k}, lexical={settings.retrieval_lexical_weight}"
        meta["retriever_class"] = type(retriever).__name__
        payload = {"meta": meta, "results": records}
        report = retrieval_markdown(records, label)
    else:
        runs = make_runs(args, settings, items, retriever)
        results = run_agent(runs, items, settings, sleep_s)
        payload = {
            "meta": {**meta, "retriever_class": type(retriever).__name__},
            "results": results,
            "aggregates": {m: aggregate(r) for m, r in results.items()},
        }
        report = agent_markdown(results)

    out = Path(args.out) if args.out else RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"raw results: {out}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
