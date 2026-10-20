"""Scope-classifier evaluation: `uv run python -m evals.scope_run --classifier llm|local|cascade`.

Runs ONLY the classify decision (not retrieval, SQL or the answer) over evals/scope_dataset.yaml
and reports how often the scope decision is right:

  accuracy         correct decisions / decided items
  OOS recall       out-of-scope questions that were rejected / out-of-scope questions
  false rejection  answerable questions that were rejected / answerable questions
  locally          share of decisions taken by the local model without the cloud LLM
  cloud calls      real cloud requests (cached decisions are not counted)
  latency          wall time of the classify node, mean and p95

`--classifier local` needs a llama-server (see README), `llm` and `cascade` need a provider key.
Items the cloud could not decide (quota exhausted, provider down) are not scored: the report says
how many were reached. `--cache FILE` stores every cloud decision (keyed by provider, model and
question) and reuses it, which makes a run resumable and lets `cascade` reuse the `llm` run.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from t2sql.config import Settings, get_settings
from t2sql.llm import PROVIDERS, LLM, LLMResult, LLMUnavailableError, Message, build_llm
from t2sql.local_classifier import build_scorer
from t2sql.nodes.classify import ScopeDecision, make_classify
from t2sql.state import Deps

SCOPE_DATASET_PATH = Path(__file__).with_name("scope_dataset.yaml")
RESULTS_DIR = Path(__file__).with_name("results")
SPLITS = ("dev", "test", "all")


@dataclass(frozen=True)
class ScopeItem:
    id: str
    question: str
    language: str
    in_scope: bool
    category: str
    split: str


def load_scope_dataset(path: Path | str = SCOPE_DATASET_PATH) -> list[ScopeItem]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = [
        ScopeItem(
            id=r["id"], question=r["q"], language=r["lang"], in_scope=bool(r["scope"]),
            category=r["cat"], split=r["split"],
        )  # fmt: skip
        for r in raw
    ]
    if len({i.id for i in items}) != len(items):
        raise ValueError("duplicate ids in the scope dataset")
    bad = {i.split for i in items} - {"dev", "test"}
    if bad:
        raise ValueError(f"unknown split(s) in the scope dataset: {sorted(bad)}")
    return items


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- LLM wrappers


class NoLLM:
    """For `--classifier local`: the classify node must never reach the cloud."""

    def structured(self, messages: list[Message], schema: type) -> LLMResult:
        raise LLMUnavailableError("the cloud LLM is not available in local mode")

    def text(self, messages: list[Message]) -> LLMResult[str]:
        raise LLMUnavailableError("the cloud LLM is not available in local mode")


class CachingLLM:
    """Remembers every ScopeDecision by (provider, model, question), optionally persisted as JSON.

    A hit costs nothing and reports provider "cache", so the report separates real cloud calls
    from reused decisions. Only the classify call goes through this wrapper.
    """

    def __init__(self, inner: LLM, label: str, path: Path | None) -> None:
        self._inner, self._label, self._path = inner, label, path
        self._data: dict[str, dict[str, Any]] = {}
        if path and path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def _key(self, messages: list[Message]) -> str:
        return f"{self._label}\n{messages[-1].content}"

    def structured(self, messages: list[Message], schema: type) -> LLMResult:
        key = self._key(messages)
        if key in self._data:
            return LLMResult(output=schema(**self._data[key]), provider="cache", model=self._label)
        res = self._inner.structured(messages, schema)
        self._data[key] = res.output.model_dump()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
        return res

    def text(self, messages: list[Message]) -> LLMResult[str]:
        return self._inner.text(messages)


# ---------------------------------------------------------------- evaluation


def evaluate_item(classify: Any, item: ScopeItem) -> dict[str, Any]:
    """One classify decision; `error` is set (and `pred` is None) when no decision was possible."""
    start = time.perf_counter()
    rec: dict[str, Any] = {
        "id": item.id, "category": item.category, "language": item.language,
        "in_scope": item.in_scope, "question": item.question,
    }  # fmt: skip
    try:
        update = classify({"question": item.question})
    except LLMUnavailableError as e:
        rec |= {"pred": None, "error": str(e)[:300], "quota": e.quota}
        return rec
    entry = update["trace"][0]
    details = entry["details"] or {}
    llm = entry["llm"]
    rec |= {
        "pred": update["in_scope"],
        "decided_by": details.get("decided_by"),
        "p_in": details.get("p_in"),
        "local_ms": details.get("local_ms"),
        "local_unavailable": bool(details.get("local_unavailable")),
        "cloud_call": bool(llm and llm["provider"] != "cache"),
        "cached": bool(llm and llm["provider"] == "cache"),
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "tokens": (llm["input_tokens"] or 0) + (llm["output_tokens"] or 0) if llm else 0,
    }
    return rec


def run_items(classify: Any, items: list[ScopeItem], cloud_sleep_s: float, label: str) -> list[dict[str, Any]]:
    """Evaluate in order; stops at the first quota error (the rest are reported as not reached)."""
    out: list[dict[str, Any]] = []
    for n, item in enumerate(items, 1):
        rec = evaluate_item(classify, item)
        out.append(rec)
        if rec.get("error"):
            log(f"[{label}] {n}/{len(items)} {item.id} ERROR: {rec['error'][:160]}")
            if rec.get("quota"):
                log(f"[{label}] quota exhausted: stopping after {n - 1} decided items")
                break
        else:
            ok = rec["pred"] == rec["in_scope"]
            log(f"[{label}] {n}/{len(items)} {item.id} {'ok' if ok else 'WRONG'} ({rec['decided_by']})")
        if cloud_sleep_s and rec.get("cloud_call") and n < len(items):
            time.sleep(cloud_sleep_s)
    return out


def _pct(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def aggregate(records: list[dict[str, Any]], n_selected: int | None = None) -> dict[str, Any]:
    """Metrics over the records that carry a decision; undecided ones are only counted."""
    done = [r for r in records if r.get("pred") is not None]
    oos = [r for r in done if not r["in_scope"]]
    ins = [r for r in done if r["in_scope"]]
    lat = sorted(r["latency_ms"] for r in done)
    local = [r for r in done if r["decided_by"] == "local"]
    return {
        "n_selected": n_selected if n_selected is not None else len(records),
        "n_decided": len(done),
        "n_errors": sum(1 for r in records if r.get("error")),
        "accuracy": _pct(sum(r["pred"] == r["in_scope"] for r in done), len(done)),
        "oos_recall": _pct(sum(not r["pred"] for r in oos), len(oos)),
        "false_rejection": _pct(sum(not r["pred"] for r in ins), len(ins)),
        "n_oos": len(oos),
        "n_in": len(ins),
        "decided_locally": _pct(len(local), len(done)),
        "confident_local_errors": sum(r["pred"] != r["in_scope"] for r in local),
        "cloud_calls": sum(bool(r.get("cloud_call")) for r in done),
        "cached_decisions": sum(bool(r.get("cached")) for r in done),
        "local_unavailable": sum(bool(r.get("local_unavailable")) for r in done),
        "tokens": sum(r.get("tokens", 0) for r in done),
        "latency_mean_ms": round(statistics.fmean(lat), 1) if lat else None,
        "latency_p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
    }


def breakdown(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("pred") is not None:
            groups[r[key]].append(r)
    rows = []
    for name, rs in sorted(groups.items()):
        wrong = [r for r in rs if r["pred"] != r["in_scope"]]
        rows.append({"name": name, "n": len(rs), "correct": len(rs) - len(wrong), "wrong_ids": [r["id"] for r in wrong]})
    return rows


def _f(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if pct else f"{x:.0f}"


def markdown(runs: dict[str, list[dict[str, Any]]], n_selected: int) -> str:
    aggs = {name: aggregate(recs, n_selected) for name, recs in runs.items()}
    head = (
        "| classifier | decided / selected | accuracy | OOS recall | false rejection | decided locally "
        "| local errors | cloud calls | cached | mean ms | p95 ms |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    out = "### Scope decision\n\n" + head
    for name, a in aggs.items():
        out += (
            f"| {name} | {a['n_decided']} / {a['n_selected']} | {_f(a['accuracy'])} | {_f(a['oos_recall'])} "
            f"({a['n_oos']} OOS) | {_f(a['false_rejection'])} ({a['n_in']} in) | {_f(a['decided_locally'])} "
            f"| {a['confident_local_errors']} | {a['cloud_calls']} | {a['cached_decisions']} "
            f"| {_f(a['latency_mean_ms'], False)} | {_f(a['latency_p95_ms'], False)} |\n"
        )
    for key, title in (("category", "By category"), ("language", "By language")):
        names = [name for name in runs]
        tables = {name: {b["name"]: b for b in breakdown(recs, key)} for name, recs in runs.items()}
        keys = sorted({k for t in tables.values() for k in t})
        out += f"\n### {title} (correct / n)\n\n| {key} | " + " | ".join(names) + " |\n|---|" + "---|" * len(names) + "\n"
        for k in keys:
            cells = []
            for name in names:
                b = tables[name].get(k)
                cells.append("-" if not b else f"{b['correct']}/{b['n']}")
            out += f"| {k} | " + " | ".join(cells) + " |\n"
    errs = [(name, r) for name, recs in runs.items() for r in recs if r.get("pred") is not None and r["pred"] != r["in_scope"]]
    if errs:
        out += "\n### Wrong decisions\n\n"
        for name, r in errs:
            exp = "in scope" if r["in_scope"] else "out of scope"
            p = f", P(in)={r['p_in']}" if r.get("p_in") is not None else ""
            out += f"- [{name}] `{r['id']}` ({r['category']}, expected {exp}, decided by {r['decided_by']}{p}): {r['question'][:110]}\n"
    return out


# ---------------------------------------------------------------- driver


def make_classify_for(settings: Settings, llm: LLM):
    deps = Deps(llm=llm, retriever=None, settings=settings, executor=None, scorer=build_scorer(settings))  # type: ignore[arg-type]
    return make_classify(deps)


def select_items(items: list[ScopeItem], split: str, ids: str | None, limit: int | None) -> list[ScopeItem]:
    if split != "all":
        items = [i for i in items if i.split == split]
    if ids:
        wanted = {x.strip() for x in ids.split(",") if x.strip()}
        unknown = wanted - {i.id for i in items}
        if unknown:
            raise SystemExit(f"unknown ids (in this split): {sorted(unknown)}")
        items = [i for i in items if i.id in wanted]
    return items[:limit] if limit else items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.scope_run", description=__doc__.split("\n\n")[0])
    p.add_argument("--classifier", choices=["llm", "local", "cascade"], required=True)
    p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument("--dataset", default=str(SCOPE_DATASET_PATH))
    p.add_argument("--ids", help="comma-separated item ids")
    p.add_argument("--limit", type=int, help="only the first N items after filtering")
    p.add_argument("--providers", help="comma-separated cloud providers, one separate run each (default: T2SQL_LLM_PROVIDERS)")
    p.add_argument("--sleep", type=float, default=None, help="seconds to wait after each real cloud call (default 2)")
    p.add_argument("--cache", help="JSON file with cloud decisions to reuse and extend (makes runs resumable)")
    p.add_argument("--out", help="raw results JSON (default: evals/results/<timestamp>-scope-<classifier>.json)")
    args = p.parse_args(argv)

    settings = get_settings().model_copy(update={"classifier": args.classifier})
    items = select_items(load_scope_dataset(args.dataset), args.split, args.ids, args.limit)
    if not items:
        raise SystemExit("no items selected")
    sleep_s = 2.0 if args.sleep is None else args.sleep
    cache = Path(args.cache) if args.cache else None

    runs: dict[str, list[dict[str, Any]]] = {}
    if args.classifier == "local":
        label = f"local/{settings.local_classifier_model}"
        runs[label] = run_items(make_classify_for(settings, NoLLM()), items, 0.0, label)
    else:
        names = [n.strip().lower() for n in args.providers.split(",")] if args.providers else list(settings.llm_providers)
        for name in names:
            if name not in PROVIDERS:
                raise SystemExit(f"unknown provider '{name}' (known: {', '.join(PROVIDERS)})")
            s = settings.model_copy(update={"llm_providers": [name]})
            try:
                llm = build_llm(s)
            except LLMUnavailableError as e:
                if args.providers:
                    raise SystemExit(str(e)) from e
                log(f"skipping {name}: no API key")
                continue
            model = getattr(s, f"{name}_model")
            label = f"{args.classifier}/{name}/{model}" if args.classifier == "llm" else f"cascade/{settings.local_classifier_model}+{name}"
            runs[label] = run_items(make_classify_for(s, CachingLLM(llm, f"{name}/{model}", cache)), items, sleep_s, label)
        if not runs:
            raise SystemExit("No LLM provider has an API key (see .env.example); use --classifier local")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), "classifier": args.classifier, "split": args.split,
        "n_items": len(items), "dataset": args.dataset,
        "local": [settings.local_classifier_model, settings.cascade_low, settings.cascade_high, settings.local_threshold],
    }  # fmt: skip
    out = Path(args.out) if args.out else RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-scope-{args.classifier}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "results": runs, "aggregates": {k: aggregate(v, len(items)) for k, v in runs.items()}}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"raw results: {out}")
    print(markdown(runs, len(items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
