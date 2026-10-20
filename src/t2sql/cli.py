"""Command line entry point: `t2sql ask "question"`, `t2sql index` and `t2sql record`."""

import json
from collections.abc import Callable
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table

from t2sql.api import build_runtime, stream_lines
from t2sql.config import get_settings
from t2sql.db import DatabaseUnavailableError
from t2sql.graph import build_graph, run_question
from t2sql.llm import LLMUnavailableError, build_llm
from t2sql.replay import DEFAULT_QUESTIONS, recording_path, variant
from t2sql.schema_index import DEFAULT_TABLE, META_SCHEMA, FastEmbedEmbedder, build_index

MAX_TABLE_ROWS = 20
MAX_CELL_CHARS = 40

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
err = Console(stderr=True)


CLASSIFIER_HELP = "Scope classifier for this question: llm, local or cascade (default: T2SQL_CLASSIFIER)"
LOCAL_MODEL_HELP = "Local model for local/cascade: tev1 (default: T2SQL_LOCAL_CLASSIFIER_MODEL)"


def _one_of(name: str, allowed: tuple[str, ...]) -> Callable[[str], str]:
    def check(value: str) -> str:
        if value and value not in allowed:
            raise typer.BadParameter(f"{value!r} is not one of {', '.join(allowed)}")
        return value

    return check


_check_classifier = _one_of("classifier", ("llm", "local", "cascade"))
_check_local_model = _one_of("local model", ("tev1",))


def _overrides(classifier: str, local_model: str) -> dict[str, str]:
    """Only the options the user gave; the rest stays the server settings."""
    return {k: v for k, v in (("classifier", classifier), ("local_model", local_model)) if v}


def _cell(v: Any) -> str:
    s = "" if v is None else str(v)
    return s if len(s) <= MAX_CELL_CHARS else s[: MAX_CELL_CHARS - 1] + "…"


def _results_table(columns: list[str], rows: list[dict[str, Any]]) -> Table:
    shown = rows[:MAX_TABLE_ROWS]
    table = Table(
        *columns,
        caption=f"{len(rows)} row(s)" + (f", showing {len(shown)}" if len(rows) > len(shown) else ""),
    )
    for r in shown:
        table.add_row(*(_cell(r[c]) for c in columns))
    return table


def _money(v: float | None) -> str:
    return "n/a" if v is None else f"${v:.6f}"


def _detail(e: dict[str, Any]) -> str:
    d = e.get("details") or {}
    if "decided_by" in d:
        scored = f" (local {d['local_model']} p={d['p_in']:.2f}, {d['local_ms']:.0f} ms)" if d.get("p_in") is not None else ""
        if d["decided_by"] == "local":
            return f"decided locally{scored}"
        return "decided by the cloud LLM" + (scored or (" (local classifier unavailable)" if d.get("local_unavailable") else ""))
    if "scores" in d:
        scores = " ".join(f"{t}={s:.2f}" for t, s in list(d["scores"].items())[:6])
        return f"top-k: {', '.join(d['selected'])} | +bridges: {', '.join(d['expanded']) or '-'} | {scores}"
    if "kind" in d:
        return f"repair after attempt {d['failed_attempt']} ({d['kind']})"
    if "row_count" in d:
        return f"{d['row_count']} row(s)"
    return ""


def _print_trace(trace: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    console.print("[bold]Path:[/] " + " -> ".join(e["node"] for e in trace))
    table = Table("node", "att", "provider/model", "tokens in/out", "latency ms", "cost", "error")
    for e in trace:
        m = e["llm"] or {}
        local = (e.get("details") or {}).get("decided_by") == "local"
        table.add_row(
            e["node"],
            str(e["attempt"]),
            f"{m['provider']}/{m['model']}" if m else f"local/{e['details']['local_model']}" if local else "",
            f"{m['input_tokens']}/{m['output_tokens']}" if m else "",
            str(m.get("latency_ms", e["latency_ms"])) if m else str(e["latency_ms"]),
            _money(e["cost_usd"]) if m else "",
            escape(_cell(e["error"])),
            end_section=e is trace[-1],
        )
    s = summary
    table.add_row(
        "[bold]total[/]",
        str(s["attempts"]),
        f"{s['llm_calls']} LLM call(s)",
        f"{s['input_tokens']}/{s['output_tokens']}",
        str(s["latency_ms"]),
        _money(s["cost_usd"]),
        f"repaired={s['repaired']} first_try_valid={s['first_try_valid']}",
    )
    console.print(table)
    for e in trace:
        if detail := _detail(e):
            console.print(f"[dim]{e['node']}[/]: {escape(detail)}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question in English or Spanish")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show per-node trace, cost and retrieval")] = False,
    classifier: Annotated[str, typer.Option(help=CLASSIFIER_HELP, callback=_check_classifier)] = "",
    local_model: Annotated[str, typer.Option(help=LOCAL_MODEL_HELP, callback=_check_local_model)] = "",
) -> None:
    """Answer a question with SQL over the database."""
    settings = get_settings()
    try:
        llm = build_llm(settings)
    except LLMUnavailableError as e:
        err.print(f"[red]{escape(str(e))}[/]")
        raise typer.Exit(2) from e

    try:
        state = run_question(build_graph(llm, settings=settings), question, settings, **_overrides(classifier, local_model))
    except DatabaseUnavailableError as e:  # not repairable: the SQL is not the problem
        err.print(f"[red]Database unavailable: {escape(str(e))}[/]")
        raise typer.Exit(1) from e
    except LLMUnavailableError as e:  # provider outage or quota: safe message, no payloads
        err.print(f"[red]{escape(e.public_detail)}[/]")
        raise typer.Exit(1) from e
    except Exception as e:
        err.print(f"[red]{type(e).__name__}: {escape(str(e))}[/]")
        raise typer.Exit(1) from e

    # Text from the LLM or Postgres may contain "[...]", which rich would read as markup.
    # Markdown (not markup) renders **bold** without printing the asterisks and keeps "[...]" literal.
    console.print()
    console.print(Markdown(state["answer"]))
    console.print()
    if state.get("sql"):
        console.print(Syntax(state["sql"], "sql", theme="ansi_dark", word_wrap=True))
    if state.get("rows") is not None:
        console.print(_results_table(state["columns"], state["rows"]))
    if verbose:
        console.print()
        _print_trace(state["trace"], state["summary"])
    if state.get("error"):
        raise typer.Exit(1)


@app.command()
def index(
    table: Annotated[str, typer.Option(help="Index table inside t2sql_meta")] = DEFAULT_TABLE,
    keywords: Annotated[
        bool, typer.Option(help="Also embed the EN/ES keyword vocabulary in table documents (experiment)")
    ] = False,
) -> None:
    """(Re)build the schema index (pgvector) used for schema RAG. Idempotent."""
    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    console.print(f"Embedding with {embedder.name} (downloads ~220 MB on first use)...")
    try:
        n = build_index(embedder, settings, table=table, keywords=keywords)
    except Exception as e:
        err.print(f"[red]{type(e).__name__}: {escape(str(e))}[/]")
        raise typer.Exit(1) from e
    console.print(f"Indexed {n} documents into {META_SCHEMA}.{table}.")


@app.command()
def record(
    questions: Annotated[
        list[str] | None, typer.Argument(help="Questions to record (default: the web UI's examples)")
    ] = None,
    directory: Annotated[
        str, typer.Option("--dir", "-d", help="Where to save (default: T2SQL_RECORD_DIR, else ./recordings)")
    ] = "",
    classifier: Annotated[str, typer.Option(help=CLASSIFIER_HELP, callback=_check_classifier)] = "",
    local_model: Annotated[str, typer.Option(help=LOCAL_MODEL_HELP, callback=_check_local_model)] = "",
) -> None:
    """Record /ask/stream runs for the offline demo (see T2SQL_REPLAY_DIR). Needs the LLM and database.

    A recording is keyed by question AND classifier mode / local model: run this once per mode
    you want to be able to replay."""
    runtime = build_runtime()
    target = directory or runtime.settings.record_dir or "recordings"
    runtime.settings = runtime.settings.model_copy(update={"record_dir": target, "replay_dir": None})
    failed = 0
    for q in questions or DEFAULT_QUESTIONS:
        last = ""
        for last in stream_lines(runtime, q, **_overrides(classifier, local_model)):
            pass
        ev = json.loads(last) if last else {}
        if ev.get("type") != "final":
            failed += 1
            err.print(f"[red]not recorded[/] {escape(q)}: {escape(str(ev.get('detail', 'no events')))}")
            continue
        note = "out of scope" if not ev["in_scope"] else f"error: {ev['error']}" if ev["error"] else f"{ev['row_count']} row(s)"
        console.print(f"recorded {escape(q)} -> {recording_path(target, q, variant(classifier or runtime.settings.classifier, local_model or runtime.settings.local_classifier_model))} ({escape(note)})", soft_wrap=True)
    if failed:
        raise typer.Exit(1)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
