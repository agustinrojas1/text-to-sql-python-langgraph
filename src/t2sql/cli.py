"""Command line entry point: `t2sql ask "question"` and `t2sql index`."""

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from t2sql.config import get_settings
from t2sql.graph import build_graph, run_question
from t2sql.llm import LLMUnavailableError, build_llm
from t2sql.schema_index import DEFAULT_TABLE, META_SCHEMA, FastEmbedEmbedder, build_index

MAX_TABLE_ROWS = 20
MAX_CELL_CHARS = 40

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
err = Console(stderr=True)


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
        table.add_row(
            e["node"],
            str(e["attempt"]),
            f"{m['provider']}/{m['model']}" if m else "",
            f"{m['input_tokens']}/{m['output_tokens']}" if m else "",
            str(m.get("latency_ms", e["latency_ms"])) if m else str(e["latency_ms"]),
            _money(e["cost_usd"]) if m else "",
            _cell(e["error"]),
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
            console.print(f"[dim]{e['node']}[/]: {detail}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question in English or Spanish")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show per-node trace, cost and retrieval")] = False,
) -> None:
    """Answer a question with SQL over the database."""
    settings = get_settings()
    try:
        llm = build_llm(settings)
    except LLMUnavailableError as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e

    try:
        state = run_question(build_graph(llm, settings=settings), question, settings)
    except Exception as e:  # LLM outage, DB down...
        err.print(f"[red]{type(e).__name__}: {e}[/]")
        raise typer.Exit(1) from e

    console.print(f"\n[bold]{state['answer']}[/]\n")
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
    embedder = FastEmbedEmbedder(settings.embedding_model)
    console.print(f"Embedding with {embedder.name} (downloads ~220 MB on first use)...")
    try:
        n = build_index(embedder, settings, table=table, keywords=keywords)
    except Exception as e:
        err.print(f"[red]{type(e).__name__}: {e}[/]")
        raise typer.Exit(1) from e
    console.print(f"Indexed {n} documents into {META_SCHEMA}.{table}.")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
