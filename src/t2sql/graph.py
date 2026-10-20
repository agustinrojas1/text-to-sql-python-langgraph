"""LangGraph definition.

classify -> (out of scope -> respond_out_of_scope -> END)
         -> retrieve_schema -> generate_sql -> validate_sql -> execute -> respond -> END
A guard rejection, an execution error or a (once) empty result goes to repair ->
generate_sql while attempts remain (T2SQL_MAX_REPAIRS); then respond_error / respond.
A database outage (DatabaseUnavailableError) is not repairable: it propagates out of the graph.
"""

from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from t2sql.config import Settings, get_settings
from t2sql.db import execute_readonly
from t2sql.llm import LLM
from t2sql.local_classifier import ScopeScorer, build_scorer
from t2sql.nodes.classify import make_classify, make_respond_out_of_scope, route_after_classify
from t2sql.nodes.execute import make_execute
from t2sql.nodes.generate import make_generate_sql
from t2sql.nodes.repair import make_repair, make_route_after_execute, make_route_after_validate
from t2sql.nodes.respond import make_respond, make_respond_error
from t2sql.nodes.retrieve import make_retrieve_schema
from t2sql.nodes.validate import make_validate_sql
from t2sql.retrieval import SchemaRetriever
from t2sql.schema_index import default_retriever
from t2sql.tracing import append_trace_file, summarize_trace
from t2sql.state import AgentState, Deps, Executor


RECURSION_MARGIN = 3


def recursion_limit(max_repairs: int) -> int:
    """Supersteps of the longest legal path, plus a small margin.

    Every node is one superstep. The path is classify, retrieve_schema, then up to
    1 + max_repairs rounds of generate_sql, validate_sql, execute with a repair between
    rounds, and finally respond / respond_error. The `attempts` counter is the normal loop
    guard; this is the backstop if a router ever stops honouring it (the run then fails
    with GraphRecursionError instead of looping, and spending LLM calls, indefinitely).
    """
    generations = 1 + max_repairs
    return 2 + 3 * generations + max_repairs + 1 + RECURSION_MARGIN


def build_graph(
    llm: LLM,
    retriever: SchemaRetriever | None = None,
    settings: Settings | None = None,
    executor: Executor | None = None,
    scorer: ScopeScorer | None = None,
) -> CompiledStateGraph:
    settings = settings or get_settings()
    deps = Deps(
        llm=llm,
        retriever=retriever or default_retriever(settings),
        settings=settings,
        executor=executor or (lambda sql: execute_readonly(sql, settings)),
        scorer=scorer or build_scorer(settings),
    )

    g = StateGraph(AgentState)
    g.add_node("classify", make_classify(deps))
    g.add_node("respond_out_of_scope", make_respond_out_of_scope())
    g.add_node("retrieve_schema", make_retrieve_schema(deps))
    g.add_node("generate_sql", make_generate_sql(deps))
    g.add_node("validate_sql", make_validate_sql(deps))
    g.add_node("execute", make_execute(deps))
    g.add_node("repair", make_repair(deps))
    g.add_node("respond", make_respond(deps))
    g.add_node("respond_error", make_respond_error())

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify, ["retrieve_schema", "respond_out_of_scope"])
    g.add_edge("respond_out_of_scope", END)
    g.add_edge("retrieve_schema", "generate_sql")
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges(
        "validate_sql", make_route_after_validate(deps), ["execute", "repair", "respond_error"]
    )
    g.add_conditional_edges(
        "execute", make_route_after_execute(deps), ["respond", "repair", "respond_error"]
    )
    g.add_edge("repair", "generate_sql")
    g.add_edge("respond", END)
    g.add_edge("respond_error", END)
    return g.compile().with_config({"recursion_limit": recursion_limit(settings.max_repairs)})


def _initial_state(question: str) -> dict[str, Any]:
    return {"question": question, "attempts": 0, "attempt_history": [], "trace": []}


def _finish(state: AgentState, question: str, settings: Settings | None) -> AgentState:
    """Attach the trace summary and (if configured) append the trace file."""
    state["summary"] = summarize_trace(state["trace"])
    if settings and settings.trace_file:
        append_trace_file(settings.trace_file, question, state)
    return state


def run_question(
    graph: CompiledStateGraph, question: str, settings: Settings | None = None
) -> AgentState:
    """Run the graph (its recursion limit is set in build_graph), attach the trace summary and (if configured) append the trace file."""
    return _finish(graph.invoke(_initial_state(question)), question, settings)


def node_event(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """One streaming event from a node's state update (its trace entry plus a small payload)."""
    entry = update["trace"][-1]
    details = entry.get("details") or {}
    err = update.get("error")
    if node == "classify":
        data = {
            "in_scope": update["in_scope"],
            "language": update["language"],
            "reason": update["scope_reason"],
            "decided_by": details.get("decided_by", "llm"),  # "local" | "llm"
            "p_in": details.get("p_in"),  # local model's P(in scope), if it was asked
            "local_ms": details.get("local_ms"),
            "local_model": details.get("local_model"),
            "local_unavailable": bool(details.get("local_unavailable")),
        }
    elif node == "retrieve_schema":
        scores = details.get("scores") or {}
        data = {
            "tables": update["candidate_tables"],
            "selected": details.get("selected") or update["candidate_tables"],
            "bridges": details.get("expanded") or [],
            "top_scores": dict(list(scores.items())[:6]),
        }
    elif node == "generate_sql":
        data = {"sql": update["sql"], "assumptions": update.get("assumptions") or []}
    elif node == "validate_sql":
        data = {"ok": not err, "error": err, "sql": None if err else update["sql"]}
    elif node == "execute":
        data = {"ok": not err, "error": err, "row_count": details.get("row_count")}
    elif node == "repair":
        failed = update["attempt_history"][-1]
        data = {"kind": failed["kind"], "error": failed["error"], "failed_sql": failed["sql"]}
    else:  # respond, respond_error, respond_out_of_scope
        data = {"answer": update["answer"]}
    return {
        "type": "node",
        "node": node,
        "attempt": entry["attempt"],
        "latency_ms": entry["latency_ms"],
        "llm": entry["llm"],
        "cost_usd": entry["cost_usd"],
        "error": err,
        "data": data,
    }


def stream_question(
    graph: CompiledStateGraph, question: str, settings: Settings | None = None
) -> Iterator[dict[str, Any]]:
    """Like run_question, but yields one `node` event per node execution as it finishes,
    then a `final` event carrying the complete state (with summary). Exceptions from the
    graph propagate to the caller, as with run_question.

    "updates" gives each node's own update; "values" gives the full state after each step,
    so the last one is the final state without re-implementing the reducers here.
    """
    state: AgentState = {}
    for mode, chunk in graph.stream(_initial_state(question), stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk
        else:
            for node, update in chunk.items():
                yield node_event(node, update)
    yield {"type": "final", "state": _finish(state, question, settings)}
