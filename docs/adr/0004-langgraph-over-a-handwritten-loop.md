# 0004. LangGraph instead of a hand-written loop

Status: accepted

## Context

The pipeline is small: classify, retrieve schema, generate, validate, execute, maybe
repair, respond. A plain Python function with a `for attempt in range(3)` loop would do it
in about 80 lines and has no dependency. The project also exists to learn and demonstrate
LangGraph, so the question is whether it earns its place beyond that.

## Decision

Use LangGraph (`StateGraph`) for orchestration, keeping the rest of the code independent
of it.

- **Explicit state.** `AgentState` is a `TypedDict` that names everything the pipeline
  knows (question, candidate tables, SQL, error, attempts, rows, trace...). Nodes are
  functions `state -> partial update`, easy to test one at a time with a fake LLM.
- **Conditional edges for the control flow.** Out-of-scope routing, "valid or repair",
  "execute succeeded, failed or came back empty" and the retry budget are routing
  functions on the graph rather than nested `if`s inside a loop. The diagram in the README
  is generated from the compiled graph, so it cannot drift from the code.
- **Reducers.** `trace` and `attempt_history` use `operator.add`, so each node appends its
  entry and none can overwrite another's.
- **Room to grow.** `interrupt` and checkpointers (human approval before an expensive
  query, multi-turn memory) are the next features in the plan; they are a few lines on a
  graph and a rewrite on a hand-made loop.
- **Boundaries.** Nodes depend on the small `LLM` protocol (`llm.py`), not on LangChain
  types, and tracing, the guard and the database layer know nothing about LangGraph.
  LangChain is used only inside the provider adapters.

## Consequences

- More machinery than the problem needs today: a framework, its state model and its
  failure modes (for example, node update keys must exist in the state). The same behavior
  in a plain loop would be shorter.
- What LangGraph does not give us for free, and had to be built here: cost and latency
  tracing (a `traced` wrapper), provider fallback, the SQL safety layers and the evals.
  The graph is structure, not intelligence.
- Debugging flows through the trace we record rather than through a hosted tool.
- Testing stays cheap because the graph accepts any `LLM`, retriever and executor:
  `FakeLLM` scripts a whole run, including repair paths, with no network.
- If the pipeline stayed linear and never needed interrupts, a hand-written loop would be
  the better choice; the decision is bet on the roadmap (HITL, memory), stated as such.
