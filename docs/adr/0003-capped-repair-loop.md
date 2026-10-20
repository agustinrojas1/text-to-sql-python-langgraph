# 0003. A capped repair loop that is fully traced

Status: accepted

## Context

Generated SQL often fails the first time: a wrong column, a bad join, a value spelled
differently from the data, a guard rejection. Giving the model the error and letting it
try again fixes most of these, but an unbounded loop can burn tokens and time forever, and
a loop that hides its retries makes the system impossible to evaluate or debug.

## Decision

`generate_sql` -> `validate_sql` -> `execute` is wrapped in a repair loop with these rules:

- **Cap.** At most `T2SQL_MAX_REPAIRS` (default 2) repairs, so at most 3 generations per
  question. When the budget is gone the graph ends in `respond_error`, a deterministic
  message with no further LLM call.
- **What triggers a repair.** A guard rejection, a database error in the query (the Postgres
  message is passed back to the model), or an empty result, the last one only once. An empty result
  is suspicious (a typo in a literal, a wrong date range) but it is also a legitimate
  answer, so after one retry an empty result is accepted and reported as "no data matched".
- **Query errors are repaired, infrastructure failures are not.** `db.py` splits psycopg
  errors in two: `QueryError` (bad SQL, permission, statement timeout) and
  `DatabaseUnavailableError` (cannot connect, connection dropped, too many connections,
  server shutdown or crash). The latter is not caught by the `execute` node: it propagates
  out of the graph, the CLI prints "Database unavailable" and exits 1, and the API answers
  503. The reason is cost: valid SQL cannot be fixed by rewriting it, so "repairing" an
  outage would spend up to 2 LLM calls per question to produce the same failure, and the
  model might even turn correct SQL into wrong SQL while trying to please the error. The
  split uses the sqlstate (classes 08, 53, 58 and 57P0x; no sqlstate plus
  `OperationalError`/`InterfaceError` means the error never came from the server), not the
  exception class, because a statement timeout (57014) is also an `OperationalError` and
  is a query problem. Timeouts stay repairable: the message gets "write a cheaper one"
  appended, which steers the repair prompt towards fewer joins or a filter.
- **A second safety net.** `attempts` is the normal loop guard, but if a router ever stopped
  honouring it the graph would loop and spend LLM calls. `build_graph` therefore sets an
  explicit LangGraph `recursion_limit` computed from the cap: 2 supersteps (classify,
  retrieve) + 3 per generation (generate, validate, execute) x (1 + max_repairs) + one
  repair per repair round + 1 respond + a margin of 3; 17 for 2 repairs. The run then
  fails with `GraphRecursionError` instead of running on. LangGraph's default is 25, which
  would silently stop being a bound if `T2SQL_MAX_REPAIRS` were raised.
- **What the model sees.** The repair prompt carries the question, the schema context and
  the full history of failed attempts (SQL and reason), not only the last one, so it does
  not oscillate between two wrong queries.
- **Everything is recorded.** Each attempt appends to `attempt_history` and every node run
  appends a trace entry (node, attempt number, latency, provider, model, tokens, cost,
  error). `summarize_trace` derives `attempts`, `repaired` and `first_try_valid` from the
  trace, and the evals report "first-try valid" and "recovered by repair" from it.

## Consequences

- Worst case cost and latency per question are bounded (3 generations plus classify and
  respond), which matters on free-tier rate limits.
- The repair loop is measurable: the eval table separates queries that were right the
  first time from queries that were rescued, and counts the ones that were not.
- A repair can turn a wrong query into a different wrong query that runs fine. The loop
  only knows about errors and emptiness, not about correctness; execution accuracy in the
  evals is the check for that.
- The empty-result rule costs one extra LLM call on questions that truly have no data.
- During a database outage a question fails fast with a clear message and no extra LLM
  spend beyond classify/generate, instead of consuming the repair budget.
- Two retries is a judgment call, not a tuned value. The raw eval results (`--out`) include
  the attempt history, so the number can be revisited with data.
