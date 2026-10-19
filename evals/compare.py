"""Result-set comparison for execution accuracy.

The generated SQL is judged by what it returns, not by its text. Rules (documented in
the README, "Evals"):

* Values are normalised first: numbers (int / Decimal / float) are rounded to 2 decimals
  and compared as numbers (5 == 5.00); timestamps at midnight compare equal to the date;
  other timestamps/dates become ISO strings; strings are stripped (case-sensitive);
  NULL equals NULL.
* Column names, aliases and column order are ignored. The generated columns are matched
  to the reference columns by *values*: a reference column can only map to a generated
  column holding the same multiset of values, and the mapping must reproduce the rows
  of the reference exactly (a small backtracking search, so "name, count" vs
  "count, name" matches but a wrong pairing does not).
* Rows are compared as multisets (order-insensitive) unless the item has
  `order_matters`, in which case the sequence of rows must be identical, except that rows
  tied on a sort key may come in any order (the question rarely says how to break ties, so
  the reference's tie-break is arbitrary). A sort key is a reference column the reference
  rows are monotone in; the generated sequence must have the same key values in the same
  order and the same set of rows inside every group of equal key values.
* Lenient headline rule: the generated result may have EXTRA columns (e.g. an id the
  question did not ask for) as long as some subset of its columns reproduces the
  reference result. `strict` additionally requires the same number of columns; both
  are reported. Missing columns always fail.
* Row counts must be identical (the agent forces LIMIT 100 and the reference goes
  through the same guard, so both sides are capped alike).

Known limits: rows come as dicts (psycopg `dict_row`), so duplicate output column
names collapse into one; a query that returns the same column name twice is judged
on the surviving column only.
"""

import itertools
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

DECIMALS = 2
MAX_ASSIGNMENTS = 5000  # cap on column mappings tried (pathological all-equal columns)


@dataclass(frozen=True)
class Comparison:
    match: bool  # lenient: extra columns allowed
    strict: bool  # match and same number of columns
    reason: str  # why it failed ("" when it matched)


def normalize_value(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, int | float | Decimal):
        x = round(float(v), DECIMALS)
        return 0.0 if x == 0 else x  # no -0.0
    if isinstance(v, datetime):
        return v.date().isoformat() if v.time() == time(0) else v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, str):
        return v.strip()
    return str(v)


def _sort_key(v: Any) -> tuple[int, Any]:
    if v is None:
        return (3, "")
    if isinstance(v, bool):
        return (0, float(v))
    if isinstance(v, float):
        return (0, v)
    return (1, str(v))


def to_tuples(columns: list[str], rows: list[dict[str, Any]]) -> list[tuple]:
    return [tuple(normalize_value(r[c]) for c in columns) for r in rows]


def _tie_equivalent(ref: list[tuple], gen: list[tuple]) -> bool:
    """`gen` equals `ref` up to reordering rows that tie on some sort key of `ref`."""
    for k in range(len(ref[0])):
        keys = [r[k] for r in ref]
        try:
            monotone = all(a <= b for a, b in itertools.pairwise(keys)) or all(
                a >= b for a, b in itertools.pairwise(keys)
            )
        except TypeError:  # NULLs or mixed types: not a usable sort key
            continue
        if not monotone or [g[k] for g in gen] != keys:
            continue
        # Same key sequence; compare the rows of each run of equal keys as multisets.
        start = 0
        for end in range(1, len(ref) + 1):
            if end == len(ref) or keys[end] != keys[start]:
                if Counter(ref[start:end]) != Counter(gen[start:end]):
                    break
                start = end
        else:
            return True
    return False


def compare_results(
    ref_columns: list[str],
    ref_rows: list[dict[str, Any]],
    gen_columns: list[str],
    gen_rows: list[dict[str, Any]],
    order_matters: bool = False,
) -> Comparison:
    ref = to_tuples(ref_columns, ref_rows)
    gen = to_tuples(gen_columns, gen_rows)
    n_ref, n_gen = len(ref_columns), len(gen_columns)

    if len(ref) != len(gen):
        return Comparison(False, False, f"row count {len(gen)} != {len(ref)}")
    if n_gen < n_ref:
        return Comparison(False, False, f"{n_gen} column(s), expected at least {n_ref}")
    if not ref:  # both empty
        return Comparison(True, n_gen == n_ref, "")

    def signature(rows: list[tuple], j: int) -> tuple:
        return tuple(sorted((r[j] for r in rows), key=_sort_key))

    ref_sigs = [signature(ref, j) for j in range(n_ref)]
    gen_sigs = [signature(gen, i) for i in range(n_gen)]
    candidates = [[i for i in range(n_gen) if gen_sigs[i] == ref_sigs[j]] for j in range(n_ref)]
    if any(not c for c in candidates):
        missing = [ref_columns[j] for j, c in enumerate(candidates) if not c]
        return Comparison(False, False, f"no generated column has the values of {missing}")

    ref_cmp = ref if order_matters else Counter(ref)
    tried = 0
    for assignment in itertools.product(*candidates):
        if len(set(assignment)) < n_ref:  # injective mapping only
            continue
        tried += 1
        if tried > MAX_ASSIGNMENTS:
            break
        projected = [tuple(r[i] for i in assignment) for r in gen]
        if (projected if order_matters else Counter(projected)) == ref_cmp or (
            order_matters and _tie_equivalent(ref, projected)
        ):
            return Comparison(True, n_gen == n_ref, "")
    return Comparison(False, False, "values match per column but rows differ (wrong pairing or order)")
