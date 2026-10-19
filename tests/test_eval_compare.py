from datetime import date, datetime
from decimal import Decimal

from evals.compare import compare_results, normalize_value


def rows(cols, *tuples):
    return [dict(zip(cols, t)) for t in tuples]


def cmp(ref, gen, order=False):
    (rc, rr), (gc, gr) = ref, gen
    return compare_results(rc, rows(rc, *rr), gc, rows(gc, *gr), order)


def test_normalize_numbers_and_dates():
    assert normalize_value(Decimal("5.00")) == normalize_value(5) == normalize_value(5.0)
    assert normalize_value(Decimal("1.0508050242649158")) == 1.05
    assert normalize_value(-0.001) == 0.0
    assert normalize_value(datetime(2003, 5, 3)) == normalize_value(date(2003, 5, 3)) == "2003-05-03"
    assert normalize_value(datetime(2003, 5, 3, 10, 30)) == "2003-05-03T10:30:00"
    assert normalize_value(" Rock ") == "Rock" and normalize_value(None) is None


def test_aliases_and_column_order_are_ignored():
    ref = (["name", "n"], [("Rock", 3), ("Jazz", 2)])
    gen = (["total", "genre"], [(2, "Jazz"), (3, "Rock")])  # swapped columns, other aliases, other row order
    c = cmp(ref, gen)
    assert c.match and c.strict


def test_row_order_matters_only_when_asked():
    ref = (["n"], [(1,), (2,), (3,)])
    gen = (["x"], [(3,), (1,), (2,)])
    assert cmp(ref, gen).match
    assert not cmp(ref, gen, order=True).match
    assert cmp(ref, (["x"], [(1,), (2,), (3,)]), order=True).match


def test_multiset_not_set_semantics():
    assert not cmp((["n"], [(1,), (1,), (2,)]), (["n"], [(1,), (2,), (2,)])).match
    assert not cmp((["n"], [(1,), (1,)]), (["n"], [(1,)])).match  # DISTINCT vs not


def test_numeric_tolerance_is_two_decimals():
    assert cmp((["a"], [(Decimal("49.62"),)]), (["b"], [(49.6200001,)])).match
    assert cmp((["a"], [(1.05080,)]), (["b"], [(Decimal("1.05"),)])).match
    assert not cmp((["a"], [(1.05,)]), (["b"], [(1.06,)])).match


def test_wrong_pairing_of_columns_fails():
    ref = (["name", "n"], [("Rock", 3), ("Jazz", 2)])
    gen = (["name", "n"], [("Rock", 2), ("Jazz", 3)])  # same values per column, wrong rows
    c = cmp(ref, gen)
    assert not c.match and "rows differ" in c.reason


def test_extra_columns_are_lenient_but_not_strict():
    ref = (["name"], [("Rock",), ("Jazz",)])
    gen = (["id", "name"], [(1, "Rock"), (2, "Jazz")])
    c = cmp(ref, gen)
    assert c.match and not c.strict


def test_missing_column_or_wrong_row_count_fails():
    ref = (["name", "n"], [("Rock", 3)])
    assert not cmp(ref, (["name"], [("Rock",)])).match
    assert "row count" in cmp((["n"], [(1,), (2,)]), (["n"], [(1,)])).reason


def test_duplicate_valued_columns_still_match():
    ref = (["a", "b"], [(1, 2), (2, 1)])
    gen = (["x", "y"], [(2, 1), (1, 2)])
    assert cmp(ref, gen).match


def test_nulls_compare_equal_and_empty_results_match():
    assert cmp((["a"], [(None,), (1,)]), (["b"], [(1,), (None,)])).match
    assert cmp((["a"], []), (["b"], [])).match


def test_ordered_comparison_tolerates_reordering_inside_ties():
    cols = ["country", "total"]
    ref = (cols, [("USA", 523.06), ("Canada", 303.96), ("Chile", 37.62), ("Spain", 37.62), ("Italy", 37.62)])
    # Same order by total, different tie-break among the 37.62 rows: fine.
    tie = (cols, [("USA", 523.06), ("Canada", 303.96), ("Spain", 37.62), ("Italy", 37.62), ("Chile", 37.62)])
    assert cmp(ref, tie, order=True).match
    # Wrong direction or a displaced non-tied row is still wrong.
    asc = (cols, list(reversed(ref[1])))
    assert not cmp(ref, asc, order=True).match
    swapped = (cols, [("Canada", 303.96), ("USA", 523.06), ("Chile", 37.62), ("Spain", 37.62), ("Italy", 37.62)])
    assert not cmp(ref, swapped, order=True).match
    # Without order_matters nothing changes.
    assert cmp(ref, asc).match
