"""The evaluation datasets are data, but wrong data silently ruins the numbers: check them."""

from collections import Counter

import pytest

from evals.dataset import DATASET_PATH, load_dataset, reference_tables
from t2sql.config import get_settings
from t2sql.db import execute_readonly
from t2sql.sql_guard import validate_sql

HOLDOUT_PATH = DATASET_PATH.with_name("holdout.yaml")
MAIN = load_dataset(DATASET_PATH)
HOLDOUT = load_dataset(HOLDOUT_PATH)
ANSWERABLE = [i for i in MAIN + HOLDOUT if i.in_scope]


def test_composition_matches_the_documented_numbers():
    assert Counter(i.level for i in MAIN) == {"easy": 20, "medium": 20, "hard": 10, "out_of_scope": 5}
    assert len(MAIN) == 55 and sum(i.in_scope for i in MAIN) == 50
    langs = Counter(i.language for i in MAIN if i.in_scope)
    assert langs == {"en": 25, "es": 25}
    assert {i.language for i in MAIN} == {"en", "es"}


def test_ids_and_questions_are_unique_across_files():
    everything = MAIN + HOLDOUT
    assert len({i.id for i in everything}) == len(everything)
    assert len({i.question for i in everything}) == len(everything)


def test_out_of_scope_items_have_no_sql():
    for i in MAIN:
        assert (i.sql is None) == (i.level == "out_of_scope"), i.id
        assert (not i.tables) == (i.level == "out_of_scope"), i.id


@pytest.mark.parametrize("item", ANSWERABLE, ids=lambda i: i.id)
def test_declared_tables_match_the_reference_sql(item):
    """`tables` is hand-readable data, but it must equal what sqlglot derives from the SQL."""
    assert set(item.tables) == set(reference_tables(item.sql, get_settings())), item.id


def test_single_table_level_really_is_single_table():
    for i in MAIN:
        if i.level == "easy":
            assert len(i.tables) == 1, i.id
        if i.level == "medium":
            assert len(i.tables) >= 2, i.id


@pytest.mark.integration
@pytest.mark.parametrize("item", ANSWERABLE, ids=lambda i: i.id)
def test_reference_sql_passes_guard_and_returns_a_sensible_result(item):
    s = get_settings()
    q = validate_sql(item.sql, set(s.allowed_tables), s.max_rows)  # raises if the guard rejects it
    result = execute_readonly(q.sql, s)
    assert result.row_count > 0, "empty reference result"
    assert result.row_count < s.max_rows, "reference hits the forced LIMIT: the comparison would be arbitrary"
