import pytest

from t2sql.sql_explain import CHECKS, explain_validation
from t2sql.sql_guard import SqlGuardError, validate_sql

ALLOWED = {"artist", "album", "track", "invoice_line", "customer"}


def table(name, *columns):
    return {"name": name, "columns": [{"name": c, "type": "integer", "pk": False, "fk": None} for c in columns]}


SCHEMA = [
    table("artist", "artist_id", "name"),
    table("album", "album_id", "title", "artist_id"),
    table("track", "track_id", "name", "album_id", "unit_price"),
    table("invoice_line", "invoice_line_id", "track_id", "unit_price", "quantity"),
]


def explain(sql, schema=SCHEMA, max_rows=100, allowed=ALLOWED):
    """Run the real guard, then explain its outcome, like the validate node does."""
    try:
        q = validate_sql(sql, set(allowed), max_rows)
    except SqlGuardError as e:
        return explain_validation(sql, None, str(e.code), allowed, max_rows, schema, e.message)
    return explain_validation(sql, q.sql, None, allowed, max_rows, schema)


def verdicts(result):
    return [(c["name"], c["ok"]) for c in result["checks"]]


def test_happy_path_with_joins_aggregates_and_cte():
    r = explain(
        "WITH rev AS (SELECT ar.artist_id, SUM(il.unit_price * il.quantity) AS total"
        " FROM invoice_line il JOIN track t ON t.track_id = il.track_id"
        " JOIN album al ON al.album_id = t.album_id JOIN artist ar ON ar.artist_id = al.artist_id"
        " GROUP BY ar.artist_id)"
        " SELECT a.name, total FROM rev JOIN artist a ON a.artist_id = rev.artist_id"
        " WHERE total > 1 ORDER BY total DESC LIMIT 3"
    )
    assert [c["name"] for c in r["checks"]] == list(CHECKS) and all(c["ok"] is True for c in r["checks"])
    assert r["tables"] == ["album", "artist", "invoice_line", "track"]  # sorted, the CTE "rev" excluded
    assert r["columns"] == [
        "album.album_id", "album.artist_id", "artist.artist_id", "artist.name",
        "invoice_line.quantity", "invoice_line.track_id", "invoice_line.unit_price",
        "track.album_id", "track.track_id",
    ]  # fmt: skip
    assert r["unresolved_columns"] == []
    c = r["clauses"]
    assert c["ctes"] == ["rev"] and c["select"] == ["a.name", "total"] and c["from"] == "rev"
    assert c["joins"] == ["JOIN artist AS a ON a.artist_id = rev.artist_id"]
    assert c["where"] == "total > 1" and c["order_by"] == ["total DESC"] and c["limit"] == 3
    assert c["group_by"] == [] and c["having"] is None
    assert r["limit_enforced"] == 3 and r["limit_added"] is False


def test_group_by_and_having_clauses():
    r = explain("SELECT artist_id, count(*) AS n FROM album GROUP BY artist_id HAVING count(*) > 1")
    assert r["clauses"]["group_by"] == ["artist_id"] and r["clauses"]["having"] == "COUNT(*) > 1"
    assert r["columns"] == ["album.artist_id"] and r["unresolved_columns"] == []


@pytest.mark.parametrize(
    ("sql", "code", "failing"),
    [
        ("SELEC FROM", "parse_error", "parse"),
        ("", "empty_query", "parse"),
        ("SELECT 1 FROM artist; SELECT 2 FROM artist", "multiple_statements", "single_statement"),
        ("DELETE FROM artist", "not_select", "select_only"),
        ("SELECT * INTO new_t FROM artist", "forbidden_clause", "no_forbidden_clauses"),
        ("SELECT pg_sleep(5) FROM artist", "forbidden_function", "no_forbidden_functions"),
        ("SELECT * FROM employee", "forbidden_table", "tables_allowed"),
    ],
)
def test_each_failure_code_fails_its_check_and_skips_the_rest(sql, code, failing):
    with pytest.raises(SqlGuardError) as raised:
        validate_sql(sql, set(ALLOWED), 100)
    assert raised.value.code == code  # the table below is keyed on the guard's real codes
    r = explain(sql)
    names = list(CHECKS)
    cut = names.index(failing)
    assert verdicts(r) == (
        [(n, True) for n in names[:cut]] + [(failing, False)] + [(n, None) for n in names[cut + 1 :]]
    )
    failed = r["checks"][cut]
    assert failed["detail"]  # the guard's own message
    assert r["tables"] == [] and r["columns"] == [] and r["unresolved_columns"] == []
    assert r["clauses"] is None and r["limit_enforced"] is None and r["limit_added"] is False


def test_failure_detail_is_the_guard_message():
    r = explain("SELECT * FROM employee")
    assert r["checks"][5]["detail"] == "table not allowed: employee"


def test_unknown_code_or_missing_message_still_gives_seven_checks():
    r = explain_validation("x", None, "something_new", ALLOWED, 100, SCHEMA)
    assert len(r["checks"]) == 7 and r["checks"][0] == {"name": "parse", "ok": False, "detail": "something_new"}
    r = explain_validation("x", None, None, ALLOWED, 100, SCHEMA)
    assert len(r["checks"]) == 7 and r["checks"][0]["ok"] is False


def test_unresolvable_columns_are_listed_without_hiding_the_rest():
    r = explain("SELECT ar.name, ar.nickname, bogus FROM artist ar")
    assert r["columns"] == ["artist.name"]
    assert r["unresolved_columns"] == ["bogus", "nickname"]


def test_ambiguous_column_is_unresolved_and_alias_references_are_not():
    r = explain("SELECT name FROM artist, track")  # both tables have `name`
    assert r["columns"] == [] and r["unresolved_columns"] == ["name"]
    r = explain("SELECT sum(unit_price * quantity) AS total FROM invoice_line ORDER BY total")
    assert r["columns"] == ["invoice_line.quantity", "invoice_line.unit_price"]
    assert r["unresolved_columns"] == []  # `total` is a select alias, not a column


def test_correlated_subquery_resolves_against_the_outer_table():
    r = explain(
        "SELECT a.name FROM artist a WHERE a.artist_id IN"
        " (SELECT artist_id FROM album WHERE album.title = a.name)"
    )
    assert r["columns"] == ["album.artist_id", "album.title", "artist.artist_id", "artist.name"]
    assert r["unresolved_columns"] == []


def test_table_missing_from_the_schema_cannot_vouch_for_its_columns():
    r = explain("SELECT c.email FROM customer c")  # allowed, but not among the retrieved tables
    assert r["tables"] == ["customer"] and r["columns"] == ["customer.email"]
    r = explain("SELECT email FROM customer")  # nothing says which table owns it
    assert r["columns"] == [] and r["unresolved_columns"] == ["email"]


def test_no_schema_means_everything_is_unresolved_but_nothing_raises():
    r = explain("SELECT a.name FROM artist a WHERE artist_id = 1", schema=None)
    assert r["tables"] == ["artist"]
    assert r["columns"] == ["artist.name"]  # qualified by an alias of a real table
    assert r["unresolved_columns"] == ["artist_id"]
    assert explain("SELECT 1 FROM artist", schema=[])["unresolved_columns"] == []


def test_star_is_not_a_column():
    r = explain("SELECT * FROM artist")
    assert r["columns"] == [] and r["unresolved_columns"] == [] and r["clauses"]["select"] == ["*"]
    r = explain("SELECT a.* FROM artist a")
    assert r["columns"] == [] and r["unresolved_columns"] == []


def test_set_operation_reports_first_select_and_the_union_limit():
    r = explain("SELECT name FROM artist UNION SELECT name FROM track ORDER BY name LIMIT 5")
    assert r["tables"] == ["artist", "track"] and r["columns"] == ["artist.name", "track.name"]
    c = r["clauses"]
    assert c["select"] == ["name"] and c["from"] == "artist" and c["limit"] == 5
    assert c["order_by"] == ["name"]
    assert r["limit_enforced"] == 5 and r["limit_added"] is False


def test_limit_added_kept_and_lowered():
    added = explain("SELECT name FROM artist")
    assert added["limit_added"] is True and added["limit_enforced"] == 100
    assert added["checks"][6] == {"name": "limit", "ok": True, "detail": "LIMIT 100 added"}

    kept = explain("SELECT name FROM artist LIMIT 3")
    assert kept["limit_added"] is False and kept["limit_enforced"] == 3
    assert kept["checks"][6]["detail"] == "LIMIT 3 kept (max 100)"

    exact = explain("SELECT name FROM artist LIMIT 100")
    assert exact["limit_added"] is False and exact["checks"][6]["detail"] == "LIMIT 100 kept (max 100)"

    lowered = explain("SELECT name FROM artist LIMIT 5000")
    assert lowered["limit_added"] is True and lowered["limit_enforced"] == 100
    assert lowered["checks"][6]["detail"] == "LIMIT 5000 lowered to 100"

    replaced = explain("SELECT name FROM artist LIMIT ALL")  # not a literal: the guard replaces it
    assert replaced["limit_added"] is True and replaced["limit_enforced"] == 100


def test_tables_check_lists_the_tables():
    r = explain("SELECT ar.name FROM artist ar JOIN album al ON al.artist_id = ar.artist_id")
    assert r["checks"][5] == {"name": "tables_allowed", "ok": True, "detail": "album, artist"}


def test_never_raises_on_garbage_inputs():
    for validated in ("SELECT FROM WHERE", "not sql at all ((", ""):
        r = explain_validation("SELECT 1", validated, None, ALLOWED, 100, SCHEMA)
        assert len(r["checks"]) == 7 and isinstance(r["tables"], list)
    r = explain_validation("SELECT 1", "SELECT 1 FROM artist LIMIT 5", None, ALLOWED, 100, [{"bad": "schema"}])
    assert r["unresolved_columns"] == [] and r["limit_enforced"] == 5
