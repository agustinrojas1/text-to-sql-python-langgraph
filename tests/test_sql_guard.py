import pytest

from t2sql.config import CHINOOK_TABLES
from t2sql.sql_guard import GuardCode, SqlGuardError, validate_sql

ALLOWED = set(CHINOOK_TABLES)
MAX = 100


def ok(sql: str):
    return validate_sql(sql, ALLOWED, MAX)


def rejected(sql: str) -> GuardCode:
    with pytest.raises(SqlGuardError) as exc:
        validate_sql(sql, ALLOWED, MAX)
    return exc.value.code


# --- accepted queries -------------------------------------------------------

VALID = [
    "SELECT * FROM track",
    "select name from artist where artist_id = 1",
    "SeLeCt Name FrOm Artist",
    "SELECT t.name, a.title FROM track t JOIN album a ON a.album_id = t.album_id",
    "SELECT g.name, COUNT(*) FROM track t JOIN genre g USING (genre_id) GROUP BY g.name HAVING COUNT(*) > 5",
    "SELECT * FROM public.track",
    "SELECT * FROM PUBLIC.TRACK",
    'SELECT * FROM "track"',
    'SELECT * FROM "public"."track"',
    "WITH c AS (SELECT customer_id FROM customer) SELECT * FROM c",
    "WITH a AS (SELECT * FROM track), b AS (SELECT * FROM a) SELECT * FROM b JOIN album USING (album_id)",
    "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
    "SELECT name FROM artist UNION SELECT title FROM album",
    "SELECT name FROM artist INTERSECT SELECT name FROM genre",
    "SELECT name FROM artist EXCEPT SELECT name FROM genre",
    "SELECT * FROM track WHERE album_id IN (SELECT album_id FROM album WHERE artist_id = 1)",
    "SELECT * FROM customer c WHERE EXISTS (SELECT 1 FROM invoice i WHERE i.customer_id = c.customer_id)",
    "SELECT name, RANK() OVER (PARTITION BY genre_id ORDER BY milliseconds DESC) FROM track",
    "SELECT * FROM (SELECT * FROM track) s",
    "SELECT * FROM generate_series(1, 3) AS g(n)",
    "SELECT date_trunc('year', invoice_date), SUM(total) FROM invoice GROUP BY 1",
    "SELECT 1",
    "(SELECT name FROM artist)",
    "SELECT * FROM track;",
    "SELECT 1; -- trailing comment",
    "SELECT 1; /* trailing block comment */",
    "SELECT 1;;",
    "-- hello\nSELECT * FROM track /* inline */",
]


@pytest.mark.parametrize("sql", VALID)
def test_valid_queries_are_accepted(sql):
    result = ok(sql)
    assert result.sql.upper().startswith(("SELECT", "WITH"))
    assert "LIMIT" in result.sql


def test_comments_are_stripped_from_output():
    assert "hello" not in ok("-- hello\nSELECT * FROM track /* inline */").sql


def test_tables_are_collected_and_normalised():
    result = ok(
        "WITH c AS (SELECT * FROM public.Customer) "
        "SELECT * FROM c JOIN INVOICE i ON i.customer_id = c.customer_id "
        "WHERE i.invoice_id IN (SELECT invoice_id FROM invoice_line)"
    )
    assert result.tables == {"customer", "invoice", "invoice_line"}


def test_cte_names_are_not_reported_as_tables():
    assert ok("WITH x AS (SELECT * FROM track) SELECT * FROM x").tables == {"track"}


# --- rejections -------------------------------------------------------------

REJECTIONS = [
    # parse / empty
    ("SELEC * FROM track", GuardCode.PARSE_ERROR),
    ("SELECT * FROM", GuardCode.PARSE_ERROR),
    ("SELECT 'unterminated FROM track", GuardCode.PARSE_ERROR),
    ("", GuardCode.EMPTY_QUERY),
    ("   ", GuardCode.EMPTY_QUERY),
    ("-- only a comment", GuardCode.EMPTY_QUERY),
    (";", GuardCode.EMPTY_QUERY),
    # multiple statements
    ("SELECT 1; DROP TABLE track", GuardCode.MULTIPLE_STATEMENTS),
    ("SELECT 1; SELECT 2", GuardCode.MULTIPLE_STATEMENTS),
    ("SELECT * FROM track; -- ok\nDELETE FROM track", GuardCode.MULTIPLE_STATEMENTS),
    ("SELECT * FROM track /* x */ ; /* y */ DROP TABLE track", GuardCode.MULTIPLE_STATEMENTS),
    # not a select
    ("INSERT INTO genre VALUES (99, 'x')", GuardCode.NOT_SELECT),
    ("UPDATE track SET name = 'x'", GuardCode.NOT_SELECT),
    ("DELETE FROM track", GuardCode.NOT_SELECT),
    ("MERGE INTO track t USING album a ON t.album_id = a.album_id WHEN MATCHED THEN DELETE", GuardCode.NOT_SELECT),
    ("DROP TABLE track", GuardCode.NOT_SELECT),
    ("CREATE TABLE x (a int)", GuardCode.NOT_SELECT),
    ("CREATE TABLE x AS SELECT * FROM track", GuardCode.NOT_SELECT),
    ("ALTER TABLE track ADD COLUMN x int", GuardCode.NOT_SELECT),
    ("TRUNCATE track", GuardCode.NOT_SELECT),
    ("COPY track TO '/tmp/x'", GuardCode.NOT_SELECT),
    ("COPY (SELECT * FROM track) TO PROGRAM 'id'", GuardCode.NOT_SELECT),
    ("GRANT ALL ON track TO public", GuardCode.NOT_SELECT),
    ("SET statement_timeout = 0", GuardCode.NOT_SELECT),
    ("SHOW ALL", GuardCode.NOT_SELECT),
    ("EXPLAIN SELECT * FROM track", GuardCode.NOT_SELECT),
    ("EXPLAIN ANALYZE DELETE FROM track", GuardCode.NOT_SELECT),
    ("VACUUM", GuardCode.NOT_SELECT),
    ("DO $$ BEGIN PERFORM 1; END $$", GuardCode.NOT_SELECT),
    ("CALL f()", GuardCode.NOT_SELECT),
    ("BEGIN", GuardCode.NOT_SELECT),
    ("VALUES (1)", GuardCode.NOT_SELECT),
    # forbidden clauses
    ("WITH x AS (DELETE FROM track RETURNING *) SELECT * FROM x", GuardCode.FORBIDDEN_CLAUSE),
    ("WITH x AS (INSERT INTO genre VALUES (99, 'x') RETURNING *) SELECT * FROM x", GuardCode.FORBIDDEN_CLAUSE),
    ("WITH x AS (UPDATE track SET name = 'x' RETURNING *) SELECT * FROM x", GuardCode.FORBIDDEN_CLAUSE),
    ("SELECT * INTO newtable FROM track", GuardCode.FORBIDDEN_CLAUSE),
    ("SELECT * INTO TEMP newtable FROM track", GuardCode.FORBIDDEN_CLAUSE),
    ("SELECT * FROM track FOR UPDATE", GuardCode.FORBIDDEN_CLAUSE),
    ("SELECT * FROM track FOR SHARE", GuardCode.FORBIDDEN_CLAUSE),
    ("SELECT * FROM track FOR NO KEY UPDATE", GuardCode.FORBIDDEN_CLAUSE),
    ("SELECT * FROM (SELECT * FROM track FOR UPDATE) s", GuardCode.FORBIDDEN_CLAUSE),
    # forbidden functions
    ("SELECT pg_sleep(10)", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT PG_SLEEP(10)", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT pg_catalog.pg_sleep(10)", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT * FROM pg_sleep(10)", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT * FROM track WHERE (SELECT pg_sleep(1)) IS NULL", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT pg_read_file('/etc/passwd')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT pg_read_binary_file('/etc/passwd')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT pg_ls_dir('.')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT lo_import('/etc/passwd')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT lo_export(1, '/tmp/x')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT dblink('host=x', 'select 1')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT dblink_exec('host=x', 'drop table x')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT set_config('statement_timeout', '0', false)", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT current_setting('server_version')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT pg_terminate_backend(1)", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT pg_cancel_backend(1)", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT nextval('s')", GuardCode.FORBIDDEN_FUNCTION),
    ("SELECT query_to_xml('select * from pg_shadow', true, true, '')", GuardCode.FORBIDDEN_FUNCTION),
    ('SELECT "pg_sleep"(1)', GuardCode.FORBIDDEN_FUNCTION),
    # forbidden tables
    ("SELECT * FROM pg_user", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM pg_shadow", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM pg_catalog.pg_user", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM pg_catalog.pg_tables", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM information_schema.tables", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM information_schema.columns", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM other_schema.track", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM chinook.public.track", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM secrets", GuardCode.FORBIDDEN_TABLE),
    ('SELECT * FROM "Track"', GuardCode.FORBIDDEN_TABLE),  # quoted = case-sensitive
    ('SELECT * FROM "pg_user"', GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM track t JOIN pg_user u ON true", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM track WHERE name IN (SELECT usename FROM pg_user)", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM (SELECT * FROM pg_user) s", GuardCode.FORBIDDEN_TABLE),
    ("WITH x AS (SELECT * FROM pg_user) SELECT * FROM x", GuardCode.FORBIDDEN_TABLE),
    ("SELECT name FROM artist UNION SELECT usename FROM pg_user", GuardCode.FORBIDDEN_TABLE),
    ("SELECT * FROM track, LATERAL (SELECT * FROM pg_user) u", GuardCode.FORBIDDEN_TABLE),
    ("SELECT /* track */ * FROM pg_user -- FROM track", GuardCode.FORBIDDEN_TABLE),
    # a CTE named like a catalog table must not hide the real table elsewhere
    (
        "SELECT * FROM pg_user, (WITH pg_user AS (SELECT 1 a) SELECT * FROM pg_user) x",
        GuardCode.FORBIDDEN_TABLE,
    ),
    (
        "SELECT * FROM track WHERE EXISTS (WITH pg_user AS (SELECT 1) SELECT 1 FROM pg_user) "
        "AND name IN (SELECT usename FROM pg_user)",
        GuardCode.FORBIDDEN_TABLE,
    ),
    # a CTE name does not hide a schema-qualified table
    ("WITH pg_user AS (SELECT 1) SELECT * FROM pg_catalog.pg_user", GuardCode.FORBIDDEN_TABLE),
]


@pytest.mark.parametrize(("sql", "code"), REJECTIONS)
def test_rejections(sql, code):
    assert rejected(sql) == code


def test_cte_shadowing_an_allowed_table_name_is_fine():
    assert ok("WITH track AS (SELECT 1 AS a) SELECT * FROM track").tables == frozenset()


def test_custom_allow_list_is_respected():
    with pytest.raises(SqlGuardError) as exc:
        validate_sql("SELECT * FROM album", {"track"}, MAX)
    assert exc.value.code is GuardCode.FORBIDDEN_TABLE
    assert validate_sql("SELECT * FROM TRACK", {"Track"}, MAX).tables == {"track"}


def test_error_has_code_and_message():
    with pytest.raises(SqlGuardError) as exc:
        ok("DROP TABLE track")
    assert exc.value.code == "not_select"
    assert "Drop" in exc.value.message


# --- LIMIT enforcement ------------------------------------------------------

LIMITS = [
    ("SELECT * FROM track", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track LIMIT 5", "SELECT * FROM track LIMIT 5"),
    ("SELECT * FROM track LIMIT 100", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track LIMIT 101", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track LIMIT 1000000", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track LIMIT ALL", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track LIMIT (SELECT 5000)", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track LIMIT 2 + 3", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track LIMIT 50 OFFSET 10", "SELECT * FROM track LIMIT 50 OFFSET 10"),
    ("SELECT * FROM track LIMIT 5000 OFFSET 10", "SELECT * FROM track LIMIT 100 OFFSET 10"),
    ("SELECT * FROM track OFFSET 10", "SELECT * FROM track LIMIT 100 OFFSET 10"),
    ("SELECT * FROM track FETCH FIRST 5 ROWS ONLY", "SELECT * FROM track LIMIT 5"),
    ("SELECT * FROM track FETCH FIRST 5000 ROWS ONLY", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track FETCH FIRST ROW ONLY", "SELECT * FROM track LIMIT 1"),
    ("SELECT * FROM track FETCH FIRST 5 ROWS WITH TIES", "SELECT * FROM track LIMIT 100"),
    ("SELECT * FROM track ORDER BY name LIMIT 7", "SELECT * FROM track ORDER BY name LIMIT 7"),
    # only the outermost query is touched
    (
        "SELECT * FROM (SELECT * FROM track LIMIT 5000) s",
        "SELECT * FROM (SELECT * FROM track LIMIT 5000) AS s LIMIT 100",
    ),
    (
        "WITH x AS (SELECT * FROM track LIMIT 3) SELECT * FROM x",
        "WITH x AS (SELECT * FROM track LIMIT 3) SELECT * FROM x LIMIT 100",
    ),
    # set operations: LIMIT applies to the whole result
    (
        "SELECT name FROM artist UNION SELECT title FROM album",
        "SELECT name FROM artist UNION SELECT title FROM album LIMIT 100",
    ),
    (
        "SELECT name FROM artist UNION SELECT title FROM album LIMIT 9999",
        "SELECT name FROM artist UNION SELECT title FROM album LIMIT 100",
    ),
    (
        "SELECT name FROM artist UNION SELECT title FROM album LIMIT 10",
        "SELECT name FROM artist UNION SELECT title FROM album LIMIT 10",
    ),
    (
        "SELECT name FROM artist UNION ALL SELECT title FROM album ORDER BY 1 LIMIT 10 OFFSET 5",
        "SELECT name FROM artist UNION ALL SELECT title FROM album ORDER BY 1 LIMIT 10 OFFSET 5",
    ),
    (
        "(SELECT name FROM artist LIMIT 5000) UNION (SELECT title FROM album LIMIT 5000)",
        "(SELECT name FROM artist LIMIT 5000) UNION (SELECT title FROM album LIMIT 5000) LIMIT 100",
    ),
    (
        "SELECT name FROM artist INTERSECT SELECT name FROM genre",
        "SELECT name FROM artist INTERSECT SELECT name FROM genre LIMIT 100",
    ),
    ("(SELECT * FROM track LIMIT 7)", "SELECT * FROM track LIMIT 7"),
]


@pytest.mark.parametrize(("sql", "expected"), LIMITS)
def test_limit_is_forced(sql, expected):
    assert ok(sql).sql == expected


def test_limit_uses_given_max():
    assert validate_sql("SELECT * FROM track", ALLOWED, 25).sql == "SELECT * FROM track LIMIT 25"
    assert validate_sql("SELECT * FROM track LIMIT 30", ALLOWED, 25).sql == "SELECT * FROM track LIMIT 25"


def test_validation_is_idempotent():
    once = ok("SELECT name FROM artist UNION SELECT title FROM album").sql
    assert ok(once).sql == once


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; -- x\nDROP TABLE track",
        "SELECT 1; /* x */ SELECT 2",
        "SELECT 1;; SELECT 2",
        "SELECT 1; -- c\nSELECT 2; -- d",
    ],
)
def test_comment_after_semicolon_does_not_hide_a_second_statement(sql):
    assert rejected(sql) == GuardCode.MULTIPLE_STATEMENTS
