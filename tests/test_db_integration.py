import pytest

from t2sql.config import get_settings
from t2sql.db import QueryError, execute_readonly, fetch_schema_metadata
from t2sql.sql_guard import validate_sql

pytestmark = pytest.mark.integration


def test_reader_can_select():
    result = execute_readonly("SELECT artist_id, name FROM artist ORDER BY artist_id LIMIT 3")
    assert result.columns == ["artist_id", "name"]
    assert result.row_count == 3
    assert result.rows[0] == {"artist_id": 1, "name": "AC/DC"}


def test_guard_output_runs_against_db():
    settings = get_settings()
    q = validate_sql("SELECT * FROM track", set(settings.allowed_tables), settings.max_rows)
    assert execute_readonly(q.sql).row_count == settings.max_rows


def test_statement_timeout_fires():
    # Bypasses the guard on purpose. A short per-call timeout (below the role's 5s)
    # proves the helper's own SET LOCAL works, not just the role default.
    settings = get_settings().model_copy(update={"statement_timeout_ms": 300})
    with pytest.raises(QueryError) as exc:
        execute_readonly("SELECT pg_sleep(3)", settings)
    assert exc.value.is_timeout
    assert "statement timeout" in exc.value.message


def test_role_has_its_own_statement_timeout():
    # The role has statement_timeout = 5s; check it is configured.
    assert execute_readonly("SHOW statement_timeout").rows == [{"statement_timeout": "5s"}]


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO genre (genre_id, name) VALUES (9999, 'x')",
        "UPDATE track SET name = 'x'",
        "DELETE FROM track",
        "DROP TABLE track",
        "CREATE TABLE evil (a int)",
        "TRUNCATE track",
    ],
)
def test_writes_fail(sql):
    with pytest.raises(QueryError) as exc:
        execute_readonly(sql)
    assert "read-only" in exc.value.message


def test_sql_error_message_is_surfaced():
    with pytest.raises(QueryError) as exc:
        execute_readonly("SELECT nope FROM track")
    assert 'column "nope" does not exist' in exc.value.message
    assert exc.value.sqlstate == "42703"


def test_reader_cannot_read_system_secrets():
    with pytest.raises(QueryError):
        execute_readonly("SELECT * FROM pg_shadow")


def test_schema_metadata():
    tables = {t.name: t for t in fetch_schema_metadata()}
    assert set(tables) == set(get_settings().allowed_tables)
    track = tables["track"]
    assert {c.name for c in track.columns} >= {"track_id", "name", "album_id", "unit_price"}
    assert next(c for c in track.columns if c.name == "track_id").is_primary_key
    fks = {(f.column, f.ref_table, f.ref_column) for f in track.foreign_keys}
    assert ("album_id", "album", "album_id") in fks
    pt = tables["playlist_track"]
    assert {c.name for c in pt.columns if c.is_primary_key} == {"playlist_id", "track_id"}
