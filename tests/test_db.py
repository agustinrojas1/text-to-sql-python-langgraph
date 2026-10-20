"""db.execute_readonly error classification, without a database (psycopg.connect is faked)."""

import psycopg
import pytest
from psycopg import errors

from t2sql.config import Settings
from t2sql.db import DatabaseUnavailableError, QueryError, execute_readonly, fetch_schema_metadata

SETTINGS = Settings(_env_file=None)


def raising(exc):
    def connect(*args, **kwargs):
        raise exc

    return connect


def test_connection_refused_is_database_unavailable():
    # Nothing listens on port 1: a real OperationalError with no sqlstate.
    settings = Settings(_env_file=None, readonly_database_url="postgresql://u:p@127.0.0.1:1/x?connect_timeout=2")
    with pytest.raises(DatabaseUnavailableError):
        execute_readonly("SELECT 1", settings)


@pytest.mark.parametrize(
    "exc",
    [
        psycopg.OperationalError("connection failed"),  # no sqlstate: never reached the server
        psycopg.InterfaceError("connection already closed"),
        errors.ConnectionFailure("08006"),
        errors.AdminShutdown(),  # 57P01
        errors.CrashShutdown(),  # 57P02
        errors.CannotConnectNow(),  # 57P03
        errors.TooManyConnections(),  # 53300
    ],
)
def test_infrastructure_errors_are_database_unavailable(monkeypatch, exc):
    monkeypatch.setattr(psycopg, "connect", raising(exc))
    with pytest.raises(DatabaseUnavailableError):
        execute_readonly("SELECT 1", SETTINGS)
    with pytest.raises(DatabaseUnavailableError):
        fetch_schema_metadata(SETTINGS)


@pytest.mark.parametrize(
    "exc,state",
    [
        (errors.SyntaxError(), "42601"),
        (errors.UndefinedColumn(), "42703"),
        (errors.DivisionByZero(), "22012"),
        (errors.InsufficientPrivilege(), "42501"),
        (errors.QueryCanceled(), "57014"),  # an OperationalError subclass, but a query problem
    ],
)
def test_query_errors_stay_query_errors(monkeypatch, exc, state):
    monkeypatch.setattr(psycopg, "connect", raising(exc))
    with pytest.raises(QueryError) as e:
        execute_readonly("SELECT 1", SETTINGS)
    assert e.value.sqlstate == state


def test_only_statement_timeout_is_a_timeout():
    assert QueryError("x", "57014").is_timeout
    assert not QueryError("x", "42703").is_timeout and not QueryError("x").is_timeout
