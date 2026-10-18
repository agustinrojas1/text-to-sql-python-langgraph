import psycopg
import pytest

from t2sql.config import get_settings


def _db_reachable() -> bool:
    try:
        with psycopg.connect(get_settings().readonly_database_url, connect_timeout=2):
            return True
    except psycopg.Error:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip `integration` tests when the docker-compose database is not up."""
    if not any("integration" in item.keywords for item in items) or _db_reachable():
        return
    skip = pytest.mark.skip(reason="Postgres not reachable (run `docker compose up -d`)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
