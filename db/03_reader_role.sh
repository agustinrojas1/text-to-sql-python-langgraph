#!/bin/bash
# Creates the read-only role used by the agent to execute generated SQL.
# Runs once, on first container start, after 02_chinook.sql has loaded the tables.
set -euo pipefail

psql -v ON_ERROR_STOP=1 -v reader_password="${T2SQL_READER_PASSWORD:-t2sql_reader}" \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE ROLE t2sql_reader LOGIN PASSWORD :'reader_password'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;

-- Defense in depth: even if SQL slips through the guard, the role itself
-- cannot write and cannot run long queries.
ALTER ROLE t2sql_reader SET statement_timeout = '5s';
ALTER ROLE t2sql_reader SET default_transaction_read_only = on;

-- Lock down the database and schema, then grant only what is needed.
REVOKE ALL ON DATABASE chinook FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE chinook TO t2sql_reader;
GRANT USAGE ON SCHEMA public TO t2sql_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO t2sql_reader;
SQL
