"""All prompts and canned messages in one place."""

import json
from typing import Any

from t2sql.catalog import TABLE_DESCRIPTIONS
from t2sql.llm import Message

CLASSIFY_SYSTEM = """You decide whether a user question can be answered by querying a SQL database.
The database is a digital music store (Chinook) with these tables:
{tables}

in_scope is true only if the question is about data this database holds
(customers, sales, invoices, music catalog, employees, playlists).
Greetings, general knowledge, coding help, opinions or requests to change data are out of scope.
Also report the language of the question: "en" or "es" (use "en" if neither)."""

GENERATE_SYSTEM = """You are a PostgreSQL expert. Write ONE read-only SQL query that answers the question.
Rules:
- PostgreSQL dialect. A single SELECT statement (CTEs allowed). No semicolon-separated statements.
- Use ONLY the tables and columns listed in the schema. Never invent columns or tables.
- Table and column names are lowercase snake_case, exactly as in the schema.
- Prefer explicit JOINs using the foreign keys shown. Alias aggregates with readable names.
- Add ORDER BY when the question implies ranking ("top", "most", "best").
In your answer: sql = the query; tables_used = tables it reads; assumptions = short notes on
any interpretation you had to make (empty list if none).

Schema:
{schema}"""

REPAIR_ADDENDUM = """

Your previous attempt(s) failed. Write a corrected query that fixes the problem.
- Read the error message carefully; fix the exact cause (wrong column, wrong table, bad join, syntax).
- If a query returned 0 rows, check literal values (spelling, case, accents: prefer ILIKE for names),
  date ranges and join conditions, and make sure the filter matches how the data is stored.
- If you are confident the query is right and the data simply has no match, return the same query.
Return the same structured answer (sql, tables_used, assumptions)."""

RESPOND_SYSTEM = """You answer a user's question from the result of a SQL query.
Reply in the same language as the question (English or Spanish), in 1 to 3 short sentences.
Use only the data shown; do not invent numbers. If the result is empty, say that no data matched.
If the result is truncated, say that you are showing a sample. Do not repeat the SQL."""

OUT_OF_SCOPE = {
    "en": "I can't answer that with this database.",
    "es": "No puedo responder eso con esta base de datos.",
}
FAILURE = {
    "en": "I couldn't get a result for that question: {error}",
    "es": "No pude obtener un resultado para esa pregunta: {error}",
}


def classify_messages(question: str, tables: list[str]) -> list[Message]:
    listing = "\n".join(f"- {t}: {TABLE_DESCRIPTIONS.get(t, '')}".rstrip(": ") for t in tables)
    return [Message("system", CLASSIFY_SYSTEM.format(tables=listing)), Message("user", question)]


def generate_messages(question: str, schema: str) -> list[Message]:
    return [Message("system", GENERATE_SYSTEM.format(schema=schema)), Message("user", question)]


def repair_messages(question: str, schema: str, history: list[dict[str, Any]]) -> list[Message]:
    attempts = "\n\n".join(
        f"Attempt {h['attempt']}:\nSQL:\n{h['sql']}\nProblem: {h['error']}" for h in history
    )
    user = f"Question: {question}\n\nFailed attempts so far:\n\n{attempts}\n\nWrite the corrected query."
    return [Message("system", GENERATE_SYSTEM.format(schema=schema) + REPAIR_ADDENDUM), Message("user", user)]


def respond_messages(
    question: str, sql: str, columns: list[str], rows: list[dict[str, Any]], total_rows: int
) -> list[Message]:
    data = json.dumps(rows, default=str, ensure_ascii=False)
    note = f" (showing the first {len(rows)} of {total_rows})" if total_rows > len(rows) else ""
    user = f"Question: {question}\n\nSQL:\n{sql}\n\nColumns: {columns}\nRows{note}:\n{data}"
    return [Message("system", RESPOND_SYSTEM), Message("user", user)]
