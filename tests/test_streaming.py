import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import END, START, StateGraph

from t2sql.graph import stream_question
from t2sql.nodes.common import emit, traced
from t2sql.state import AgentState
from t2sql.streaming import TokenStream, chunk_text, decode_sql_prefix


def test_decode_none_until_the_sql_field_opens():
    assert decode_sql_prefix("") is None
    assert decode_sql_prefix('{"assumptions": ["a"], "sq') is None
    assert decode_sql_prefix('{"sql"') is None
    assert decode_sql_prefix('{"sql":') is None
    assert decode_sql_prefix('{"sql": "') == ""
    assert decode_sql_prefix('{\n  "sql" :  "SEL') == "SEL"


def test_decode_grows_monotonically_over_a_whole_document():
    raw = json.dumps({"tables_used": ["t"], "sql": 'SELECT "a\\b",\n\t\'x\' / é 😀 FROM t', "assumptions": ['say "hi"']})
    expected = 'SELECT "a\\b",\n\t\'x\' / é 😀 FROM t'
    prefixes = [decode_sql_prefix(raw[:i]) for i in range(len(raw) + 1)]
    seen = [p for p in prefixes if p is not None]
    assert seen[-1] == expected
    for earlier, later in zip(seen, seen[1:]):
        assert later.startswith(earlier)  # never retracts, even in the middle of an escape
    # ascii-escaped output too (json.dumps default): \uXXXX and surrogate pairs
    raw = json.dumps({"sql": expected}, ensure_ascii=True)
    prefixes = [p for p in (decode_sql_prefix(raw[:i]) for i in range(len(raw) + 1)) if p is not None]
    assert prefixes[-1] == expected
    assert all(b.startswith(a) for a, b in zip(prefixes, prefixes[1:]))


@pytest.mark.parametrize(
    ("raw", "sql"),
    [
        ('{"sql": "a\\', "a"),  # lone backslash
        ('{"sql": "a\\u00', "a"),  # half a unicode escape
        ('{"sql": "a\\u00e9', "aé"),
        ('{"sql": "a\\ud83d', "a"),  # high surrogate waiting for its low half
        ('{"sql": "a\\ud83d\\ude', "a"),
        ('{"sql": "a\\ud83d\\ude00b', "a😀b"),
        ('{"sql": "x\\"y\\\\z', 'x"y\\z'),
        ('{"sql": "line1\\nline2\\t\\/', "line1\nline2\t/"),
        ('{"sql": "done", "assumptions": ["has \\"sql\\": \\"no\\""]}', "done"),  # stops at the closing quote
        ('{"sql": "bad \\q escape', "bad q escape"),  # invalid JSON escape: keep the character
    ],
)
def test_decode_escapes(raw, sql):
    assert decode_sql_prefix(raw) == sql


def test_chunk_text_reads_strings_blocks_and_tool_call_fragments():
    assert chunk_text(SimpleNamespace(content="abc")) == "abc"
    blocks = [{"type": "text", "text": "x"}, "y", {"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "z"}]
    assert chunk_text(SimpleNamespace(content=blocks)) == "xyz"
    chunk = AIMessageChunk(content="", tool_call_chunks=[
        {"name": "G", "args": '{"sql": "SEL', "id": "1", "index": 0},
        {"name": None, "args": None, "id": None, "index": 1},
    ])  # fmt: skip
    assert chunk_text(chunk) == '{"sql": "SEL'
    assert chunk_text(SimpleNamespace()) == ""


def test_token_stream_generate_sql_events():
    ts = TokenStream()
    ts.start("generate_sql", 2)
    ev = ts.feed("generate_sql", AIMessageChunk(content='{"tables_used": [], '))
    assert ev == {
        "type": "token", "node": "generate_sql", "attempt": 2,
        "data": {"delta": '{"tables_used": [], ', "sql": None, "text": None},
    }  # fmt: skip
    ev = ts.feed("generate_sql", AIMessageChunk(content='"sql": "SELECT 1'))
    assert ev["data"]["sql"] == "SELECT 1" and ev["data"]["text"] is None
    ev = ts.feed("generate_sql", AIMessageChunk(content=' FROM t"}'))
    assert ev["data"]["sql"] == "SELECT 1 FROM t" and ev["data"]["delta"] == ' FROM t"}'


def test_token_stream_respond_accumulates_text_and_ignores_other_nodes_and_empty_chunks():
    ts = TokenStream()
    ts.start("respond", 1)
    assert ts.feed("classify", AIMessageChunk(content="x")) is None
    assert ts.feed("respond", AIMessageChunk(content="")) is None
    a = ts.feed("respond", AIMessageChunk(content="Three "))
    b = ts.feed("respond", AIMessageChunk(content="artists."))
    assert a["data"] == {"delta": "Three ", "sql": None, "text": "Three "}
    assert b["data"] == {"delta": "artists.", "sql": None, "text": "Three artists."}


def test_token_stream_resets_on_start_and_on_a_new_message_id():
    ts = TokenStream()
    ts.start("generate_sql", 1)
    ts.feed("generate_sql", AIMessageChunk(content='{"sql": "BAD', id="m1"))
    # a parse retry (new message) must not be glued to the first answer
    ev = ts.feed("generate_sql", AIMessageChunk(content='{"sql": "GOOD', id="m2"))
    assert ev["data"]["sql"] == "GOOD"
    ts.start("generate_sql", 2)  # the repair attempt starts clean
    ev = ts.feed("generate_sql", AIMessageChunk(content='{"sql": "NEXT', id="m3"))
    assert ev["data"]["sql"] == "NEXT" and ev["attempt"] == 2


def _tiny_graph(chat):
    """A graph whose generate_sql / respond nodes call a real (fake) LangChain chat model."""

    @traced("generate_sql", new_attempt=True)
    def generate_sql(state: AgentState) -> dict:
        emit({"type": "prompt", "node": "generate_sql", "attempt": 1, "data": {"messages": [], "chars": 0}})
        msg = chat.invoke("q")
        return {"sql": json.loads(msg.content)["sql"], "attempts": 1}

    @traced("respond")
    def respond(state: AgentState) -> dict:
        return {"answer": chat.invoke("q2").content}

    g = StateGraph(AgentState)
    g.add_node("generate_sql", generate_sql)
    g.add_node("respond", respond)
    g.add_edge(START, "generate_sql")
    g.add_edge("generate_sql", "respond")
    g.add_edge("respond", END)
    return g.compile()


def test_stream_question_turns_langgraph_messages_into_token_events():
    chat = GenericFakeChatModel(messages=iter([
        AIMessage(content='{"sql": "SELECT count(*) FROM customer", "assumptions": []}'),
        AIMessage(content="There are 59 customers."),
    ]))  # fmt: skip
    evs = list(stream_question(_tiny_graph(chat), "How many customers?"))
    kinds = [(e["type"], e.get("node")) for e in evs]
    assert kinds[:3] == [("start", "generate_sql"), ("prompt", "generate_sql"), ("token", "generate_sql")]
    assert kinds.index(("node", "generate_sql")) > max(i for i, k in enumerate(kinds) if k == ("token", "generate_sql"))
    assert kinds[-1] == ("final", None)
    sql_tokens = [e["data"]["sql"] for e in evs if e["type"] == "token" and e["node"] == "generate_sql"]
    assert sql_tokens[0] is None  # the first fragments are before the `"sql": "` opening
    sql_tokens = [t for t in sql_tokens if t is not None]
    assert sql_tokens[-1] == "SELECT count(*) FROM customer"
    assert all(b.startswith(a) for a, b in zip(sql_tokens, sql_tokens[1:]))  # the sql prefix only grows
    texts = [e["data"]["text"] for e in evs if e["type"] == "token" and e["node"] == "respond"]
    assert texts[-1] == "There are 59 customers." and len(texts) > 1
    assert all(e["attempt"] == 1 for e in evs if e["type"] == "token")


def test_emit_is_a_noop_outside_a_run():
    emit({"type": "start", "node": "x", "attempt": 0})  # no runnable context: must not raise
