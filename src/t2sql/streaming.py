"""Live token streaming helpers: turn LLM message chunks into `token` events.

The structured output of generate_sql arrives as JSON text (`{"sql": "SELECT ...`) or, with
tool-calling providers, as tool-call argument fragments. Neither is valid JSON until the
very end, so `decode_sql_prefix` reads the `"sql"` string out of the partial text. Pure
functions plus a small accumulator; no LangChain imports, chunks are read by duck typing.
"""

import re
from typing import Any

_SQL_FIELD = re.compile(r'"sql"\s*:\s*"')
_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}


def decode_sql_prefix(raw: str) -> str | None:
    """The value of the `"sql"` JSON string field in partial JSON `raw`, as far as it goes.

    Returns None until the opening `"sql": "` has arrived. Escapes are decoded; the string
    ends at the first unescaped quote (or at the end of `raw` while still streaming). An
    incomplete trailing escape (a lone backslash, half a \\uXXXX) is dropped, so the result
    only ever grows as more text arrives.
    """
    m = _SQL_FIELD.search(raw)
    if m is None:
        return None
    out: list[str] = []
    i, n = m.end(), len(raw)
    while i < n:
        ch = raw[i]
        if ch == '"':
            break
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            break  # lone backslash at the end
        esc = raw[i + 1]
        if esc in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[esc])
            i += 2
        elif esc == "u":
            code = _hex4(raw, i + 2)
            if code is None:
                break  # incomplete (or invalid) \uXXXX
            i += 6
            if 0xD800 <= code < 0xDC00:  # high surrogate: needs its low half to be a character
                low = _hex4(raw, i + 2) if raw[i : i + 2] == "\\u" else None
                if low is None or not 0xDC00 <= low < 0xE000:
                    break
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                i += 6
            out.append(chr(code))
        else:  # not valid JSON; keep the character rather than lose the text
            out.append(esc)
            i += 2
    return "".join(out)


def _hex4(raw: str, start: int) -> int | None:
    digits = raw[start : start + 4]
    if len(digits) < 4:
        return None
    try:
        return int(digits, 16)
    except ValueError:
        return None


def chunk_text(chunk: Any) -> str:
    """The raw text a message chunk adds: content text plus tool-call argument fragments."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        text = content
    else:  # a list of content blocks: plain strings or {"type": "text", "text": ...} dicts
        text = "".join(
            b if isinstance(b, str) else (b.get("text") or "") if b.get("type", "text") == "text" else ""
            for b in content
            if isinstance(b, str | dict)
        )
    args = "".join(tc.get("args") or "" for tc in getattr(chunk, "tool_call_chunks", None) or [])
    return text + args


class TokenStream:
    """Accumulates the chunks of one LLM call per node and builds `token` events.

    `generate_sql` events carry the decoded `sql` so far, `respond` events the whole text.
    A new message id (a parse retry, or the next provider after a failure) starts a fresh
    accumulation, so two attempts never get glued into one broken JSON.
    """

    NODES = ("generate_sql", "respond")

    def __init__(self) -> None:
        self._raw: dict[str, str] = {}
        self._msg_id: dict[str, str | None] = {}
        self._attempt: dict[str, int] = {}

    def start(self, node: str, attempt: int) -> None:
        """A node began (its `start` event): forget the previous generation of that node."""
        self._raw[node] = ""
        self._msg_id.pop(node, None)
        self._attempt[node] = attempt

    def feed(self, node: str, chunk: Any) -> dict[str, Any] | None:
        """The `token` event for one message chunk, or None if it adds nothing to show."""
        if node not in self.NODES:
            return None
        delta = chunk_text(chunk)
        if not delta:
            return None
        msg_id = getattr(chunk, "id", None)
        if msg_id is not None and self._msg_id.get(node) not in (None, msg_id):
            self._raw[node] = ""
        if msg_id is not None:
            self._msg_id[node] = msg_id
        raw = self._raw[node] = self._raw.get(node, "") + delta
        data = {"delta": delta, "sql": None, "text": None}
        if node == "generate_sql":
            data["sql"] = decode_sql_prefix(raw)
        else:
            data["text"] = raw
        return {"type": "token", "node": node, "attempt": self._attempt.get(node, 0), "data": data}
