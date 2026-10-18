"""Test doubles shared by the test suite (and usable from evals later)."""

import hashlib
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from t2sql.llm import LLMResult, Message

Scripted = BaseModel | str | Exception


@dataclass
class FakeCall:
    kind: str  # "structured" | "text"
    messages: list[Message]
    schema: type[BaseModel] | None = None

    @property
    def prompt(self) -> str:
        return "\n".join(m.content for m in self.messages)


@dataclass
class FakeLLM:
    """Returns scripted responses in call order and records every prompt.

    Each response is a BaseModel instance (for `structured`), a str (for `text`),
    or an Exception to raise.
    """

    responses: list[Scripted]
    calls: list[FakeCall] = field(default_factory=list)
    provider: str = "fake"
    model: str = "fake-model"

    def _next(self, call: FakeCall) -> Scripted:
        self.calls.append(call)
        if not self.responses:
            raise AssertionError(f"FakeLLM: no scripted response left for {call.kind} call")
        return self.responses.pop(0)

    def _result[T](self, output: T) -> LLMResult[T]:
        return LLMResult(output, self.provider, self.model, 10, 5, 1.0)

    def structured[S: BaseModel](self, messages: list[Message], schema: type[S]) -> LLMResult[S]:
        r = self._next(FakeCall("structured", messages, schema))
        if isinstance(r, Exception):
            raise r
        if not isinstance(r, schema):
            raise AssertionError(f"FakeLLM: expected {schema.__name__}, got {r!r}")
        return self._result(r)

    def text(self, messages: list[Message]) -> LLMResult[str]:
        r = self._next(FakeCall("text", messages))
        if isinstance(r, Exception):
            raise r
        if not isinstance(r, str):
            raise AssertionError(f"FakeLLM: expected str, got {r!r}")
        return self._result(r)


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _bucket(word: str, dim: int) -> int:
    return int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "big") % dim


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder: no model download, no network."""

    def __init__(self, dim: int = 256) -> None:
        self.name, self.dim = f"fake-bow-{dim}", dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for w in _TOKEN.findall(text.lower()):
                v[_bucket(w, self.dim)] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out
