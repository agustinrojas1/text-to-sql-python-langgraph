from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from t2sql.config import Settings
from t2sql.llm import (
    ChatModelLLM,
    FallbackLLM,
    LLMUnavailableError,
    Message,
    build_llm,
)
from t2sql.testing import FakeLLM

MSGS = [Message("system", "s"), Message("user", "u")]


class Out(BaseModel):
    x: int


def settings(**kw) -> Settings:
    base = dict(_env_file=None, google_api_key=None, groq_api_key=None, anthropic_api_key=None)
    return Settings(**{**base, **kw})


class StubChat:
    """Mimics the two LangChain chat-model methods the adapter uses."""

    def __init__(self, parsed=None, text="hi", error: Exception | None = None):
        self.parsed, self.text, self.error = parsed, text, error
        self.structured_kwargs = None

    def with_structured_output(self, schema, **kw):
        self.structured_kwargs = kw
        return SimpleNamespace(invoke=lambda m: self._call(
            {"raw": SimpleNamespace(usage_metadata={"input_tokens": 7, "output_tokens": 3}),
             "parsed": self.parsed, "parsing_error": None}))

    def invoke(self, messages):
        return self._call(SimpleNamespace(content=self.text, usage_metadata={"input_tokens": 4, "output_tokens": 2}))

    def _call(self, value):
        if self.error:
            raise self.error
        return value


def test_adapter_captures_metadata():
    llm = ChatModelLLM("p", "m", StubChat(parsed=Out(x=1)))
    r = llm.structured(MSGS, Out)
    assert r.output == Out(x=1)
    assert (r.provider, r.model, r.input_tokens, r.output_tokens) == ("p", "m", 7, 3)
    assert r.latency_ms >= 0
    t = ChatModelLLM("p", "m", StubChat(text=" hello ")).text(MSGS)
    assert t.output == "hello" and t.input_tokens == 4 and t.meta["output_tokens"] == 2


def test_adapter_unparsed_output_raises():
    with pytest.raises(ValueError):
        ChatModelLLM("p", "m", StubChat(parsed=None)).structured(MSGS, Out)


def test_fallback_to_second_provider():
    first = ChatModelLLM("a", "ma", StubChat(error=RuntimeError("429 rate limit"), parsed=Out(x=0)))
    second = ChatModelLLM("b", "mb", StubChat(parsed=Out(x=2), text="ok"))
    llm = FallbackLLM([first, second])
    r = llm.structured(MSGS, Out)
    assert r.provider == "b" and r.output.x == 2
    assert llm.text(MSGS).provider == "b"


def test_fallback_uses_first_when_healthy():
    llm = FallbackLLM([FakeLLM([Out(x=1)], provider="a"), FakeLLM([], provider="b")])
    assert llm.structured(MSGS, Out).provider == "a"


def test_all_providers_failing_raises_with_details():
    llm = FallbackLLM([
        FakeLLM([TimeoutError("slow")], provider="a"),
        FakeLLM([ConnectionError("boom")], provider="b"),
    ])
    with pytest.raises(LLMUnavailableError) as e:
        llm.text(MSGS)
    assert "a: TimeoutError" in str(e.value) and "b: ConnectionError" in str(e.value)


def test_providers_without_keys_are_skipped():
    built = []

    def factory(name):
        def make(s):
            built.append(name)
            return FakeLLM([], provider=name)
        return make

    registry = {
        "gemini": ("google_api_key", factory("gemini")),
        "groq": ("groq_api_key", factory("groq")),
    }
    llm = build_llm(settings(groq_api_key="k"), registry)
    assert built == ["groq"]
    assert llm.provider == "groq"  # single provider: no fallback wrapper

    built.clear()
    llm = build_llm(settings(groq_api_key="k", google_api_key="k"), registry)
    assert built == ["gemini", "groq"] and isinstance(llm, FallbackLLM)


def test_no_keys_raises_clear_error():
    with pytest.raises(LLMUnavailableError, match=r"GOOGLE_API_KEY.*\.env\.example"):
        build_llm(settings())


def test_unknown_provider_raises():
    with pytest.raises(LLMUnavailableError, match="nope"):
        build_llm(settings(llm_providers="nope"))


def test_provider_order_from_env_string(monkeypatch):
    monkeypatch.setenv("T2SQL_LLM_PROVIDERS", "groq, Gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    s = Settings(_env_file=None)
    assert s.llm_providers == ["groq", "gemini"] and s.google_api_key == "g"


def test_real_factories_construct_without_network():
    s = settings(google_api_key="g", groq_api_key="q", anthropic_api_key="a",
                 llm_providers="gemini,groq,anthropic")
    llm = build_llm(s)
    assert [p.provider for p in llm.providers] == ["gemini", "groq", "anthropic"]
    assert [p.model_name for p in llm.providers] == [s.gemini_model, s.groq_model, s.anthropic_model]
