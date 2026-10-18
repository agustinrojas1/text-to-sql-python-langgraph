"""LLM layer: a small provider-agnostic interface plus LangChain-backed providers.

Graph nodes depend only on the `LLM` protocol and `Message`/`LLMResult`, never on
LangChain. `build_llm` assembles the configured providers behind `FallbackLLM`.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from t2sql.config import Settings, get_settings


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class LLMResult[T]:
    """Parsed output plus the metadata that Phase 3 tracing consumes."""

    output: T
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0

    @property
    def meta(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
        }


class LLM(Protocol):
    def structured[S: BaseModel](
        self, messages: list[Message], schema: type[S]
    ) -> LLMResult[S]: ...

    def text(self, messages: list[Message]) -> LLMResult[str]: ...


class LLMUnavailableError(RuntimeError):
    """No provider is configured, or every configured provider failed."""


def _to_lc(messages: list[Message]) -> list[SystemMessage | HumanMessage]:
    return [(SystemMessage if m.role == "system" else HumanMessage)(m.content) for m in messages]


def _tokens(msg: Any) -> tuple[int | None, int | None]:
    usage = getattr(msg, "usage_metadata", None) or {}
    return usage.get("input_tokens"), usage.get("output_tokens")


def _text_of(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    # Some providers return a list of content blocks.
    return "".join(
        b if isinstance(b, str) else b.get("text", "") for b in content if isinstance(b, (str, dict))
    )


class ChatModelLLM:
    """Adapts one LangChain chat model to the `LLM` protocol."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        chat: BaseChatModel,
        structured_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self._chat = chat
        self._structured_kwargs = structured_kwargs or {}

    def structured[S: BaseModel](self, messages: list[Message], schema: type[S]) -> LLMResult[S]:
        runnable = self._chat.with_structured_output(
            schema, include_raw=True, **self._structured_kwargs
        )
        start = time.perf_counter()
        out = runnable.invoke(_to_lc(messages))
        latency = (time.perf_counter() - start) * 1000
        if out.get("parsed") is None:
            raise ValueError(f"{self.provider}: could not parse structured output: {out.get('parsing_error')}")
        tin, tout = _tokens(out.get("raw"))
        return LLMResult(out["parsed"], self.provider, self.model_name, tin, tout, latency)

    def text(self, messages: list[Message]) -> LLMResult[str]:
        start = time.perf_counter()
        msg = self._chat.invoke(_to_lc(messages))
        latency = (time.perf_counter() - start) * 1000
        tin, tout = _tokens(msg)
        return LLMResult(_text_of(msg).strip(), self.provider, self.model_name, tin, tout, latency)


class FallbackLLM:
    """Tries each provider in order; any exception moves on to the next one."""

    def __init__(self, providers: list[LLM]) -> None:
        if not providers:
            raise ValueError("FallbackLLM needs at least one provider")
        self.providers = providers

    def _first_ok[R](self, call: Callable[[LLM], R]) -> R:
        errors: list[str] = []
        for p in self.providers:
            try:
                return call(p)
            except Exception as e:  # rate limit, 5xx, timeout, unparseable output...
                name = getattr(p, "provider", type(p).__name__)
                errors.append(f"{name}: {type(e).__name__}: {e}")
        raise LLMUnavailableError("All LLM providers failed:\n" + "\n".join(errors))

    def structured[S: BaseModel](self, messages: list[Message], schema: type[S]) -> LLMResult[S]:
        return self._first_ok(lambda p: p.structured(messages, schema))

    def text(self, messages: list[Message]) -> LLMResult[str]:
        return self._first_ok(lambda p: p.text(messages))


# --- provider construction -------------------------------------------------


def _make_gemini(s: Settings) -> ChatModelLLM:
    from langchain_google_genai import ChatGoogleGenerativeAI

    chat = ChatGoogleGenerativeAI(
        model=s.gemini_model,
        api_key=s.google_api_key,
        temperature=s.llm_temperature,
        timeout=s.llm_timeout_s,
        retries=s.llm_max_retries,
    )
    return ChatModelLLM("gemini", s.gemini_model, chat)


def _make_groq(s: Settings) -> ChatModelLLM:
    from langchain_groq import ChatGroq

    chat = ChatGroq(
        model=s.groq_model,
        api_key=s.groq_api_key,
        temperature=s.llm_temperature,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
    )
    # json_schema mode is the one Groq supports natively (gpt-oss models).
    return ChatModelLLM("groq", s.groq_model, chat, {"method": "json_schema"})


def _make_anthropic(s: Settings) -> ChatModelLLM:
    from langchain_anthropic import ChatAnthropic

    # No temperature: recent Claude models use adaptive thinking and may reject it.
    chat = ChatAnthropic(
        model=s.anthropic_model,
        api_key=s.anthropic_api_key,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
    )
    return ChatModelLLM("anthropic", s.anthropic_model, chat)


# provider name -> (settings attribute with the API key, factory)
PROVIDERS: dict[str, tuple[str, Callable[[Settings], ChatModelLLM]]] = {
    "gemini": ("google_api_key", _make_gemini),
    "groq": ("groq_api_key", _make_groq),
    "anthropic": ("anthropic_api_key", _make_anthropic),
}


def build_llm(
    settings: Settings | None = None,
    providers: dict[str, tuple[str, Callable[[Settings], LLM]]] | None = None,
) -> LLM:
    """Build the configured providers (skipping those without a key) behind a fallback.

    `providers` is injectable so tests can avoid constructing real clients.
    """
    settings = settings or get_settings()
    registry = providers or PROVIDERS
    built: list[LLM] = []
    skipped: list[str] = []
    for name in settings.llm_providers:
        if name not in registry:
            raise LLMUnavailableError(
                f"Unknown LLM provider '{name}' in T2SQL_LLM_PROVIDERS (known: {', '.join(registry)})"
            )
        key_attr, factory = registry[name]
        if getattr(settings, key_attr):
            built.append(factory(settings))
        else:
            skipped.append(f"{name} ({key_attr.upper()})")
    if not built:
        raise LLMUnavailableError(
            "No LLM provider is available: missing API key for "
            + (", ".join(skipped) or "any provider")
            + ". Set it in .env (see .env.example) or check T2SQL_LLM_PROVIDERS."
        )
    return built[0] if len(built) == 1 else FallbackLLM(built)
