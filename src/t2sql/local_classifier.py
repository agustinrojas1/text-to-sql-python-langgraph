"""Optional local "System 1" scope classifier: a small model served by llama.cpp's `llama-server`.

A scorer answers one question, "P(this question is about the database)", in a few hundred
milliseconds on a local GPU, with no tokens spent. The model is Tev1-4B (GGUF), queried through
`/completion`; P(in scope) is read from the probabilities of the option letters A / B. (A second
model, LiquidAI d1-3B, was evaluated and its support removed; see docs/system1-classifier-study.md.)

The scorers never decide anything: the classify node applies the thresholds. Any failure
(server down, timeout, malformed answer) raises `LocalClassifierUnavailable`, which the node
treats as "ask the cloud LLM instead".

HTTP uses the standard library (urllib): one JSON POST and one GET do not justify a new
runtime dependency, and tests inject a fake `http` callable.
"""

import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from t2sql import prompts
from t2sql.config import Settings

# (url, JSON payload or None for GET, timeout in seconds) -> parsed JSON body
Http = Callable[[str, dict[str, Any] | None, float], Any]


class LocalClassifierUnavailable(Exception):
    """The local classifier could not produce a score (the caller falls back to the cloud LLM)."""


@dataclass(frozen=True)
class ScopeScore:
    p_in: float  # probability that the database can answer the question, 0..1
    latency_ms: float
    model: str  # "tev1"


class ScopeScorer(Protocol):
    model: str

    def score(self, question: str) -> ScopeScore: ...

    def available(self) -> bool:
        """Is the server reachable and ready? Short timeout, never raises."""
        ...


def http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    """GET (payload None) or POST JSON; returns the decoded body. Errors become LocalClassifierUnavailable."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (operator-configured URL)
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as e:  # URLError, timeouts, bad JSON, bad status
        raise LocalClassifierUnavailable(f"{url}: {type(e).__name__}: {e}") from e


class _LlamaServerScorer:
    model: str

    def __init__(self, settings: Settings, http: Http = http_json, url: str | None = None) -> None:
        self._base = (url or settings.local_classifier_url).rstrip("/")
        self._timeout = settings.local_classifier_timeout_s
        self._tables = sorted(settings.allowed_tables)
        self._http = http

    def _request(self, question: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def _parse(self, body: Any) -> float:
        raise NotImplementedError

    def score(self, question: str) -> ScopeScore:
        path, payload = self._request(question)
        start = time.perf_counter()
        body = self._http(self._base + path, payload, self._timeout)
        latency = (time.perf_counter() - start) * 1000
        try:
            p = self._parse(body)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LocalClassifierUnavailable(f"unexpected {self.model} response: {type(e).__name__}: {e}") from e
        return ScopeScore(p_in=p, latency_ms=round(latency, 1), model=self.model)

    def available(self) -> bool:
        try:
            body = self._http(self._base + "/health", None, min(self._timeout, 1.0))
        except LocalClassifierUnavailable:
            return False
        return not isinstance(body, dict) or body.get("status", "ok") == "ok"


class Tev1Scorer(_LlamaServerScorer):
    """One generated token per request; P(in scope) is the mean over both option orders.

    Order 1 lists in_scope first (P(in) = P("A") / (P("A") + P("B"))); order 2 lists out_of_scope
    first (P(in) = P("B") / (P("A") + P("B"))). Averaging cancels the model's position bias; this
    is exactly what the benchmark measured, so thresholds tuned there apply (2 requests, ~200 ms each).
    """

    model = "tev1"
    MIN_LETTER_MASS = 0.5  # below this the model did not answer with A or B: the score is meaningless

    def _request(self, question: str, swap: bool = False) -> tuple[str, dict[str, Any]]:
        return "/completion", {
            "prompt": prompts.tev1_prompt(question, self._tables, swap=swap),
            "n_predict": 1,
            "n_probs": 10,
            "temperature": 0,
            "cache_prompt": False,  # the question comes first in the prompt, nothing to reuse
        }

    def _parse(self, body: Any, swap: bool = False) -> float:
        probs: dict[str, float] = {}
        for t in body["completion_probabilities"][0]["top_logprobs"]:
            p = math.exp(t["logprob"]) if "logprob" in t else float(t["prob"])
            probs[t["token"].strip()] = max(probs.get(t["token"].strip(), 0.0), p)
        a, b = probs.get("A", 0.0), probs.get("B", 0.0)
        if a + b < self.MIN_LETTER_MASS:
            raise ValueError(f"probability mass on A/B is only {a + b:.3f}")
        return (b if swap else a) / (a + b)

    def score(self, question: str) -> ScopeScore:
        ps: list[float] = []
        latency = 0.0
        for swap in (False, True):
            path, payload = self._request(question, swap)
            start = time.perf_counter()
            body = self._http(self._base + path, payload, self._timeout)
            latency += (time.perf_counter() - start) * 1000
            try:
                ps.append(self._parse(body, swap))
            except (KeyError, IndexError, TypeError, ValueError) as e:
                raise LocalClassifierUnavailable(f"unexpected {self.model} response: {type(e).__name__}: {e}") from e
        return ScopeScore(p_in=sum(ps) / 2, latency_ms=round(latency, 1), model=self.model)


SCORERS: dict[str, type[_LlamaServerScorer]] = {"tev1": Tev1Scorer}


def build_scorer(settings: Settings, http: Http = http_json) -> ScopeScorer | None:
    """The configured scorer, or None when the classifier is the cloud LLM only."""
    if settings.classifier == "llm":
        return None
    model = settings.local_classifier_model
    return SCORERS[model](settings, http, settings.local_url(model))


class ScorerPool:
    """One scorer per local model, built on first use and then reused.

    `preset` supplies ready-made scorers (tests, or the one the caller already built). Each model
    talks to its own llama-server URL (`Settings.local_url`).
    """

    def __init__(
        self, settings: Settings, http: Http = http_json, preset: dict[str, ScopeScorer] | None = None
    ) -> None:
        self._settings, self._http = settings, http
        self._scorers: dict[str, ScopeScorer] = dict(preset or {})
        self._lock = threading.Lock()

    def get(self, model: str) -> ScopeScorer:
        with self._lock:
            if model not in self._scorers:
                self._scorers[model] = SCORERS[model](self._settings, self._http, self._settings.local_url(model))
            return self._scorers[model]

    def availability(self) -> dict[str, bool]:
        """Is each model's server up? Probed in parallel, so /health waits for one short timeout, not the sum."""
        models = list(SCORERS)
        with ThreadPoolExecutor(len(models)) as pool:
            return dict(zip(models, pool.map(lambda m: self.get(m).available(), models), strict=True))


# ---------------------------------------------------------------- language


# The local models output no language, and the answer is in the user's language. This only has to
# choose between "en" and "es" (like the cloud prompt: "en" if neither), so a word vote is enough.
_ES_WORDS = frozenset(
    "el la los las un una unos unas de del al y o que qué cuál cuáles cual cuales cuánto cuánta cuántos "
    "cuántas cuanto cuantos cuantas quién quiénes quien quienes cómo como dónde donde cuándo cuando por "
    "para con sin sobre entre desde hasta es son fue fueron hay tiene tienen tenés tenes tengo sos vos "
    "dame damelo dime decime mostrame muéstrame muestrame muestra listame lista más mas menos mucho "
    "muchos año años mes meses hoy ayer mi mis su sus nuestros cada todos todas también tambien pero "
    "hicieron vendieron compraron compró compro gastó gasto hizo fue sido están estan está".split()
)
_EN_WORDS = frozenset(
    "the an of and or to in on for with without from by per is are was were be been do does did have has "
    "had what which who whom whose how many much when where why show me list give get tell find top most "
    "least more less all each every any total average there their his her its our your my this that these "
    "those than then also but not can could would should will bought sold made".split()
)
_WORD = re.compile(r"[a-záéíóúüñ]+")


def detect_language(question: str) -> str:
    """"es" or "en" from punctuation, accents and a stop-word vote; ties and no evidence give "en".

    ¿ and ¡ are decisive; ñ is one vote for Spanish; otherwise the language with more of its
    function words wins (includes rioplatense forms such as "tenés", "mostrame", "decime").
    """
    q = question.lower()
    if "¿" in q or "¡" in q:
        return "es"
    words = _WORD.findall(q)
    es = sum(w in _ES_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if "ñ" in q:
        es += 1
    return "es" if es > en else "en"
