"""Local scope scorers and the language heuristic. No network except a loopback test server."""

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from t2sql import prompts
from t2sql.config import CHINOOK_TABLES, Settings
from t2sql.local_classifier import (
    LocalClassifierUnavailable,
    Tev1Scorer,
    build_scorer,
    detect_language,
    http_json,
)

TABLES = sorted(CHINOOK_TABLES)


def settings(**kw):
    return Settings(_env_file=None, classifier="cascade", local_classifier_url="http://llama:8090/", **kw)


class FakeHttp:
    """Records (url, payload, timeout) and returns the scripted body (or raises it)."""

    def __init__(self, body):
        self.body, self.calls = body, []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def tev1_body(a, b, extra=()):
    """Shape of llama-server /completion with n_predict=1, n_probs=10 (log-probabilities)."""
    top = [{"token": t, "logprob": math.log(p)} for t, p in (("A", a), ("B", b), *extra) if p > 0]
    return {"content": "A", "tokens_evaluated": 455, "completion_probabilities": [{"token": "A", "top_logprobs": top}]}


# ---------------------------------------------------------------- Tev1


class SeqHttp(FakeHttp):
    """Returns the scripted bodies in order, one per call."""

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return self.body[len(self.calls) - 1]


def test_tev1_makes_two_requests_with_the_benchmarked_prompts():
    http = FakeHttp(tev1_body(0.9, 0.1))
    Tev1Scorer(settings(), http).score("How many customers?")
    assert len(http.calls) == 2
    for (url, payload, timeout), swap in zip(http.calls, (False, True)):
        assert url == "http://llama:8090/completion" and timeout == 2.0
        assert payload["n_predict"] == 1 and payload["n_probs"] == 10
        assert payload["temperature"] == 0 and payload["cache_prompt"] is False
        assert payload["prompt"] == prompts.tev1_prompt("How many customers?", TABLES, swap=swap)
    prompt = http.calls[0][1]["prompt"]
    assert prompt.startswith("<|im_start|>system\nEvaluate the supplied decision task.")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    user = json.loads(prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
    assert list(user) == ["state", "question", "options"]  # state first, per the model card
    assert user["state"] == "How many customers?"
    assert [(o["label"], o["key"]) for o in user["options"]] == [("A", "in_scope"), ("B", "out_of_scope")]
    assert "- track: individual songs" in user["question"] and "- album:" in user["question"]


def test_tev1_swapped_prompt_puts_out_of_scope_first():
    user = json.loads(prompts.tev1_prompt("q", TABLES, swap=True).split("<|im_start|>user\n")[1].split("<|im_end|>")[0])
    assert [(o["label"], o["key"]) for o in user["options"]] == [("A", "out_of_scope"), ("B", "in_scope")]


def test_tev1_probability_is_the_mean_of_both_orders():
    # in-first: A=0.9 -> 0.9 ; swapped: B=0.7 of (0.3+0.7) -> 0.7 ; mean 0.8
    http = SeqHttp([tev1_body(0.9, 0.1), tev1_body(0.3, 0.7)])
    s = Tev1Scorer(settings(), http).score("q")
    assert s.p_in == pytest.approx(0.8) and s.model == "tev1" and s.latency_ms >= 0


@pytest.mark.parametrize(
    "a, b, expected",
    [(0.97, 0.03, 0.5), (0.5, 0.5, 0.5), (0.6, 0.2, 0.5)],  # same body for both orders: (A + B) / (A + B) / 2
)
def test_tev1_renormalises_over_the_letters(a, b, expected):
    s = Tev1Scorer(settings(), FakeHttp(tev1_body(a, b, extra=[("The", 0.00001)]))).score("q")
    assert s.p_in == pytest.approx(expected, abs=1e-6)


def test_tev1_accepts_prob_instead_of_logprob_and_padded_tokens():
    body = {"completion_probabilities": [{"top_logprobs": [{"token": " A", "prob": 0.8}, {"token": "B", "prob": 0.2}]}]}
    assert Tev1Scorer(settings(), FakeHttp(body)).score("q").p_in == pytest.approx(0.5)  # (0.8 + 0.2) / 2


def test_tev1_second_request_failure_is_unavailable():
    http = SeqHttp([tev1_body(0.9, 0.1), {"oops": 1}])
    with pytest.raises(LocalClassifierUnavailable, match="unexpected tev1 response"):
        Tev1Scorer(settings(), http).score("q")


def test_tev1_missing_letter_mass_is_unavailable():
    with pytest.raises(LocalClassifierUnavailable, match="mass"):
        Tev1Scorer(settings(), FakeHttp(tev1_body(0.0, 0.0, extra=[("The", 0.9)]))).score("q")


@pytest.mark.parametrize("body", [{}, {"completion_probabilities": []}, {"completion_probabilities": [{}]}, "oops", None])
def test_tev1_malformed_response_is_unavailable(body):
    with pytest.raises(LocalClassifierUnavailable, match="unexpected tev1 response"):
        Tev1Scorer(settings(), FakeHttp(body)).score("q")


# ---------------------------------------------------------------- transport, availability, factory


def test_transport_errors_propagate_as_unavailable():
    err = LocalClassifierUnavailable("connection refused")
    with pytest.raises(LocalClassifierUnavailable, match="connection refused"):
        Tev1Scorer(settings(), FakeHttp(err)).score("q")


def test_available_asks_health_with_a_short_timeout_and_never_raises():
    http = FakeHttp({"status": "ok"})
    assert Tev1Scorer(settings(local_classifier_timeout_s=5), http).available() is True
    assert http.calls == [("http://llama:8090/health", None, 1.0)]
    assert Tev1Scorer(settings(), FakeHttp({"status": "loading model"})).available() is False
    assert Tev1Scorer(settings(), FakeHttp(LocalClassifierUnavailable("down"))).available() is False


def test_build_scorer_by_mode_and_model():
    assert build_scorer(Settings(_env_file=None)) is None  # default: classifier=llm
    assert isinstance(build_scorer(settings(local_classifier_model="tev1")), Tev1Scorer)
    assert isinstance(build_scorer(Settings(_env_file=None, classifier="local")), Tev1Scorer)


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"status": "ok"})

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        got = json.loads(self.rfile.read(n))
        if self.path == "/boom":
            self._send(500, {"error": "x"})
        elif self.path == "/garbage":
            self._send(200, b"<html>")
        else:
            self._send(200, {"echo": got})

    def log_message(self, *a):
        pass


def test_http_json_against_a_loopback_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        assert http_json(base + "/x", {"a": "ñ"}, 2) == {"echo": {"a": "ñ"}}
        assert http_json(base + "/health", None, 2) == {"status": "ok"}
        for path in ("/boom", "/garbage"):
            with pytest.raises(LocalClassifierUnavailable):
                http_json(base + path, {}, 2)
        # End to end through a scorer: the echo has no probabilities, so it is "unexpected".
        with pytest.raises(LocalClassifierUnavailable, match="unexpected tev1 response"):
            Tev1Scorer(Settings(_env_file=None, classifier="local", local_classifier_url=base)).score("q")
    finally:
        srv.shutdown()
        srv.server_close()
    with pytest.raises(LocalClassifierUnavailable):  # nothing listens there any more
        http_json(base + "/x", {}, 0.5)


# ---------------------------------------------------------------- language


@pytest.mark.parametrize(
    "q",
    [
        "How many customers are there?",
        "Which 3 artists generated the most revenue?",
        "Show me the top 5 tracks by sales",
        "List employees hired after 2003",
        "Hello",
        "",
        "AC/DC",  # no evidence: default to en, like the cloud prompt
        "What is the weather in Madrid today?",
    ],
)
def test_detect_language_english(q):
    assert detect_language(q) == "en"


@pytest.mark.parametrize(
    "q",
    [
        "¿Cuántos clientes hay?",
        "¡Hola!",
        "Cuántos empleados hay",
        "Cuales son los 5 artistas con más ventas",
        "Mostrame las canciones de Rock",  # rioplatense imperative
        "Decime cuántas facturas tiene cada cliente",
        "Qué países tienen más compras",
        "Tenés los ingresos por año?",  # voseo, no inverted mark
        "Listame los álbumes del año 2010",
        "Compró alguien en Argentina",
        "Mañana",  # ñ alone
    ],
)
def test_detect_language_spanish(q):
    assert detect_language(q) == "es"


def test_detect_language_mixed_follows_the_majority():
    assert detect_language("Show me los clientes de Brasil") == "en"  # tie -> en
    assert detect_language("Cuántos customers hay por país") == "es"
    assert detect_language("What is the total of ventas?") == "en"
    assert detect_language("Dame the top artists") == "en"
    assert detect_language("¿Which artist has more albums?") == "es"  # ¿ is decisive
