"""Runs the pure formatAnswer() from the web UI under node (skipped when node is missing)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parents[1] / "src/t2sql/web/index.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def fmt(text: str) -> str:
    src = re.search(r"// FORMAT_ANSWER_START(.*?)// FORMAT_ANSWER_END", HTML.read_text(), re.S).group(1)
    js = src + f"\nprocess.stdout.write(formatAnswer({json.dumps(text)}));"
    return subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize(
    "text, expected",
    [
        ("es **U2**, con **15.84**", "es <strong>U2</strong>, con <strong>15.84</strong>"),
        ("__bold__ y *it* y `x<y`", "<strong>bold</strong> y <em>it</em> y <code>x&lt;y</code>"),
        ("<b>hi</b> & \"q\"", "&lt;b&gt;hi&lt;/b&gt; &amp; &quot;q&quot;"),
        ("2*3*4 y a * b * c", "2*3*4 y a * b * c"),
        ("customer_id y my__var__x", "customer_id y my__var__x"),
        ("sin cerrar **U2", "sin cerrar **U2"),
        ("sin cerrar `code", "sin cerrar `code"),
        ("**<script>**", "<strong>&lt;script&gt;</strong>"),
    ],
)
def test_format_answer(text, expected):
    assert fmt(text) == expected


def test_partial_stream_prefixes_never_produce_broken_html():
    full = "El top es **U2** con *15.84* y `x`."
    for i in range(len(full) + 1):
        out = fmt(full[:i])
        for tag in ("strong", "em", "code"):
            assert out.count(f"<{tag}>") == out.count(f"</{tag}>")


def classify_summary(d: dict) -> str:
    src = re.search(r"// CLASSIFY_SUMMARY_START(.*?)// CLASSIFY_SUMMARY_END", HTML.read_text(), re.S).group(1)
    js = src + f"\nprocess.stdout.write(classifySummary({json.dumps(d)}));"
    return subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize(
    "d, expected",
    [
        ({"decided_by": "local", "local_model": "tev1", "p_in": 0.97}, "local tev1 p=0.97"),
        ({"decided_by": "local", "local_model": "tev1"}, "local tev1"),
        ({"decided_by": "llm", "local_model": "tev1", "p_in": 0.71}, "tev1 unsure (p=0.71) → cloud LLM"),
        ({"decided_by": "llm", "local_model": "tev1", "local_unavailable": True}, "tev1 offline → cloud LLM"),
        ({"decided_by": "llm", "classifier": "llm"}, "cloud LLM"),
    ],
)
def test_classify_summary(d, expected):
    assert classify_summary(d) == expected
