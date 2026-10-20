import json
import re
from pathlib import Path

from t2sql.replay import (
    DEFAULT_QUESTIONS,
    load_recording,
    normalize,
    record_lines,
    recording_path,
    replay_lines,
)

FINAL = json.dumps({"type": "final", "answer": "ok"}) + "\n"
START = json.dumps({"type": "start", "node": "classify"}) + "\n"


def test_normalize_and_path():
    assert normalize("  Top 3\tArtists \n by REVENUE ") == "top 3 artists by revenue"
    a = recording_path("d", "Top 3 artists by revenue")
    assert a == recording_path("d", "  top 3 artists   BY revenue")
    assert a.name.startswith("top-3-artists-by-revenue-") and a.suffix == ".ndjson"
    # Accents are folded in the slug; the hash keeps look-alikes apart.
    assert recording_path("d", "¿Quiénes compraron música de Jazz?").name.startswith("quienes-compraron-musica-de-jazz-")
    assert recording_path("d", "a?") != recording_path("d", "a!")
    assert recording_path("d", "???").name.startswith("question-")


def test_default_questions_match_the_web_ui_examples():
    html = (Path(__file__).parents[1] / "src/t2sql/web/index.html").read_text(encoding="utf-8")
    assert tuple(re.findall(r'<button class="link ex">(.*?)</button>', html)) == DEFAULT_QUESTIONS


def test_record_roundtrip_keeps_exact_lines_and_timing(tmp_path):
    lines = [START, "{\"type\": \"x\", \"s\": \"caf\u00e9 \\n\"}\n", FINAL]
    assert list(record_lines(iter(lines), tmp_path, "Q one")) == lines
    events = load_recording(tmp_path, "q   ONE")
    assert [line for _, line in events] == lines
    assert [t for t, _ in events] == sorted(t for t, _ in events)


def test_incomplete_run_is_not_saved(tmp_path):
    list(record_lines(iter([START]), tmp_path, "q"))
    assert list(tmp_path.iterdir()) == []
    gen = record_lines(iter([START, FINAL]), tmp_path, "q")
    next(gen)
    gen.close()  # client went away mid-run
    assert list(tmp_path.iterdir()) == []


def test_load_ignores_missing_corrupt_and_mismatched_files(tmp_path):
    assert load_recording(tmp_path, "q") is None
    path = recording_path(tmp_path, "q")
    path.write_text("not json\n")
    assert load_recording(tmp_path, "q") is None
    path.write_text(json.dumps({"question": "other", "format": 1}) + "\n" + json.dumps({"t_ms": 0, "line": FINAL}) + "\n")
    assert load_recording(tmp_path, "q") is None  # slug collision with another question
    path.write_text(json.dumps({"question": "q", "format": 1}) + "\n" + json.dumps({"t_ms": 0, "line": START}) + "\n")
    assert load_recording(tmp_path, "q") is None  # no final event


def test_replay_caps_each_gap(monkeypatch):
    sleeps = []
    monkeypatch.setattr("t2sql.replay.time.sleep", sleeps.append)
    events = [(0, START), (200, START), (5200, START), (5200, FINAL)]
    assert list(replay_lines(events, 1.5)) == [START, START, START, FINAL]
    assert sleeps == [0.2, 1.5]  # zero gaps do not sleep
