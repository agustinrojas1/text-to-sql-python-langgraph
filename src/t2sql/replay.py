"""Record and replay of `/ask/stream` runs, so a demo works without network or LLM.

A recording is `<dir>/<slug>-<hash>-<variant>.ndjson`, one file per normalised question and
classifier variant ("llm", "local-tev1", "cascade-tev1"...), overwritten on every new recording.
Its first line is a header naming the normalised question and the variant (so a slug collision
or a different mode can never replay the wrong run); every other line wraps one event line
exactly as it was sent to the client:

    {"question": "top 3 artists by revenue", "format": 1, "classifier": "cascade", "local_model": "tev1"}
    {"t_ms": 12, "line": "{\\"type\\": \\"start\\", ...}\\n"}

`t_ms` is the time since the run began. Replay sends each `line` unchanged, sleeping the
recorded gap between lines (capped), so the browser shows the same board at a similar pace.
A run is saved only if it ends with a `final` event: a failed run never replaces a good one.

Backward compatibility: recordings made before the selector existed are `<slug>-<hash>.ndjson`
with no classifier in the header. They are replayed only for requests whose variant equals the
server's default one (T2SQL_CLASSIFIER / T2SQL_LOCAL_CLASSIFIER_MODEL), as before.
"""

import hashlib
import json
import logging
import re
import time
import unicodedata
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger("t2sql.replay")

FORMAT = 1

# The example questions of the web UI (test_replay checks they match index.html).
DEFAULT_QUESTIONS: tuple[str, ...] = (
    "How many customers are there per country?",
    "¿Quiénes compraron música de Jazz?",
    "Top 3 artists by revenue",
    "Revenue per year, with the change vs the previous year",
    "What is the weather in Madrid today?",
)


def normalize(question: str) -> str:
    """Lowercase, trimmed, whitespace collapsed: what makes two questions 'the same'."""
    return " ".join(question.lower().split())


def variant(classifier: str, local_model: str) -> str:
    """The key of a classifier choice: the model only matters when a local model is involved."""
    return "llm" if classifier == "llm" else f"{classifier}-{local_model}"


def recording_path(directory: str | Path, question: str, var: str | None = None) -> Path:
    """Where the recording of `question` lives; `var` None is the legacy (variant-less) file."""
    norm = normalize(question)
    ascii_text = unicodedata.normalize("NFKD", norm).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")[:60].strip("-") or "question"
    digest = hashlib.sha1(norm.encode()).hexdigest()[:8]  # keeps similar questions apart
    return Path(directory) / f"{slug}-{digest}{'-' + var if var else ''}.ndjson"


def _final_ok(last_line: str | None) -> bool:
    try:
        return last_line is not None and json.loads(last_line).get("type") == "final"
    except ValueError:
        return False


def record_lines(
    lines: Iterator[str], directory: str | Path, question: str, classifier: str = "llm", local_model: str = "tev1"
) -> Iterator[str]:
    """Pass `lines` through unchanged while writing them to the question's recording.

    Written to a `.part` file and renamed when the run ends with a `final` event. An
    I/O problem is logged and recording stops; it never breaks the stream.
    """
    path = recording_path(directory, question, variant(classifier, local_model))
    part = path.with_suffix(".ndjson.part")
    out = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = part.open("w", encoding="utf-8")
        out.write(json.dumps(
            {"question": normalize(question), "format": FORMAT, "classifier": classifier, "local_model": local_model},
            ensure_ascii=False,
        ) + "\n")
    except OSError as e:
        log.warning("Cannot record to %s: %s", part, e)
        out = None

    start = time.monotonic()
    last: str | None = None
    try:
        for line in lines:
            last = line
            if out is not None:
                try:
                    t_ms = round((time.monotonic() - start) * 1000)
                    out.write(json.dumps({"t_ms": t_ms, "line": line}, ensure_ascii=False) + "\n")
                    out.flush()
                except OSError as e:
                    log.warning("Recording to %s stopped: %s", part, e)
                    out.close()
                    out = None
            yield line
    finally:
        if out is not None:
            out.close()
            if _final_ok(last):
                part.replace(path)
                log.info("Recorded %r to %s", question, path)
            else:
                part.unlink(missing_ok=True)  # incomplete or failed run: keep any older recording


def load_recording(
    directory: str | Path,
    question: str,
    classifier: str = "llm",
    local_model: str = "tev1",
    default: tuple[str, str] | None = None,
) -> list[tuple[int, str]] | None:
    """The `(t_ms, line)` pairs recorded for `question` in this classifier variant, or None.

    `default` is the server's own (classifier, local_model); a legacy recording (no classifier in
    its header) is used only when the request resolves to the same variant as `default`.
    """
    var = variant(classifier, local_model)
    candidates = [(recording_path(directory, question, var), var)]
    if default is not None and variant(*default) == var:
        candidates.append((recording_path(directory, question), None))
    for path, expected in candidates:
        if path.is_file() and (events := _read(path, question, expected)):
            return events
    return None


def _read(path: Path, question: str, var: str | None) -> list[tuple[int, str]] | None:
    try:
        with path.open(encoding="utf-8") as f:
            header = json.loads(f.readline())
            if header.get("format") != FORMAT or header.get("question") != normalize(question):
                return None
            if var is not None and variant(header.get("classifier", ""), header.get("local_model", "")) != var:
                return None
            events = []
            for raw in f:
                rec = json.loads(raw)
                events.append((int(rec["t_ms"]), str(rec["line"])))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        log.warning("Ignoring unreadable recording %s: %s", path, e)
        return None
    if not events or not _final_ok(events[-1][1]):
        return None
    return events


def replay_lines(events: list[tuple[int, str]], max_gap_s: float) -> Iterator[str]:
    """The recorded lines, unchanged, each after its recorded gap (at most `max_gap_s`)."""
    previous = 0
    for t_ms, line in events:
        gap = min(max(t_ms - previous, 0) / 1000, max_gap_s)
        previous = t_ms
        if gap > 0:
            time.sleep(gap)
        yield line
