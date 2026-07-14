"""CLI wiring: progress events, the --log-file audit trail, and output files.

``run_pipeline`` is replaced with a fake that replays a realistic event
sequence (ingest -> plan -> dispatch/extract rounds -> reduce -> gapfill ->
compose), so the whole console layer — progress display, logging setup,
summary table, file writing — runs for real with no endpoint.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import meeting_assistant.graph.builder as builder
from meeting_assistant.cli import app
from meeting_assistant.graph.merge import consolidate
from meeting_assistant.ingest.loader import build_inventory
from meeting_assistant.model.extraction import ActionItem, ChunkExtraction, MeetingSynthesis
from meeting_assistant.render.document import render_json, render_report

FIXTURE = Path(__file__).parent / "fixtures" / "sample-standup.txt"

runner = CliRunner()


def _fake_run_pipeline(**kw):
    settings = kw["settings"]
    on_event = kw.get("on_event") or (lambda node, payload: None)
    inv = build_inventory(
        kw.get("transcript_text", ""), settings, source_file=kw.get("source_file", "")
    )
    extraction = ChunkExtraction(
        action_items=[ActionItem(task="File infra ticket", owner="Bob", segment_ids=["S2"])]
    )
    items, count = consolidate([extraction])
    synthesis = MeetingSynthesis(
        title="Sprint Standup",
        meeting_type="standup",
        executive_summary="The team planned the release.",
        detailed_summary="Details.",
    )
    state = {
        "inventory": inv,
        "items": items,
        "item_count": count,
        "rounds_done": 2,
        "synthesis": synthesis,
        "signals": {},
        "warnings": ["endpoint hiccup on chunk 1"],
    }

    on_event("ingest", {"inventory": inv})
    on_event("plan", {"plan": None})
    payloads = [{"chunk_index": c.index} for c in inv.chunks]
    on_event("dispatch", {"current_chunks": payloads, "rounds_done": 1, "item_count": 0})
    for _ in payloads:
        on_event("extract_chunk", {"harvest": [extraction]})
    on_event("dispatch", {"current_chunks": [], "item_count": count, "items": items})
    on_event("reduce", {"synthesis": synthesis})
    on_event("gapfill", {"rescued": {"risks": 1}})
    state["report_md"] = render_report(state)
    state["report_json"] = render_json(state)
    on_event("compose", {"report_md": state["report_md"]})
    return state


def test_cli_writes_outputs_and_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "run_pipeline", _fake_run_pipeline)
    transcript = tmp_path / "meeting.txt"
    transcript.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    out_dir = tmp_path / "out"
    log_path = tmp_path / "run.log"

    result = runner.invoke(
        app,
        ["process", str(transcript), "-o", str(out_dir), "--log-file", str(log_path)],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "Sprint-Standup-Report.md").exists()
    assert (out_dir / "Sprint-Standup-Report.json").exists()
    # Console shows ingest stats, the summary table and the sanitized warning.
    assert "ingest" in result.output
    assert "Action items" in result.output
    assert "endpoint hiccup" in result.output
    # The log file carries the run bookends for later review.
    text = log_path.read_text(encoding="utf-8")
    assert "run started" in text
    assert "run finished" in text


def test_cli_runs_without_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "run_pipeline", _fake_run_pipeline)
    transcript = tmp_path / "meeting.txt"
    transcript.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    result = runner.invoke(app, ["process", str(transcript), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert "Report written" in result.output


def test_cli_creates_log_file_parent_directory(tmp_path, monkeypatch):
    """--log-file inside a not-yet-existing directory (e.g. the output dir) must work."""
    monkeypatch.setattr(builder, "run_pipeline", _fake_run_pipeline)
    transcript = tmp_path / "meeting.txt"
    transcript.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = tmp_path / "out" / "logs" / "run.log"  # nothing under out/ exists yet

    result = runner.invoke(
        app, ["process", str(transcript), "-o", str(tmp_path / "out"), "--log-file", str(log_path)]
    )

    assert result.exit_code == 0, result.output
    assert log_path.exists()
    assert "run started" in log_path.read_text(encoding="utf-8")


def test_cli_unopenable_log_file_fails_cleanly(tmp_path, monkeypatch):
    """A log path that cannot be opened exits 2 with a message, not a traceback."""
    monkeypatch.setattr(builder, "run_pipeline", _fake_run_pipeline)
    transcript = tmp_path / "meeting.txt"
    transcript.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")

    result = runner.invoke(
        app,
        ["process", str(transcript), "--log-file", str(blocker / "run.log")],
    )

    assert result.exit_code == 2
    # A clean typer.Exit surfaces as SystemExit through the runner — anything
    # else (e.g. FileNotFoundError) would be the raw-traceback bug.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Could not open log file" in result.output


# --------------------------------------------------------------------------- #
# _ProgressView unit tests (driven directly, no CLI)
# --------------------------------------------------------------------------- #

import io

from rich.console import Console
from rich.progress import Progress

from meeting_assistant.cli import _ProgressView


def _view(max_rounds: int = 4) -> _ProgressView:
    out = Console(file=io.StringIO(), width=120)
    return _ProgressView(Progress(console=out), out, max_rounds)


def test_progress_view_pass_deltas():
    view = _view()
    view.on_event("dispatch", {"current_chunks": [{}, {}], "rounds_done": 1, "item_count": 0})
    view.on_event("extract_chunk", {})
    view.on_event("extract_chunk", {})
    view.on_event("dispatch", {"current_chunks": [{}, {}], "rounds_done": 2, "item_count": 10})
    assert view.last_count == 10
    assert "(+10)" in view.progress.tasks[0].description  # pass 1 closed with its delta
    view.on_event("extract_chunk", {})
    view.on_event("extract_chunk", {})
    view.on_event("dispatch", {"current_chunks": [], "item_count": 12})
    assert "(+2)" in view.progress.tasks[1].description


def test_progress_view_resumed_run_baseline():
    """A resumed run's first event can be a mid-loop dispatch: the item count it
    carries must become the delta baseline, not 0."""
    view = _view()
    view.on_event("dispatch", {"current_chunks": [{}], "rounds_done": 2, "item_count": 7})
    assert view.last_count == 7
    view.on_event("extract_chunk", {})
    view.on_event("dispatch", {"current_chunks": [], "item_count": 7})
    assert "(+0)" in view.progress.tasks[0].description


def test_progress_view_extract_before_any_dispatch():
    """Resumed mid-round: extract_chunk events with no open bar get a fallback task."""
    view = _view()
    view.on_event("extract_chunk", {})
    view.on_event("extract_chunk", {})
    assert view.round_task is not None
    view.on_event("dispatch", {"current_chunks": [], "item_count": 5})
    assert view.round_task is None  # closed cleanly despite the unknown total
