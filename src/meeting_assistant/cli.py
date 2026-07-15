"""Command-line entry point.

    meeting-assistant process transcript.vtt [--context notes.md] [-o out/]

Console output strategy: the live progress display (one bar per review pass,
spinner rows for the plan/synthesis/gapfill/compose phases) is driven by the
pipeline's node events; textual detail goes through ``logging`` — warnings
(including endpoint retries) always reach the console, ``--verbose`` mirrors
info-level events too, and ``--log-file`` appends the full DEBUG audit trail
(every LLM call, retry, cache hit and pass decision) for later review.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__
from .config import load_settings
from .model.extraction import CATEGORIES

app = typer.Typer(add_completion=False, help="Extract everything from a meeting transcript with a local LLM.")
console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def _root() -> None:
    """meeting-assistant: multi-pass meeting transcript analysis."""


def _setup_logging(log_path: str, verbose: bool) -> None:
    """Route agent logs.

    The ``meeting_assistant`` logger tree gets: a DEBUG file handler when a log
    file is configured (the full audit trail), and a console handler that shows
    warnings live — endpoint retries, chunk failures — or info too with -v.
    """
    root = logging.getLogger("meeting_assistant")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    if log_path:
        target = Path(log_path)
        try:
            if target.parent and str(target.parent) not in ("", "."):
                target.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(target, encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Could not open log file {target}:[/red] {exc}")
            raise typer.Exit(code=2)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
    console_handler = RichHandler(
        console=console, show_time=False, show_path=False, rich_tracebacks=False
    )
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    root.addHandler(console_handler)


class _ProgressView:
    """Translates pipeline node events into the live progress display.

    One bar per review pass (annotated with the running item count and the
    delta the pass added), spinner rows for the plan/synthesis/gapfill/compose
    phases. Tolerates resumed runs, where the event stream can start mid-loop.
    """

    def __init__(self, progress: Progress, out: Console, max_rounds: int) -> None:
        self.progress = progress
        self.console = out
        self.max_rounds = max_rounds
        self.phase_task = None
        self.round_task = None
        self.round_desc = ""
        self.round_total: int | None = None
        self.last_count = 0
        self.chunks = 0

    def _finish_phase(self) -> None:
        if self.phase_task is not None:
            self.progress.update(self.phase_task, completed=1)
            self.phase_task = None

    def _start_phase(self, description: str) -> None:
        self._finish_phase()
        self.phase_task = self.progress.add_task(description, total=1)

    def _finish_round(self, item_count: object) -> None:
        # Record the baseline even when no bar is open (a resumed run's first
        # event can be a mid-loop dispatch), so later deltas stay truthful.
        delta_note = ""
        if isinstance(item_count, int):
            delta = item_count - self.last_count
            self.last_count = item_count
            delta_note = f"  [dim]{item_count} items (+{delta})[/dim]"
        if self.round_task is None:
            return
        total = self.round_total
        if total is None:  # indeterminate fallback task: close it where it stands
            task = next(t for t in self.progress.tasks if t.id == self.round_task)
            total = task.completed or 1
        self.progress.update(
            self.round_task, total=total, completed=total,
            description=self.round_desc + delta_note,
        )
        self.round_task = None

    def on_event(self, node: str, payload: object) -> None:
        p = payload if isinstance(payload, dict) else {}
        if node == "ingest":
            inv = p.get("inventory")
            if inv is not None:
                self.chunks = len(inv.chunks)
                self.console.print(
                    f"[cyan]ingest[/cyan]: {len(inv.segments)} segments, {inv.total_words} words, "
                    f"format={inv.source_format}, {len(inv.chunks)} chunks"
                )
            self._start_phase("Planning the analysis")
            return
        if node == "plan":
            self._finish_phase()
            return
        if node == "dispatch":
            chunks = p.get("current_chunks")
            self._finish_round(p.get("item_count"))
            if chunks:
                launched = p.get("rounds_done", 1)  # rounds launched, incl. this one
                kind = "initial extraction" if launched == 1 else "sweep for missed items"
                self.round_desc = f"Pass {launched}/{self.max_rounds}: {kind}"
                self.round_total = len(chunks)
                self.round_task = self.progress.add_task(self.round_desc, total=len(chunks))
            else:
                self._start_phase("Writing the report prose")
            return
        if node == "extract_chunk":
            if self.round_task is None:  # resumed run: the dispatch event was checkpointed away
                self.round_desc = "Reviewing transcript chunks"
                self.round_total = self.chunks or None
                self.round_task = self.progress.add_task(self.round_desc, total=self.round_total)
            self.progress.advance(self.round_task)
            return
        if node == "reduce":
            self._start_phase("Gap-fill: re-checking weak categories")
            return
        if node == "gapfill":
            rescued = p.get("rescued") or {}
            if rescued:
                self.console.print(
                    "[cyan]gapfill[/cyan] rescued: "
                    + ", ".join(f"{cat} +{n}" for cat, n in rescued.items())
                )
            self._start_phase("Composing the report")
            return
        if node == "compose":
            self._finish_phase()


@app.command()
def process(
    transcript: Annotated[Path, typer.Argument(help="Transcript file (.vtt, .srt, .txt, or any plain-text dialogue).")],
    context: Annotated[Optional[str], typer.Option("--context", "-c", help="Background context: a path to a text file, or the text itself (names, agenda, prior notes).")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Directory where the report is written.")] = Path("out"),
    log_file: Annotated[Optional[Path], typer.Option("--log-file", help="Append the detailed run log (every LLM call, retry, cache hit and pass decision) to this file for later review.")] = None,
    language: Annotated[Optional[str], typer.Option("--language", "-l", help="Output language for the report: 'auto' (follow the transcript), a code like 'es'/'en', or any language name. Quotes are never translated.")] = None,
    base_url: Annotated[Optional[str], typer.Option(help="OpenAI-compatible endpoint base URL (overrides .env).")] = None,
    worker_model: Annotated[Optional[str], typer.Option(help="Model for per-chunk extraction (overrides .env).")] = None,
    lead_model: Annotated[Optional[str], typer.Option(help="Model for planning, synthesis and sweeps (overrides .env).")] = None,
    max_concurrency: Annotated[Optional[int], typer.Option(help="Parallel LLM calls in the map/sweep phases. Lower to 1-2 if the endpoint returns 429/504.")] = None,
    max_sweeps: Annotated[Optional[int], typer.Option(help="Extra full-transcript coverage passes after the first (default 3).")] = None,
    request_timeout: Annotated[Optional[float], typer.Option(help="HTTP timeout in seconds per LLM call (default 300; large local models are slow).")] = None,
    chunk_chars: Annotated[Optional[int], typer.Option(help="Character budget per transcript chunk. Lower it if the endpoint times out on full chunks.")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Resume the previous interrupted run of this transcript.")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Ignore cached per-chunk extractions.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Also mirror info-level log events to the console.")] = False,
) -> None:
    """Analyze a meeting transcript and write a Markdown report and JSON export."""
    from .graph.builder import run_pipeline

    if not transcript.is_file():
        console.print(f"[red]Not a file:[/red] {transcript}")
        raise typer.Exit(code=2)

    try:
        transcript_text = transcript.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        console.print(f"[red]Could not read transcript:[/red] {exc}")
        raise typer.Exit(code=2)

    context_text = ""
    if context:
        candidate = Path(context)
        if candidate.is_file():
            context_text = candidate.read_text(encoding="utf-8", errors="replace")
        else:
            context_text = context

    settings = load_settings(
        openai_base_url=base_url,
        MEETING_WORKER_MODEL=worker_model,
        MEETING_LEAD_MODEL=lead_model,
        MEETING_MAX_CONCURRENCY=max_concurrency,
        MEETING_MAX_SWEEPS=max_sweeps,
        MEETING_REQUEST_TIMEOUT=request_timeout,
        MEETING_CHUNK_MAX_CHARS=chunk_chars,
        MEETING_LOG_FILE=str(log_file) if log_file else None,
        MEETING_LANGUAGE=language,
        MEETING_USE_CACHE=False if no_cache else None,
    )
    _setup_logging(settings.log_file, verbose)

    console.print(
        f"[bold]meeting-assistant v{__version__}[/bold] — "
        f"worker: {settings.resolved_worker_model()}, lead: {settings.resolved_lead_model()}"
    )
    if settings.log_file:
        console.print(f"[dim]Detailed log: {settings.log_file}[/dim]")
    logger.info(
        "run started: transcript=%s (%d chars), output=%s, worker=%s, lead=%s, "
        "concurrency=%d, sweeps=%d, timeout=%.0fs, retries=%d, resume=%s",
        transcript, len(transcript_text), output,
        settings.resolved_worker_model(), settings.resolved_lead_model(),
        settings.max_concurrency, settings.max_sweeps,
        settings.request_timeout, settings.llm_retries, resume,
    )

    progress = Progress(
        SpinnerColumn(finished_text="[green]ok[/green]"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    view = _ProgressView(progress, console, max_rounds=1 + max(0, settings.max_sweeps))

    try:
        with progress:
            state = run_pipeline(
                transcript_text=transcript_text,
                settings=settings,
                source_file=str(transcript.resolve()),
                context_text=context_text,
                on_event=view.on_event,
                resume=resume,
            )
    except ValueError as exc:
        logger.error("run failed: %s", exc)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    # -------------------------------------------------------------- outputs
    output.mkdir(parents=True, exist_ok=True)
    synthesis = state.get("synthesis")
    inv = state.get("inventory")
    stem = _safe_name((synthesis.title if synthesis else "") or (inv.title_hint if inv else "") or transcript.stem) or "Meeting"

    items = state.get("items", {}) or {}
    if any(items.get(field) for field, _l, _n in CATEGORIES):
        table = Table(show_header=True, header_style="bold")
        table.add_column("Category")
        table.add_column("Items", justify="right")
        for field, label, _noun in CATEGORIES:
            table.add_row(label, str(len(items.get(field) or [])))
        console.print(table)
    console.print(
        f"Review passes: {state.get('rounds_done', 0)} — items extracted: {state.get('item_count', 0)}"
    )

    report_md = state.get("report_md", "")
    if report_md:
        md_path = output / f"{stem}-Report.md"
        md_path.write_text(report_md, encoding="utf-8")
        console.print(f"[green]Report written:[/green] {md_path}")
    else:
        console.print("[red]No report was produced.[/red]")

    report_json = state.get("report_json", "")
    if report_json:
        json_path = output / f"{stem}-Report.json"
        json_path.write_text(report_json, encoding="utf-8")
        console.print(f"[green]JSON export written:[/green] {json_path}")

    warnings = state.get("warnings", []) or []
    for warning in warnings:
        # Endpoint errors can carry whole HTML pages; keep warnings to one line.
        clean = re.sub(r"\s+", " ", str(warning)).strip()[:300]
        console.print(f"[yellow]warning:[/yellow] {clean}")

    logger.info(
        "run finished: %d pass(es), %d items, %d warning(s), report=%s",
        state.get("rounds_done", 0), state.get("item_count", 0),
        len(warnings), bool(report_md),
    )
    if settings.log_file:
        console.print(f"[dim]Detailed log: {settings.log_file}[/dim]")

    if not report_md:
        raise typer.Exit(code=1)


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "-", name.strip()).strip("-")


if __name__ == "__main__":
    app()
