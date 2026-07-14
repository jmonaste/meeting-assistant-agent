"""Command-line entry point.

    meeting-assistant process transcript.vtt [--context notes.md] [-o out/]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import __version__
from .config import load_settings

app = typer.Typer(add_completion=False, help="Extract everything from a meeting transcript with a local LLM.")
console = Console()


@app.callback()
def _root() -> None:
    """meeting-assistant: multi-pass meeting transcript analysis."""


@app.command()
def process(
    transcript: Annotated[Path, typer.Argument(help="Transcript file (.vtt, .srt, .txt, or any plain-text dialogue).")],
    context: Annotated[Optional[str], typer.Option("--context", "-c", help="Background context: a path to a text file, or the text itself (names, agenda, prior notes).")] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Directory where the report is written.")] = Path("out"),
    base_url: Annotated[Optional[str], typer.Option(help="OpenAI-compatible endpoint base URL (overrides .env).")] = None,
    worker_model: Annotated[Optional[str], typer.Option(help="Model for per-chunk extraction (overrides .env).")] = None,
    lead_model: Annotated[Optional[str], typer.Option(help="Model for planning, synthesis and sweeps (overrides .env).")] = None,
    max_concurrency: Annotated[Optional[int], typer.Option(help="Parallel LLM calls in the map/sweep phases.")] = None,
    max_sweeps: Annotated[Optional[int], typer.Option(help="Extra full-transcript coverage passes after the first (default 3).")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Resume the previous interrupted run of this transcript.")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Ignore cached per-chunk extractions.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show every pipeline event.")] = False,
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
        MEETING_USE_CACHE=False if no_cache else None,
    )

    console.print(
        f"[bold]meeting-assistant v{__version__}[/bold] — "
        f"worker: {settings.resolved_worker_model()}, lead: {settings.resolved_lead_model()}"
    )

    progress = {"round_calls": 0}

    def on_event(node: str, payload: object) -> None:
        if node == "ingest" and isinstance(payload, dict):
            inv = payload.get("inventory")
            if inv is not None:
                console.print(
                    f"[cyan]ingest[/cyan]: {len(inv.segments)} segments, {inv.total_words} words, "
                    f"format={inv.source_format}, {len(inv.chunks)} chunks"
                )
            return
        if node == "extract_chunk":
            progress["round_calls"] += 1
            console.print(f"[cyan]review[/cyan]: {progress['round_calls']} chunk passes done", end="\r")
            return
        if node == "dispatch" and isinstance(payload, dict) and verbose:
            console.print(f"[dim]dispatch[/dim]: {payload.get('item_count', '?')} items so far")
            return
        if verbose:
            console.print(f"[dim]{node}[/dim]")
        elif node in ("plan", "reduce", "gapfill", "compose"):
            console.print(f"[cyan]{node}[/cyan] done")

    try:
        state = run_pipeline(
            transcript_text=transcript_text,
            settings=settings,
            source_file=str(transcript.resolve()),
            context_text=context_text,
            on_event=on_event,
            resume=resume,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    synthesis = state.get("synthesis")
    inv = state.get("inventory")
    stem = _safe_name((synthesis.title if synthesis else "") or (inv.title_hint if inv else "") or transcript.stem) or "Meeting"

    report_md = state.get("report_md", "")
    if report_md:
        md_path = output / f"{stem}-Report.md"
        md_path.write_text(report_md, encoding="utf-8")
        console.print(f"\n[green]Report written:[/green] {md_path}")
    else:
        console.print("[red]No report was produced.[/red]")

    report_json = state.get("report_json", "")
    if report_json:
        json_path = output / f"{stem}-Report.json"
        json_path.write_text(report_json, encoding="utf-8")
        console.print(f"[green]JSON export written:[/green] {json_path}")

    for warning in state.get("warnings", []) or []:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    if not report_md:
        raise typer.Exit(code=1)


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "-", name.strip()).strip("-")


if __name__ == "__main__":
    app()
