"""Turn a transcript file (and optional context) into a TranscriptInventory.

This is the single deterministic ingestion entry point: read the file, parse it
into utterances, merge fragmented same-speaker turns, assign stable segment ids,
aggregate speakers, and chunk the result. No LLM is involved.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..model.transcript import Segment, Speaker, TranscriptInventory
from .chunker import build_chunks
from .parser import parse_transcript

MERGE_MAX_CHARS = 1_200  # cap when merging consecutive same-speaker utterances


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _merge_turns(utts: list) -> list:
    """Merge consecutive utterances by the same non-empty speaker into one turn.

    Cue formats (VTT/SRT) fragment a single spoken turn across many short cues;
    merging them yields cleaner, more meaningful segments. Bounded so a
    monologue does not become one enormous segment.
    """
    merged: list = []
    for u in utts:
        if (
            merged
            and u.speaker
            and merged[-1].speaker == u.speaker
            and len(merged[-1].text) + len(u.text) + 1 <= MERGE_MAX_CHARS
        ):
            merged[-1].text = f"{merged[-1].text} {u.text}".strip()
            merged[-1].end = u.end or merged[-1].end
        else:
            merged.append(u)
    return merged


def build_inventory(
    transcript_text: str,
    settings: Settings,
    source_file: str = "",
    context_text: str = "",
    fmt: str = "",
) -> TranscriptInventory:
    """Parse and assemble the full inventory from raw transcript text."""
    parsed = parse_transcript(transcript_text, fmt=fmt)
    utts = _merge_turns(parsed.utterances)

    segments: list[Segment] = []
    for i, u in enumerate(utts):
        segments.append(
            Segment(sid=f"S{i}", index=i, speaker=u.speaker, text=u.text, start=u.start, end=u.end)
        )

    has_speakers = any(s.speaker for s in segments)
    has_timestamps = any(s.start for s in segments)

    # Aggregate speakers in first-appearance order.
    order: list[str] = []
    agg: dict[str, Speaker] = {}
    for s in segments:
        if not s.speaker:
            continue
        sp = agg.get(s.speaker)
        if sp is None:
            sp = Speaker(name=s.speaker, first_sid=s.sid)
            agg[s.speaker] = sp
            order.append(s.speaker)
        sp.turns += 1
        sp.words += len(s.text.split())
    speakers = [agg[name] for name in order]

    starts = [s.start for s in segments if s.start]
    duration = f"{starts[0]} .. {starts[-1]}" if starts else ""

    total_chars = sum(len(s.text) for s in segments)
    total_words = sum(len(s.text.split()) for s in segments)

    chunks = build_chunks(
        segments,
        with_timestamp=has_timestamps,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )

    root = str(Path(source_file).resolve().parent) if source_file else ""
    title_hint = Path(source_file).stem.replace("_", " ").replace("-", " ").strip() if source_file else ""

    warnings = list(parsed.warnings)
    if not segments:
        warnings.append("no segments were produced from the transcript")

    return TranscriptInventory(
        root=root,
        source_file=source_file,
        source_format=parsed.fmt,
        title_hint=title_hint,
        segments=segments,
        speakers=speakers,
        chunks=chunks,
        context_text=context_text,
        has_speakers=has_speakers,
        has_timestamps=has_timestamps,
        duration=duration,
        total_chars=total_chars,
        total_words=total_words,
        warnings=warnings,
    )
