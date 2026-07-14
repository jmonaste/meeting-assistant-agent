"""Read-only, bounded transcript tools for the evidence-gathering agents.

Everything is sandboxed to the parsed inventory, output is capped so a long
transcript cannot overflow a limited context window, and reads are recorded so
evidence citations (segment ids) are collected deterministically rather than
trusted from the model.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

from .model.transcript import TranscriptInventory

SEARCH_MAX = 30
WINDOW_MAX_SEGMENTS = 40


def build_transcript_tools(
    inv: TranscriptInventory,
    recorder: list[str] | None = None,
) -> list:
    """Create the tool set, closed over the inventory.

    ``recorder`` (when given) accumulates the segment ids actually read, so
    callers can cite evidence deterministically.
    """
    by_sid = inv.by_sid()
    with_ts = inv.has_timestamps

    def _record(sid: str) -> None:
        if recorder is not None and sid and sid not in recorder:
            recorder.append(sid)

    @tool
    def search_transcript(query: str, regex: bool = False) -> str:
        """Search the transcript for a word or phrase (case-insensitive).
        Returns up to 30 matching segments as ``Sid speaker: text``. Use this to
        find where something was discussed before reading around it."""
        try:
            pat = re.compile(query if regex else re.escape(query), re.IGNORECASE)
        except re.error as exc:
            return f"ERROR: invalid regex: {exc}"
        hits: list[str] = []
        for seg in inv.segments:
            if pat.search(seg.text) or (seg.speaker and pat.search(seg.speaker)):
                _record(seg.sid)
                hits.append(seg.render(with_ts))
                if len(hits) >= SEARCH_MAX:
                    break
        if not hits:
            return f"(no matches for {query!r})"
        footer = f"\n(capped at {SEARCH_MAX} matches; narrow the query)" if len(hits) >= SEARCH_MAX else ""
        return "\n".join(hits) + footer

    @tool
    def read_segments(first_sid: str, last_sid: str = "") -> str:
        """Read an inclusive range of transcript segments by id (e.g. first_sid='S10',
        last_sid='S18'). Omit last_sid to read a single segment. Capped at 40 segments."""
        first = first_sid.strip()
        last = (last_sid or first_sid).strip()
        if first not in by_sid:
            return f"ERROR: unknown segment id {first!r}"
        idx_first = by_sid[first].index
        idx_last = by_sid[last].index if last in by_sid else idx_first
        if idx_last < idx_first:
            idx_first, idx_last = idx_last, idx_first
        idx_last = min(idx_last, idx_first + WINDOW_MAX_SEGMENTS - 1)
        lines = []
        for seg in inv.segments[idx_first : idx_last + 1]:
            _record(seg.sid)
            lines.append(seg.render(with_ts))
        return "\n".join(lines)

    @tool
    def read_around(sid: str, context: int = 3) -> str:
        """Read a segment plus a few segments of context before and after it.
        Useful to understand who said something and what prompted it."""
        target = by_sid.get(sid.strip())
        if target is None:
            return f"ERROR: unknown segment id {sid!r}"
        lo = max(0, target.index - max(0, context))
        hi = min(len(inv.segments) - 1, target.index + max(0, context))
        lines = []
        for seg in inv.segments[lo : hi + 1]:
            _record(seg.sid)
            marker = "  <-- here" if seg.sid == target.sid else ""
            lines.append(seg.render(with_ts) + marker)
        return "\n".join(lines)

    @tool
    def list_speakers() -> str:
        """List the participants detected in the transcript with turn/word counts."""
        return inv.speaker_context()

    @tool
    def read_background() -> str:
        """Read the optional background context the user provided about this meeting."""
        return inv.context_block() or "(no background context was provided)"

    return [search_transcript, read_segments, read_around, list_speakers, read_background]
