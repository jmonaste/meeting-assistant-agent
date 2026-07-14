"""Intermediate representation (IR) of a parsed meeting transcript.

Raw transcripts come in many shapes (WebVTT, SRT, ``Name: text`` dialogue,
timestamped lines, or one undifferentiated wall of text). The ingestion layer
normalizes every shape *deterministically* — before a single LLM token is spent
— into an ordered list of :class:`Segment` (speaker turns / utterances) and a
:class:`TranscriptInventory`. Everything the LLM sees downstream is rendered
from these models, and every extracted item can point back to a segment id, so
citations are grounded in the actual text rather than trusted from the model.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """One utterance: a contiguous span of speech attributed to a speaker.

    ``sid`` is a stable, human-readable id (``S0``, ``S1``, ...) used both in the
    text handed to the model and as the citation key extractions point back to.
    """

    sid: str
    index: int
    speaker: str = ""  # normalized speaker/display name, "" when unknown
    text: str = ""
    start: str = ""  # timestamp as it appeared (e.g. "00:12:34"), "" if none
    end: str = ""

    def render(self, with_timestamp: bool = True) -> str:
        """Render this segment as one line of transcript for the model."""
        prefix = f"[{self.sid}]"
        if with_timestamp and self.start:
            prefix += f" {self.start}"
        who = f" {self.speaker}:" if self.speaker else ""
        return f"{prefix}{who} {self.text}".rstrip()


class Speaker(BaseModel):
    """A participant, aggregated from the segments they spoke."""

    name: str
    turns: int = 0
    words: int = 0
    first_sid: str = ""
    aliases: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A window over consecutive segments handed to the worker model.

    Chunks overlap by a configurable number of characters so an item that
    straddles a boundary is still seen whole by at least one chunk. ``text`` is
    the already-rendered block; ``content_hash`` keys the extraction cache.
    """

    index: int
    first_sid: str
    last_sid: str
    text: str
    content_hash: str = ""

    def label(self) -> str:
        return f"chunk {self.index} [{self.first_sid}..{self.last_sid}]"


def hash_text(text: str) -> str:
    """Stable content hash of a text block (keys the extraction cache)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class TranscriptInventory(BaseModel):
    """Everything the pipeline knows about the transcript after ingestion."""

    root: str = ""  # directory of the source file (holds the cache)
    source_file: str = ""
    source_format: str = "plain"  # vtt | srt | labeled | timestamped | plain
    title_hint: str = ""  # filename-derived title, refined later by the model

    segments: list[Segment] = Field(default_factory=list)
    speakers: list[Speaker] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)

    context_text: str = ""  # optional user-provided background (may be empty)

    has_speakers: bool = False
    has_timestamps: bool = False
    duration: str = ""  # "HH:MM:SS .. HH:MM:SS" when timestamps exist
    total_chars: int = 0
    total_words: int = 0
    warnings: list[str] = Field(default_factory=list)

    # -- lookup helpers ----------------------------------------------------- #

    def by_sid(self) -> dict[str, Segment]:
        return {s.sid: s for s in self.segments}

    def full_text(self) -> str:
        """The whole transcript rendered as one block (with segment ids)."""
        return "\n".join(s.render(self.has_timestamps) for s in self.segments)

    def window(self, first_sid: str = "", last_sid: str = "") -> str:
        """Render an inclusive segment range as text (for the read tools)."""
        started = not first_sid
        lines: list[str] = []
        for s in self.segments:
            if not started and s.sid == first_sid:
                started = True
            if started:
                lines.append(s.render(self.has_timestamps))
            if last_sid and s.sid == last_sid:
                break
        return "\n".join(lines)

    # -- context rendering -------------------------------------------------- #

    def census(self) -> str:
        """Compact overview used by the plan node."""
        lines = [
            f"Source: {self.source_file or '(unknown)'} (format: {self.source_format})",
            f"Segments: {len(self.segments)} | words: {self.total_words} | chars: {self.total_chars}",
            f"Speakers detected: {self.has_speakers} | timestamps: {self.has_timestamps}",
        ]
        if self.duration:
            lines.append(f"Time span: {self.duration}")
        if self.speakers:
            who = ", ".join(
                f"{sp.name} ({sp.turns} turns, {sp.words} words)" for sp in self.speakers[:20]
            )
            lines.append(f"Participants: {who}")
        lines.append(f"Chunks to review: {len(self.chunks)}")
        return "\n".join(lines)

    def speaker_context(self) -> str:
        if not self.speakers:
            return "(no distinct speakers were detected in the transcript)"
        return "\n".join(
            f"  - {sp.name}: {sp.turns} turns, {sp.words} words"
            + (f" (aliases: {', '.join(sp.aliases)})" if sp.aliases else "")
            for sp in self.speakers
        )

    def context_block(self) -> str:
        """The optional user-provided background, ready to drop into a prompt."""
        if not self.context_text.strip():
            return ""
        return (
            "BACKGROUND CONTEXT PROVIDED BY THE USER (use it to resolve names, "
            "acronyms and references; do not treat it as spoken in the meeting):\n"
            f"{self.context_text.strip()}"
        )
