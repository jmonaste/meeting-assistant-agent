"""Deterministic signal scan over the transcript.

These cheap, explainable regex passes run without any LLM. They do two jobs:

* they seed the coverage check — if the scan finds 12 date mentions but the
  model only captured 3, that category clearly deserves another re-read;
* they back the "signal coverage" appendix, so a reader can see the raw cues the
  analysis was working from.

A signal hit is never trusted as a finished extraction (the phrasing is fuzzy);
it is a pointer that says "there is probably something here, look again".
"""

from __future__ import annotations

import re

from .model.transcript import Segment, TranscriptInventory

_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_WEEKDAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"

_PATTERNS: dict[str, re.Pattern[str]] = {
    "action_items": re.compile(
        r"\b(action item|to-?do|follow[- ]?up|i'?ll |we'?ll |we need to|we have to|"
        r"let'?s |assign(?:ed)? to|you (?:should|need to|will)|please |take care of|"
        r"will (?:handle|own|take|do|send|prepare|update|create|review))\b",
        re.IGNORECASE,
    ),
    "decisions": re.compile(
        r"\b(we (?:decided|agreed|concluded)|(?:it'?s |we'?re )?agreed|"
        r"decision is|let'?s go with|we'?ll go with|the plan is|we choose|"
        r"sign(?:ed)? off|approved|final decision)\b",
        re.IGNORECASE,
    ),
    "open_questions": re.compile(r"\?\s*$"),
    "key_dates": re.compile(
        rf"\b(?:{_MONTHS})\b|\b(?:{_WEEKDAYS})\b|\b(?:today|tomorrow|yesterday|"
        r"next (?:week|month|quarter|sprint|year)|end of (?:the )?(?:day|week|month|quarter)|"
        r"eod|eow|q[1-4]|by (?:the )?\w+|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
        r"\d{4}-\d{2}-\d{2}|deadline|due date)\b",
        re.IGNORECASE,
    ),
    "key_facts": re.compile(r"(\$\s?\d|\d+\s?%|\b\d{3,}\b|\b\d+(?:\.\d+)?\s?(?:k|m|bn|million|billion|users|hours|days|weeks)\b)", re.IGNORECASE),
    "risks": re.compile(
        r"\b(risk|blocker|blocked|concern|worried|issue|problem|depend(?:s|ency|encies)?|"
        r"bottleneck|delay(?:ed)?|at risk|fall(?:s|ing)? behind|technical debt)\b",
        re.IGNORECASE,
    ),
}

MAX_HITS_PER_CATEGORY = 40
SNIPPET_CHARS = 140


def _snippet(seg: Segment) -> str:
    who = f"{seg.speaker}: " if seg.speaker else ""
    text = f"{who}{seg.text}"
    return f"{seg.sid}: {text[:SNIPPET_CHARS]}"


def scan_signals(inv: TranscriptInventory) -> dict[str, list[str]]:
    """Return, per category, a bounded list of ``'Sxx: snippet'`` cue matches."""
    hits: dict[str, list[str]] = {cat: [] for cat in _PATTERNS}
    for seg in inv.segments:
        text = seg.text
        for cat, pat in _PATTERNS.items():
            if len(hits[cat]) >= MAX_HITS_PER_CATEGORY:
                continue
            if pat.search(text):
                hits[cat].append(_snippet(seg))
    return {cat: h for cat, h in hits.items() if h}
