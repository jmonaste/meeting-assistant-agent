"""Deterministic transcript parsing: raw text -> ordered utterances.

Transcripts arrive in many shapes and none of them is XML-clean, so format
detection is heuristic and every branch degrades to "plain text" rather than
failing. The output is a list of :class:`_Utt` (speaker, text, start, end); the
loader turns those into :class:`~meeting_assistant.model.transcript.Segment`s.

Supported shapes:

* **WebVTT** (``WEBVTT`` header, ``-->`` cues, ``<v Speaker>`` voice tags)
* **SRT** (numbered cues, ``HH:MM:SS,mmm --> ...``)
* **Labeled dialogue** (``Speaker: text``, optionally timestamped)
* **Timestamped lines** (``[HH:MM:SS] text`` with no speaker)
* **Plain text** (paragraphs, or one undifferentiated block)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TS_TOKEN = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?"
_CUE_RE = re.compile(rf"({_TS_TOKEN})\s*-->\s*({_TS_TOKEN})")
_TAG_RE = re.compile(r"<[^>]+>")
_VOICE_RE = re.compile(r"<v\s+([^>]+)>", re.IGNORECASE)
# Leading timestamp on a line: "[00:12:34]", "(00:12)", "00:12:34 -"
_LEAD_TS_RE = re.compile(rf"^\s*[\[(]?({_TS_TOKEN})[\])]?\s*[-–]?\s*")
# "Speaker:" or "Speaker (00:12):" label at the start of a line.
_SPEAKER_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9][\w .,'&/\-]{0,39}?)\s*(?:\((?P<ts>[^)]*)\))?\s*:\s+(?P<text>\S.*)$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

PLAIN_SEGMENT_CHARS = 500  # cap for a synthetic segment when text has no structure


@dataclass
class _Utt:
    speaker: str = ""
    text: str = ""
    start: str = ""
    end: str = ""


@dataclass
class ParseResult:
    fmt: str
    utterances: list[_Utt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _fmt_ts(raw: str) -> str:
    """Normalize a timestamp token to ``HH:MM:SS`` (drop milliseconds)."""
    raw = raw.strip().replace(",", ".")
    core = raw.split(".", 1)[0]
    parts = core.split(":")
    if len(parts) == 2:
        parts = ["00", *parts]
    try:
        h, m, s = (int(p) for p in parts[:3])
        return f"{h:02d}:{m:02d}:{s:02d}"
    except ValueError:
        return core


def _clean(text: str) -> str:
    return _WS(_TAG_RE.sub("", text)).strip()


def _WS(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text)


def _looks_like_speaker(name: str) -> bool:
    """A conservative test so 'Note: ...' style lines are not read as speakers."""
    name = name.strip()
    if not name or len(name) > 40:
        return False
    if len(name.split()) > 5:
        return False
    # Reject if it ends like a sentence fragment rather than a name/label.
    return bool(re.match(r"^[A-Za-z0-9][\w .,'&/\-]*$", name))


# --------------------------------------------------------------------------- #
# Format-specific parsers
# --------------------------------------------------------------------------- #

def _split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line.rstrip())
    if current:
        blocks.append(current)
    return blocks


def _parse_cue_blocks(text: str) -> list[_Utt]:
    """Shared VTT/SRT cue handling."""
    utts: list[_Utt] = []
    for block in _split_blocks(text):
        lines = list(block)
        if lines and lines[0].strip().upper().startswith("WEBVTT"):
            continue
        if lines and lines[0].strip().upper() in {"NOTE", "STYLE", "REGION"}:
            continue
        # Drop a leading numeric cue index (SRT) or cue identifier (VTT).
        if lines and _CUE_RE.search(lines[0]) is None and lines[0].strip().isdigit():
            lines = lines[1:]
        if not lines:
            continue
        m = _CUE_RE.search(lines[0])
        if m is None:
            continue
        start, end = _fmt_ts(m.group(1)), _fmt_ts(m.group(2))
        payload_lines = lines[1:]
        speaker = ""
        vm = _VOICE_RE.search("\n".join(payload_lines))
        if vm:
            speaker = _clean(vm.group(1))
        payload = _clean(" ".join(payload_lines))
        if not speaker:
            sm = _SPEAKER_RE.match(payload)
            if sm and _looks_like_speaker(sm.group("name")):
                speaker = sm.group("name").strip()
                payload = sm.group("text").strip()
        if payload:
            utts.append(_Utt(speaker=speaker, text=payload, start=start, end=end))
    return utts


def _parse_lines(text: str, expect_speakers: bool) -> list[_Utt]:
    """Line-oriented parsing for labeled / timestamped / plain dialogue."""
    utts: list[_Utt] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        start = ""
        ts = _LEAD_TS_RE.match(line)
        if ts:
            start = _fmt_ts(ts.group(1))
            line = line[ts.end():]
        sm = _SPEAKER_RE.match(line)
        if sm and _looks_like_speaker(sm.group("name")):
            inline_ts = sm.group("ts")
            if inline_ts and not start:
                tsm = re.search(_TS_TOKEN, inline_ts)
                if tsm:
                    start = _fmt_ts(tsm.group(0))
            utts.append(_Utt(speaker=sm.group("name").strip(), text=sm.group("text").strip(), start=start))
        else:
            body = _clean(line)
            if body:
                utts.append(_Utt(speaker="", text=body, start=start))
    return utts


def _parse_plain(text: str) -> list[_Utt]:
    """No speakers, no timestamps: split into paragraph- or sentence-sized units."""
    utts: list[_Utt] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        # One wall of text: break into sentence groups so segment ids are useful.
        blob = _WS(text.replace("\n", " ")).strip()
        sentences = _SENT_SPLIT_RE.split(blob)
        buf = ""
        for sent in sentences:
            if buf and len(buf) + len(sent) + 1 > PLAIN_SEGMENT_CHARS:
                utts.append(_Utt(text=buf.strip()))
                buf = ""
            buf = f"{buf} {sent}".strip()
        if buf.strip():
            utts.append(_Utt(text=buf.strip()))
    else:
        for para in paragraphs:
            utts.append(_Utt(text=_WS(para.replace("\n", " ")).strip()))
    return utts


# --------------------------------------------------------------------------- #
# Detection + entry point
# --------------------------------------------------------------------------- #

def detect_format(text: str) -> str:
    head = text.lstrip()[:64].upper()
    if head.startswith("WEBVTT"):
        return "vtt"
    cue_lines = _CUE_RE.findall(text)
    if cue_lines:
        # SRT uses comma milliseconds and integer index lines; VTT uses dots.
        if re.search(rf"^\s*\d+\s*$\n\s*{_TS_TOKEN}\s*-->", text, re.MULTILINE) and "," in "".join(
            f"{a}{b}" for a, b in cue_lines
        ):
            return "srt"
        return "vtt"

    content_lines = [ln for ln in text.splitlines() if ln.strip()]
    if not content_lines:
        return "plain"
    speaker_hits = sum(
        1 for ln in content_lines
        if (m := _SPEAKER_RE.match(_LEAD_TS_RE.sub("", ln))) and _looks_like_speaker(m.group("name"))
    )
    if speaker_hits >= 3 and speaker_hits >= 0.25 * len(content_lines):
        return "labeled"
    ts_hits = sum(1 for ln in content_lines if _LEAD_TS_RE.match(ln))
    if ts_hits >= 3 and ts_hits >= 0.4 * len(content_lines):
        return "timestamped"
    return "plain"


def parse_transcript(text: str, fmt: str = "") -> ParseResult:
    """Detect the format (unless given) and parse the transcript into utterances."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    fmt = fmt or detect_format(text)
    warnings: list[str] = []

    if fmt in {"vtt", "srt"}:
        utts = _parse_cue_blocks(text)
        if not utts:
            warnings.append(f"no {fmt.upper()} cues parsed; falling back to plain text")
            fmt, utts = "plain", _parse_plain(text)
    elif fmt == "labeled":
        utts = _parse_lines(text, expect_speakers=True)
    elif fmt == "timestamped":
        utts = _parse_lines(text, expect_speakers=False)
    else:
        fmt, utts = "plain", _parse_plain(text)

    utts = [u for u in utts if u.text.strip()]
    if not utts:
        warnings.append("transcript produced no usable content")
    return ParseResult(fmt=fmt, utterances=utts, warnings=warnings)
