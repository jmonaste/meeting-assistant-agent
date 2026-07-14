"""Transcript parsing and format detection."""

from __future__ import annotations

from meeting_assistant.ingest.parser import detect_format, parse_transcript

VTT = """WEBVTT

00:00:01.000 --> 00:00:04.000
<v Alice>Good morning, let's begin.</v>

00:00:04.000 --> 00:00:07.000
<v Bob>Morning. I have an update.</v>
"""

SRT = """1
00:00:01,000 --> 00:00:04,000
Alice: Good morning, let's begin.

2
00:00:04,000 --> 00:00:07,000
Bob: Morning. I have an update.
"""

LABELED = """[00:00:01] Alice: Good morning.
[00:00:04] Bob: Morning, I have an update on the blocker.
Alice: Thanks Bob.
"""

PLAIN = """The team met to discuss the release. They agreed to ship in August.
QA needs another week for regression testing before sign-off.
"""


def test_detect_vtt():
    assert detect_format(VTT) == "vtt"


def test_detect_srt():
    assert detect_format(SRT) == "srt"


def test_detect_labeled():
    assert detect_format(LABELED) == "labeled"


def test_detect_plain():
    assert detect_format(PLAIN) == "plain"


def test_parse_vtt_speakers_and_timestamps():
    res = parse_transcript(VTT)
    assert res.fmt == "vtt"
    assert [u.speaker for u in res.utterances] == ["Alice", "Bob"]
    assert res.utterances[0].start == "00:00:01"
    assert "Good morning" in res.utterances[0].text


def test_parse_srt_inline_speaker_labels():
    res = parse_transcript(SRT)
    assert res.fmt == "srt"
    assert res.utterances[0].speaker == "Alice"
    assert res.utterances[1].speaker == "Bob"


def test_parse_labeled_with_leading_timestamp():
    res = parse_transcript(LABELED)
    assert res.fmt == "labeled"
    assert res.utterances[0].speaker == "Alice"
    assert res.utterances[0].start == "00:00:01"


def test_plain_falls_back_and_segments():
    res = parse_transcript(PLAIN)
    assert res.fmt == "plain"
    assert len(res.utterances) >= 1
    assert all(u.speaker == "" for u in res.utterances)


def test_empty_input_is_safe():
    res = parse_transcript("")
    assert res.utterances == []
    assert res.warnings
