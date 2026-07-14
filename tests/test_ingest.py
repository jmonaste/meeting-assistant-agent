"""Inventory assembly, chunking and the deterministic signal scan."""

from __future__ import annotations

from meeting_assistant.extractors import scan_signals
from meeting_assistant.ingest.chunker import build_chunks
from meeting_assistant.ingest.loader import build_inventory
from meeting_assistant.model.transcript import Segment


def test_inventory_speakers_and_timestamps(sample_inventory):
    inv = sample_inventory
    assert inv.source_format == "labeled"
    assert {sp.name for sp in inv.speakers} == {"Alice", "Bob", "Carol", "Dave"}
    assert inv.has_timestamps
    assert inv.duration.startswith("00:00:01")
    assert inv.total_words > 100


def test_chunks_cover_every_segment(sample_inventory):
    inv = sample_inventory
    covered = set()
    by_index = {s.sid: s.index for s in inv.segments}
    for ch in inv.chunks:
        lo, hi = by_index[ch.first_sid], by_index[ch.last_sid]
        covered.update(range(lo, hi + 1))
    assert covered == set(range(len(inv.segments)))


def test_chunks_overlap(settings):
    segs = [Segment(sid=f"S{i}", index=i, speaker="X", text="word " * 60) for i in range(10)]
    chunks = build_chunks(segs, with_timestamp=False, max_chars=700, overlap_chars=200)
    assert len(chunks) > 1
    # Adjacent chunks should share at least one segment id (the overlap).
    by = {f"S{i}": i for i in range(10)}
    for a, b in zip(chunks, chunks[1:]):
        assert by[b.first_sid] <= by[a.last_sid]


def test_single_giant_segment_still_chunks():
    segs = [Segment(sid="S0", index=0, text="x" * 5000)]
    chunks = build_chunks(segs, with_timestamp=False, max_chars=700, overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].first_sid == "S0"


def test_signal_scan_finds_actions_and_dates(sample_inventory):
    sig = scan_signals(sample_inventory)
    assert sig["action_items"]
    assert sig["key_dates"]
    assert sig["decisions"]
    # Every hit is anchored to a segment id.
    assert all(hit.startswith("S") for hits in sig.values() for hit in hits)


def test_context_text_is_carried(settings, sample_text):
    inv = build_inventory(sample_text, settings, context_text="Priya is the product owner.")
    assert "Priya" in inv.context_block()
