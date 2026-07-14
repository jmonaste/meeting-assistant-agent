"""Consolidation / de-duplication of extracted items."""

from __future__ import annotations

from meeting_assistant.graph.merge import consolidate, render_known_items
from meeting_assistant.model.extraction import ActionItem, ChunkExtraction, Decision


def test_dedup_keeps_one_and_unions_segments():
    a = ChunkExtraction(action_items=[ActionItem(task="Send the report", segment_ids=["S1"])])
    b = ChunkExtraction(action_items=[ActionItem(task="send the report", owner="Bob", segment_ids=["S2"])])
    items, count = consolidate([a, b])
    assert count == 1
    action = items["action_items"][0]
    assert action.owner == "Bob"  # backfilled from the duplicate
    assert set(action.segment_ids) == {"S1", "S2"}  # unioned


def test_distinct_items_are_kept():
    a = ChunkExtraction(decisions=[Decision(decision="Ship in August")])
    b = ChunkExtraction(decisions=[Decision(decision="Adopt the new API")])
    items, count = consolidate([a, b])
    assert count == 2


def test_blank_keys_are_dropped():
    items, count = consolidate([ChunkExtraction(action_items=[ActionItem(task="")])])
    assert count == 0


def test_render_known_items_lists_categories():
    items, _ = consolidate([ChunkExtraction(action_items=[ActionItem(task="Do X")])])
    text = render_known_items(items)
    assert "Action items" in text
    assert "Do X" in text
