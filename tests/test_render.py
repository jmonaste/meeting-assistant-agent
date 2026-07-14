"""Deterministic report and JSON rendering."""

from __future__ import annotations

import json

from meeting_assistant.graph.merge import consolidate
from meeting_assistant.model.extraction import (
    ActionItem,
    ChunkExtraction,
    Decision,
    MeetingSynthesis,
)
from meeting_assistant.render.document import render_json, render_report


def _state(sample_inventory):
    items, count = consolidate([
        ChunkExtraction(
            action_items=[ActionItem(task="File infra ticket", owner="Bob", due="Wednesday", segment_ids=["S2"])],
            decisions=[Decision(decision="Ship 2.0 on August 15th", segment_ids=["S8"])],
        )
    ])
    synthesis = MeetingSynthesis(
        title="Sprint Planning",
        meeting_type="standup",
        executive_summary="Planned the release.",
        detailed_summary="Details here.",
        next_steps=["File infra ticket"],
    )
    return {
        "inventory": sample_inventory,
        "items": items,
        "item_count": count,
        "synthesis": synthesis,
        "rounds_done": 3,
        "signals": {"action_items": ["S2: ..."]},
        "warnings": [],
    }


def test_report_has_sections_and_tables(sample_inventory):
    md = render_report(_state(sample_inventory))
    assert "# Sprint Planning — Meeting Report" in md
    assert "## Action items" in md
    assert "File infra ticket" in md
    assert "Ship 2.0 on August 15th" in md
    assert "## Contents" in md
    assert "Review passes" in md


def test_report_drops_hallucinated_segment_ids(sample_inventory):
    items, count = consolidate([
        ChunkExtraction(action_items=[ActionItem(task="Do thing", segment_ids=["S9999"])])
    ])
    state = {"inventory": sample_inventory, "items": items, "item_count": count,
             "synthesis": None, "rounds_done": 1}
    md = render_report(state)
    assert "S9999" not in md  # invalid id filtered out


def test_report_is_emoji_free(sample_inventory):
    synthesis = MeetingSynthesis(
        title="T", meeting_type="x",
        executive_summary="All good \U0001f600 shipping now",
        detailed_summary="d",
    )
    state = {"inventory": sample_inventory, "items": {}, "item_count": 0,
             "synthesis": synthesis, "rounds_done": 1}
    md = render_report(state)
    assert "\U0001f600" not in md


def test_json_export_is_valid(sample_inventory):
    data = json.loads(render_json(_state(sample_inventory)))
    assert data["title"] == "Sprint Planning"
    assert data["items"]["action_items"][0]["owner"] == "Bob"
    assert data["source"]["review_passes"] == 3
    assert {p["name"] for p in data["participants"]} == {"Alice", "Bob", "Carol", "Dave"}
