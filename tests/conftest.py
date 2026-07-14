"""Shared test fixtures: an injectable fake LLM and a sample inventory.

The fake LLM lets the whole graph run with no endpoint. It returns canned
structured objects keyed by the requested schema, and derives per-chunk items
from the segment ids present in the prompt so the map/dedup path is exercised
with realistic, distinct items.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from meeting_assistant.config import load_settings
from meeting_assistant.ingest.loader import build_inventory
from meeting_assistant.llm import MeetingLLM
from meeting_assistant.model.extraction import (
    ActionItem,
    ChunkExtraction,
    Decision,
    Entity,
    KeyDate,
    MeetingPlan,
    MeetingSynthesis,
    OpenQuestion,
    ParticipantRole,
    Risk,
    Topic,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeLLM(MeetingLLM):
    """A MeetingLLM that never touches the network."""

    def __init__(self, settings) -> None:  # noqa: D107 - no super().__init__ endpoint work needed
        self.settings = settings
        self._models = {}
        self.calls: list[str] = []

    def structured(self, schema, system, user, role="worker"):  # type: ignore[override]
        self.calls.append(schema.__name__)
        if schema is MeetingPlan:
            return MeetingPlan(
                title="Sprint Planning Standup",
                meeting_type="standup",
                focus_areas=["release date", "blockers"],
                likely_participants=["Alice", "Bob"],
            )
        if schema is MeetingSynthesis:
            return MeetingSynthesis(
                title="Sprint Planning Standup",
                meeting_type="standup",
                executive_summary="The team planned the 2.0 release.",
                detailed_summary="Alice opened; Bob and Carol reported; a date was set.",
                overall_sentiment="collaborative and decisive",
                next_steps=["File infra ticket", "Send test plan"],
                participants=[ParticipantRole(name="Alice", role="facilitator")],
                coverage_gaps=[],
            )
        if schema is ChunkExtraction:
            if "ALREADY CAPTURED" in user or "EVIDENCE GATHERED" in user:
                return ChunkExtraction()  # sweeps/rescue find nothing new -> loop terminates
            sids = re.findall(r"\[(S\d+)\]", user)
            first = sids[0] if sids else "S0"
            return ChunkExtraction(
                topics=[Topic(title=f"Topic {first}", summary="discussed", segment_ids=[first])],
                decisions=[Decision(decision=f"Decision at {first}", segment_ids=[first])],
                action_items=[ActionItem(task=f"Task from {first}", owner="Bob", due="Friday", segment_ids=[first])],
                open_questions=[OpenQuestion(question=f"Question at {first}?", segment_ids=[first])],
                risks=[Risk(description=f"Risk at {first}", segment_ids=[first])],
                key_dates=[KeyDate(when="August 15th", what="release", segment_ids=[first])],
                entities=[Entity(name="Priya", kind="person", segment_ids=[first])],
            )
        raise AssertionError(f"unexpected schema {schema!r}")

    def tool_loop(self, system, user, tools, role="lead", max_iterations=None):  # type: ignore[override]
        self.calls.append("tool_loop")
        return "No additional items were found in the transcript.", []


@pytest.fixture
def settings():
    return load_settings(
        MEETING_CHUNK_MAX_CHARS=700,
        MEETING_CHUNK_OVERLAP_CHARS=150,
        MEETING_MAX_SWEEPS=2,
        MEETING_USE_CACHE=False,
        MEETING_MAX_CONCURRENCY=2,
    )


@pytest.fixture
def sample_text():
    return (FIXTURES / "sample-standup.txt").read_text(encoding="utf-8")


@pytest.fixture
def sample_inventory(settings, sample_text):
    return build_inventory(sample_text, settings, source_file=str(FIXTURES / "sample-standup.txt"))


@pytest.fixture
def fake_llm(settings):
    return FakeLLM(settings)
