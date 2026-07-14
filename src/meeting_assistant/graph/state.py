"""Shared pipeline state.

The state carries compact artifacts only: the parsed inventory, per-chunk
extractions and the consolidated item lists — never the raw transcript passed
around redundantly. Map/sweep nodes run in parallel via the Send API, so the
accumulator they write (``harvest``) uses an additive reducer; the consolidated
``items`` are rebuilt from that harvest by the (single-threaded) dispatch node.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..model.extraction import ChunkExtraction, MeetingPlan, MeetingSynthesis
from ..model.transcript import TranscriptInventory


class MeetingState(TypedDict, total=False):
    # inputs
    transcript_text: str  # raw transcript, when the inventory is not pre-built
    source_file: str
    context_text: str

    # ingestion artifacts (deterministic)
    inventory: TranscriptInventory
    signals: dict[str, list[str]]

    # analysis artifacts
    plan: MeetingPlan
    # Every chunk extraction from every round, appended as it completes. The
    # dispatch node rebuilds `items` from the full harvest, so consolidation is
    # idempotent and safe to repeat between rounds.
    harvest: Annotated[list[ChunkExtraction], operator.add]
    current_chunks: list["ChunkPayload"]  # payloads to extract this round
    rounds_done: int
    items: dict[str, list]  # category -> deduped list of item models
    item_count: int  # size of `items` after the last consolidation

    # synthesis + rescue
    synthesis: MeetingSynthesis
    rescued: dict[str, int]  # category -> number of items rescued by the gapfill agent
    gap_notes: list[str]

    # outputs
    report_md: str
    report_json: str
    warnings: Annotated[list[str], operator.add]


class ChunkPayload(TypedDict):
    """Isolated input for one chunk extraction (built by dispatch)."""

    chunk_index: int
    chunk_text: str
    round: int  # 0 = initial full extraction; >=1 = coverage sweep
    known_items: str  # digest of items already found (empty on round 0)
    chunk_hash: str
    context: str  # user-provided background block (may be empty)
