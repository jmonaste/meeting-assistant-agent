"""Wire the pipeline graph and provide the single ``run_pipeline`` entry point.

Graph shape:

    START -> ingest -> plan -> dispatch --(Send per chunk)--> extract_chunk -> dispatch
                                   \\--(rounds done)--> reduce -> gapfill -> compose -> END

``dispatch`` is the loop head: it consolidates the harvest and either launches
another full-transcript pass (round 0 = extract everything; later rounds = find
what was missed) or falls through to ``reduce`` once sweeping is exhausted.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from ..config import Settings, cache_dir
from ..llm import MeetingLLM
from ..model.transcript import TranscriptInventory
from .nodes import PipelineNodes
from .state import MeetingState

logger = logging.getLogger(__name__)


def build_graph(settings: Settings, llm: MeetingLLM | None = None, checkpointer=None):
    """Compile the pipeline; ``llm`` is injectable for tests."""
    nodes = PipelineNodes(settings, llm=llm)

    builder = StateGraph(MeetingState)
    builder.add_node("ingest", nodes.ingest)
    builder.add_node("plan", nodes.plan)
    builder.add_node("dispatch", nodes.dispatch)
    builder.add_node("extract_chunk", nodes.extract_chunk)
    builder.add_node("reduce", nodes.reduce)
    builder.add_node("gapfill", nodes.gapfill)
    builder.add_node("compose", nodes.compose)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "plan")
    builder.add_edge("plan", "dispatch")
    builder.add_conditional_edges("dispatch", nodes.route_map, ["extract_chunk", "reduce"])
    builder.add_edge("extract_chunk", "dispatch")
    builder.add_edge("reduce", "gapfill")
    builder.add_edge("gapfill", "compose")
    builder.add_edge("compose", END)

    return builder.compile(checkpointer=checkpointer)


def _serde():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    tr = "meeting_assistant.model.transcript"
    ex = "meeting_assistant.model.extraction"
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            (tr, "Segment"), (tr, "Speaker"), (tr, "Chunk"), (tr, "TranscriptInventory"),
            (ex, "Topic"), (ex, "Decision"), (ex, "ActionItem"), (ex, "OpenQuestion"),
            (ex, "Risk"), (ex, "KeyFact"), (ex, "KeyDate"), (ex, "Entity"),
            (ex, "Quote"), (ex, "Disagreement"), (ex, "GlossaryTerm"),
            (ex, "ChunkExtraction"), (ex, "SweepResult"), (ex, "MeetingPlan"),
            (ex, "ParticipantRole"), (ex, "MeetingSynthesis"),
        ]
    )


def run_pipeline(
    transcript_text: str = "",
    settings: Settings | None = None,
    source_file: str = "",
    context_text: str = "",
    inventory: TranscriptInventory | None = None,
    llm: MeetingLLM | None = None,
    on_event: Callable[[str, object], None] | None = None,
    resume: bool = False,
    cache_base: Path | None = None,
) -> MeetingState:
    """Run the full pipeline and return the final state.

    A SQLite checkpointer records every super-step under ``.meeting_cache/`` so a
    crashed run on a long transcript can be resumed with ``resume=True``.
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    if settings is None:
        raise ValueError("settings is required")

    base = cache_base or (Path(source_file).resolve().parent if source_file else Path("."))
    ckpt_dir = cache_dir(base)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_file = ckpt_dir / "last_run_id"

    thread_id = None
    if resume and run_file.exists():
        thread_id = run_file.read_text(encoding="utf-8").strip() or None
    if thread_id is None:
        thread_id = uuid.uuid4().hex
        run_file.write_text(thread_id, encoding="utf-8")

    # check_same_thread=False: map-phase nodes run in worker threads.
    conn = sqlite3.connect(str(ckpt_dir / "checkpoints.sqlite"), check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=_serde())
        graph = build_graph(settings, llm=llm, checkpointer=saver)
        config = {
            "configurable": {"thread_id": thread_id},
            "max_concurrency": settings.max_concurrency,
            "recursion_limit": 200,
        }
        inputs: MeetingState | None = {
            "transcript_text": transcript_text,
            "source_file": source_file,
            "context_text": context_text,
        }
        if inventory is not None:
            inputs["inventory"] = inventory
        if resume and _has_progress(saver, thread_id):
            inputs = None  # continue from the last checkpoint
            logger.info("resuming run %s from its last checkpoint", thread_id)
        else:
            logger.debug("starting run thread %s", thread_id)

        for update in graph.stream(inputs, config=config, stream_mode="updates"):
            for node, payload in update.items():
                if on_event is not None:
                    on_event(node, payload)
        state = graph.get_state(config)
        return dict(state.values) if state and state.values else {}
    finally:
        conn.close()


def _has_progress(saver, thread_id: str) -> bool:
    try:
        return saver.get({"configurable": {"thread_id": thread_id}}) is not None
    except Exception:  # noqa: BLE001
        return False
