"""Full pipeline with an injected fake LLM — no endpoint required."""

from __future__ import annotations

import json

from meeting_assistant.graph.builder import run_pipeline


def _run(fake_llm, settings, sample_inventory, tmp_path, **kw):
    return run_pipeline(
        settings=settings,
        inventory=sample_inventory,
        llm=fake_llm,
        cache_base=tmp_path,
        **kw,
    )


def test_pipeline_produces_report_and_json(fake_llm, settings, sample_inventory, tmp_path):
    state = _run(fake_llm, settings, sample_inventory, tmp_path)
    assert state["report_md"].startswith("# Sprint Planning Standup — Meeting Report")
    assert "## Action items" in state["report_md"]
    data = json.loads(state["report_json"])
    assert data["meeting_type"] == "standup"
    assert data["items"]["action_items"]


def test_pipeline_runs_multiple_review_passes(fake_llm, settings, sample_inventory, tmp_path):
    state = _run(fake_llm, settings, sample_inventory, tmp_path)
    # Round 0 extracts; at least one sweep round runs, then it goes dry and stops.
    assert state["rounds_done"] >= 2
    # ChunkExtraction was requested for round 0 AND at least one sweep round.
    n_chunks = len(sample_inventory.chunks)
    assert fake_llm.calls.count("ChunkExtraction") >= 2 * n_chunks


def test_pipeline_stops_when_sweeps_go_dry(fake_llm, settings, sample_inventory, tmp_path):
    state = _run(fake_llm, settings, sample_inventory, tmp_path)
    # The fake returns nothing new on sweeps, so it must stop before the cap
    # (1 + max_sweeps = 3) — it terminates one round after the first dry sweep.
    assert state["rounds_done"] <= 1 + settings.max_sweeps


def test_pipeline_consolidates_items_across_chunks(fake_llm, settings, sample_inventory, tmp_path):
    state = _run(fake_llm, settings, sample_inventory, tmp_path)
    # One action item per distinct chunk-leading segment id.
    tasks = {a.task for a in state["items"]["action_items"]}
    assert len(tasks) == len(sample_inventory.chunks)


def test_pipeline_resume_reuses_thread(fake_llm, settings, sample_inventory, tmp_path):
    first = _run(fake_llm, settings, sample_inventory, tmp_path)
    resumed = _run(fake_llm, settings, sample_inventory, tmp_path, resume=True)
    assert resumed["report_md"] == first["report_md"]
