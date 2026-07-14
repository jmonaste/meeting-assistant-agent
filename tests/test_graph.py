"""Full pipeline with an injected fake LLM — no endpoint required."""

from __future__ import annotations

import json

from conftest import FakeLLM

from meeting_assistant.graph.builder import run_pipeline
from meeting_assistant.llm import MeetingLLMUnavailable
from meeting_assistant.model.extraction import ChunkExtraction, Quote


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


class _RescueLLM(FakeLLM):
    """The gapfill evidence pass finds one item the chunk passes missed."""

    def structured(self, schema, system, user, role="worker"):  # type: ignore[override]
        if schema is ChunkExtraction and "EVIDENCE GATHERED" in user:
            return ChunkExtraction(
                quotes=[Quote(text="A unique rescued statement", segment_ids=["S1"])]
            )
        return super().structured(schema, system, user, role)


def test_gapfill_rescue_keeps_item_count_consistent(settings, sample_inventory, tmp_path):
    llm = _RescueLLM(settings)
    state = _run(llm, settings, sample_inventory, tmp_path)
    assert state["rescued"]  # the rescue actually happened
    total = sum(len(bucket) for bucket in state["items"].values())
    assert state["item_count"] == total  # summary count includes rescued items


class _DownLLM(FakeLLM):
    """Every chunk extraction fails like a dead endpoint (504/429 after retries)."""

    def structured(self, schema, system, user, role="worker"):  # type: ignore[override]
        if schema is ChunkExtraction:
            self.calls.append("ChunkExtraction")
            raise MeetingLLMUnavailable("endpoint still failing after 5 attempts: 504 Gateway Time-out")
        return super().structured(schema, system, user, role)


def test_pipeline_writes_detailed_log(fake_llm, settings, sample_inventory, tmp_path):
    import logging

    log_path = tmp_path / "run.log"
    lg = logging.getLogger("meeting_assistant")
    old_level = lg.level
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        _run(fake_llm, settings, sample_inventory, tmp_path)
    finally:
        lg.removeHandler(handler)
        handler.close()
        lg.setLevel(old_level)
    text = log_path.read_text(encoding="utf-8")
    assert "starting pass 1/" in text
    assert "review loop finished" in text
    assert "compose: report" in text


def test_pipeline_stops_sweeping_when_endpoint_is_unhealthy(settings, sample_inventory, tmp_path):
    llm = _DownLLM(settings)
    state = _run(llm, settings, sample_inventory, tmp_path)
    # Round 0 fails on every chunk -> no further sweep rounds are launched.
    assert state["rounds_done"] == 1
    n_chunks = len(sample_inventory.chunks)
    assert llm.calls.count("ChunkExtraction") <= n_chunks + 1  # +1 for the gapfill structuring
    # The run still completes with a report and a clear warning.
    assert state["report_md"]
    assert any("endpoint looks unhealthy" in w for w in state["warnings"])
