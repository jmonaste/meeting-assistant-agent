"""Web UI: job lifecycle, report serving and endpoint diagnostics.

The app runs the real pipeline in its background worker with the injected
fake LLM (see ``conftest.FakeLLM``), so these tests cover the full path a
browser takes: upload -> queued -> running -> done -> rendered report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meeting_assistant.config import load_settings
from meeting_assistant.web import create_app
from meeting_assistant.web.app import _render_markdown
from meeting_assistant.web.jobs import JobManager

from conftest import FakeLLM

FIXTURE = Path(__file__).parent / "fixtures" / "sample-standup.txt"


def _settings(**overrides):
    base = {
        "MEETING_CHUNK_MAX_CHARS": 700,
        "MEETING_CHUNK_OVERLAP_CHARS": 150,
        "MEETING_MAX_SWEEPS": 2,
        "MEETING_USE_CACHE": False,
        "MEETING_MAX_CONCURRENCY": 2,
        "openai_base_url": "http://127.0.0.1:9/v1",  # nothing listens on the discard port
    }
    base.update(overrides)
    return load_settings(**base)


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, settings_factory=_settings, llm_factory=FakeLLM)
    with TestClient(app) as c:
        yield c


def _wait(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def test_upload_runs_the_pipeline_and_serves_the_report(client):
    with FIXTURE.open("rb") as handle:
        res = client.post(
            "/api/jobs",
            files={"transcript": ("standup.txt", handle, "text/plain")},
            data={"context": "Priya is the PO.", "language": "en", "max_sweeps": "2"},
        )
    assert res.status_code == 201, res.text
    job = _wait(client, res.json()["id"])

    assert job["status"] == "done", job
    assert job["title"] == "Sprint Planning Standup"
    assert job["options"] == {"language": "en", "max_sweeps": 2}
    assert job["has_context"] is True
    # The review loop is visible: an initial pass plus a sweep that went dry.
    assert [p["kind"] for p in job["passes"]][:2] == ["initial", "sweep"]
    assert job["passes"][-1]["new_items"] == 0
    assert job["counts"]["Action items"] > 0
    assert job["segments"] > 0 and job["words"] > 0

    md = client.get(f"/api/jobs/{job['id']}/report?format=md")
    assert md.status_code == 200
    assert md.text.startswith("# Sprint Planning Standup")
    assert "attachment" in md.headers["content-disposition"]

    exported = client.get(f"/api/jobs/{job['id']}/report?format=json").json()
    assert exported["title"] == "Sprint Planning Standup"

    html = client.get(f"/api/jobs/{job['id']}/report?format=html").text
    assert '<h2 id="action-items">Action items</h2>' in html
    assert "<table>" in html

    log = client.get(f"/api/jobs/{job['id']}/log").text
    assert "run started" in log and "run finished" in log


def test_pasted_text_is_accepted(client):
    text = FIXTURE.read_text(encoding="utf-8")
    res = client.post("/api/jobs", data={"transcript_text": text, "transcript_name": "notes.txt"})
    assert res.status_code == 201, res.text
    job = _wait(client, res.json()["id"])
    assert job["status"] == "done"
    assert job["transcript_name"] == "notes.txt"
    assert client.get(f"/api/jobs/{job['id']}/transcript").text == text


def test_rejects_empty_and_word_uploads(client):
    assert client.post("/api/jobs", data={"transcript_text": "   "}).status_code == 422
    docx = client.post("/api/jobs", files={"transcript": ("minutes.docx", b"PK\x03\x04rest", "application/zip")})
    assert docx.status_code == 415
    assert ".vtt" in docx.json()["detail"]


def test_ansi_transcript_is_normalized_to_utf8(client):
    ansi = "Ana: Revisamos la migración del módulo.\nLuis: De acuerdo.\n".encode("cp1252")
    res = client.post("/api/jobs", files={"transcript": ("reunion.txt", ansi, "text/plain")})
    job = _wait(client, res.json()["id"])
    assert "migración" in client.get(f"/api/jobs/{job['id']}/transcript").text


def test_failed_job_can_be_resumed_and_deleted(tmp_path):
    calls = []

    def flaky_llm(settings):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("endpoint down")
        return FakeLLM(settings)

    app = create_app(tmp_path, settings_factory=_settings, llm_factory=flaky_llm)
    with TestClient(app) as client:
        res = client.post("/api/jobs", data={"transcript_text": FIXTURE.read_text(encoding="utf-8")})
        job = _wait(client, res.json()["id"])
        assert job["status"] == "failed"
        assert job["error"] == "endpoint down"

        assert client.post(f"/api/jobs/{job['id']}/resume").status_code == 200
        resumed = _wait(client, job["id"])
        assert resumed["status"] == "done", resumed
        assert resumed["error"] == ""

        assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404
        assert client.post("/api/jobs/missing/resume").status_code == 404


def test_done_job_cannot_be_resumed(client):
    res = client.post("/api/jobs", data={"transcript_text": FIXTURE.read_text(encoding="utf-8")})
    job = _wait(client, res.json()["id"])
    assert client.post(f"/api/jobs/{job['id']}/resume").status_code == 409


def test_restart_marks_running_jobs_interrupted(tmp_path):
    manager = JobManager(tmp_path, settings_factory=_settings, llm_factory=FakeLLM)
    job = manager.submit("t.txt", FIXTURE.read_bytes(), "", {})
    job.status = "running"
    manager._save(job)

    reloaded = JobManager(tmp_path, settings_factory=_settings).get(job.id)
    assert reloaded.status == "interrupted"
    assert json.loads((tmp_path / "jobs" / job.id / "job.json").read_text())["status"] == "interrupted"


def test_upload_cannot_overwrite_job_files(tmp_path):
    manager = JobManager(tmp_path, settings_factory=_settings)
    job = manager.submit("job.json", FIXTURE.read_bytes(), "", {})
    assert job.transcript_name == "transcript-job.json"
    assert (tmp_path / "jobs" / job.id / "transcript-job.json").read_bytes() == FIXTURE.read_bytes()


def test_queued_job_can_be_cancelled(tmp_path):
    manager = JobManager(tmp_path, settings_factory=_settings, llm_factory=FakeLLM)  # worker not started
    job = manager.submit("t.txt", FIXTURE.read_bytes(), "", {})
    assert manager.cancel(job.id).status == "cancelled"
    with pytest.raises(ValueError):
        manager.cancel(job.id)


def test_report_html_escapes_model_output():
    html = _render_markdown("# Title\n\n<script>alert(1)</script>\n\n| a | b |\n|---|---|\n| <img src=x onerror=1> | 2 |\n")
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html


def test_llm_check_reports_unreachable_endpoint(client):
    result = client.get("/api/llm-check").json()
    assert result["ok"] is False
    assert result["url"] == "http://127.0.0.1:9/v1/models"
    assert result["error"]


def test_config_masks_proxy_credentials(client, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:secret@proxy.corp:8080")
    config = client.get("/api/config").json()
    assert config["proxy"]["HTTPS_PROXY"] == "http://***@proxy.corp:8080"
    assert "secret" not in json.dumps(config)


def test_index_page_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "<title>Meeting Assistant</title>" in page.text
    assert client.get("/healthz").json()["status"] == "ok"
