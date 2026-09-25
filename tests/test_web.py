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


# ------------------------------------------------------------ settings dialog


def _fields(client):
    return {f["key"]: f for f in client.get("/api/settings").json()["fields"]}


def test_settings_saved_in_the_ui_override_the_deployment(client, tmp_path):
    before = _fields(client)
    assert before["MEETING_MAX_SWEEPS"]["source"] == "env"  # set by the test's base factory
    assert before["MEETING_LEAD_MODEL"]["source"] == "default"

    res = client.put("/api/settings", json={"values": {
        "OPENAI_BASE_URL": "http://llm.internal:8000/v1",
        "MEETING_LEAD_MODEL": "gpt-oss-120b",
        "MEETING_MAX_CONCURRENCY": "1",
        "MEETING_VERIFY_SSL": "false",
    }})
    assert res.status_code == 200, res.text
    after = {f["key"]: f for f in res.json()["fields"]}
    assert after["MEETING_LEAD_MODEL"] == {**after["MEETING_LEAD_MODEL"], "source": "ui", "value": "gpt-oss-120b"}
    assert after["MEETING_MAX_CONCURRENCY"]["value"] == 1  # stored typed
    assert after["MEETING_VERIFY_SSL"]["value"] is False

    config = client.get("/api/config").json()
    assert config["base_url"] == "http://llm.internal:8000/v1"
    assert config["lead_model"] == "gpt-oss-120b"
    assert config["max_concurrency"] == 1

    # A run uses them, and per-analysis options still win.
    manager = client.app.state.jobs
    job = manager.submit("t.txt", FIXTURE.read_bytes(), "", {"max_concurrency": 2})
    manager.cancel(job.id)
    settings = manager.settings_for(job)
    assert settings.resolved_lead_model() == "gpt-oss-120b"
    assert settings.verify_ssl is False
    assert settings.max_concurrency == 2

    # Reset falls back to the deployment value.
    client.put("/api/settings", json={"values": {"MEETING_LEAD_MODEL": None, "MEETING_MAX_CONCURRENCY": ""}})
    reset = _fields(client)
    assert reset["MEETING_LEAD_MODEL"]["source"] == "default"
    assert reset["MEETING_MAX_CONCURRENCY"]["value"] == 2  # the base factory's value

    stored = json.loads((tmp_path / "settings.json").read_text())
    assert "MEETING_LEAD_MODEL" not in stored and stored["OPENAI_BASE_URL"] == "http://llm.internal:8000/v1"


def test_api_key_is_write_only_and_private(client, tmp_path):
    client.put("/api/settings", json={"values": {"OPENAI_API_KEY": "sk-secret-value-1234"}})
    body = client.get("/api/settings").text
    assert "sk-secret-value" not in body
    key = _fields(client)["OPENAI_API_KEY"]
    assert key["configured"] is True and key["masked"] == "…1234" and "value" not in key

    path = tmp_path / "settings.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert client.app.state.jobs.settings_for(
        client.app.state.jobs.submit("t.txt", b"Ana: hola", "", {})
    ).openai_api_key == "sk-secret-value-1234"

    client.put("/api/settings", json={"values": {"OPENAI_API_KEY": None}})
    assert _fields(client)["OPENAI_API_KEY"]["configured"] is False


def test_invalid_settings_are_rejected_without_saving(client):
    res = client.put("/api/settings", json={"values": {"MEETING_MAX_CONCURRENCY": "0", "MEETING_TEMPERATURE": "hot"}})
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert "MEETING_TEMPERATURE" in detail
    assert _fields(client)["MEETING_TEMPERATURE"]["source"] == "default"

    res = client.put("/api/settings", json={"values": {"MEETING_MAX_CONCURRENCY": "0"}})
    assert res.status_code == 422 and "MEETING_MAX_CONCURRENCY" in res.json()["detail"]
    assert client.put("/api/settings", json={"values": {"NOT_A_SETTING": "x"}}).status_code == 422


def test_settings_survive_a_restart(tmp_path):
    first = create_app(tmp_path, settings_factory=_settings, llm_factory=FakeLLM)
    with TestClient(first) as c:
        c.put("/api/settings", json={"values": {"MEETING_WORKER_MODEL": "gemma-3"}})
    with TestClient(create_app(tmp_path, settings_factory=_settings, llm_factory=FakeLLM)) as c:
        assert c.get("/api/config").json()["worker_model"] == "gemma-3"


def test_connection_can_be_tested_with_unsaved_values(client):
    client.put("/api/settings", json={"values": {"OPENAI_BASE_URL": "http://saved.invalid/v1"}})
    result = client.post("/api/llm-check", json={"values": {"OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}}).json()
    assert result["url"] == "http://127.0.0.1:9/v1/models"
    assert result["ok"] is False
    assert _fields(client)["OPENAI_BASE_URL"]["value"] == "http://saved.invalid/v1"  # nothing saved


# ------------------------------------------------------------ connection check


class _MockedLLM:
    """Stands in for MeetingLLM: same ChatOpenAI wiring, but a mocked HTTP transport."""

    def __init__(self, handler):
        self.handler = handler

    def __call__(self, settings):
        import httpx
        from langchain_openai import ChatOpenAI

        handler = self.handler

        class _Wrapped:
            def _chat_model(self, role):
                return ChatOpenAI(
                    model="gpt-oss", base_url=settings.openai_base_url, api_key=settings.openai_api_key,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                )

        return _Wrapped()


@pytest.mark.parametrize(
    ("status", "headers", "ok", "expect"),
    [
        (200, {}, True, None),
        (302, {"location": "https://sso.corp.example/login"}, False, "redirect to https://sso.corp.example/login"),
        (401, {}, False, "HTTP 401"),
        (404, {}, True, "HTTP 404 on /models"),
    ],
)
def test_llm_check_goes_through_the_pipeline_client(client, monkeypatch, status, headers, ok, expect):
    import httpx

    seen = []

    def handler(request):
        seen.append(str(request.url))
        if status == 200:
            return httpx.Response(200, json={"object": "list", "data": [{"id": "gpt-oss", "object": "model"}]})
        return httpx.Response(status, headers=headers, json={"error": {"message": "nope"}})

    monkeypatch.setattr("meeting_assistant.llm.MeetingLLM", _MockedLLM(handler))
    result = client.get("/api/llm-check").json()
    assert seen == ["http://127.0.0.1:9/v1/models"]
    assert result["ok"] is ok
    if expect:
        assert expect in result["error"]
    else:
        assert result["models"] == ["gpt-oss"]
    assert result["route"].startswith(("direct", "through proxy"))


def test_route_reports_proxy_and_no_proxy(monkeypatch):
    from meeting_assistant.web.app import _route

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pw@proxy.corp:8080")
    assert _route("https://llm.corp.internal/v1") == (
        "through proxy http://***@proxy.corp:8080 (add llm.corp.internal to NO_PROXY to go direct)"
    )
    monkeypatch.setenv("NO_PROXY", ".corp.internal,localhost")
    assert _route("https://llm.corp.internal/v1").startswith("direct")
