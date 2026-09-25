"""FastAPI application: a single page plus a small JSON API over ``JobManager``.

The page polls plain HTTP endpoints (no websockets, no external assets), so it
works unchanged behind ``oc port-forward``, a corporate proxy, or a browser
that cannot reach any CDN.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from .. import __version__
from ..config import Settings, load_settings
from ..render.lint import md_anchor
from .jobs import JobManager

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy")


def _mask(url: str) -> str:
    """Hide credentials embedded in a URL (``http://user:pass@proxy``)."""
    return re.sub(r"//[^/@\s]+@", "//***@", url or "")


def _to_utf8(data: bytes) -> bytes:
    """Transcripts saved on Windows are often ANSI; normalize them to UTF-8."""
    try:
        data.decode("utf-8")
        return data
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace").encode("utf-8")


async def _read_upload(upload: UploadFile | None) -> bytes:
    if upload is None or not upload.filename:
        return b""
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if data[:2] == b"PK":
        raise HTTPException(
            415,
            f"{upload.filename} looks like a Word/zip file. Export the transcript as "
            ".vtt, .srt or .txt (in Teams: Transcript > Download as .vtt).",
        )
    return _to_utf8(data)


def _render_markdown(text: str) -> str:
    """Report Markdown to HTML. Raw HTML is disabled: model output is untrusted."""
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])
    tokens = md.parse(text)
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and i + 1 < len(tokens):
            token.attrSet("id", md_anchor(tokens[i + 1].content))
    return md.renderer.render(tokens, md.options, {})


def _int_or_none(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        return int(value)
    except ValueError:
        raise HTTPException(422, f"not a number: {value!r}") from None


def create_app(data_dir: Path | str | None = None, settings_factory=load_settings, llm_factory=None) -> FastAPI:
    data_path = Path(data_dir or os.environ.get("MEETING_DATA_DIR") or "meeting-data")
    manager = JobManager(data_path, settings_factory=settings_factory, llm_factory=llm_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        manager.start()
        yield

    app = FastAPI(
        title="meeting-assistant", version=__version__, docs_url="/api/docs", redoc_url=None, lifespan=lifespan
    )
    app.state.jobs = manager

    def _job_or_404(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    # ------------------------------------------------------------ page

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files(__package__).joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    # ------------------------------------------------------------ endpoint

    @app.get("/api/config")
    def config() -> dict:
        s: Settings = settings_factory()
        return {
            "version": __version__,
            "base_url": _mask(s.openai_base_url),
            "worker_model": s.resolved_worker_model(),
            "lead_model": s.resolved_lead_model(),
            "verify_ssl": s.verify_ssl,
            "request_timeout": s.request_timeout,
            "max_concurrency": s.max_concurrency,
            "max_sweeps": s.max_sweeps,
            "chunk_max_chars": s.chunk_max_chars,
            "language": s.language,
            "ca_bundle": os.environ.get("SSL_CERT_FILE", ""),
            "proxy": {k: _mask(v) for k in _PROXY_VARS if (v := os.environ.get(k))},
            "data_dir": str(data_path.resolve()),
        }

    @app.get("/api/llm-check")
    def llm_check() -> dict:
        """Probe ``GET {base_url}/models`` with the same TLS/proxy settings as a run."""
        s: Settings = settings_factory()
        url = s.openai_base_url.rstrip("/") + "/models"
        result: dict = {"url": _mask(url), "ok": False}
        started = time.perf_counter()
        try:
            with httpx.Client(verify=s.verify_ssl, timeout=15.0) as client:
                response = client.get(url, headers={"Authorization": f"Bearer {s.openai_api_key}"})
        except httpx.ProxyError as exc:
            result.update(error=f"proxy error: {exc}", hint="Check HTTPS_PROXY / NO_PROXY for this endpoint.")
        except httpx.TimeoutException:
            result.update(
                error="timed out after 15s",
                hint="The endpoint is not reachable from here: a firewall, a missing proxy, "
                "or an internal host that needs to be in NO_PROXY.",
            )
        except httpx.ConnectError as exc:
            text = str(exc)
            hint = "Check the host name and that it is reachable from this machine/pod."
            if "CERTIFICATE_VERIFY_FAILED" in text or "certificate" in text.lower():
                hint = (
                    "TLS verification failed: the endpoint (or a proxy doing SSL inspection) "
                    "uses a certificate from a corporate CA. Mount that CA and point "
                    "SSL_CERT_FILE at a bundle that includes it."
                )
            result.update(error=f"connection failed: {text[:300]}", hint=hint)
        else:
            result["status_code"] = response.status_code
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            if response.status_code in (401, 403):
                result.update(error=f"HTTP {response.status_code}", hint="The endpoint rejected OPENAI_API_KEY.")
            elif response.status_code >= 400:
                # Reachable, but no /models route: some gateways only proxy /chat/completions.
                result.update(
                    ok=response.status_code == 404,
                    error=f"HTTP {response.status_code} on /models",
                    hint="The endpoint answered, so network and TLS work; it just does not list models.",
                )
            else:
                result["ok"] = True
                try:
                    ids = [m.get("id") for m in response.json().get("data", []) if isinstance(m, dict)]
                except ValueError:
                    ids = []
                result["models"] = ids
                missing = [m for m in {s.resolved_worker_model(), s.resolved_lead_model()} if ids and m not in ids]
                if missing:
                    result["hint"] = "Configured model(s) not listed by the endpoint: " + ", ".join(sorted(missing))
        return result

    # ------------------------------------------------------------ jobs

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [job.to_dict() for job in manager.list()]

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        transcript: UploadFile | None = File(None),
        transcript_text: str = Form(""),
        transcript_name: str = Form(""),
        context: str = Form(""),
        context_file: UploadFile | None = File(None),
        language: str = Form(""),
        max_sweeps: str = Form(""),
        max_concurrency: str = Form(""),
        chunk_chars: str = Form(""),
        worker_model: str = Form(""),
        lead_model: str = Form(""),
    ) -> dict:
        data = await _read_upload(transcript)
        name = transcript.filename if transcript is not None and transcript.filename else ""
        if not data and transcript_text.strip():
            data = transcript_text.encode("utf-8")
            name = transcript_name.strip() or "pasted-transcript.txt"
        if not data.strip():
            raise HTTPException(422, "Upload a transcript file or paste the transcript text.")

        context_text = context
        context_data = await _read_upload(context_file)
        if context_data:
            context_text = (context_data.decode("utf-8") + "\n\n" + context).strip()

        job = manager.submit(
            name,
            data,
            context_text,
            {
                "language": language.strip(),
                "max_sweeps": _int_or_none(max_sweeps),
                "max_concurrency": _int_or_none(max_concurrency),
                "chunk_chars": _int_or_none(chunk_chars),
                "worker_model": worker_model.strip(),
                "lead_model": lead_model.strip(),
            },
        )
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return _job_or_404(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str, lines: int = 200) -> str:
        _job_or_404(job_id)
        return manager.log_tail(job_id, lines=max(1, min(lines, 5000)))

    @app.get("/api/jobs/{job_id}/report")
    def job_report(job_id: str, format: str = "html"):
        _job_or_404(job_id)
        kind = "json" if format == "json" else "md"
        path = manager.report_path(job_id, kind)
        if path is None:
            raise HTTPException(404, "this job has no report yet")
        if format == "html":
            return HTMLResponse(_render_markdown(path.read_text(encoding="utf-8")))
        media = "application/json" if kind == "json" else "text/markdown; charset=utf-8"
        return FileResponse(path, media_type=media, filename=path.name)

    @app.get("/api/jobs/{job_id}/transcript", response_class=PlainTextResponse)
    def job_transcript(job_id: str) -> str:
        job = _job_or_404(job_id)
        path = manager.job_dir(job_id) / job.transcript_name
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

    def _action(fn, job_id: str) -> dict:
        _job_or_404(job_id)
        try:
            result = fn(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return result.to_dict() if result is not None else {"deleted": job_id}

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str) -> dict:
        return _action(manager.resume, job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        return _action(manager.cancel, job_id)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> JSONResponse:
        return JSONResponse(_action(manager.delete, job_id))

    return app
