"""FastAPI application: a single page plus a small JSON API over ``JobManager``.

The page polls plain HTTP endpoints (no websockets, no external assets), so it
works unchanged behind ``oc port-forward``, a corporate proxy, or a browser
that cannot reach any CDN.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.request
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from .. import __version__
from ..config import Settings, load_settings
from ..render.lint import md_anchor
from .jobs import JobManager
from .settings_store import SettingsError, SettingsStore

logger = logging.getLogger(__name__)

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


def _cause_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in chain and len(chain) < 10:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    return chain


def _route(url: str) -> str:
    """Whether a request to ``url`` goes direct or through a proxy.

    Uses the same sources httpx reads with ``trust_env``: the proxy environment
    variables and, on Windows, the system (registry) proxy settings.
    """
    parts = urlsplit(url)
    proxies = urllib.request.getproxies()
    proxy = proxies.get(parts.scheme) or proxies.get("all")
    if not proxy:
        return "direct (no proxy configured)"
    no_proxy = proxies.get("no", "")
    host = parts.hostname or ""
    bypass = any(
        entry and (host == entry.lstrip(".") or host.endswith("." + entry.lstrip(".")) or entry == "*")
        for entry in (e.strip() for e in no_proxy.split(","))
    ) or urllib.request.proxy_bypass(host)
    if bypass:
        return f"direct ({host} is excluded from the proxy)"
    return f"through proxy {_mask(proxy)} (add {host} to NO_PROXY to go direct)"


def _probe_endpoint(s: Settings) -> dict:
    """List the endpoint's models through the *pipeline's own* client.

    The OpenAI client and HTTP client come from ``MeetingLLM`` exactly as a run
    builds them, so whatever ``llm.py`` configures there — TLS verification or a
    CA path, redirects, proxies, timeouts — is what gets tested. Only the
    timeout (15 s) and retries (none) are shortened for the probe.
    """
    from ..llm import MeetingLLM

    url = s.openai_base_url.rstrip("/") + "/models"
    result: dict = {"url": _mask(url), "ok": False, "route": _route(s.openai_base_url)}
    started = time.perf_counter()
    try:
        client = MeetingLLM(s)._chat_model("lead").root_client.with_options(max_retries=0, timeout=15.0)
        page = client.models.list()
        ids = [getattr(m, "id", None) for m in getattr(page, "data", [])]
    except Exception as exc:  # noqa: BLE001 - classified below for the user
        chain = _cause_chain(exc)
        text = " | ".join(str(e) for e in chain)
        root = chain[-1]
        status = getattr(exc, "status_code", None)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        if status is not None:
            result["status_code"] = status
            response = getattr(exc, "response", None)
            location = response.headers.get("location", "") if response is not None else ""
            if status in (401, 403):
                result.update(error=f"HTTP {status}", hint="The endpoint rejected the API key.")
            elif status == 404:
                result.update(
                    ok=True,
                    error="HTTP 404 on /models",
                    hint="The endpoint answered, so network and TLS work; it just does not list models.",
                )
            elif 300 <= status < 400:
                result.update(
                    error=f"HTTP {status} redirect" + (f" to {_mask(location)}" if location else ""),
                    hint="The endpoint answered with a redirect instead of the API: usually a login/SSO "
                    "page, a proxy block page, or a base URL that is slightly off (http vs https, "
                    "missing /v1).",
                )
            else:
                result.update(error=f"HTTP {status}: {_mask(str(exc))[:200]}")
        elif any(isinstance(e, httpx.ProxyError) for e in chain):
            result.update(error=f"proxy error: {root}", hint="Check HTTPS_PROXY / NO_PROXY for this endpoint.")
        elif any("Timeout" in type(e).__name__ for e in chain):
            result.update(
                error="timed out after 15s",
                hint="Not reachable from here: a firewall, a missing proxy, or an internal host "
                "that needs to be in NO_PROXY.",
            )
        elif "CERTIFICATE_VERIFY_FAILED" in text or "certificate verify" in text.lower():
            result.update(
                error=f"TLS verification failed: {str(root)[:200]}",
                hint="The endpoint (or a proxy doing SSL inspection) uses a certificate from a "
                "corporate CA that this process does not trust. Point SSL_CERT_FILE (or the CA "
                "path your llm.py uses) at a bundle that includes it.",
            )
        else:
            result.update(
                error=f"{type(root).__name__}: {str(root)[:300]}",
                hint="Check the host name and that it is reachable from this machine/pod.",
            )
        return result

    result["ok"] = True
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    result["models"] = [i for i in ids if i]
    missing = [m for m in {s.resolved_worker_model(), s.resolved_lead_model()} if ids and m not in ids]
    if missing:
        result["hint"] = "Configured model(s) not listed by the endpoint: " + ", ".join(sorted(missing))
    return result


def create_app(data_dir: Path | str | None = None, settings_factory=load_settings, llm_factory=None) -> FastAPI:
    data_path = Path(data_dir or os.environ.get("MEETING_DATA_DIR") or "meeting-data")
    store = SettingsStore(data_path / "settings.json")

    def effective_settings(**job_overrides) -> Settings:
        """Environment/defaults, then the values saved in the UI, then per-job options."""
        return settings_factory(**{**store.init_overrides(), **job_overrides})

    manager = JobManager(data_path, settings_factory=effective_settings, llm_factory=llm_factory)

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
        s = effective_settings()
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
        return _probe_endpoint(effective_settings())

    @app.post("/api/llm-check")
    def llm_check_values(payload: dict = Body(default_factory=dict)) -> dict:
        """Probe with unsaved values from the Settings dialog merged over the saved ones."""
        try:
            values = store.validate(store.merged(payload.get("values") or {}), settings_factory)
        except SettingsError as exc:
            raise HTTPException(422, exc.errors) from None
        return _probe_endpoint(settings_factory(**store.init_overrides(values)))

    # ------------------------------------------------------------ settings

    @app.get("/api/settings")
    def get_settings() -> dict:
        return {"fields": store.describe(settings_factory())}

    @app.put("/api/settings")
    def put_settings(payload: dict = Body(...)) -> dict:
        """Save changed values; ``null`` or ``""`` resets a field to its ConfigMap/default value."""
        changes = payload.get("values")
        if not isinstance(changes, dict):
            raise HTTPException(422, "expected {\"values\": {...}}")
        try:
            store.update(changes, settings_factory)
        except SettingsError as exc:
            raise HTTPException(422, exc.errors) from None
        logger.info("settings updated from the UI: %s", ", ".join(sorted(changes)) or "(none)")
        return {"fields": store.describe(settings_factory())}

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
