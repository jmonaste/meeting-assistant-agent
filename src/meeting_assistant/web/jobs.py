"""Background job runner for the web UI.

A run takes minutes to tens of minutes against a local model, far longer than
a browser tab or an ``oc port-forward`` tunnel can be trusted to stay open, so
the web layer never runs the pipeline inside a request. Each submission becomes
a *job*: a directory on disk holding the transcript, the context, the run log,
the pipeline's checkpoint cache and the finished reports, plus a ``job.json``
with its status and live progress. A single worker thread drains the queue one
job at a time — the LLM endpoint is the bottleneck, and ``max_concurrency``
already parallelizes the calls *within* a run.

Everything the UI shows is read back from these files, so closing the browser,
dropping the tunnel or restarting the pod loses nothing: an interrupted job is
resumable from its last checkpoint.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Settings, load_settings
from ..model.extraction import CATEGORIES

logger = logging.getLogger(__name__)

# Terminal states a job can be resumed from (its checkpoint survives).
RESUMABLE = frozenset({"failed", "interrupted", "cancelled"})
ACTIVE = frozenset({"queued", "running"})

# Per-job overrides the form may send, mapped to their Settings aliases.
OPTION_ALIASES = {
    "language": "MEETING_LANGUAGE",
    "max_sweeps": "MEETING_MAX_SWEEPS",
    "max_concurrency": "MEETING_MAX_CONCURRENCY",
    "chunk_chars": "MEETING_CHUNK_MAX_CHARS",
    "worker_model": "MEETING_WORKER_MODEL",
    "lead_model": "MEETING_LEAD_MODEL",
}

_LIVE_WARNINGS_KEPT = 20
# Files the job directory itself uses; an upload must never overwrite them.
_RESERVED_NAMES = frozenset({"job.json", "job.tmp", "context.txt", "run.log", "out"})


class JobCancelled(Exception):
    """Raised from the progress callback to stop a running pipeline."""


@dataclass
class PassInfo:
    number: int
    kind: str  # "initial" | "sweep"
    chunks: int
    items_after: int | None = None
    new_items: int | None = None


@dataclass
class Job:
    id: str
    transcript_name: str
    created: float
    options: dict = field(default_factory=dict)
    has_context: bool = False
    status: str = "queued"  # queued | running | done | failed | interrupted | cancelled
    resume: bool = False
    cancel_requested: bool = False
    started: float | None = None
    finished: float | None = None
    # live progress
    phase: str = ""
    segments: int = 0
    words: int = 0
    source_format: str = ""
    max_passes: int = 0
    passes: list[PassInfo] = field(default_factory=list)
    chunks_done: int = 0
    chunks_total: int = 0
    item_count: int = 0
    rescued: dict = field(default_factory=dict)
    live_warnings: list[str] = field(default_factory=list)
    # outcome
    title: str = ""
    counts: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    report_md: str = ""
    report_json: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        data = dict(data)
        data["passes"] = [PassInfo(**p) for p in data.get("passes", [])]
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


class _Progress:
    """Translates pipeline node events into job fields.

    Same event semantics as the CLI's progress view (see ``cli._ProgressView``),
    minus the terminal rendering: one entry per review pass with the running
    item count, and a phase label for the plan/synthesis/gapfill/compose steps.
    """

    def __init__(self, job: Job, save, cancel: threading.Event) -> None:
        self.job = job
        self.save = save
        self.cancel = cancel

    def _close_pass(self, item_count: object) -> None:
        if not isinstance(item_count, int):
            return
        if self.job.passes and self.job.passes[-1].items_after is None:
            last = self.job.passes[-1]
            last.items_after = item_count
            last.new_items = item_count - self.job.item_count
        self.job.item_count = item_count

    def on_event(self, node: str, payload: object) -> None:
        if self.cancel.is_set():
            raise JobCancelled()
        p = payload if isinstance(payload, dict) else {}
        job = self.job
        if node == "ingest":
            inv = p.get("inventory")
            if inv is not None:
                job.segments = len(inv.segments)
                job.words = inv.total_words
                job.source_format = inv.source_format
                job.chunks_total = len(inv.chunks)
            job.phase = "plan"
        elif node == "plan":
            job.phase = ""
        elif node == "dispatch":
            self._close_pass(p.get("item_count"))
            chunks = p.get("current_chunks")
            if chunks:
                number = p.get("rounds_done", len(job.passes) + 1)
                job.passes.append(
                    PassInfo(number=number, kind="initial" if number == 1 else "sweep", chunks=len(chunks))
                )
                job.chunks_total = len(chunks)
                job.chunks_done = 0
                job.phase = "review"
            else:
                job.phase = "reduce"
        elif node == "extract_chunk":
            job.chunks_done += 1
            if job.phase != "review":  # resumed run: the dispatch event was checkpointed away
                job.phase = "review"
        elif node == "reduce":
            job.phase = "gapfill"
        elif node == "gapfill":
            job.rescued = dict(p.get("rescued") or {})
            job.phase = "compose"
        elif node == "compose":
            job.phase = ""
        self.save(job)


class _JobLogHandler(logging.Handler):
    """Surfaces pipeline warnings (endpoint retries, chunk failures) live."""

    def __init__(self, job: Job) -> None:
        super().__init__(level=logging.WARNING)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = re.sub(r"\s+", " ", record.getMessage()).strip()[:300]
        except Exception:  # noqa: BLE001 - a bad log record must never kill the run
            return
        self.job.live_warnings = (self.job.live_warnings + [message])[-_LIVE_WARNINGS_KEPT:]


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "-", name.strip()).strip("-")


class JobManager:
    """Owns the job directories under ``data_dir/jobs`` and the worker thread."""

    def __init__(self, data_dir: Path, settings_factory=load_settings, llm_factory=None) -> None:
        self.root = Path(data_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings_factory = settings_factory
        self.llm_factory = llm_factory  # tests inject a fake LLM here
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel: dict[str, threading.Event] = {}
        self._worker: threading.Thread | None = None
        self._load_existing()

    # ------------------------------------------------------------ storage

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        with self._lock:
            target = self.job_dir(job.id) / "job.json"
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(target)

    def _load_existing(self) -> None:
        """Reload jobs after a restart; a job caught mid-run becomes resumable."""
        for path in sorted(self.root.glob("*/job.json")):
            try:
                job = Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("skipping unreadable job file %s: %s", path, exc)
                continue
            if job.status == "running":
                job.status = "interrupted"
                job.error = "The server restarted while this job was running."
                job.phase = ""
                self._save(job)
            self._jobs[job.id] = job
            if job.status == "queued":
                self._queue.put(job.id)

    # ------------------------------------------------------------ public API

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, name="job-worker", daemon=True)
            self._worker.start()

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, transcript_name: str, transcript: bytes, context: str, options: dict) -> Job:
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        name = _safe_name(Path(transcript_name).name) or "transcript.txt"
        if name in _RESERVED_NAMES or name.startswith("."):
            name = "transcript-" + name.lstrip(".")
        folder = self.job_dir(job_id)
        folder.mkdir(parents=True)
        (folder / name).write_bytes(transcript)
        if context.strip():
            (folder / "context.txt").write_text(context, encoding="utf-8")
        clean = {k: v for k, v in options.items() if k in OPTION_ALIASES and v not in (None, "")}
        job = Job(
            id=job_id,
            transcript_name=name,
            created=time.time(),
            options=clean,
            has_context=bool(context.strip()),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._queue.put(job_id)
        logger.info("job %s queued (%s, %d bytes)", job_id, name, len(transcript))
        return job

    def resume(self, job_id: str) -> Job:
        with self._lock:
            job = self._require(job_id)
            if job.status not in RESUMABLE:
                raise ValueError(f"job is {job.status}; only failed, interrupted or cancelled jobs can be resumed")
            job.status = "queued"
            job.resume = True
            job.error = ""
            job.live_warnings = []
            self._save(job)
        self._queue.put(job_id)
        logger.info("job %s re-queued for resume", job_id)
        return job

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._require(job_id)
            if job.status == "queued":
                job.status = "cancelled"
                self._save(job)
            elif job.status == "running":
                job.cancel_requested = True
                self._cancel.setdefault(job_id, threading.Event()).set()
            else:
                raise ValueError(f"job is already {job.status}")
            return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self._require(job_id)
            if job.status in ACTIVE:
                raise ValueError("cancel the job before deleting it")
            del self._jobs[job_id]
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        logger.info("job %s deleted", job_id)

    def log_tail(self, job_id: str, lines: int = 200) -> str:
        path = self.job_dir(job_id) / "run.log"
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 256 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def report_path(self, job_id: str, kind: str) -> Path | None:
        job = self._require(job_id)
        name = job.report_md if kind == "md" else job.report_json
        if not name:
            return None
        path = self.job_dir(job_id) / "out" / name
        return path if path.is_file() else None

    def settings_for(self, job: Job) -> Settings:
        overrides = {OPTION_ALIASES[k]: v for k, v in job.options.items()}
        return self.settings_factory(**overrides)

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    # ------------------------------------------------------------ worker

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                job = self.get(job_id)
                if job is None or job.status != "queued":
                    continue  # deleted or cancelled while waiting
                self._run(job)
            except Exception:  # noqa: BLE001 - the worker must survive any job
                logger.exception("job %s crashed the worker loop", job_id)
            finally:
                self._queue.task_done()

    def _run(self, job: Job) -> None:
        from ..graph.builder import run_pipeline

        folder = self.job_dir(job.id)
        cancel = self._cancel.setdefault(job.id, threading.Event())
        cancel.clear()

        pkg_logger = logging.getLogger("meeting_assistant")
        file_handler = logging.FileHandler(folder / "run.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s")
        )
        live_handler = _JobLogHandler(job)
        pkg_logger.addHandler(file_handler)
        pkg_logger.addHandler(live_handler)

        resume = job.resume
        with self._lock:
            job.status = "running"
            job.started = time.time()
            job.finished = None
            job.error = ""
            job.resume = False
            job.cancel_requested = False
            if not resume:
                job.passes, job.item_count, job.chunks_done = [], 0, 0
            self._save(job)

        try:
            settings = self.settings_for(job)
            job.max_passes = 1 + max(0, settings.max_sweeps)
            transcript = folder / job.transcript_name
            context_file = folder / "context.txt"
            logger.info(
                "run started: job=%s transcript=%s worker=%s lead=%s concurrency=%d sweeps=%d resume=%s",
                job.id, job.transcript_name, settings.resolved_worker_model(),
                settings.resolved_lead_model(), settings.max_concurrency, settings.max_sweeps, resume,
            )
            progress = _Progress(job, self._save, cancel)
            state = run_pipeline(
                transcript_text=transcript.read_text(encoding="utf-8", errors="replace"),
                settings=settings,
                source_file=str(transcript.resolve()),
                context_text=context_file.read_text(encoding="utf-8") if context_file.is_file() else "",
                llm=self.llm_factory(settings) if self.llm_factory else None,
                on_event=progress.on_event,
                resume=resume,
                cache_base=folder,
            )
            self._finish(job, state)
        except JobCancelled:
            logger.info("job %s cancelled", job.id)
            job.status = "cancelled"
        except Exception as exc:  # noqa: BLE001 - reported on the job, full trace in run.log
            logger.exception("job %s failed", job.id)
            job.status = "failed"
            job.error = re.sub(r"\s+", " ", str(exc)).strip()[:500] or type(exc).__name__
        finally:
            job.finished = time.time()
            job.phase = ""
            job.cancel_requested = False
            with self._lock:
                self._save(job)
            pkg_logger.removeHandler(file_handler)
            pkg_logger.removeHandler(live_handler)
            file_handler.close()

    def _finish(self, job: Job, state: dict) -> None:
        folder = self.job_dir(job.id) / "out"
        folder.mkdir(exist_ok=True)
        synthesis = state.get("synthesis")
        inv = state.get("inventory")
        job.title = (synthesis.title if synthesis else "") or (inv.title_hint if inv else "") or ""
        stem = _safe_name(job.title or Path(job.transcript_name).stem) or "Meeting"
        items = state.get("items", {}) or {}
        job.counts = {label: len(items.get(key) or []) for key, label, _noun in CATEGORIES}
        job.item_count = state.get("item_count", job.item_count)
        job.warnings = [re.sub(r"\s+", " ", str(w)).strip()[:300] for w in state.get("warnings", []) or []]

        report_md = state.get("report_md", "")
        if report_md:
            job.report_md = f"{stem}-Report.md"
            (folder / job.report_md).write_text(report_md, encoding="utf-8")
        report_json = state.get("report_json", "")
        if report_json:
            job.report_json = f"{stem}-Report.json"
            (folder / job.report_json).write_text(report_json, encoding="utf-8")

        if report_md:
            job.status = "done"
        else:
            job.status = "failed"
            job.error = "The pipeline finished without producing a report; see the run log."
        logger.info(
            "run finished: job=%s status=%s items=%d warnings=%d",
            job.id, job.status, job.item_count, len(job.warnings),
        )
