"""Disk cache for per-chunk extractions.

Keyed by (chunk content hash, model, prompt version): re-running the agent on a
transcript only re-extracts chunks whose text actually changed. Only the
first-pass full extraction is cached; coverage sweeps depend on what earlier
passes already found, so they are always recomputed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ..model.extraction import ChunkExtraction

PROMPT_VERSION = "1"


class ExtractionCache:
    def __init__(self, path: Path, model: str, enabled: bool = True) -> None:
        self.path = path
        self.model = model
        self.enabled = enabled
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if enabled and path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def _key(self, content_hash: str) -> str:
        return f"{content_hash}:{self.model}:{PROMPT_VERSION}"

    def get(self, content_hash: str) -> ChunkExtraction | None:
        if not self.enabled:
            return None
        with self._lock:
            raw = self._data.get(self._key(content_hash))
        if raw is None:
            return None
        try:
            return ChunkExtraction.model_validate(raw)
        except Exception:  # noqa: BLE001 - stale schema
            return None

    def put(self, content_hash: str, extraction: ChunkExtraction) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[self._key(content_hash)] = extraction.model_dump()

    def save(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._data), encoding="utf-8")
            except OSError:
                pass
