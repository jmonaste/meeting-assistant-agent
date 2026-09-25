"""Settings edited from the web UI, persisted next to the jobs.

Precedence for every run, highest first:

    per-analysis options (the New analysis form)
    > values saved from the Settings dialog (this module, ``settings.json``)
    > environment variables (the ConfigMap/Secret in OpenShift, ``.env`` locally)
    > code defaults (``config.Settings``)

Only the endpoint, the models and the pipeline knobs are editable here. Network
plumbing that must exist before the process starts — proxy variables, the
corporate CA bundle, ``LANGCHAIN_OPENAI_TCP_KEEPALIVE`` — stays in the
deployment configuration.

The API key is write-only through the API: it is stored in ``settings.json``
(mode 0600, on the pod's volume) and never sent back to the browser.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ..config import Settings


@dataclass(frozen=True)
class EditableField:
    attr: str  # Settings attribute
    label: str
    group: str
    kind: str  # text | secret | bool | int | float | language
    minimum: float | None = None

    @property
    def init_key(self) -> str:
        """Keyword ``Settings(...)`` accepts for this field (its alias, if any)."""
        return Settings.model_fields[self.attr].alias or self.attr

    @property
    def env(self) -> str:
        """Environment variable name, also the key used in settings.json and the API."""
        return self.init_key.upper()

    @property
    def help(self) -> str:
        return Settings.model_fields[self.attr].description or ""


FIELDS: tuple[EditableField, ...] = (
    EditableField("openai_base_url", "Base URL", "Endpoint", "text"),
    EditableField("openai_api_key", "API key", "Endpoint", "secret"),
    EditableField("verify_ssl", "Verify TLS certificate", "Endpoint", "bool"),
    EditableField("worker_model", "Worker model", "Models", "text"),
    EditableField("lead_model", "Lead model", "Models", "text"),
    EditableField("temperature", "Temperature", "Models", "float", minimum=0),
    EditableField("max_tokens", "Max completion tokens", "Models", "int", minimum=256),
    EditableField("language", "Report language", "Review", "language"),
    EditableField("max_sweeps", "Extra review passes", "Review", "int", minimum=0),
    EditableField("agent_max_iterations", "Gap-fill tool-loop cap", "Review", "int", minimum=1),
    EditableField("use_cache", "Reuse cached chunk extractions", "Review", "bool"),
    EditableField("max_concurrency", "Parallel LLM calls", "Endpoint tuning", "int", minimum=1),
    EditableField("chunk_max_chars", "Chunk size (chars)", "Endpoint tuning", "int", minimum=500),
    EditableField("chunk_overlap_chars", "Chunk overlap (chars)", "Endpoint tuning", "int", minimum=0),
    EditableField("request_timeout", "Request timeout (s)", "Endpoint tuning", "float", minimum=1),
    EditableField("llm_retries", "Retries per call", "Endpoint tuning", "int", minimum=0),
    EditableField("llm_retry_base_delay", "Retry base delay (s)", "Endpoint tuning", "float", minimum=0),
)
BY_ENV = {f.env: f for f in FIELDS}


class SettingsError(ValueError):
    """Rejected settings; ``errors`` maps each env key to its message."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def _mask(secret: str) -> str:
    return "…" + secret[-4:] if len(secret) > 8 else "set"


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._values: dict[str, object] = {}
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._values = {k: v for k, v in raw.items() if k in BY_ENV}
            except (OSError, ValueError):
                self._values = {}

    def saved(self) -> dict[str, object]:
        with self._lock:
            return dict(self._values)

    def init_overrides(self, values: dict[str, object] | None = None) -> dict[str, object]:
        """Saved values (or ``values``) as keyword arguments for ``load_settings``."""
        source = self.saved() if values is None else values
        return {BY_ENV[k].init_key: v for k, v in source.items() if k in BY_ENV}

    def merged(self, changes: dict[str, object]) -> dict[str, object]:
        """Saved values with ``changes`` applied: ``None`` or ``""`` removes a value."""
        values = self.saved()
        for key, value in changes.items():
            if key not in BY_ENV:
                raise SettingsError({key: "not an editable setting"})
            if value is None or (isinstance(value, str) and not value.strip()):
                values.pop(key, None)
            else:
                values[key] = value.strip() if isinstance(value, str) else value
        return values

    def validate(self, values: dict[str, object], base_factory) -> dict[str, object]:
        """Coerce ``values`` through ``Settings``; returns them typed, or raises."""
        try:
            settings = base_factory(**self.init_overrides(values))
        except ValidationError as exc:
            errors = {}
            for err in exc.errors():
                loc = str(err["loc"][0]) if err.get("loc") else "?"
                env = next((f.env for f in FIELDS if loc in (f.init_key, f.attr)), loc)
                errors[env] = err["msg"]
            raise SettingsError(errors) from None
        typed = {key: getattr(settings, BY_ENV[key].attr) for key in values}
        errors = {
            key: f"must be at least {BY_ENV[key].minimum:g}"
            for key, value in typed.items()
            if BY_ENV[key].minimum is not None and isinstance(value, (int, float)) and value < BY_ENV[key].minimum
        }
        if errors:
            raise SettingsError(errors)
        return typed

    def update(self, changes: dict[str, object], base_factory) -> None:
        typed = self.validate(self.merged(changes), base_factory)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(typed, handle, indent=1)
            tmp.replace(self.path)
            self._values = typed

    def describe(self, base: Settings) -> list[dict]:
        """Every editable field with its effective value and where it comes from.

        ``base`` is the configuration *without* the UI values (environment +
        defaults), so ``fallback`` is what applies if the saved value is reset.
        """
        saved = self.saved()
        out = []
        for f in FIELDS:
            default = Settings.model_fields[f.attr].default
            fallback = getattr(base, f.attr)
            source = "ui" if f.env in saved else ("env" if fallback != default else "default")
            row = {
                "key": f.env, "label": f.label, "group": f.group, "kind": f.kind,
                "help": f.help, "minimum": f.minimum, "source": source,
            }
            if f.kind == "secret":
                current = str(saved.get(f.env, fallback) or "")
                row["configured"] = bool(current) and current != default
                row["masked"] = _mask(current) if row["configured"] else ""
                row["fallback_configured"] = bool(fallback) and fallback != default
            else:
                row["value"] = saved.get(f.env, fallback)
                row["fallback"] = fallback
            out.append(row)
        return out
