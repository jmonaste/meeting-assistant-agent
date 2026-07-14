"""Typed configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Agent configuration.

    Values come from environment variables or a local ``.env`` file. CLI flags
    can override individual fields at runtime (see ``cli.py``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM endpoint (OpenAI-compatible) ---
    openai_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL of the OpenAI-compatible endpoint (must end in /v1).",
    )
    openai_api_key: str = Field(
        default="not-needed",
        description="API key; any non-empty string if the endpoint ignores it.",
    )
    worker_model: str = Field(
        default="",
        alias="MEETING_WORKER_MODEL",
        description="Model for the per-chunk map phase (many small parallel calls).",
    )
    lead_model: str = Field(
        default="",
        alias="MEETING_LEAD_MODEL",
        description="Model for planning, synthesis and coverage sweeps (fewer, harder calls).",
    )
    temperature: float = Field(
        default=0.0,
        alias="MEETING_TEMPERATURE",
        description="Sampling temperature (keep 0 for reproducible reports).",
    )
    verify_ssl: bool = Field(
        default=True,
        alias="MEETING_VERIFY_SSL",
        description="Verify the endpoint's TLS certificate. Set false only for a "
        "trusted local/corporate endpoint with a self-signed certificate.",
    )
    request_timeout: float = Field(
        default=300.0,
        alias="MEETING_REQUEST_TIMEOUT",
        description="HTTP timeout in seconds for each LLM call. Large local reasoning "
        "models can take minutes on a full chunk; the proxy/gateway in front of the "
        "model must allow at least this long too, or it will return 504s first.",
    )
    llm_retries: int = Field(
        default=4,
        alias="MEETING_LLM_RETRIES",
        description="Retries per LLM call on transient endpoint errors (429 rate "
        "limits, 502/503/504 gateway errors, timeouts), with exponential backoff.",
    )
    llm_retry_base_delay: float = Field(
        default=2.0,
        alias="MEETING_LLM_RETRY_BASE_DELAY",
        description="Initial backoff delay in seconds; doubles on each retry, capped at 60s.",
    )
    max_tokens: int = Field(
        default=8192,
        alias="MEETING_MAX_TOKENS",
        description="Max completion tokens per call. Reasoning models (e.g. gpt-oss) "
        "spend tokens thinking before answering; too low a budget truncates the JSON "
        "output and validation fails. Raise it if extractions still come back truncated.",
    )

    # --- Pipeline limits ---
    max_concurrency: int = Field(
        default=4,
        alias="MEETING_MAX_CONCURRENCY",
        description="Max parallel LLM calls during the map and sweep phases.",
    )
    chunk_max_chars: int = Field(
        default=9_000,
        alias="MEETING_CHUNK_MAX_CHARS",
        description="Character budget for one transcript chunk handed to the worker model.",
    )
    chunk_overlap_chars: int = Field(
        default=800,
        alias="MEETING_CHUNK_OVERLAP_CHARS",
        description="Characters of overlap between adjacent chunks so an item that "
        "straddles a chunk boundary is still seen whole by at least one chunk.",
    )
    max_sweeps: int = Field(
        default=3,
        alias="MEETING_MAX_SWEEPS",
        description="Extra full-transcript coverage sweeps after the first pass. Each "
        "sweep re-reads the WHOLE transcript looking for items earlier passes missed; "
        "sweeping stops early once a full round finds nothing new.",
    )
    agent_max_iterations: int = Field(
        default=8,
        alias="MEETING_AGENT_MAX_ITERATIONS",
        description="Tool-loop cap for the evidence/clarification agents.",
    )
    use_cache: bool = Field(
        default=True,
        alias="MEETING_USE_CACHE",
        description="Reuse cached per-chunk extractions when the chunk text has not changed.",
    )

    def resolved_worker_model(self) -> str:
        return self.worker_model or self.lead_model or "gpt-oss"

    def resolved_lead_model(self) -> str:
        return self.lead_model or self.worker_model or "gpt-oss"


def load_settings(**overrides: object) -> Settings:
    """Load settings, applying any non-None CLI overrides on top of env/.env."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**clean)


def cache_dir(base: Path) -> Path:
    """Per-run cache directory (chunk extractions, checkpoints, run metadata).

    ``base`` is the directory holding the transcript file, so re-processing the
    same transcript reuses its cache.
    """
    return base / ".meeting_cache"
