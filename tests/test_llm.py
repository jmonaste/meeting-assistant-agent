"""Retry/backoff behavior of the LLM gateway on transient endpoint errors."""

from __future__ import annotations

import pytest

import meeting_assistant.llm as llm_mod
from meeting_assistant.config import load_settings
from meeting_assistant.llm import (
    MeetingLLM,
    MeetingLLMUnavailable,
    _error_summary,
    _is_retryable,
)
from meeting_assistant.model.extraction import MeetingPlan


class _Runner:
    """Stub runner: raises ``exc`` for the first ``failures`` invokes, then returns ``result``."""

    def __init__(self, failures: int, exc: Exception, result="ok") -> None:
        self.failures = failures
        self.exc = exc
        self.result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return self.result


@pytest.fixture
def fast_llm(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)  # no real waiting
    settings = load_settings(MEETING_LLM_RETRIES=3, MEETING_LLM_RETRY_BASE_DELAY=0.01)
    return MeetingLLM(settings)


def test_is_retryable_matches_gateway_and_rate_limit_errors():
    assert _is_retryable(RuntimeError("Error code: 429 - {'detail': 'Rate limit exceeded'}"))
    assert _is_retryable(RuntimeError("<html><body><h1>504 Gateway Time-out</h1></body></html>"))
    assert _is_retryable(RuntimeError("The read operation timed out"))
    assert not _is_retryable(ValueError("1 validation error for ChunkExtraction"))


def test_is_retryable_walks_cause_chain():
    inner = RuntimeError("504 Gateway Time-out")
    outer = RuntimeError("call failed")
    outer.__cause__ = inner
    assert _is_retryable(outer)


def test_error_summary_strips_html_pages():
    text = _error_summary(RuntimeError("<html><body><h1>504 Gateway Time-out</h1>\nslow</body></html>"))
    assert "<html>" not in text
    assert "504" in text


def test_invoke_retries_then_succeeds(fast_llm):
    runner = _Runner(2, RuntimeError("429 rate limit"))
    assert fast_llm._invoke_with_retry(runner, []) == "ok"
    assert runner.calls == 3


def test_invoke_raises_unavailable_after_exhaustion(fast_llm):
    runner = _Runner(99, RuntimeError("504 Gateway Time-out"))
    with pytest.raises(MeetingLLMUnavailable):
        fast_llm._invoke_with_retry(runner, [])
    assert runner.calls == 4  # 1 attempt + 3 retries


def test_non_retryable_errors_propagate_immediately(fast_llm):
    runner = _Runner(99, ValueError("validation error"))
    with pytest.raises(ValueError):
        fast_llm._invoke_with_retry(runner, [])
    assert runner.calls == 1


class _StructStub:
    """Chat-model stub whose every structured runner fails with the given error."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.methods: list[str] = []

    def with_structured_output(self, schema, method):
        self.methods.append(method)
        return _Runner(999, self.exc)

    def invoke(self, messages):  # the plain-JSON fallback path
        raise self.exc


def test_structured_aborts_method_ladder_when_endpoint_is_down(fast_llm, monkeypatch):
    stub = _StructStub(RuntimeError("504 Gateway Time-out"))
    monkeypatch.setattr(fast_llm, "_chat_model", lambda role: stub)
    with pytest.raises(MeetingLLMUnavailable):
        fast_llm.structured(MeetingPlan, "sys", "user")
    # A dead transport must NOT fall through to function_calling / JSON fallback.
    assert stub.methods == ["json_schema"]
