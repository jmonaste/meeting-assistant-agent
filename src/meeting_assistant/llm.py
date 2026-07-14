"""Single gateway for every LLM call: structured output and tool loops.

Local models behind OpenAI-compatible endpoints are less reliable than hosted
frontier models, so all robustness lives here in one place:

* ``structured()`` asks for a Pydantic schema via tool calling, retries once
  with the validation error, then falls back to parsing JSON out of a plain
  completion.
* ``tool_loop()`` runs a bounded think-act-observe loop and returns the final
  answer plus the transcript of what happened.

Tests inject a fake subclass, so no other module talks to the model directly.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from .config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Reasoning models (e.g. gpt-oss) may wrap their answer in a <think>…</think>
# block; strip it before hunting for the JSON object.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _iter_json_objects(text: str):
    """Yield every balanced ``{...}`` substring, ignoring braces inside strings."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]


def _candidate_payloads(data: object):
    """The object itself, plus its one-level-nested dicts.

    Local models often wrap the real object under a key named after the schema
    (``{"ChunkExtraction": {...}}``) or echo the schema's ``{"properties": {...}}``;
    trying the nested dicts recovers the intended payload.
    """
    if isinstance(data, dict):
        yield data
        for value in data.values():
            if isinstance(value, dict):
                yield value


class MeetingLLMError(RuntimeError):
    """Raised when the model could not produce usable output after retries."""


class MeetingLLMUnavailable(MeetingLLMError):
    """The endpoint kept failing with transient errors (rate limit, gateway timeout).

    Distinct from a schema/validation failure: when the transport itself is
    down, trying a more permissive structured-output method is pointless, so
    ``structured()`` aborts its method ladder instead of burning more calls.
    """


# Transient conditions worth retrying: rate limits, gateway errors, timeouts.
_RETRYABLE_TYPES = frozenset({
    "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "ServiceUnavailableError",
    "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException",
})
_RETRYABLE_MARKERS = (
    "429", "rate limit", "too many requests",
    "502", "503", "504", "bad gateway", "service unavailable",
    "gateway time", "timed out", "timeout", "connection error", "overloaded",
)

_TAGISH_RE = re.compile(r"<[^>]{1,120}>")


def _is_retryable(exc: BaseException) -> bool:
    """True when the error looks transient (walks the exception cause chain)."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _RETRYABLE_TYPES:
            return True
        text = str(cur).lower()
        if any(marker in text for marker in _RETRYABLE_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _error_summary(exc: BaseException | None) -> str:
    """One clean line: gateways return whole HTML error pages, strip that."""
    text = _TAGISH_RE.sub(" ", str(exc))
    return re.sub(r"\s+", " ", text).strip()[:200]


class MeetingLLM:
    """Two model slots (worker/lead) over one OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[str, object] = {}

    def _chat_model(self, role: str):
        if role not in self._models:
            from langchain_openai import ChatOpenAI

            model_name = (
                self.settings.resolved_worker_model()
                if role == "worker"
                else self.settings.resolved_lead_model()
            )
            http_client = httpx.Client(
                verify=self.settings.verify_ssl,
                timeout=self.settings.request_timeout,
            )
            self._models[role] = ChatOpenAI(
                model=model_name,
                base_url=self.settings.openai_base_url,
                api_key=self.settings.openai_api_key,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                http_client=http_client,
            )
        return self._models[role]

    # ------------------------------------------------------------------ #

    def _invoke_with_retry(self, runner, messages):
        """Invoke with exponential backoff on transient endpoint errors.

        Local endpoints under load answer with 429s and gateways in front of
        slow models answer with 504s; both usually succeed on a later attempt
        once pressure drops. Non-transient errors propagate immediately.
        """
        retries = max(0, self.settings.llm_retries)
        delay = max(0.1, self.settings.llm_retry_base_delay)
        last: BaseException | None = None
        for attempt in range(retries + 1):
            started = time.perf_counter()
            try:
                result = runner.invoke(messages)
                logger.debug(
                    "LLM call ok in %.1fs (attempt %d/%d)",
                    time.perf_counter() - started, attempt + 1, retries + 1,
                )
                return result
            except Exception as exc:  # noqa: BLE001 - classified below
                if not _is_retryable(exc):
                    logger.debug(
                        "LLM call failed with non-retryable error after %.1fs: %s",
                        time.perf_counter() - started, _error_summary(exc),
                    )
                    raise
                last = exc
                if attempt < retries:
                    logger.warning(
                        "transient endpoint error (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1, retries + 1, delay, _error_summary(exc),
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
        logger.error("giving up after %d attempts: %s", retries + 1, _error_summary(last))
        raise MeetingLLMUnavailable(
            f"endpoint still failing after {retries + 1} attempts: {_error_summary(last)}"
        ) from last

    # ------------------------------------------------------------------ #

    def structured(self, schema: type[T], system: str, user: str, role: str = "worker") -> T:
        """Return an instance of ``schema`` produced by the model.

        Local models vary in how well they honor structured output, so we try
        increasingly permissive strategies: guided JSON (``json_schema``, which
        constrains decoding to the schema on capable endpoints like vLLM), then
        tool calling, then a plain completion whose JSON we parse ourselves.
        """
        llm = self._chat_model(role)
        result = self._try_structured(llm, schema, system, user, "json_schema", retry=False)
        if result is not None:
            return result
        logger.debug("structured %s: json_schema failed, trying function_calling", schema.__name__)
        result = self._try_structured(llm, schema, system, user, "function_calling", retry=True)
        if result is not None:
            return result
        logger.debug("structured %s: function_calling failed, parsing plain JSON", schema.__name__)
        return self._structured_via_json(schema, system, user, role)

    def _try_structured(
        self, llm, schema: type[T], system: str, user: str, method: str, retry: bool
    ) -> T | None:
        """One structured-output attempt via ``method``, with an optional single retry.

        Returns the validated instance, or ``None`` if the endpoint does not
        support the method or the model did not produce a valid instance.
        ``MeetingLLMUnavailable`` (transport dead after backoff) propagates:
        falling through to a more permissive method cannot fix a dead endpoint
        and would just burn more slow, failing calls.
        """
        try:
            runner = llm.with_structured_output(schema, method=method)
            result = self._invoke_with_retry(
                runner, [SystemMessage(content=system), HumanMessage(content=user)]
            )
            if isinstance(result, schema):
                return result
        except MeetingLLMUnavailable:
            raise
        except Exception as first_error:  # noqa: BLE001 - endpoint/validation quirks
            if not retry:
                return None
            retry_note = (
                f"\n\nYour previous attempt failed with: {first_error}."
                " Respond again, satisfying the schema exactly."
            )
            try:
                runner = llm.with_structured_output(schema, method=method)
                result = self._invoke_with_retry(
                    runner,
                    [SystemMessage(content=system), HumanMessage(content=user + retry_note)],
                )
                if isinstance(result, schema):
                    return result
            except MeetingLLMUnavailable:
                raise
            except Exception:  # noqa: BLE001
                pass
        return None

    def _structured_via_json(self, schema: type[T], system: str, user: str, role: str) -> T:
        """Fallback: plain completion asked to emit JSON matching the schema.

        Local models are inconsistent here: they wrap the object under a key
        named after the schema, prepend a reasoning block, or emit several
        objects. So we strip reasoning, collect every balanced JSON object, and
        accept the first candidate (or one-level-nested dict) that validates.
        """
        llm = self._chat_model(role)
        prompt = (
            f"{user}\n\nRespond with ONLY a JSON object matching this schema"
            f" (no prose, no code fences):\n{json.dumps(schema.model_json_schema(), indent=1)}"
        )
        reply = self._invoke_with_retry(
            llm, [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        text = _THINK_RE.sub("", text)

        first_error: ValidationError | None = None
        for block in _iter_json_objects(text):
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue
            for payload in _candidate_payloads(data):
                try:
                    return schema.model_validate(payload)
                except ValidationError as exc:
                    first_error = first_error or exc
        if first_error is not None:
            raise MeetingLLMError(f"invalid {schema.__name__} JSON: {first_error}") from first_error
        raise MeetingLLMError(f"model returned no JSON for {schema.__name__}")

    # ------------------------------------------------------------------ #

    def tool_loop(
        self,
        system: str,
        user: str,
        tools: list,
        role: str = "lead",
        max_iterations: int | None = None,
    ) -> tuple[str, list[BaseMessage]]:
        """Bounded think-act-observe loop; returns (final_text, transcript)."""
        limit = max_iterations or self.settings.agent_max_iterations
        llm = self._chat_model(role).bind_tools(tools)
        by_name = {t.name: t for t in tools}
        messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=user)]

        for _ in range(limit):
            reply = self._invoke_with_retry(llm, messages)
            messages.append(reply)
            calls = getattr(reply, "tool_calls", None) or []
            if not calls:
                text = reply.content if isinstance(reply.content, str) else str(reply.content)
                return text, messages
            for call in calls:
                logger.debug("tool call: %s(%s)", call["name"], str(call.get("args", {}))[:200])
                tool = by_name.get(call["name"])
                if tool is None:
                    output = f"ERROR: unknown tool {call['name']!r}"
                else:
                    try:
                        output = str(tool.invoke(call.get("args", {})))
                    except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
                        output = f"ERROR: {exc}"
                messages.append(ToolMessage(content=output, tool_call_id=call["id"], name=call["name"]))

        # Out of budget: ask for a final answer without tools.
        logger.debug("tool loop budget exhausted after %d iterations", limit)
        messages.append(HumanMessage(content="Tool budget exhausted. Answer now with what you have."))
        reply = self._invoke_with_retry(self._chat_model(role), messages)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        messages.append(AIMessage(content=text))
        return text, messages
