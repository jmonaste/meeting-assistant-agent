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
import re
from typing import TypeVar

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from .config import Settings

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
        result = self._try_structured(llm, schema, system, user, "function_calling", retry=True)
        if result is not None:
            return result
        return self._structured_via_json(schema, system, user, role)

    def _try_structured(
        self, llm, schema: type[T], system: str, user: str, method: str, retry: bool
    ) -> T | None:
        """One structured-output attempt via ``method``, with an optional single retry.

        Returns the validated instance, or ``None`` if the endpoint does not
        support the method or the model did not produce a valid instance.
        """
        try:
            runner = llm.with_structured_output(schema, method=method)
            result = runner.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            if isinstance(result, schema):
                return result
        except Exception as first_error:  # noqa: BLE001 - endpoint/validation quirks
            if not retry:
                return None
            retry_note = (
                f"\n\nYour previous attempt failed with: {first_error}."
                " Respond again, satisfying the schema exactly."
            )
            try:
                runner = llm.with_structured_output(schema, method=method)
                result = runner.invoke(
                    [SystemMessage(content=system), HumanMessage(content=user + retry_note)]
                )
                if isinstance(result, schema):
                    return result
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
        reply = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
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
            reply = llm.invoke(messages)
            messages.append(reply)
            calls = getattr(reply, "tool_calls", None) or []
            if not calls:
                text = reply.content if isinstance(reply.content, str) else str(reply.content)
                return text, messages
            for call in calls:
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
        messages.append(HumanMessage(content="Tool budget exhausted. Answer now with what you have."))
        reply = self._chat_model(role).invoke(messages)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        messages.append(AIMessage(content=text))
        return text, messages
