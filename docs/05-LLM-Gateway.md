# 05 — LLM gateway

Every call to the model goes through `llm.py:MeetingLLM`. No other module talks
to the endpoint. This keeps all the robustness needed for less-reliable local
models in one tested place, and lets tests inject a fake by subclassing.

## Two roles, one endpoint

`MeetingLLM` lazily builds two `ChatOpenAI` clients over the same
OpenAI-compatible endpoint:

- `worker` — the model named by `MEETING_WORKER_MODEL`, used for the many
  per-chunk map/sweep extractions.
- `lead` — `MEETING_LEAD_MODEL`, used for the few harder calls: planning,
  synthesis, and the gap-fill agent.

Both are configured with the shared `temperature`, `max_tokens`, timeout and TLS
settings. `verify_ssl=false` exists only for trusted local endpoints with
self-signed certificates.

## Transient failures: retry with backoff

Every invoke goes through `_invoke_with_retry()`. An error classified as
transient by `_is_retryable()` — 429 rate limits, 502/503/504 gateway errors,
timeouts, connection errors, matched by exception type name or message marker
across the whole `__cause__` chain — is retried up to `MEETING_LLM_RETRIES`
times with exponential backoff (base `MEETING_LLM_RETRY_BASE_DELAY`, doubling,
capped at 60 s). Non-transient errors propagate immediately.

When the retries are exhausted, the gateway raises `MeetingLLMUnavailable`
(a `MeetingLLMError` subclass) with a cleaned one-line summary — gateways
answer with whole HTML error pages, which are stripped. This distinction
matters for the ladder below: a schema/validation failure is worth retrying
with a more permissive output method, but a dead transport is not, so
`MeetingLLMUnavailable` aborts the ladder instead of burning two more slow,
failing calls per chunk.

## `structured()` — a schema-out retry ladder

Local models honor structured output unevenly, so `structured()` tries three
increasingly permissive strategies and returns the first validated instance:

1. **`json_schema`** — guided decoding constrained to the schema (capable
   endpoints like vLLM enforce this). No retry.
2. **`function_calling`** — tool-calling structured output, with one retry that
   feeds the validation error back to the model.
3. **JSON fallback** (`_structured_via_json`) — a plain completion asked to emit
   only JSON. The reply is cleaned (a `<think>...</think>` reasoning block from
   models like gpt-oss is stripped), every balanced `{...}` object is extracted
   (`_iter_json_objects`, which ignores braces inside strings), and the first
   candidate — or a one-level-nested dict — that validates against the schema is
   returned. Local models often wrap the object under a key named after the
   schema or emit several objects; this recovers the intended payload.

If nothing validates, it raises `MeetingLLMError`. Every pipeline node catches
that and degrades, so a single bad call never fails the run.

## `tool_loop()` — bounded think-act-observe

Used by the gap-fill agent. It binds the read-only transcript tools, then loops
up to `agent_max_iterations` times: invoke the model, run any tool calls it
requested, feed the results back. Unknown tools and tool exceptions are turned
into error observations rather than crashes, so the model can recover. When the
model stops requesting tools it returns its final text plus the full transcript.
If the budget is exhausted it makes one final, tool-free call for an answer.

The loop returns the text; the *evidence* (which segments were actually read) is
captured separately by the `recorder` list passed into `build_transcript_tools`,
so citations come from real reads, not from the model's claims.

## Why the gateway matters

Centralizing model contact means:

- **One place to harden.** Every quirk of local models — truncated JSON, wrapped
  objects, reasoning preambles, flaky tool calling — is handled once.
- **Deterministic tests.** `tests/conftest.py:FakeLLM` subclasses `MeetingLLM`
  and overrides `structured()` and `tool_loop()`, so the entire graph runs with
  no endpoint and fully predictable outputs (chapter 06).
- **A single cost/latency surface.** Concurrency, timeouts and token budgets are
  all configured in one object.
