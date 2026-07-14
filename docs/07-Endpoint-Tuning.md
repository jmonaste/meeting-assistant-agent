# 07 — Endpoint tuning: from a safe baseline to full speed

A practical guide for running the agent against a slow or rate-limited local
endpoint (a single big model such as gpt-oss-120B behind a gateway), written
after a real run that produced `504 Gateway Time-out` and
`429 Rate limit exceeded` warnings. The strategy is always the same: **start
from a baseline that cannot overload the endpoint, measure with the log file,
then increase one knob at a time.**

## Understand the two errors first

They come from different places and need different fixes:

| Error | Who produces it | What it means |
|-------|-----------------|---------------|
| `504 Gateway Time-out` | The **proxy/gateway in front of the model** | The model was still generating when the gateway's own clock ran out. The client-side `MEETING_REQUEST_TIMEOUT` never got a say — the gateway hung up first. |
| `429 Rate limit exceeded` | The **endpoint/gateway rate limiter** | Requests are arriving faster than the server accepts. With a single big model, this is almost always the map phase's parallel calls piling up. |

The agent retries both with exponential backoff (2s, 4s, 8s, ... capped at
60s), so isolated occurrences are absorbed silently. *Sustained* occurrences
mean the settings ask more of the endpoint than it can give — backoff cannot
fix that, only tuning can.

## Step 0 — the safe baseline

```bash
meeting-assistant process meeting.txt \
  --max-concurrency 1 \
  --chunk-chars 5000 \
  --log-file tuning-run.log \
  -o out/
```

- `--max-concurrency 1` — exactly one request in flight. A single-instance
  120B model serves one generation at a time anyway; parallel requests just
  queue inside the server until the gateway times them out or rate-limits.
- `--chunk-chars 5000` — smaller chunks mean shorter generations per call, so
  each call finishes well inside the gateway's window. (More chunks per pass,
  but each one is fast and safe.)
- `--log-file` — the audit trail you will tune from.

And configure the **gateway's own timeout to at least ~5 minutes per request**
(match or exceed `MEETING_REQUEST_TIMEOUT`, default 300s). This is set on the
gateway, not in this tool. Common places:

| Gateway | Setting |
|---------|---------|
| nginx | `proxy_read_timeout 300s;` (and `proxy_send_timeout`) |
| Traefik | `--serversTransport.forwardingTimeouts.responseHeaderTimeout=300s` |
| LiteLLM proxy | `request_timeout: 300` in the proxy config |
| HAProxy | `timeout server 300s` |

If you cannot change the gateway, work backwards instead: measure your real
per-call durations (next step) and shrink `--chunk-chars` until every call
fits inside the gateway's existing window with ~2x headroom.

## Step 1 — measure the baseline

Run once on a representative transcript, then read the log:

```bash
# how long does a typical call actually take?
grep "LLM call ok" tuning-run.log

# did anything still need retrying?
grep -c "retrying in" tuning-run.log

# did any chunk fail outright?
grep "failed after" tuning-run.log
```

You want to see, for example:

```
... meeting_assistant.llm: LLM call ok in 41.3s (attempt 1/5)
... meeting_assistant.llm: LLM call ok in 58.9s (attempt 1/5)
```

Two numbers matter:

- **Slowest call duration.** Your gateway timeout should be at least ~2x this
  value. If the slowest call is 150s and the gateway allows 300s, you have
  healthy headroom. If it is 280s, you are one long meeting away from 504s.
- **Retry count.** A clean baseline run should show zero (or nearly zero)
  retries. If the baseline itself retries, fix the gateway timeout or shrink
  chunks further before touching anything else.

Tips that make tuning runs cheap:

- Use `--max-sweeps 0` while tuning — one full pass is enough to measure call
  behavior, and it cuts the run to a fraction of the calls.
- Use `--no-cache` so every call really hits the endpoint (otherwise the
  second run replays cached extractions and measures nothing).
- Keep using the **same transcript** across runs so timings are comparable.

## Step 2 — increase one knob at a time

Change a single setting per run, in this order, and re-check the same three
grep commands after each run.

**2a. Concurrency: 1 → 2 → 4.** Only worth raising if your endpoint can truly
serve parallel requests (vLLM with continuous batching: yes; a llama.cpp-style
single-stream server: no — stay at 1).

Accept the step if, compared to the previous run:

- retries stayed at (near) zero — a handful of isolated `429` retries that
  backoff absorbs is acceptable; a retry on most calls is not;
- no `504` appeared;
- the run got meaningfully faster (check the pass timings in the progress
  display, or the first/last timestamps in the log). If doubling concurrency
  did not speed the run up, the server was already saturated — go back down,
  you are only adding queueing risk.

**2b. Chunk size: 5000 → 7000 → 9000.** Bigger chunks mean fewer, longer
calls (less per-call overhead, more context per extraction). Accept the step
if the slowest `LLM call ok` duration still leaves ~2x headroom against the
gateway timeout and no truncation warnings appear. If extractions start
coming back truncated (validation retries in the log), also raise
`MEETING_MAX_TOKENS`.

**Rollback rule:** if a step produces sustained retries, any 504, or the
"endpoint looks unhealthy" stop, go back to the previous value and stay
there. The knobs interact — high concurrency multiplies the effective load of
big chunks — which is why you change only one at a time.

## Step 3 — freeze the result in .env

When you have found the stable settings, persist them so every future run
uses them without flags:

```ini
MEETING_MAX_CONCURRENCY=2
MEETING_CHUNK_MAX_CHARS=7000
MEETING_REQUEST_TIMEOUT=300
MEETING_LOG_FILE=
```

## Symptom → action quick reference

| Symptom | Action |
|---------|--------|
| `504 Gateway Time-out` | Raise the **gateway's** timeout; if you can't, lower `--chunk-chars`. Client timeout alone won't fix it. |
| Sustained `429` on most calls | Lower `--max-concurrency` (1 for a single big model). |
| Isolated `429`/`504`, run completes | Nothing — backoff absorbed it. Check the log to confirm attempts stayed low. |
| "endpoint looks unhealthy" stop | The endpoint failed half a pass even after retries. Fix it (or the settings), then `--resume` — completed chunks are not re-paid. |
| Extractions truncated / JSON validation retries | Raise `MEETING_MAX_TOKENS`; consider smaller chunks for a reasoning model that thinks at length. |
| Doubling concurrency didn't speed anything up | The server is saturated — go back down. |
