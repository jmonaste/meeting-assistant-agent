# meeting-assistant-agent

A LangGraph agent that reads a **meeting transcript as many times as it takes**
and extracts everything worth keeping, using a **local LLM** behind an
OpenAI-compatible endpoint (gemma, gpt-oss, ...). It produces:

1. **`<Meeting>-Report.md`** — a professional, Obsidian-ready report: executive
   summary, minutes, decisions, action items, open questions, risks, key facts,
   dates, people, quotes, disagreements and a glossary — every item citing the
   transcript segments it came from.
2. **`<Meeting>-Report.json`** — the same extraction as structured data, ready
   to pipe into a tracker, a database or another tool.

Built on the same stack and philosophy as
[rpa-code-guardian-agent](https://github.com/jmonaste/rpa-code-guardian-agent):
LangGraph + `ChatOpenAI(base_url=...)` + typed settings + typer CLI, designed to
run on restricted corporate machines against a local model.

> **Studying the implementation?** [`docs/`](docs/README.md) contains full
> engineering documentation: the requirements and design decisions, the pipeline
> architecture, the transcript ingestion deep dive, the multi-pass coverage
> strategy, the LLM gateway, and the testing approach.

## Why multiple passes

Handing a whole transcript to an LLM in one shot reliably *loses detail*: the
model summarizes the loudest parts and quietly drops the throwaway action item,
the number said once, the question nobody answered. This agent is built around
that failure. It never relies on a single read:

```
transcript (+ optional context)
   |
   v
 ingest (deterministic, no LLM)
   detect format (VTT / SRT / labeled / plain), split into speaker turns
   with stable segment ids (S0, S1, ...), pack into overlapping chunks,
   run a deterministic cue scan (dates, actions, decisions, risks, numbers)
   |
   v
 plan (LLM)          what kind of meeting this is, a title, what to watch for
   |
   v
 REVIEW LOOP  (this is the point of the tool)
   round 0 - map (LLM, parallel):  extract EVERYTHING from each chunk in an
             isolated context; results cached on disk by chunk hash
   round 1..N - sweep (LLM, parallel):  re-read the WHOLE transcript, each pass
             told what is already found and asked ONLY for what was missed;
             stops as soon as a full round finds nothing new (loop-until-dry)
   |
   v
 reduce (LLM)        writes the prose (summary, minutes, sentiment, next steps)
   |                 from the consolidated items only; may flag a category as
   |                 still-incomplete to force one more targeted re-read
   v
 gapfill (LLM agent, bounded tool loop)
   for any category the model flagged or the cue scan shows is under-covered,
   an agent re-searches the transcript with read-only tools and rescues the
   missed items; evidence is recorded from actual reads, not model claims
   |
   v
 compose (deterministic)
   Markdown skeleton, tables, and JSON export built from the items; hallucinated
   segment citations are dropped; emoji and marketing language stripped by lint
```

De-duplication across all those passes is deterministic (by a normalized key per
item), so re-reading the same content many times converges instead of piling up
duplicates. A SQLite checkpointer under `.meeting_cache/` records every step, so
an interrupted run on a long transcript resumes with `--resume`.

## Setup

Requires Python 3.12+. Pick whichever package manager your machine allows.

### Option A — pip + venv (works on restricted / corporate laptops)

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .

cp .env.example .env              # then edit .env  (Windows: copy .env.example .env)
```

For development extras (tests): `pip install -e ".[dev]"`.

> **Behind a corporate proxy / internal index?** Point pip at your mirror, e.g.
> `pip install -e . --index-url https://<your-artifactory>/api/pypi/pypi/simple`.
> If SSL inspection breaks TLS, add `--trusted-host <host>`.

### Option B — uv

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env
```

Fill in `.env`:

| Var | Meaning |
|-----|---------|
| `OPENAI_BASE_URL` | Your OpenAI-compatible endpoint (ends in `/v1`). |
| `OPENAI_API_KEY` | Any non-empty string if the endpoint ignores it. |
| `MEETING_WORKER_MODEL` | Model for the per-chunk map/sweep phases (e.g. `gemma`). |
| `MEETING_LEAD_MODEL` | Model for planning/synthesis/gapfill (e.g. `gpt-oss`). |
| `MEETING_MAX_SWEEPS` | Extra full-transcript passes after the first (default 3). |
| `MEETING_MAX_CONCURRENCY` | Parallel LLM calls in the map/sweep phases (default 4; use 1-2 for a single big local model). |
| `MEETING_CHUNK_MAX_CHARS` | Char budget per transcript chunk (default 9000). |
| `MEETING_REQUEST_TIMEOUT` | HTTP timeout per LLM call in seconds (default 300). |
| `MEETING_LLM_RETRIES` | Retries per call on 429/5xx/timeouts, with exponential backoff (default 4). |
| `MEETING_LOG_FILE` | File to append the detailed run log to (same as `--log-file`). |
| `MEETING_LANGUAGE` | Report language: `auto` (follow the transcript), a code (`es`, `en`, ...) or a name. |

Both models must support OpenAI-style **tool calling**; if a structured call
fails, the client retries once and then falls back to JSON parsing.

## Usage

```bash
# a plain transcript
meeting-assistant process standup.txt -o out/

# a Zoom/Teams caption file, with background context to resolve names and jargon
meeting-assistant process meeting.vtt --context "Priya is the PO. Project Phoenix is the mobile app." -o out/

# context can also be a file (agenda, prior minutes, participant list)
meeting-assistant process meeting.srt --context ./agenda.md -o out/

# useful flags
meeting-assistant process ... --language es      # force the report language (default: follow the transcript)
meeting-assistant process ... --max-sweeps 5     # review even more thoroughly
meeting-assistant process ... --log-file run.log # full call/retry/decision audit trail
meeting-assistant process ... --resume           # continue an interrupted run
meeting-assistant process ... --no-cache         # ignore cached chunk extractions
meeting-assistant process ... -v                 # mirror info-level log events to the console
```

Outputs land in `out/` as `<Meeting>-Report.md` and `<Meeting>-Report.json`.
The Markdown drops straight into any Obsidian vault.

### Following and reviewing a run

The console shows a live progress view: one bar per review pass (with the
running item count and how many were new after each pass — you can watch the
loop go dry), plus spinner rows for the planning, synthesis, gap-fill and
compose phases, and a category summary table at the end. Endpoint retries and
chunk failures surface live as warnings while the bars keep moving.

For a full audit trail, pass `--log-file run.log` (or set `MEETING_LOG_FILE`).
Every LLM call (with duration), every retry (with its backoff delay and the
error that caused it), every cache hit, pass decision, gap-fill query and
failure is appended with a timestamp and thread name, so a slow or degraded
run can be reviewed after the fact. `-v` additionally mirrors info-level
events to the console.

### Supported transcript formats

Format is auto-detected; no flag needed.

| Format | Looks like |
|--------|-----------|
| WebVTT | `WEBVTT` header, `00:00:01.000 --> ...` cues, `<v Speaker>` tags |
| SRT | numbered cues, `00:00:01,000 --> ...` |
| Labeled dialogue | `Alice: ...` lines, optionally timestamped `[00:12] Alice: ...` |
| Timestamped | `[00:12:34] ...` lines with no speaker |
| Plain text | paragraphs, or one undifferentiated block |

The optional **context** input is background only — participant names, an agenda,
prior minutes, a glossary. It is used to resolve references and acronyms; it is
never treated as something said in the meeting.

### Language

By default the report is written in **the language the meeting was held in**
(every prompt carries an explicit output-language rule, so the model does not
drift into English on non-English meetings). Force a language with
`--language es`, `--language en`, or any language name / `MEETING_LANGUAGE`.
Verbatim quotes are never translated — they are the grounding evidence. The
deterministic cue scan that backs the coverage check is bilingual
(English + Spanish).

## Troubleshooting slow or rate-limited endpoints

> Step-by-step version with measurements and a safe ramp-up procedure:
> [docs/07-Endpoint-Tuning.md](docs/07-Endpoint-Tuning.md).

Every LLM call already retries transient failures (429 rate limits, 502/503/504
gateway errors, timeouts) with exponential backoff, and if half or more of a
review round still fails, the agent stops sweeping, finishes the report with
what it has, and tells you to fix the endpoint and re-run with `--resume`
(cached round-0 extractions are not re-paid). If you see these warnings:

- **`504 Gateway Time-out`** — the *gateway in front of the model* gave up
  before the model finished a chunk. Raise the gateway's own timeout, and/or
  give the model less work per call: `--chunk-chars 5000`, `--max-concurrency 1`.
  The client-side `MEETING_REQUEST_TIMEOUT` (default 300 s) must also be at
  least as long as a real completion takes.
- **`429 Rate limit exceeded`** — the endpoint cannot take the parallelism.
  Lower `--max-concurrency` to 1 or 2; the backoff will absorb occasional 429s
  but sustained ones mean the concurrency is simply too high for the server.
- A run interrupted or degraded by endpoint problems is resumable:
  `meeting-assistant process ... --resume` continues from the last checkpoint.

## What gets extracted

Everything below, each item grounded in the transcript segments (`Sxx`) it came
from:

- **Summary** — executive summary and a chronological detailed summary (minutes)
- **Topics** — the agenda, reconstructed from what was discussed
- **Decisions** — what was decided, why, and by whom
- **Action items** — task, owner, due date, priority, status
- **Open questions** — raised but not resolved
- **Risks and blockers** — with severity and mitigation
- **Key facts and figures** — metrics, numbers, money, quantities
- **Dates and deadlines** — everything time-bound
- **People and entities** — people, orgs, products, systems, tools, documents
- **Notable quotes** — verbatim, attributed
- **Disagreements** — points of tension and how they resolved
- **Glossary** — domain terms and acronyms
- **Next steps**, **overall sentiment**, and **per-speaker contributions**

## Development

```bash
pip install -e ".[dev]"
pytest          # 54 tests: parser, ingestion, chunker, merge, renderer, LLM gateway retries, CLI, language, full graph (fake LLM)
```

The suite runs the entire pipeline against a bundled sample transcript using an
injected fake LLM — no endpoint needed — and asserts that the review loop makes
multiple passes, consolidates without duplicating, and stops once a sweep goes
dry.

## Project layout

```
src/meeting_assistant/
  config.py            # typed settings from .env / CLI
  llm.py               # single LLM gateway: structured output + tool loops
  extractors.py        # deterministic cue scan over the transcript
  tools.py             # bounded read-only transcript tools for the gapfill agent
  ingest/              # loader, format-sniffing parser, overlapping chunker
  model/               # transcript IR + LLM structured-output schemas
  graph/               # LangGraph state, nodes, merge/dedup, builder, cache
  render/              # deterministic Markdown + JSON assembly, style lint
  cli.py               # typer entrypoint (meeting-assistant process)
tests/                 # sample transcript + fake-LLM pipeline tests
```
