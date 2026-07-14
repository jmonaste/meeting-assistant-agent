# 06 — Testing and verification

The suite (`tests/`, 46 tests) runs the entire pipeline with **no endpoint**, by
injecting a fake LLM. It proves the deterministic machinery exactly and the
model-facing control flow structurally.

```bash
pip install -e ".[dev]"
pytest
```

## The injectable fake LLM (`tests/conftest.py`)

`FakeLLM` subclasses `MeetingLLM` and overrides only `structured()` and
`tool_loop()`, so it never touches the network. It returns canned objects keyed
by the requested schema:

- `MeetingPlan` / `MeetingSynthesis` — fixed, realistic objects.
- `ChunkExtraction` — on round 0 it derives items from the segment ids present in
  the prompt (one action item, decision, topic, etc. per chunk-leading id), so the
  map and de-dup paths run on realistic, distinct items. On a **sweep** prompt (it
  detects `"ALREADY CAPTURED"`) or a **gap-fill** prompt (`"EVIDENCE GATHERED"`) it
  returns an empty extraction — modelling "nothing new was found", which is what
  makes the loop go dry so termination can be asserted.

This mirrors the rpa-code-guardian testing approach: the fake makes the model's
*behavior* deterministic without stubbing out any of the real pipeline code.

## What each test file proves

- **test_parser.py** — format detection for VTT/SRT/labeled/plain, correct
  speaker/timestamp extraction per format, and safe handling of empty input.
- **test_ingest.py** — inventory assembly (speakers, timestamps, span), that the
  chunks cover every segment and adjacent chunks overlap, that a single giant
  segment still chunks without stalling, and that the cue scan finds actions,
  dates and decisions anchored to segment ids. Also that context text is carried.
- **test_merge.py** — de-duplication keeps one item and unions its segment ids,
  distinct items are kept, blank keys are dropped, and the known-items digest
  lists categories.
- **test_render.py** — the report has the expected sections and tables, invented
  segment ids are dropped from the output, emoji are stripped, and the JSON export
  is valid and complete.
- **test_llm.py** — the gateway's transient-error handling: retryable
  classification (429/504/timeouts, including down the `__cause__` chain), retry
  then success, `MeetingLLMUnavailable` after exhaustion, immediate propagation
  of non-transient errors, HTML-page error summaries, that a dead transport
  aborts the structured-output method ladder instead of falling through, and
  that each retry emits a reviewable log record with its backoff delay.
- **test_cli.py** — the console layer with a fake `run_pipeline` replaying a
  realistic event sequence: the progress display and summary table render, the
  report/JSON files are written, warnings are sanitized to one line, and
  `--log-file` produces an audit trail with the run bookends.
- **test_graph.py** — the full pipeline via `run_pipeline` with the fake LLM:
  - it produces a Markdown report and a valid JSON export;
  - it **runs multiple review passes** (`rounds_done >= 2`) and requests chunk
    extractions for round 0 *and* at least one sweep round;
  - it **stops when sweeps go dry** rather than always running to the cap;
  - it **consolidates across chunks** (one action item per distinct chunk, no
    duplicates);
  - a `--resume` run reuses the checkpoint thread and reproduces the report;
  - when every chunk extraction fails like a dead endpoint
    (`MeetingLLMUnavailable`), it **stops sweeping after round 0**, still
    produces a report, and warns that the endpoint looks unhealthy;
  - a real pipeline run **writes the detailed log**: pass starts, the loop's
    stop reason, and the compose summary all appear in the log file;
  - when the gapfill agent rescues items, **`item_count` stays consistent**
    with the category table (the rescued extraction joins the harvest).

## Verifying against a real endpoint

The fake proves control flow and determinism; to sanity-check prompt quality
against a real local model, run the CLI on the bundled fixture:

```bash
meeting-assistant process tests/fixtures/sample-standup.txt \
  --context "Priya is the product owner." -o out/ -v
```

and inspect `out/Sprint-Planning-Standup-Report.md`. The fixture is a short
standup containing, by construction, decisions (the August 15th release), action
items with owners and due dates (Bob's infra ticket, Carol's Friday test plan),
an open question (analytics scope), a risk (staging blocker), a number ($4000/mo),
and dates — so you can check each category end to end.
