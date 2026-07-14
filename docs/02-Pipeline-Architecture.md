# 02 — Pipeline architecture

The pipeline is a LangGraph `StateGraph` compiled in
`graph/builder.py:build_graph()` and driven by `run_pipeline()`.

## Graph shape

```
START -> ingest -> plan -> dispatch --(Send per chunk)--> extract_chunk -> dispatch
                              \--(sweeps exhausted)--> reduce -> gapfill -> compose -> END
```

`dispatch` is the loop head. `route_map` (a conditional edge) either fans out one
`Send("extract_chunk", payload)` per chunk or falls through to `reduce`. The edge
`extract_chunk -> dispatch` closes the loop, so each completed round returns to
`dispatch`, which decides whether to run another.

## State (`graph/state.py`)

`MeetingState` is a `TypedDict(total=False)`. It carries compact artifacts only —
the parsed inventory, per-chunk extractions and the consolidated item lists —
never the raw transcript passed around redundantly.

Two reducer keys make the parallel map/sweep safe:

```python
harvest:  Annotated[list[ChunkExtraction], operator.add]   # every chunk, every round
warnings: Annotated[list[str], operator.add]
```

`harvest` is an **append-only accumulator**. Every `extract_chunk` call (across
every round) appends its result. The single-threaded `dispatch` node rebuilds the
deduplicated `items` from the *entire* harvest each time it runs, so consolidation
is idempotent and never double-counts overlapping chunks or repeated sweeps.

`ChunkPayload` is the isolated input for one extraction: the chunk text, the round
number, the digest of already-known items (empty on round 0), the content hash,
and the optional context block.

## Nodes (`graph/nodes.py`)

- **ingest** — builds the `TranscriptInventory` (or accepts a pre-built one for
  tests), initializes the `ExtractionCache`, runs the deterministic cue scan.
  Raises `ValueError` if the transcript has no usable content.
- **plan** — one `lead` call producing a `MeetingPlan` (type, title, focus).
  Degrades to a sensible default on any failure.
- **dispatch** — consolidates the harvest into `items`; decides whether to launch
  another pass; builds the per-chunk payloads for the next round. See chapter 04.
- **extract_chunk** — one `worker` call producing a `ChunkExtraction`. On round 0
  it checks and populates the disk cache. Degrades to an empty extraction (plus a
  warning) on failure, so one bad chunk never fails the run.
- **reduce** — one `lead` call producing the `MeetingSynthesis` prose from the
  consolidated items. Degrades to a listing fallback.
- **gapfill** — the agentic rescue for under-covered categories (chapter 04).
- **compose** — deterministic assembly of the Markdown report and JSON export.

Every LLM node follows the same rule inherited from rpa-code-guardian: **degrade
to a deterministic fallback and record a warning; never fail the whole run for
one bad model call.**

## Concurrency

The map and sweep rounds fan out with the Send API. `run_pipeline` sets
`max_concurrency` from settings, so at most N chunk extractions are in flight at
once. Because chunk extractions are independent and write only to the additive
`harvest`, there are no ordering hazards.

## Checkpointing and resume

`run_pipeline` wraps the graph in a `SqliteSaver` under `.meeting_cache/`, keyed
by a `thread_id` persisted in `last_run_id`. Every super-step is checkpointed. A
crashed run resumes with `resume=True`, which reuses the thread id and passes
`None` inputs so the graph continues from the last checkpoint.

Because the state contains Pydantic models, the serializer is given an explicit
`allowed_msgpack_modules` allowlist (`builder.py:_serde()`) naming every model
class in `model/transcript.py` and `model/extraction.py`. This is a security
boundary: only these classes may be reconstructed from a checkpoint.

## Rendering (`render/`)

`document.py` assembles the deliverables deterministically: the LLM writes prose
*sections*; the skeleton, tables, table of contents and appendices are built from
the consolidated items. Segment citations are validated against the real
transcript (`_valid_sids`) and invalid ones are dropped. `lint.py` strips emoji,
trailing whitespace and excess blank lines from the final Markdown so the
deliverable is clean regardless of what the model emitted.
