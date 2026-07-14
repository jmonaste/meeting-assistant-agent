# Documentation

Study-oriented documentation of how meeting-assistant is designed and built. It
is written to be read next to the source: every chapter names the modules,
classes and functions it explains, and every non-obvious decision is justified.

| # | Chapter | What it covers |
|---|---------|----------------|
| 01 | [Requirements and design](01-Requirements-And-Design.md) | The problem (single-pass LLMs lose detail), the requirements, and the key design decisions with their rationale. |
| 02 | [Pipeline architecture](02-Pipeline-Architecture.md) | The LangGraph graph: state, reducers, every node, the review loop, concurrency, checkpointing and resume. |
| 03 | [Transcript ingestion](03-Transcript-Ingestion.md) | Format detection, parsing VTT/SRT/labeled/plain, turn merging, segment ids, overlapping chunks, and the deterministic cue scan. |
| 04 | [Multi-pass coverage](04-Multi-Pass-Coverage.md) | The heart of the tool: the map/sweep loop, loop-until-dry, deterministic de-duplication, and the agentic gap-fill rescue. |
| 05 | [LLM gateway](05-LLM-Gateway.md) | The single point of contact with the model: structured output, the retry ladder, the JSON fallback, the tool loop, and local-model quirks. |
| 06 | [Testing and verification](06-Testing-And-Verification.md) | The injectable fake LLM, what each fixture and test proves, and how the review loop is verified without an endpoint. |

## The system in one diagram

```
 transcript file (+ optional context text)
    |
    v
 ingest        deterministic: detect format, parse into speaker turns with
    |          stable segment ids, merge fragmented turns, build overlapping
    |          chunks, run the deterministic cue scan
    v
 plan          LLM (lead): meeting type, title, focus areas
    |
    v
 dispatch <----------------------+   the review loop head
    | Send() per chunk           |   round 0: extract everything
    v                            |   round 1..N: find only what was missed
 extract_chunk (parallel) -------+   LLM (worker); round-0 results disk-cached
    |                                stops when a full round adds nothing new
    | (sweeps exhausted)
    v
 reduce        LLM (lead): prose (summary, minutes, sentiment, next steps)
    |          from the consolidated items; may flag under-covered categories
    v
 gapfill       LLM agent (lead): bounded tool loop rescues flagged categories
    |          from the transcript; evidence recorded from real reads
    v
 compose       deterministic: Markdown report + JSON export, citations
    |          validated, style linted
    v
 END
```

## Source map

```
src/meeting_assistant/
  config.py            Settings (pydantic-settings), cache_dir()
  llm.py               MeetingLLM: structured(), tool_loop()             -> ch. 05
  extractors.py        scan_signals(): deterministic cue scan            -> ch. 03
  tools.py             build_transcript_tools(): read-only tools         -> ch. 04
  ingest/
    parser.py          detect_format(), parse_transcript()               -> ch. 03
    chunker.py         build_chunks(): overlapping windows               -> ch. 03
    loader.py          build_inventory()                                 -> ch. 03
  model/
    transcript.py      Segment, Speaker, Chunk, TranscriptInventory      -> ch. 03
    extraction.py      item schemas + ChunkExtraction + syntheses        -> ch. 04, 05
  graph/
    state.py           MeetingState, reducers, ChunkPayload              -> ch. 02
    nodes.py           PipelineNodes: nodes + prompts                    -> ch. 02, 04
    merge.py           consolidate(), render_known_items()               -> ch. 04
    builder.py         build_graph(), run_pipeline(), checkpointer       -> ch. 02
    cache.py           ExtractionCache (content-hash disk cache)         -> ch. 04
  render/
    document.py        render_report(), render_json()                   -> ch. 02
    lint.py            lint_markdown(), md_cell(), md_anchor()           -> ch. 02
  cli.py               typer entry point (meeting-assistant process)
tests/                                                                   -> ch. 06
```
