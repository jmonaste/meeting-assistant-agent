# 03 — Transcript ingestion

Ingestion is fully deterministic: no LLM is involved. It turns a raw transcript
file into a `TranscriptInventory` (`model/transcript.py`) of id-tagged segments
and overlapping chunks. Entry point: `ingest/loader.py:build_inventory()`.

## Format detection (`parser.py:detect_format`)

Detection is heuristic and always has a safe fallback to plain text:

1. Starts with `WEBVTT` -> **vtt**.
2. Contains `-->` cue lines -> **srt** if it also has integer index lines and
   comma-millisecond timestamps, else **vtt**.
3. A quarter or more of the content lines match a `Speaker:` label (after any
   leading timestamp is stripped) and there are at least three -> **labeled**.
4. Forty percent or more of the lines start with a timestamp -> **timestamped**.
5. Otherwise -> **plain**.

The speaker test (`_looks_like_speaker`) is deliberately conservative — at most
five words, name-like characters only — so lines such as `Note: ...` or
`TODO: ...` are not mistaken for speakers.

## Parsing (`parser.py:parse_transcript`)

Each format has a parser producing `_Utt(speaker, text, start, end)`:

- **VTT / SRT** share `_parse_cue_blocks`: split on blank lines, skip headers and
  `NOTE`/`STYLE` blocks, drop cue index lines, read the `-->` timing, and take the
  speaker from a `<v Speaker>` voice tag or a `Speaker:` prefix in the payload.
  HTML-ish tags are stripped.
- **Labeled / timestamped** share `_parse_lines`: peel an optional leading
  timestamp, then a `Speaker:` (or `Speaker (00:12):`) label if present.
- **Plain** (`_parse_plain`): split on blank lines into paragraphs; if the text is
  one undifferentiated block, split into sentence groups capped at
  `PLAIN_SEGMENT_CHARS` so segment ids stay useful as anchors.

Every branch degrades: if a VTT/SRT file yields no cues, parsing falls back to
plain text with a warning. Timestamps are normalized to `HH:MM:SS` (ms dropped).

## From utterances to segments (`loader.py`)

- **Turn merging** (`_merge_turns`): consecutive utterances by the *same* speaker
  are merged into one turn (cue formats fragment a single turn across many short
  cues). Bounded by `MERGE_MAX_CHARS` so a monologue does not become one giant
  segment.
- **Segment ids**: merged turns become `Segment`s with ids `S0`, `S1`, ... These
  ids are the grounding currency used everywhere downstream.
- **Speaker aggregation**: turns and words per speaker, in first-appearance order.
- **Flags and span**: `has_speakers`, `has_timestamps`, and a `duration` string
  from the first and last timestamps.

## Chunking (`chunker.py:build_chunks`)

Segments are packed greedily into chunks of at most `MEETING_CHUNK_MAX_CHARS`.
Each new chunk **repeats at least the previous chunk's last segment** and extends
the overlap further back while it stays within `MEETING_CHUNK_OVERLAP_CHARS`. The
overlap guarantees an item whose statement straddles a boundary is seen whole by
at least one chunk. The loop always advances `start`, so even a single segment
larger than the whole budget cannot stall it. Each chunk's rendered text is
hashed (`hash_text`) to key the extraction cache.

## The deterministic cue scan (`extractors.py:scan_signals`)

A set of cheap regexes counts concrete cues per category: action verbs
(`I'll`, `we need to`, `assign`), decision phrases (`we agreed`, `let's go with`),
lines ending in `?`, dates and deadlines (weekdays, months, `EOD`, `Q3`,
`by Friday`, ISO dates), numbers and money, and risk words. Each hit is anchored
to its segment id.

The scan is **not** an extractor — its phrasing is too fuzzy to produce finished
items. It is a coverage signal: `gapfill` compares the cue count for a category
against how many items the model actually captured, and re-scans categories where
the model clearly fell short (chapter 04). It also backs the "signal coverage"
appendix, so a reader can see the raw cues the analysis worked from.
