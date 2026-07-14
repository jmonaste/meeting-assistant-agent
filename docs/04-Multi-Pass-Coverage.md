# 04 — Multi-pass coverage

This is the chapter that explains the point of the tool: how it reads the
transcript repeatedly, finds what earlier passes missed, and knows when to stop.

## The loop head: `dispatch`

`dispatch` (`graph/nodes.py`) runs every time control returns from a completed
round. Its logic:

```python
prev_count = state.get("item_count", 0)
items, count = consolidate(state.get("harvest", []))   # rebuild from the FULL harvest
rounds_done = state.get("rounds_done", 0)
max_rounds  = 1 + max(0, settings.max_sweeps)

if rounds_done >= 1 and (rounds_done >= max_rounds or count <= prev_count):
    return {"items": items, "item_count": count, "current_chunks": []}   # -> reduce

# else launch another full pass over every chunk
known = render_known_items(items) if rounds_done >= 1 else ""
payloads = [ChunkPayload(... round=rounds_done, known_items=known ...) for ch in chunks]
return {"items": items, "item_count": count, "rounds_done": rounds_done + 1,
        "current_chunks": payloads}
```

- **Round 0** (`rounds_done == 0`): `known` is empty, so every chunk is asked to
  extract *everything* it contains.
- **Round 1..N** (`rounds_done >= 1`): `known` is the digest of everything found
  so far, and each chunk is asked only for what is *missing*.
- **Stopping**: after at least one round, stop when the round cap is hit
  (`1 + MEETING_MAX_SWEEPS`) **or** the last round added no new items
  (`count <= prev_count`). This is loop-until-dry: extra sweeps cost nothing once
  the transcript is exhausted, because a dry round terminates immediately.
- **Endpoint health**: each failed chunk extraction records its round number in
  the additive `failures` state key. If half or more of the last round's chunks
  failed (sustained 429s / gateway timeouts, after the gateway's own backoff was
  exhausted), `dispatch` stops sweeping with a warning instead of burning more
  failing calls; the run finishes with what it has and is resumable with
  `--resume` once the endpoint recovers.

`item_count` recorded on each dispatch is the count *before* the next round's
additions, so comparing it to the freshly consolidated count detects a dry round.

## The two prompts: map vs sweep

`extract_chunk` picks the system prompt by round:

- `MAP_SYSTEM` (round 0): "Extract EVERYTHING of value in this excerpt ... it is
  better to surface a borderline item than to miss it ... ground every item
  strictly in what is said ... cite the segment ids."
- `SWEEP_SYSTEM` (round >= 1): "You are re-reading ... to catch information
  earlier passes MISSED. You are given the items already captured. Extract ONLY
  items that are NOT already in that list. If everything is already captured,
  return empty lists."

The sweep prompt is what makes a re-read productive rather than redundant: the
model spends its attention on the gaps, not on re-deriving what is already known.

## Deterministic consolidation (`graph/merge.py`)

Overlapping chunks and repeated sweeps produce heavily overlapping results.
`consolidate(harvest)` merges them into one deduplicated set of item lists:

- Each item type defines `dedup_key()` — a normalized (lowercased, whitespace-
  collapsed, truncated) key over its identifying text.
- The first occurrence of a key is kept; later duplicates **enrich** it via
  `_enrich`: segment ids are unioned and blank scalar fields are backfilled, so
  the surviving item is the most complete version seen.
- Items with an empty key are dropped.

Because consolidation reads the entire harvest every time, it is idempotent: it
can be recomputed after every round without ever double-counting. This is why the
sweep loop can re-read the same content many times and still converge.

`render_known_items(items)` produces the compact digest shown to the sweeps — one
identifying line per item, grouped by category with counts — small enough to fit
the context budget but specific enough that the model can tell new from known.

## Model-driven and signal-driven gap-fill (`gapfill`)

After `reduce` writes the prose, `gapfill` runs one more, *targeted* rescue. It
decides which categories still look under-covered from two independent signals
(`_flagged_categories`):

1. **Model-driven**: `MeetingSynthesis.coverage_gaps` — the synthesis step is
   asked to name any category that still looks incomplete given the meeting's
   size.
2. **Signal-driven**: any category where the deterministic cue scan found clearly
   more cues than the model captured (`len(cues) > len(items) + 2`).

For the flagged categories, `gapfill` runs a **bounded agentic tool loop**
(`llm.tool_loop`) over read-only transcript tools (`build_transcript_tools`:
`search_transcript`, `read_segments`, `read_around`, `list_speakers`,
`read_background`). The agent searches for the relevant cues, reads around the
matches, and reports what it finds — with the segment ids it actually read
recorded deterministically. That evidence is then structured into a
`ChunkExtraction` and merged into `items` via the same `consolidate`, and the
number of newly rescued items per category is reported in the appendix.

This gives two complementary mechanisms: the sweep loop is broad and exhaustive
(every chunk, every category); the gap-fill is narrow and investigative (one
agent, the specific categories that still look thin), and it can search across
the whole transcript at once rather than chunk by chunk.

## Caching (`graph/cache.py`)

Round-0 extractions are cached on disk (`ExtractionCache`) keyed by
`(chunk hash, worker model, prompt version)`. Re-processing an unchanged
transcript re-reads nothing on the first pass. Sweeps are never cached because
their result depends on what earlier passes already found, which is not part of
the key.
