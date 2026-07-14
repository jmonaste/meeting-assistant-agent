# 01 — Requirements and design

## The problem

A meeting transcript is long, repetitive and low-signal-per-word. The obvious
approach — paste the whole thing into an LLM and ask for "minutes and action
items" — fails in a specific, predictable way: **a single pass loses detail.**
The model latches onto the dominant thread and drops the things that were said
once: the number quoted in passing, the task someone volunteered for in half a
sentence, the question that nobody circled back to, the risk mentioned and never
addressed. These are exactly the items a meeting report exists to capture.

You cannot fix this by prompting harder ("don't miss anything!") because the
failure is structural: attention is finite and the model has no mechanism to
check its own coverage. The fix has to be structural too.

## Requirements

1. **Exhaustive extraction.** Capture everything of value a meeting can contain:
   summary, topics, decisions, action items, open questions, risks, key facts
   and figures, dates, people and entities, quotes, disagreements, glossary.
2. **Multiple reviews.** Read the transcript as many times as needed, each pass
   able to find what earlier passes missed, and stop automatically when further
   reading stops finding anything new.
3. **Optional context.** Accept optional background text (participant list,
   agenda, prior minutes, glossary) and use it to resolve names and jargon —
   without treating it as content spoken in the meeting.
4. **Format-agnostic input.** Accept VTT, SRT, speaker-labeled dialogue,
   timestamped lines, or plain text, with no manual configuration.
5. **Grounded output.** Every extracted item must point back to the transcript,
   so a reader can verify it and hallucinations are visible.
6. **Runs locally.** Work against a local, OpenAI-compatible model on a
   restricted machine — no dependency on a hosted frontier model.
7. **Resumable and cheap to re-run.** A long transcript should survive an
   interruption, and re-processing an unchanged transcript should be nearly free.

## Key design decisions

### Deterministic ingestion before any LLM call
Format detection, parsing, turn merging, segmentation and the cue scan are pure
Python (`ingest/`, `extractors.py`). The model only ever sees clean, id-tagged
segments. This makes structural facts (who spoke, when, how many turns) exact
and keeps token spend proportional to content, not to transcript noise.

### The review loop is the product
Instead of one prompt, the pipeline runs a **map/sweep loop** (chapter 04):
round 0 extracts everything from each chunk; each later round re-reads the whole
transcript, is shown what has already been found, and is asked only for what is
missing. The loop stops when a full round adds nothing new (loop-until-dry),
bounded by `MEETING_MAX_SWEEPS`. This is the direct, structural answer to
requirement 2.

### Stable segment ids as the grounding currency
Every utterance gets an id (`S0`, `S1`, ...). The model is asked to cite ids on
every item; the renderer validates those ids against the real transcript and
drops any it invented (`render/document.py`). Grounding is therefore checked, not
trusted.

### Deterministic de-duplication
Re-reading the same content repeatedly would pile up duplicates. Each item type
defines a normalized `dedup_key()`, and `graph/merge.py:consolidate()` merges the
full harvest idempotently — keeping the first occurrence, unioning its segment
ids and backfilling blank fields from duplicates. Convergence, not accumulation.

### Two model roles over one endpoint
A cheap `worker` model does the many per-chunk extractions; a stronger `lead`
model does the few hard calls (planning, synthesis, gap-fill). Both are served by
the same OpenAI-compatible endpoint through one gateway (`llm.py`), so the whole
system has a single, well-tested point of contact with the model.

### A deterministic cue scan as a safety net
Cheap regex passes (`extractors.py`) count concrete cues — dates, action verbs,
decision phrases, risk words, numbers. When the cue count for a category far
exceeds what the model captured, the gap-fill agent is sent to re-scan that
category specifically. The scan never produces finished items (its phrasing is
fuzzy); it is a coverage signal that says "look here again".

## What was deliberately left out

- **No diarization or ASR.** The tool starts from a transcript; turning audio
  into text is a separate concern.
- **No speaker identity resolution beyond labels + context.** If the transcript
  says "Speaker 1", that is what appears, unless the context maps it.
- **No runtime integrations in the core.** The JSON export is the integration
  surface; pushing items into a tracker lives outside this agent.
