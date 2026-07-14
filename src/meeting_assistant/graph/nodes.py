"""Pipeline nodes: ingest -> plan -> map/sweep loop -> reduce -> gapfill -> compose.

Context strategy in one paragraph: the ingest node compresses the transcript
deterministically (parse -> segments -> overlapping chunks); the map phase
extracts every item from one chunk at a time in an isolated context; the sweep
loop re-reads the WHOLE transcript again and again, each pass told what has
already been found and asked only for what earlier passes missed, stopping when
a full round adds nothing new; the reduce phase writes the prose from the
consolidated items only; the gapfill agent selectively rescues categories that
still look under-covered. Every LLM node degrades to a safe fallback and records
a warning instead of failing the run.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END
from langgraph.types import Send

from ..config import Settings, cache_dir
from ..extractors import scan_signals
from ..ingest.loader import build_inventory
from ..llm import MeetingLLM
from ..model.extraction import (
    CATEGORIES,
    ChunkExtraction,
    MeetingPlan,
    MeetingSynthesis,
    ParticipantRole,
)
from ..render.document import render_json, render_report
from ..tools import build_transcript_tools
from .cache import ExtractionCache
from .merge import consolidate, render_known_items
from .state import ChunkPayload, MeetingState

_FIELD_BY_NAME = {field: field for field, _l, _n in CATEGORIES}
_FIELD_BY_LABEL = {label.lower(): field for field, label, _n in CATEGORIES}

PLAN_SYSTEM = """\
You are an expert meeting analyst. You are given an overview of a meeting
transcript (participants, size, detected signal cues) and any background the
user provided. Produce a short plan for extracting everything of value from it:
what kind of meeting this is, a fitting title, and what deserves special
attention. Be specific to THIS meeting; do not restate generic advice."""

MAP_SYSTEM = """\
You are an expert meeting analyst extracting structured information from ONE
excerpt of a meeting transcript. Each line begins with its segment id in
brackets, e.g. [S12]. Extract EVERYTHING of value in this excerpt across all the
categories in the schema: topics, decisions, action items, open questions, risks
and blockers, key facts and figures, dates and deadlines, people and entities,
notable quotes, disagreements, and glossary terms. Be exhaustive: it is better
to surface a borderline item than to miss it. Ground every item strictly in what
is said — never invent people, systems or outcomes that are not in the text. For
each item, copy a short verbatim quote and list the segment ids it came from. Do
not attribute a name unless the transcript or background supports it."""

SWEEP_SYSTEM = """\
You are re-reading ONE excerpt of a meeting transcript to catch information that
earlier passes MISSED. You are given the items already captured for the whole
meeting. Extract ONLY items that appear in this excerpt and are NOT already in
that list — genuinely new decisions, action items, questions, risks, facts,
dates, entities, quotes, disagreements or terms. If everything in this excerpt is
already captured, return empty lists. Be strict about novelty but thorough about
coverage; ground every new item in the text with a short quote and segment ids."""

REDUCE_SYSTEM = """\
You are writing the prose of a meeting report for a professional audience. You
are given an overview of the meeting and the consolidated lists of everything
extracted from it. Write the requested prose sections grounded strictly in that
material (no marketing language, no emojis). Distil the next steps from the
action items and decisions. If a whole category still looks incompletely
captured given the meeting's size, name it in coverage_gaps so it gets another
re-read; otherwise leave coverage_gaps empty."""

GAP_SYSTEM = """\
You recover information a summary may have missed from a meeting transcript,
using the provided read-only tools. Search for the relevant cues, read around
the matches, and report what you find as concrete items, each with a short
verbatim quote and its segment ids. Ground everything strictly in the tool
results; if the transcript does not contain something, do not invent it."""

GAP_STRUCT_SYSTEM = """\
Convert the gathered evidence about a meeting into structured items across the
schema categories. Keep only items clearly supported by the evidence text,
preserving the quotes and segment ids it cites. Leave a category empty if the
evidence has nothing for it."""


class PipelineNodes:
    """Node implementations, closed over settings and the LLM gateway."""

    def __init__(self, settings: Settings, llm: MeetingLLM | None = None) -> None:
        self.settings = settings
        self.llm = llm or MeetingLLM(settings)
        self.cache: ExtractionCache | None = None

    # ---------------------------------------------------------------- ingest

    def ingest(self, state: MeetingState) -> dict:
        inv = state.get("inventory")
        if inv is None:
            inv = build_inventory(
                state.get("transcript_text", ""),
                self.settings,
                source_file=state.get("source_file", ""),
                context_text=state.get("context_text", ""),
            )
        if not inv.segments:
            raise ValueError("no usable transcript content was found")

        base = Path(inv.root) if inv.root else Path(".")
        self.cache = ExtractionCache(
            cache_dir(base) / "extractions.json",
            model=self.settings.resolved_worker_model(),
            enabled=self.settings.use_cache,
        )
        signals = scan_signals(inv)
        return {"inventory": inv, "signals": signals, "rounds_done": 0, "warnings": list(inv.warnings)}

    # ------------------------------------------------------------------ plan

    def plan(self, state: MeetingState) -> dict:
        inv = state["inventory"]
        parts = [inv.census(), "PARTICIPANTS:\n" + inv.speaker_context()]
        if inv.context_block():
            parts.append(inv.context_block())
        signal_summary = "; ".join(f"{cat}: {len(hits)} cues" for cat, hits in (state.get("signals") or {}).items())
        if signal_summary:
            parts.append("DETERMINISTIC SIGNAL CUES:\n" + signal_summary)
        user = "\n\n".join(parts)
        try:
            plan = self.llm.structured(MeetingPlan, PLAN_SYSTEM, user, role="lead")
        except Exception as exc:  # noqa: BLE001 - degrade, don't fail the run
            plan = MeetingPlan(
                title=inv.title_hint or "Meeting",
                meeting_type="Unknown (analysis plan unavailable)",
                focus_areas=["Extract every decision, action item, question and risk."],
            )
            return {"plan": plan, "warnings": [f"plan node fell back to defaults: {exc}"]}
        if not plan.title:
            plan.title = inv.title_hint or "Meeting"
        return {"plan": plan}

    # ------------------------------------------------- map / sweep loop

    def dispatch(self, state: MeetingState) -> dict:
        """Consolidate the harvest, then either launch another full-transcript
        pass or fall through to the reduce phase.

        Round 0 extracts everything from every chunk. Each later round is a
        coverage sweep: it re-reads every chunk, told what is already found, and
        asks only for what is missing. Sweeping stops at the configured cap or as
        soon as a whole round adds nothing new (loop-until-dry)."""
        if self.cache is not None:
            self.cache.save()
        prev_count = state.get("item_count", 0)
        items, count = consolidate(state.get("harvest", []))
        rounds_done = state.get("rounds_done", 0)
        max_rounds = 1 + max(0, self.settings.max_sweeps)
        inv = state["inventory"]

        out: dict = {"items": items, "item_count": count}
        if rounds_done >= 1:
            # Endpoint health check: if half or more of the last round's chunks
            # failed (rate limits, gateway timeouts), further sweeps would just
            # burn more failing calls — stop and let reduce work with what we have.
            last_round = rounds_done - 1
            failed_last = sum(1 for r in state.get("failures", []) if r == last_round)
            n_chunks = len(inv.chunks)
            if n_chunks and failed_last * 2 >= n_chunks:
                out["current_chunks"] = []  # -> reduce
                out["warnings"] = [
                    f"stopped reviewing after round {last_round}: {failed_last} of "
                    f"{n_chunks} chunk extractions failed; the endpoint looks unhealthy "
                    "(rate limits / gateway timeouts). Fix the endpoint and re-run with "
                    "--resume to continue this run."
                ]
                return out
            if rounds_done >= max_rounds or count <= prev_count:
                out["current_chunks"] = []  # -> reduce
                return out
        known = render_known_items(items) if rounds_done >= 1 else ""
        context = inv.context_block()
        payloads: list[ChunkPayload] = [
            ChunkPayload(
                chunk_index=ch.index,
                chunk_text=ch.text,
                round=rounds_done,
                known_items=known,
                chunk_hash=ch.content_hash,
                context=context,
            )
            for ch in inv.chunks
        ]
        out["current_chunks"] = payloads
        out["rounds_done"] = rounds_done + 1
        return out

    def route_map(self, state: MeetingState):
        chunks = state.get("current_chunks") or []
        if chunks:
            return [Send("extract_chunk", payload) for payload in chunks]
        return "reduce"

    def extract_chunk(self, payload: ChunkPayload) -> dict:
        rnd = payload["round"]
        if rnd == 0 and self.cache is not None:
            cached = self.cache.get(payload["chunk_hash"])
            if cached is not None:
                return {"harvest": [cached]}

        parts: list[str] = []
        if payload.get("context"):
            parts.append(payload["context"])
        parts.append("TRANSCRIPT EXCERPT (each line begins with its [segment id]):\n" + payload["chunk_text"])
        if rnd >= 1:
            parts.append("ALREADY CAPTURED ACROSS THE WHOLE MEETING:\n" + payload["known_items"])
            parts.append("List ONLY items from the excerpt above that are missing from that list. Empty lists if nothing is new.")
        user = "\n\n".join(parts)
        system = SWEEP_SYSTEM if rnd >= 1 else MAP_SYSTEM
        try:
            extraction = self.llm.structured(ChunkExtraction, system, user, role="worker")
        except Exception as exc:  # noqa: BLE001
            return {
                "harvest": [ChunkExtraction()],
                "failures": [rnd],
                "warnings": [f"extraction failed for chunk {payload['chunk_index']} (round {rnd}): {exc}"],
            }
        if rnd == 0 and self.cache is not None:
            self.cache.put(payload["chunk_hash"], extraction)
        return {"harvest": [extraction]}

    # ---------------------------------------------------------------- reduce

    def reduce(self, state: MeetingState) -> dict:
        if self.cache is not None:
            self.cache.save()
        inv = state["inventory"]
        items = state.get("items", {})
        plan = state.get("plan")
        parts = [inv.census(), "PARTICIPANTS:\n" + inv.speaker_context()]
        if plan is not None:
            parts.append(f"MEETING TYPE (planning guess): {plan.meeting_type}. FOCUS: {'; '.join(plan.focus_areas)}")
        if inv.context_block():
            parts.append(inv.context_block())
        parts.append(
            "CONSOLIDATED EXTRACTION (write the prose from THIS; category names in parentheses "
            "are the exact strings to use in coverage_gaps):\n" + render_known_items(items)
        )
        parts.append("Category keys: " + ", ".join(field for field, _l, _n in CATEGORIES))
        user = "\n\n".join(parts)
        try:
            synthesis = self.llm.structured(MeetingSynthesis, REDUCE_SYSTEM, user, role="lead")
        except Exception as exc:  # noqa: BLE001
            synthesis = self._fallback_synthesis(state)
            return {"synthesis": synthesis, "warnings": [f"reduce node fell back to a listing: {exc}"]}
        if not synthesis.title:
            synthesis.title = (plan.title if plan else "") or inv.title_hint or "Meeting"
        if not synthesis.participants and inv.speakers:
            synthesis.participants = [ParticipantRole(name=sp.name) for sp in inv.speakers]
        return {"synthesis": synthesis}

    def _fallback_synthesis(self, state: MeetingState) -> MeetingSynthesis:
        inv = state["inventory"]
        plan = state.get("plan")
        items = state.get("items", {})
        counts = ", ".join(f"{label.lower()}: {len(items.get(field) or [])}" for field, label, _n in CATEGORIES)
        return MeetingSynthesis(
            title=(plan.title if plan else "") or inv.title_hint or "Meeting",
            meeting_type=(plan.meeting_type if plan else "") or "Unknown",
            executive_summary=(
                "Automated analysis of the meeting transcript. The narrative synthesis step "
                "was unavailable; this report presents the extracted items directly. "
                f"Items found — {counts}."
            ),
            detailed_summary="See the extracted items below for the meeting's content.",
            participants=[ParticipantRole(name=sp.name) for sp in inv.speakers],
        )

    # -------------------------------------------------------------- gap fill

    def gapfill(self, state: MeetingState) -> dict:
        inv = state["inventory"]
        items = state.get("items", {})
        synthesis = state.get("synthesis")
        signals = state.get("signals") or {}

        flagged = self._flagged_categories(synthesis, signals, items)
        if not flagged:
            return {"rescued": {}, "gap_notes": []}

        nouns = [noun for field, _l, noun in CATEGORIES if field in flagged]
        labels = [label for field, label, _n in CATEGORIES if field in flagged]
        recorder: list[str] = []
        tools = build_transcript_tools(inv, recorder)
        question = (
            "Re-read the meeting for anything the summary may have missed, focusing on: "
            f"{', '.join(labels)}. Search the transcript for the relevant cues, read around the "
            f"matches, and list every {', '.join(nouns)} you find, each with a short verbatim "
            "quote and its segment ids."
        )
        warnings: list[str] = []
        try:
            evidence, _ = self.llm.tool_loop(GAP_SYSTEM, question, tools, role="lead")
        except Exception as exc:  # noqa: BLE001
            return {"rescued": {}, "gap_notes": [], "warnings": [f"gapfill agent failed: {exc}"]}

        try:
            rescued_extraction = self.llm.structured(
                ChunkExtraction,
                GAP_STRUCT_SYSTEM,
                f"EVIDENCE GATHERED FROM THE TRANSCRIPT (segments read: {', '.join(recorder) or 'none'}):\n{evidence}",
                role="lead",
            )
        except Exception as exc:  # noqa: BLE001
            return {"rescued": {}, "gap_notes": [question], "warnings": [f"gapfill structuring failed: {exc}"]}

        merged, _ = consolidate([*state.get("harvest", []), rescued_extraction])
        rescued = {
            field: len(merged.get(field) or []) - len(items.get(field) or [])
            for field, _l, _n in CATEGORIES
        }
        rescued = {k: v for k, v in rescued.items() if v > 0}
        result: dict = {"items": merged, "gap_notes": [question]}
        if rescued:
            result["rescued"] = rescued
        if warnings:
            result["warnings"] = warnings
        return result

    def _flagged_categories(self, synthesis, signals: dict, items: dict) -> set[str]:
        """Categories to re-scan: those the model flagged, plus those where the
        deterministic signal scan clearly found more cues than were captured."""
        flagged: set[str] = set()
        if synthesis is not None:
            for raw in synthesis.coverage_gaps:
                key = raw.strip().lower()
                if key in _FIELD_BY_NAME:
                    flagged.add(key)
                elif key in _FIELD_BY_LABEL:
                    flagged.add(_FIELD_BY_LABEL[key])
        for cat, hits in signals.items():
            if cat in _FIELD_BY_NAME and len(hits) > len(items.get(cat) or []) + 2:
                flagged.add(cat)
        return flagged

    # --------------------------------------------------------------- compose

    def compose(self, state: MeetingState) -> dict:
        return {"report_md": render_report(state), "report_json": render_json(state)}

    def route_compose(self, state: MeetingState):
        return END
