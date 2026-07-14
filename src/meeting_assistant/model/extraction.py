"""Structured outputs produced by the LLM at each stage of the pipeline.

Every schema here is used with tool-calling structured output, so field
descriptions double as instructions to the model. Keep them precise.

The item models (``ActionItem``, ``Decision``, ...) are the units of "everything
worth extracting from a meeting". They are produced per-chunk in the map phase,
then re-discovered by the full-transcript coverage sweeps, so each one carries a
``dedup_key()`` used to merge duplicates deterministically across passes. Each
also carries a verbatim ``quote`` and the ``segment_ids`` it came from, so the
final report is grounded in the transcript rather than trusted from the model.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_WS_RE = re.compile(r"\s+")


def _norm(text: str, length: int = 90) -> str:
    """Normalize a string for duplicate detection across chunks/sweeps."""
    return _WS_RE.sub(" ", text or "").strip().lower()[:length]


# --------------------------------------------------------------------------- #
# Extractable items — the meeting's content, grounded in segment ids
# --------------------------------------------------------------------------- #

class ActionItem(BaseModel):
    """A concrete task, commitment or follow-up someone is expected to do."""

    task: str = Field(description="The action to be taken, as a short imperative statement.")
    owner: str = Field(default="", description="Person or team responsible, exactly as named in the meeting. Empty if unassigned.")
    due: str = Field(default="", description="Deadline or timeframe mentioned (e.g. 'Friday', 'next sprint', '2026-08-01'). Empty if none.")
    priority: str = Field(default="", description="high | medium | low, only if the meeting signals urgency. Empty otherwise.")
    status: str = Field(default="open", description="open | in-progress | done | blocked, if stated. Default 'open'.")
    quote: str = Field(default="", description="Short verbatim snippet from the transcript that states this action.")
    segment_ids: list[str] = Field(default_factory=list, description="Ids (e.g. 'S12') of the segments this came from.")

    def dedup_key(self) -> str:
        return _norm(self.task)


class Decision(BaseModel):
    """A choice the group made or agreed on."""

    decision: str = Field(description="What was decided, in one sentence.")
    rationale: str = Field(default="", description="Why, if stated.")
    decided_by: str = Field(default="", description="Who made or drove the decision, if identifiable.")
    alternatives: str = Field(default="", description="Options considered but not chosen, if mentioned.")
    quote: str = Field(default="", description="Short verbatim snippet supporting this decision.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(self.decision)


class OpenQuestion(BaseModel):
    """A question raised but not resolved during the meeting."""

    question: str = Field(description="The unresolved question, in one sentence.")
    raised_by: str = Field(default="", description="Who raised it, if identifiable.")
    context: str = Field(default="", description="Why it matters / what it blocks, one sentence.")
    quote: str = Field(default="", description="Short verbatim snippet.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(self.question)


class Risk(BaseModel):
    """A risk, concern, blocker or dependency surfaced in the meeting."""

    description: str = Field(description="The risk, concern or blocker, in one sentence.")
    severity: str = Field(default="", description="high | medium | low, if the meeting signals it. Empty otherwise.")
    mitigation: str = Field(default="", description="Proposed mitigation or owner, if any.")
    quote: str = Field(default="", description="Short verbatim snippet.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(self.description)


class KeyFact(BaseModel):
    """A concrete fact, metric, figure or number stated in the meeting."""

    fact: str = Field(description="The fact or figure, stated precisely (keep exact numbers, units, names).")
    category: str = Field(default="", description="metric | financial | date | quantity | status | other.")
    quote: str = Field(default="", description="Short verbatim snippet.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(self.fact)


class KeyDate(BaseModel):
    """A date, deadline or scheduled event mentioned in the meeting."""

    when: str = Field(description="The date/time or relative timeframe exactly as said (e.g. 'next Tuesday', 'end of Q3').")
    what: str = Field(description="What happens at that time.")
    quote: str = Field(default="", description="Short verbatim snippet.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(f"{self.when}|{self.what}")


class Entity(BaseModel):
    """A named person, organization, product, system, tool, document or place."""

    name: str = Field(description="The entity name exactly as referenced.")
    kind: str = Field(description="person | organization | product | system | tool | document | location | other.")
    note: str = Field(default="", description="One phrase on its role in the discussion, if clear.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(f"{self.kind}|{self.name}", 60)


class Quote(BaseModel):
    """A notable, quotable statement worth preserving verbatim."""

    text: str = Field(description="The statement, verbatim.")
    speaker: str = Field(default="", description="Who said it, if identifiable.")
    why: str = Field(default="", description="One phrase on why it is notable.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(self.text)


class Disagreement(BaseModel):
    """A point of tension, disagreement or unresolved debate."""

    topic: str = Field(description="What the disagreement was about.")
    positions: str = Field(default="", description="The differing positions and who held them, if clear.")
    resolution: str = Field(default="", description="How it was resolved, or 'unresolved'.")
    quote: str = Field(default="", description="Short verbatim snippet.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(self.topic)


class GlossaryTerm(BaseModel):
    """A domain term, acronym or piece of jargon used in the meeting."""

    term: str = Field(description="The term or acronym.")
    definition: str = Field(default="", description="What it means, from context or the user-provided background. Empty if unknown.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(self.term, 60)


class Topic(BaseModel):
    """A distinct subject discussed, used to reconstruct the agenda."""

    title: str = Field(description="Short title of the topic.")
    summary: str = Field(default="", description="One or two sentences on what was said about it.")
    segment_ids: list[str] = Field(default_factory=list)

    def dedup_key(self) -> str:
        return _norm(self.title, 60)


# --------------------------------------------------------------------------- #
# Map-phase output: everything extracted from ONE chunk
# --------------------------------------------------------------------------- #

class ChunkExtraction(BaseModel):
    """The full set of items extracted from a single transcript chunk.

    Produced once per chunk in the map phase. Be exhaustive within the chunk:
    it is better to surface a borderline item than to miss it, because later
    passes deduplicate but do not re-read what was dropped here.
    """

    topics: list[Topic] = Field(default_factory=list, description="Distinct subjects discussed in this chunk.")
    decisions: list[Decision] = Field(default_factory=list, description="Decisions or agreements reached in this chunk.")
    action_items: list[ActionItem] = Field(default_factory=list, description="Tasks, commitments and follow-ups in this chunk.")
    open_questions: list[OpenQuestion] = Field(default_factory=list, description="Questions raised but not resolved in this chunk.")
    risks: list[Risk] = Field(default_factory=list, description="Risks, concerns, blockers or dependencies in this chunk.")
    key_facts: list[KeyFact] = Field(default_factory=list, description="Concrete facts, metrics and figures in this chunk.")
    key_dates: list[KeyDate] = Field(default_factory=list, description="Dates, deadlines and scheduled events in this chunk.")
    entities: list[Entity] = Field(default_factory=list, description="People, orgs, products, systems, tools, documents named in this chunk.")
    quotes: list[Quote] = Field(default_factory=list, description="Notable verbatim statements in this chunk.")
    disagreements: list[Disagreement] = Field(default_factory=list, description="Points of tension or unresolved debate in this chunk.")
    glossary: list[GlossaryTerm] = Field(default_factory=list, description="Domain terms, acronyms or jargon used in this chunk.")


# The ordered categories, shared by the map merge, the sweeps and the renderer.
# (state field, human label, singular noun for prompts)
CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("topics", "Topics", "topic discussed"),
    ("decisions", "Decisions", "decision or agreement"),
    ("action_items", "Action items", "action item, task, commitment or follow-up"),
    ("open_questions", "Open questions", "unresolved question"),
    ("risks", "Risks and blockers", "risk, concern, blocker or dependency"),
    ("key_facts", "Key facts and figures", "concrete fact, metric or figure"),
    ("key_dates", "Dates and deadlines", "date, deadline or scheduled event"),
    ("entities", "People and entities", "named person, organization, product, system, tool or document"),
    ("quotes", "Notable quotes", "notable verbatim quote"),
    ("disagreements", "Disagreements", "point of tension or unresolved debate"),
    ("glossary", "Glossary", "domain term, acronym or jargon"),
)

_ITEM_TYPES: dict[str, type[BaseModel]] = {
    "topics": Topic,
    "decisions": Decision,
    "action_items": ActionItem,
    "open_questions": OpenQuestion,
    "risks": Risk,
    "key_facts": KeyFact,
    "key_dates": KeyDate,
    "entities": Entity,
    "quotes": Quote,
    "disagreements": Disagreement,
    "glossary": GlossaryTerm,
}


def item_type(field: str) -> type[BaseModel]:
    return _ITEM_TYPES[field]


# --------------------------------------------------------------------------- #
# Sweep output: "what did earlier passes MISS for this one category?"
# --------------------------------------------------------------------------- #

class SweepResult(BaseModel):
    """Items of a single category found on a re-read that were NOT already listed.

    Return only genuinely new items. If everything is already covered, return an
    empty list — that is the signal that this category is exhausted.
    """

    new_items: list[dict] = Field(
        default_factory=list,
        description="Each new item as an object with the same fields as the category's schema "
        "(shown in the prompt). Only items missing from the provided list; [] if none.",
    )


# --------------------------------------------------------------------------- #
# Plan-phase output
# --------------------------------------------------------------------------- #

class MeetingPlan(BaseModel):
    """Output of the plan node: how to read this specific meeting."""

    title: str = Field(description="A concise title for the meeting, inferred from the content.")
    meeting_type: str = Field(description="One line naming the kind of meeting (standup, planning, review, sales call, interview, ...).")
    focus_areas: list[str] = Field(description="3-6 things the extraction should pay special attention to for THIS meeting.")
    likely_participants: list[str] = Field(default_factory=list, description="Participant names inferred from speakers and context.")
    notes: str = Field(default="", description="Any other observation useful for the downstream extraction.")


# --------------------------------------------------------------------------- #
# Reduce-phase output: the meeting's prose, synthesized from the item lists
# --------------------------------------------------------------------------- #

class ParticipantRole(BaseModel):
    """A participant and the role/contribution they played in the meeting."""

    name: str = Field(description="Participant name.")
    role: str = Field(default="", description="Their apparent role or main contribution, one phrase.")


class MeetingSynthesis(BaseModel):
    """Output of the reduce node: the report's prose, grounded in the item lists.

    Write from the consolidated items and the transcript overview only. Ground
    every statement in what was extracted; do not invent participants, systems
    or outcomes that are not present.
    """

    title: str = Field(description="Final meeting title.")
    meeting_type: str = Field(description="The kind of meeting.")
    executive_summary: str = Field(description="3-6 sentences: what the meeting was about, the main outcomes, and what happens next.")
    detailed_summary: str = Field(description="A thorough narrative of the meeting in chronological order, as flowing prose with paragraph breaks. Reference topics and decisions as they arose.")
    overall_sentiment: str = Field(default="", description="One or two sentences on the tone and level of alignment (collaborative, tense, decisive, inconclusive, ...).")
    next_steps: list[str] = Field(default_factory=list, description="The immediate next steps as short bullet statements, distilled from the action items and decisions.")
    participants: list[ParticipantRole] = Field(default_factory=list, description="Each participant and their role/contribution.")
    coverage_gaps: list[str] = Field(
        default_factory=list,
        description="Category names (from the list you were given) that still look incompletely "
        "captured and deserve another full re-read. Empty if the extraction looks complete.",
    )
