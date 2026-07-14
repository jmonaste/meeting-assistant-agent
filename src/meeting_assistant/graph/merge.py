"""Deterministic consolidation of extracted items across chunks and sweeps.

The map and sweep phases produce many :class:`ChunkExtraction` objects that
overlap heavily (chunks overlap; sweeps re-scan the whole transcript). This
module merges them into one deduplicated set of item lists per category. Merging
is idempotent: it can be recomputed from the full harvest at any point, which is
what lets the sweep loop rebuild ``items`` after every round without double
counting.

De-duplication keeps the first occurrence of an item (by its ``dedup_key``) and
enriches it from later duplicates: segment ids are unioned and empty fields are
backfilled, so the surviving item is the most complete version seen.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..model.extraction import CATEGORIES, ChunkExtraction

_FIELDS = [field for field, _label, _noun in CATEGORIES]


def _enrich(keeper: BaseModel, other: BaseModel) -> None:
    """Fill blank scalar fields on ``keeper`` from ``other`` and union segment ids."""
    for name in type(keeper).model_fields:
        kv = getattr(keeper, name)
        ov = getattr(other, name, None)
        if name == "segment_ids":
            merged = list(dict.fromkeys([*(kv or []), *(ov or [])]))
            setattr(keeper, name, merged)
        elif isinstance(kv, str) and not kv.strip() and isinstance(ov, str) and ov.strip():
            setattr(keeper, name, ov)


def consolidate(harvest: list[ChunkExtraction]) -> tuple[dict[str, list], int]:
    """Merge every extraction into deduplicated per-category lists.

    Returns ``(items, total_count)`` where ``items[field]`` is a list of item
    models and ``total_count`` is the number of items across all categories.
    """
    buckets: dict[str, dict[str, BaseModel]] = {f: {} for f in _FIELDS}
    for extraction in harvest:
        for field in _FIELDS:
            for item in getattr(extraction, field, None) or []:
                key = item.dedup_key()
                if not key:
                    continue
                existing = buckets[field].get(key)
                if existing is None:
                    buckets[field][key] = item
                else:
                    _enrich(existing, item)

    items = {f: list(buckets[f].values()) for f in _FIELDS}
    total = sum(len(v) for v in items.values())
    return items, total


def render_known_items(items: dict[str, list], max_per_category: int = 40) -> str:
    """A compact digest of what is already found, shown to a coverage sweep.

    The sweep is told to return only items NOT in this list, so the digest must
    identify each item unambiguously but stay small.
    """
    lines: list[str] = []
    for field, label, _noun in CATEGORIES:
        bucket = items.get(field) or []
        if not bucket:
            continue
        lines.append(f"{label} ({len(bucket)} already found):")
        for item in bucket[:max_per_category]:
            lines.append(f"  - {_one_line(item)}")
        if len(bucket) > max_per_category:
            lines.append(f"  (and {len(bucket) - max_per_category} more)")
    return "\n".join(lines) or "(nothing found yet)"


def _one_line(item: BaseModel) -> str:
    """Best-effort single-line identity of an item for the known-items digest."""
    for name in ("task", "decision", "question", "description", "fact", "term", "title", "name", "text", "topic"):
        val = getattr(item, name, None)
        if isinstance(val, str) and val.strip():
            return val.strip()[:140]
    # Fall back to the first non-empty string field.
    for name in type(item).model_fields:
        val = getattr(item, name, None)
        if isinstance(val, str) and val.strip():
            return val.strip()[:140]
    return "(item)"
