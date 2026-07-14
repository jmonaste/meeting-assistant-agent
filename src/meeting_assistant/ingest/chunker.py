"""Split the segment list into overlapping, budget-sized chunks.

The map phase reviews one chunk at a time in an isolated context. Chunks overlap
by a configurable number of characters so an item whose statement spans a chunk
boundary is still seen whole by at least one chunk. Chunking is deterministic
and content-hashed, so re-processing an unchanged transcript reuses the cache.
"""

from __future__ import annotations

from ..model.transcript import Chunk, Segment, hash_text


def build_chunks(
    segments: list[Segment],
    with_timestamp: bool,
    max_chars: int = 9_000,
    overlap_chars: int = 800,
) -> list[Chunk]:
    """Greedily pack rendered segments into chunks of at most ``max_chars``.

    Each new chunk begins a few segments back from where the previous one ended,
    repeating about ``overlap_chars`` of text for continuity. Always makes
    forward progress, even when a single segment exceeds the budget.
    """
    if not segments:
        return []
    lines = [s.render(with_timestamp) for s in segments]
    lengths = [len(ln) + 1 for ln in lines]  # +1 for the newline join
    n = len(segments)
    max_chars = max(max_chars, 500)
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    chunks: list[Chunk] = []
    start = 0
    while start < n:
        used = 0
        end = start
        while end < n and (used + lengths[end] <= max_chars or end == start):
            used += lengths[end]
            end += 1
        # end is exclusive; segments [start, end) form this chunk.
        text = "\n".join(lines[start:end])
        chunks.append(
            Chunk(
                index=len(chunks),
                first_sid=segments[start].sid,
                last_sid=segments[end - 1].sid,
                text=text,
                content_hash=hash_text(text),
            )
        )
        if end >= n:
            break
        # Repeat at least the previous chunk's last segment, then extend the
        # overlap further back while it stays within the overlap budget. Always
        # advance `start` so a giant single segment cannot stall the loop.
        next_start = end - 1
        back = lengths[end - 1]
        while next_start - 1 > start and back + lengths[next_start - 1] <= overlap_chars:
            back += lengths[next_start - 1]
            next_start -= 1
        start = max(next_start, start + 1)
    return chunks
