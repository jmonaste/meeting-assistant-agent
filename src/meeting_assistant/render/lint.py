"""Style lint applied to every generated document.

The deliverable must be clean and emoji-free regardless of what the model wrote,
so this is enforced here deterministically rather than requested politely in
prompts.
"""

from __future__ import annotations

import re

# Pictographs, emoji, dingbats, transport symbols, flags, variation selectors.
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U00002700-\U000027bf"
    "\U00002600-\U000026ff"
    "\U0001f1e6-\U0001f1ff"
    "\U0000fe0e-\U0000fe0f"
    "\U0000200d"
    "\U000020e3"
    "\U00002b00-\U00002bff"
    "]+"
)

_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXTRA_BLANK_RE = re.compile(r"\n{3,}")


def lint_markdown(text: str) -> str:
    """Strip emojis, trailing whitespace and excess blank lines."""
    text = _EMOJI_RE.sub("", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _EXTRA_BLANK_RE.sub("\n\n", text)
    return text.strip() + "\n"


def md_cell(text: str) -> str:
    """Make a string safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def md_anchor(heading: str) -> str:
    """GitHub/Obsidian-style anchor for a heading."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[\s]+", "-", slug)
