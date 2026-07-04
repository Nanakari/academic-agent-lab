"""Filter diagnostic placeholders that are not scientific claims."""

from __future__ import annotations


NON_VERIFIABLE_MARKERS = (
    "not explicitly stated",
    "not explicitly stated in the local evidence",
    "evidence is insufficient",
    "cannot be established",
    "cannot yet be established",
    "local evidence coverage is insufficient",
    "no relevant method was found",
    "a defensible gap cannot",
    "原文未明确说明",
    "证据不足",
    "无法建立",
    "无法从证据中确定",
)


def is_verifiable_claim(text: str) -> bool:
    """Return false for empty, very short, or evidence-status messages."""
    normalized = " ".join(str(text or "").casefold().split())
    if len(normalized) < 12:
        return False
    return not any(marker in normalized for marker in NON_VERIFIABLE_MARKERS)
