"""Small helpers for bounded JSON responses from LLM stages."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, accepting fenced Markdown wrappers."""
    cleaned = str(text or "").strip()
    if not cleaned:
        raise ValueError("LLM response was empty.")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    elif not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response did not contain a JSON object.")
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON was not an object.")
    return parsed


def list_of_strings(value: Any, *, default: list[str] | None = None) -> list[str]:
    """Normalize a JSON value into a compact list of strings."""
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value] if value.strip() else list(default or [])
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or list(default or [])
    return [str(value).strip()] if str(value).strip() else list(default or [])


def limited_evidence_payload(evidence: list[dict], limit: int = 5) -> list[dict]:
    """Keep prompts bounded while retaining auditable evidence identifiers."""
    payload = []
    for item in evidence[:limit]:
        payload.append({
            "evidence_id": item.get("evidence_id") or item.get("source_id"),
            "title": item.get("title"),
            "kind": item.get("kind") or item.get("source_type"),
            "section": item.get("section"),
            "support_level": item.get("support_level"),
            "text": str(
                item.get("excerpt")
                or item.get("text")
                or item.get("summary")
                or ""
            )[:900],
        })
    return payload
