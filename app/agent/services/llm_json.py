"""Small helpers for bounded JSON responses from LLM stages."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_value(text: str) -> Any:
    """Parse a JSON value, accepting fenced Markdown wrappers."""
    cleaned = str(text or "").strip()
    if not cleaned:
        raise ValueError("LLM response was empty.")
    fenced = re.search(
        r"```(?:json)?\s*([\[{].*?[\]}])\s*```",
        cleaned,
        re.DOTALL,
    )
    if fenced:
        cleaned = fenced.group(1)
    elif not cleaned.startswith(("{", "[")):
        object_start = cleaned.find("{")
        array_start = cleaned.find("[")
        starts = [index for index in (object_start, array_start) if index != -1]
        if not starts:
            raise ValueError("LLM response did not contain JSON.")
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end == -1 or end <= start:
            raise ValueError("LLM response did not contain complete JSON.")
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, accepting fenced Markdown wrappers."""
    parsed = parse_json_value(text)
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
        items = [_stringify_item(item) for item in value]
        items = [item for item in items if item]
        return items or list(default or [])
    item = _stringify_item(value)
    return [item] if item else list(default or [])


def _stringify_item(value: Any) -> str:
    """Render LLM JSON values without Python dict/list repr leakage."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        preferred = [
            value.get(key)
            for key in (
                "name",
                "title",
                "tool_name",
                "scenario_id",
                "component",
                "metric",
            )
            if value.get(key)
        ]
        summary_parts = [
            str(value.get(key)).strip()
            for key in (
                "description",
                "source",
                "function",
                "variation",
                "trigger",
                "risk_level",
            )
            if str(value.get(key) or "").strip()
        ]
        if preferred:
            head = str(preferred[0]).strip()
            if summary_parts:
                return f"{head}: {'; '.join(summary_parts)}"
            return head
        parts = [
            f"{key}: {_stringify_item(item)}"
            for key, item in value.items()
            if _stringify_item(item)
        ]
        return "; ".join(parts)
    if isinstance(value, list):
        return "; ".join(
            item for item in (_stringify_item(item) for item in value) if item
        )
    return str(value).strip()


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value for common LLM schema aliases."""
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return None


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
