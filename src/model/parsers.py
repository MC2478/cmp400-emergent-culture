"""Quick card: helpers to coerce LLM text into structured decisions."""

from __future__ import annotations

import json
from typing import Any, Dict


def parse_json_response(raw_text: str) -> Dict[str, Any] | None:
    """Parsing cue: do my best to fish a JSON object out of messy LLM text."""

    def _try_load(candidate: str) -> Dict[str, Any] | None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    text = (raw_text or "").strip()
    if not text:
        return None

    candidate = text
    result = _try_load(candidate)
    if result is not None:
        return result

    if "{" in candidate:
        candidate = candidate[candidate.find("{") :]
        result = _try_load(candidate)
        if result is not None:
            return result

    if "}" in candidate:
        last_brace = candidate.rfind("}")
        candidate = candidate[: last_brace + 1]
        result = _try_load(candidate)
        if result is not None:
            return result

    candidate = candidate.replace("'", '"')
    result = _try_load(candidate)
    if result is not None:
        return result

    # Last ditch: sweep the string for balanced JSON chunks and keep the last sane one.
    in_string = False
    escape = False
    depth = 0
    start = None
    last_result: Dict[str, Any] | None = None
    for idx, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start : idx + 1]
                    result = _try_load(chunk)
                    if result is not None:
                        last_result = result
                    start = None
    return last_result


def extract_action_hint(text: str, allowed_actions: tuple[str, ...]) -> Dict[str, Any] | None:
    """Heuristic cue: if JSON fails, look for a loose action keyword to salvage intent."""
    lower = (text or "").lower()
    for action in allowed_actions:
        if action in lower:
            return {"action": action}
    return None


def sanitise_allocations(raw: Any, allowed_actions: tuple[str, ...]) -> Dict[str, float]:
    """Cleanup note: filter allocation dicts and normalise shares safely."""
    allocations: Dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key not in allowed_actions:
                continue
            try:
                share = float(value)
            except (TypeError, ValueError):
                continue
            if share <= 0:
                continue
            allocations[key] = share

    total_share = sum(allocations.values())
    if total_share > 1e-6:
        if total_share > 1.0:
            allocations = {k: v / total_share for k, v in allocations.items()}
    else:
        allocations = {}
    return allocations
