"""I wrap LM Studio's OpenAI-style endpoint for the CMP400 feasibility demo so my single agent can
request structured JSON decisions without depending on llama-cpp or local DLLs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import requests

# I mirror the allowed action list locally (the source lives in ``LeaderAgent``) so the prompt
# instructions and validation stay consistent without creating circular imports.
_ALLOWED_ACTIONS: tuple[str, ...] = ("gather", "consume", "wait")

@dataclass
class LLMConfig:
    """I keep the LM Studio connection details (URL, model, decoding params) used throughout the demo."""

    base_url: str = "http://127.0.0.1:1234/v1/chat/completions"
    model: str = "meta-llama-3.1-8b-instruct"
    temperature: float = 0.1
    max_tokens: int = 128
    timeout: int = 30


class LLMDecisionClient:
    """I am a thin `requests`-powered helper that speaks OpenAI's schema to LM Studio and falls back
    cleanly when disabled so I can compare LLM and rule-based behaviour."""

    def __init__(self, config: LLMConfig | None = None, enabled: bool = False) -> None:
        """I store the configuration and whether the client is currently active."""
        self.config = config or LLMConfig()
        self.enabled = enabled

    def compose_prompt(self, state: Dict[str, Any]) -> str:
        """I describe the state, enumerate valid actions, and demand JSON output."""
        territory = state.get("territory") or state.get("name", "Unknown")
        food = state.get("food", "unknown")
        wealth = state.get("wealth", "unknown")
        relation = state.get("relation_to_neighbor", "neutral")
        step = state.get("step", "unknown")
        actions_hint = ", ".join(f'"{action}"' for action in _ALLOWED_ACTIONS)
        # I embed the current readings and remind the model of the restricted action space.
        prompt = (
            f"You govern the territory named {territory}.\n"
            f"Current step: {step}.\n"
            f"Resources -> food: {food}, wealth: {wealth}, relation to neighbor: {relation}.\n"
            f"Choose exactly one action from: {actions_hint}.\n"
            "You MUST respond with a single JSON object containing exactly the keys "
            "\"action\", \"target\", and \"reason\".\n"
            "The \"action\" must be one of the allowed actions above, \"target\" must be \"None\" "
            "for this prototype, and \"reason\" must be a single short sentence.\n"
            "Respond ONLY with that JSON object (no code fences, no commentary).\n"
        )
        # I make the JSON instructions painfully explicit so the Week 11 feasibility demo stays reliable.
        return prompt

    def _parse_response(self, raw_text: str) -> Dict[str, Any] | None:
        """I run a few safe heuristics to coerce the response into JSON."""

        def _try_load(candidate: str) -> Dict[str, Any] | None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        text = raw_text.strip()
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
        return _try_load(candidate)

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """I send the composed prompt to LM Studio and interpret the JSON reply."""
        # I ensure callers do not make HTTP requests when this client is disabled
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        # I convert the structured state into a textual prompt.
        prompt = self.compose_prompt(state)
        # I follow the OpenAI chat-completion schema so LM Studio can understand the request and I do everything via HTTP requests.
        response = requests.post(
            self.config.base_url,
            json={
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful decision-making ruler agent.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
            timeout=self.config.timeout,
        )
        # I raise if the HTTP layer fails so the caller can handle errors.
        response.raise_for_status()
        data = response.json()
        # I pull the assistant reply text from the first choice.
        text = data["choices"][0]["message"]["content"].strip()

        parsed = self._parse_response(text)
        if parsed is None:
            print("LLMDecisionClient: failed to parse JSON, falling back to rule-based policy.")
            return {"action": None, "target": "None", "reason": "LLM output was not valid JSON."}

        action_raw = str(parsed.get("action", "")).strip().lower()
        action_value = action_raw if action_raw in _ALLOWED_ACTIONS else None

        target_value = str(parsed.get("target", "None")).strip() or "None"
        if target_value.lower() != "none":
            target_value = "None"

        reason_value = str(parsed.get("reason", "")).strip()
        if not reason_value:
            reason_value = "No reason provided by model."

        # I normalize the output so downstream callers always receive defaults
        return {
            "action": action_value,
            "target": target_value,
            "reason": reason_value,
        }
