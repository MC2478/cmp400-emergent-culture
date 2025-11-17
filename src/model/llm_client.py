"""I wrap LM Studio's OpenAI-style endpoint and translate world state to JSON-friendly replies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import json

import requests


@dataclass
class LLMConfig:
    """I store the base_url, model name, temperature, max_tokens, and timeout for LM Studio at 127.0.0.1."""
    base_url: str = "http://127.0.0.1:1234/v1/chat/completions"
    model: str = "meta-llama-3.1-8b-instruct"
    temperature: float = 0.1
    max_tokens: int = 128
    timeout: int = 30


class LLMDecisionClient:
    """I am a thin HTTP client that can be toggled off to keep the simulation rule-based."""

    def __init__(self, config: LLMConfig | None = None, enabled: bool = False):
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
        # I embed the current readings and remind the model of the restricted action space.
        prompt = (
            f"You govern the territory named {territory}.\n"
            f"Current step: {step}.\n"
            f"Resources -> food: {food}, wealth: {wealth}, relation to neighbor: {relation}.\n"
            "Choose exactly one action from: \"gather\", \"consume\", \"wait\".\n"
            "Respond ONLY with JSON like {\"action\": ..., \"target\": \"None\", \"reason\": ...}."
        )
        # I restate the {"action","target","reason"} envelope so parsing stays simple
        return prompt

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """I send the composed prompt to LM Studio and interpret the JSON reply."""
        # I ensure callers do not make HTTP requests when this client is disabled
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        # I convert the structured state into a textual promp
        prompt = self.compose_prompt(state)
        # I follow the OpenAI chat-completion schema so LM Studio can understand the request
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

        decision: Dict[str, Any] = {}
        try:
            # I expect clean JSON so I try a straight parse first
            decision = json.loads(text)
        except json.JSONDecodeError:
            # I fall back to a lightweight cleanup when the model adds noise.
            text_fixed = text.replace("'", '"')
            brace_idx = text_fixed.rfind("}")
            if brace_idx != -1:
                text_fixed = text_fixed[: brace_idx + 1]
            try:
                decision = json.loads(text_fixed)
            except json.JSONDecodeError:
                decision = {}

        # I normalize the output so downstream callers always receive defaults
        return {
            "action": decision.get("action", "wait"),
            "target": decision.get("target", "None"),
            "reason": decision.get("reason", "no reason provided"),
        }
