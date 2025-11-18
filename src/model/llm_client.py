"""I wrap LM Studio's OpenAI-style endpoint for the CMP400 feasibility demo so my single agent can
request structured JSON decisions without depending on llama-cpp or local DLLs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import requests

# I mirror the allowed action list locally (the source lives in ``LeaderAgent``) so the prompt
# instructions and validation stay consistent without creating circular imports.
_ALLOWED_ACTIONS: tuple[str, ...] = (
    "focus_food",
    "focus_wealth",
    "balanced",
    "wait",
)

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
        population = state.get("population", "unknown")
        required_food = state.get("required_food", "unknown")
        relation = state.get("relation_to_neighbor", "neutral")
        step = state.get("step", "unknown")
        neighbor_name = state.get("neighbor_name", "West")
        neighbor_food = state.get("neighbor_food", "unknown")
        neighbor_wealth = state.get("neighbor_wealth", "unknown")
        neighbor_population = state.get("neighbor_population", "unknown")
        max_food_ff = state.get("max_food_if_focus_food", "unknown")
        max_food_fw = state.get("max_food_if_focus_wealth", "unknown")
        max_wealth_ff = state.get("max_wealth_if_focus_food", "unknown")
        max_wealth_fw = state.get("max_wealth_if_focus_wealth", "unknown")
        can_hit_ff = state.get("can_meet_quota_if_focus_food", "unknown")
        can_hit_fw = state.get("can_meet_quota_if_focus_wealth", "unknown")
        # I give the LLM a minimal but clear action menu so it contrasts food vs wealth priorities.
        actions_hint = (
            '- "focus_food": allocate roughly 75% of work points to food production (rest to wealth).\n'
            '- "focus_wealth": allocate roughly 75% of work points to wealth production (rest to food).\n'
            '- "balanced": split work points about evenly between food and wealth.\n'
            '- "wait": allocate zero work points; resources will not grow this step.\n'
        )
        # I embed the current readings, including the neighbour snapshot, so the LLM sees the small economic contrast.
        prompt = (
            f"You currently lead {territory} at step {step}.\n"
            f"Population: {population} people needing about {required_food} food this step to avoid starvation.\n"
            f"Work points available this step: {state.get('work_points', 'unknown')} (100 population ≈ 1 point).\n"
            f"Your resources -> food: {food}, wealth: {wealth}.\n"
            f"The neighbouring territory {neighbor_name} has food {neighbor_food}, wealth {neighbor_wealth}, "
            f"population {neighbor_population}, and the relationship is {relation}.\n"
            f"Current relationship score: {state.get('relation_score', 'unknown')}.\n"
            "Production depends entirely on work allocation: 100 population = 1 work point. "
            "There is no automatic production, so if you wait you gain nothing.\n"
            f"If you focus on food you can reach food ≈ {max_food_ff} and wealth ≈ {max_wealth_ff} "
            f"(meets quota? {can_hit_ff}). If you focus on wealth you can reach food ≈ {max_food_fw} "
            f"and wealth ≈ {max_wealth_fw} (meets quota? {can_hit_fw}).\n"
            "Choose exactly one of these actions:\n"
            f"{actions_hint}"
            "Respond with ONE JSON object like "
            '{"action": <allowed_action>, "target": "None", "reason": <short sentence>}.\n'
            "Avoiding population loss is the top priority: if focus_food can meet the quota and food is near the requirement, prioritise focus_food. "
            "If even focus_food cannot reach the quota, consider focus_wealth to earn trade resources.\n"
            "Do not output explanations or code fences, only that JSON.\n"
        )
        # I encode the survival-first goal in the prompt so the LLM keeps the population alive for the demo narrative.
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

    def compose_negotiation_prompt(self, state: Dict[str, Any]) -> str:
        """I outline the joint state so the LLM can improvise a mini dialogue and trade."""
        step = state.get("step", "unknown")
        east = state.get("east", {})
        west = state.get("west", {})
        last_actions = state.get("last_actions", {})
        prompt = (
            f"Simulate a brief negotiation at step {step} between two leaders, East and West (this happens before upkeep/starvation each tick).\n"
            f"Current relationship status: {east.get('relation_to_neighbor', 'neutral')} (score {east.get('relation_score', 0)}).\n"
            f"East -> food: {east.get('food', 'unknown')}, wealth: {east.get('wealth', 'unknown')}, "
            f"population: {east.get('population', 'unknown')}.\n"
            f"West -> food: {west.get('food', 'unknown')}, wealth: {west.get('wealth', 'unknown')}, "
            f"population: {west.get('population', 'unknown')}.\n"
            f"Last actions -> East: {last_actions.get('east')}, West: {last_actions.get('west')}.\n"
            "Output a short dialogue line for each leader plus a proposed trade of food/wealth that keeps flows small.\n"
            'Respond with JSON like {"east_line": "...", "west_line": "...", '
            '"trade": {"food_from_east_to_west": 0, "wealth_from_west_to_east": 0, "reason": "..."}}.\n'
            "Positive food_from_east_to_west means East ships food to West (negative means West ships food to East). "
            "Positive wealth_from_west_to_east means West sends wealth to East (negative means the reverse). "
            "Keep trades realistic: you cannot send more of a resource than you currently have, and the overall goal is to avoid starvation. "
            "If no trade is beneficial, set both flows to 0.\n"
            "If one side cannot meet its food requirement even when focusing on food, trading for food before upkeep is high priority.\n"
            "Generous gifts (sending food or wealth without repayment) improve relations. Exploitative deals (one side clearly gains at the other's expense) harm relations.\n"
            "Do not output explanations or code fences, only that JSON.\n"
        )
        # I keep this prompt short because I only need a concise negotiation snippet for the feasibility demo.
        return prompt

    def negotiate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """I call LM Studio to improvise a dialogue and trade between East and West."""
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        prompt = self.compose_negotiation_prompt(state)
        response = requests.post(
            self.config.base_url,
            json={
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are simulating a calm negotiation between East and West.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        parsed = self._parse_response(text)

        if parsed is None:
            print("LLMDecisionClient: failed to parse negotiation JSON, defaulting to no trade.")
            return {
                "east_line": "Let's hold steady for now.",
                "west_line": "Agreed, we can revisit later.",
                "trade": {
                    "food_from_east_to_west": 0,
                    "wealth_from_west_to_east": 0,
                    "reason": "fallback trade",
                },
            }

        trade = parsed.get("trade") or {}

        def _sanitise_flow(key: str) -> int:
            value = trade.get(key, 0)
            try:
                delta = int(value)
            except (ValueError, TypeError):
                delta = 0
            return max(-5, min(5, delta))

        reason_text = str(trade.get("reason", "LLM proposed this trade.")).strip() or "LLM proposed this trade."
        decision = {
            "east_line": str(parsed.get("east_line", "")).strip() or "Let's hold steady for now.",
            "west_line": str(parsed.get("west_line", "")).strip() or "Agreed, we can revisit later.",
            "trade": {
                "food_from_east_to_west": _sanitise_flow("food_from_east_to_west"),
                "wealth_from_west_to_east": _sanitise_flow("wealth_from_west_to_east"),
                "reason": reason_text,
            },
        }
        # I keep the negotiation output structured so the chronicle can capture both dialogue and trade succinctly.
        return decision

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
