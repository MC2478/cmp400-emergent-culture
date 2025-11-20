"""I wrap LM Studio's OpenAI-style endpoint for the CMP400 feasibility demo so my single agent can
request structured JSON decisions without depending on llama-cpp or local DLLs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict
import logging

import requests

import config

log = logging.getLogger(__name__)

# I mirror the allowed action list locally (the source lives in ``LeaderAgent``) so the prompt
# instructions and validation stay consistent without creating circular imports.
_ALLOWED_ACTIONS: tuple[str, ...] = (
    "focus_food",
    "focus_wood",
    "focus_wealth",
    "build_infrastructure",
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
        name = state.get("territory", "Unknown")
        food = state.get("food", 0.0)
        wealth = state.get("wealth", 0.0)
        wood = state.get("wood", 0.0)
        pop = state.get("population", 0.0)
        required_food = state.get("required_food", 0.0)
        infra = state.get("infra", state.get("infrastructure_level", 0))
        work_points = state.get("work_points", 0)
        step = state.get("step", "unknown")
        relation = state.get("relation_to_neighbor", "neutral")
        relation_score = state.get("relation_score", "unknown")
        yields = state.get("yields", {})
        priority_hint = state.get("priority_hint", {})

        prompt = f"""
You are the autonomous leader of the territory "{name}" at simulation step {step}.
Population: {pop:.0f} people, requiring {required_food:.2f} food per step to avoid starvation.
Current resources: food={food:.2f}, wealth={wealth:.2f}, wood={wood:.2f}, infrastructure level={infra}.
Work points available this step: {work_points} (100 population ≈ 1 work point adjusted by morale).

Per-work yields with current infrastructure:
  - focus_food:  {yields.get('food_per_work', 0.0):.3f} food/work
  - focus_wood:  {yields.get('wood_per_work', 0.0):.3f} wood/work
  - focus_wealth: {yields.get('wealth_per_work', 0.0):.3f} wealth/work

Soft priority hint (you may override this):
  - food_safety_ratio (food vs. next {config.FOOD_SAFETY_HORIZON_STEPS} steps): {priority_hint.get('food_safety_ratio', 0.0):.3f}
  - suggested weights: {priority_hint.get('priorities', {})}

Available actions (choose exactly one):
  - "focus_food": devote all work points to growing food using the food_per_work yield.
  - "focus_wood": devote all work points to growing wood using the wood_per_work yield.
  - "focus_wealth": devote all work points to growing wealth using the wealth_per_work yield.
  - "build_infrastructure": if you have at least {config.INFRA_COST_WOOD} wood and {config.INFRA_COST_WEALTH} wealth, consume them to raise infrastructure by 1, permanently boosting yields.
  - "wait": conserve resources and do nothing this step.

Objectives (in soft order):
1. Avoid starvation in the short and medium term.
2. Build resilience by investing in infrastructure when food safety allows.
3. Develop prosperity (wealth/wood) to unlock future options.

You can follow or override the hint depending on the situation. Consider trade-offs between immediate survival and longer-term strength.

Respond with a single JSON object of the form:
{{"action": "<focus_food|focus_wood|focus_wealth|build_infrastructure|wait>", "target": "None", "reason": "<why this is best now>"}}

Do not include extra keys or commentary outside this JSON.
"""
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
        try:
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
                timeout=10,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            log.warning("LLM negotiation request failed (%s); falling back to heuristic trade.", exc)
            return self._fallback_negotiation()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("LLM negotiation response parsing failed (%s); falling back to heuristic trade.", exc)
            return self._fallback_negotiation()
        parsed = self._parse_response(text)

        if parsed is None:
            log.warning("LLM negotiation JSON invalid; using fallback trade.")
            return self._fallback_negotiation()

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
        try:
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
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            log.warning("LLM decision request failed (%s); using fallback policy.", exc)
            return self._fallback_decision("request failed")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("LLM decision parsing failed (%s); using fallback policy.", exc)
            return self._fallback_decision("response parsing failed")

        parsed = self._parse_response(text)
        if parsed is None:
            log.warning("LLM decision JSON invalid; using fallback policy.")
            return self._fallback_decision("invalid JSON")

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

    def _fallback_negotiation(self) -> Dict[str, Any]:
        """I return a safe no-trade negotiation result."""
        return {
            "east_line": "Let's hold steady for now.",
            "west_line": "Agreed, we can revisit later.",
            "trade": {
                "food_from_east_to_west": 0,
                "wealth_from_west_to_east": 0,
                "reason": "fallback trade",
            },
        }

    def _fallback_decision(self, reason: str) -> Dict[str, Any]:
        """I provide a neutral fallback decision payload when the LLM is unreachable."""
        return {
            "action": None,
            "target": "None",
            "reason": f"LLM decision fallback: {reason}",
        }
