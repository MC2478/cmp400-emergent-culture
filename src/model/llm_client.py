"""I wrap LM Studio's OpenAI-style endpoint for the CMP400 feasibility demo so my single agent can
request structured JSON decisions without depending on llama-cpp or local DLLs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List
import logging

import requests

import config

log = logging.getLogger(__name__)


def summarise_memory_for_prompt(events: List[Dict[str, Any]], max_events: int = 10) -> str:
    """I condense recent in-run memory events into a readable history for the LLM."""
    if not events:
        return "No previous steps in this run."

    recent = events[-max_events:]
    lines: list[str] = []
    for event in recent:
        pop_before = float(event.get("pop_before", 0.0))
        pop_after = float(event.get("pop_after", pop_before))
        food_before = float(event.get("food_before", 0.0))
        food_after = float(event.get("food_after", food_before))
        wealth_before = float(event.get("wealth_before", 0.0))
        wealth_after = float(event.get("wealth_after", wealth_before))
        pop_delta = pop_after - pop_before
        food_delta = food_after - food_before
        wealth_delta = wealth_after - wealth_before

        outcome_bits: list[str] = []
        if event.get("starving"):
            outcome_bits.append("people starved")
        if event.get("strike"):
            outcome_bits.append("workers on strike")
        if pop_delta > 0:
            outcome_bits.append(f"population grew by {int(round(pop_delta))}")
        elif pop_delta < 0:
            outcome_bits.append(f"population fell by {int(round(abs(pop_delta)))}")
        if food_delta > 0:
            outcome_bits.append(f"food +{food_delta:.2f}")
        elif food_delta < 0:
            outcome_bits.append(f"food {food_delta:.2f}")
        if wealth_delta > 0:
            outcome_bits.append(f"wealth +{wealth_delta:.2f}")
        elif wealth_delta < 0:
            outcome_bits.append(f"wealth {wealth_delta:.2f}")
        note = event.get("note")
        if note:
            trimmed = str(note).strip()
            if len(trimmed) > 160:
                trimmed = trimmed[:157] + "..."
            if trimmed:
                outcome_bits.append(f"note: {trimmed}")

        outcome = ", ".join(outcome_bits) if outcome_bits else "no major changes"
        lines.append(
            f"Step {event.get('step')}: action={event.get('action')}, "
            f"food {food_before:.2f}->{food_after:.2f}, "
            f"wealth {wealth_before:.2f}->{wealth_after:.2f}, "
            f"pop {int(round(pop_before))}->{int(round(pop_after))} ({outcome})"
        )

    return "\n".join(lines)

# I mirror the allowed action list locally (the source lives in ``LeaderAgent``) so the prompt
# instructions and validation stay consistent without creating circular imports.
_WORK_ACTIONS: tuple[str, ...] = ("focus_food", "focus_wood", "focus_wealth")

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
        history_text = state.get("history_text") or "No previous steps in this run."
        prior_directive = state.get("self_directive") or "No prior directive recorded."
        interaction_text = state.get("interaction_text") or "No notable interactions recorded."
        current_season = state.get("current_season", "unknown")
        next_season = state.get("next_season", "unknown")
        current_season_mult = config.SEASON_MULTIPLIERS.get(current_season, 1.0)
        next_season_mult = config.SEASON_MULTIPLIERS.get(next_season, 1.0)

        horizon_steps = config.FOOD_SAFETY_HORIZON_STEPS
        food_need_horizon = required_food * horizon_steps
        food_gap = food - food_need_horizon
        wood_gap = wood - config.INFRA_COST_WOOD
        wealth_gap = wealth - config.INFRA_COST_WEALTH
        workers = pop / config.PEOPLE_PER_WORK_POINT if config.PEOPLE_PER_WORK_POINT else 0.0
        wage_bill = workers * config.WAGE_PER_WORKER
        wage_gap = wealth - wage_bill

        def _gap_status(value: float) -> str:
            return "OK" if value >= 0 else "SHORTFALL"

        feasibility_snapshot = f"""
Resource feasibility snapshot (gap = have - need):
  - Food buffer (next {horizon_steps} steps): need {food_need_horizon:.2f}, have {food:.2f}, gap {food_gap:.2f} ({_gap_status(food_gap)}).
  - Wood for infrastructure: need {config.INFRA_COST_WOOD:.2f}, have {wood:.2f}, gap {wood_gap:.2f} ({_gap_status(wood_gap)}).
  - Wealth for infrastructure: need {config.INFRA_COST_WEALTH:.2f}, have {wealth:.2f}, gap {wealth_gap:.2f} ({_gap_status(wealth_gap)}).
  - Wage bill this step (~{wage_bill:.2f} wealth): gap {wage_gap:.2f} ({_gap_status(wage_gap)}).
"""

        prompt = f"""
You are the autonomous leader of the territory "{name}" at simulation step {step}.
Population: {pop:.0f} people, requiring {required_food:.2f} food per step to avoid starvation.
Current resources: food={food:.2f}, wealth={wealth:.2f}, wood={wood:.2f}, infrastructure level={infra}.
Building infrastructure will immediately consume {config.INFRA_COST_WOOD:.0f} wood and {config.INFRA_COST_WEALTH:.0f} wealth; declaring this action without both on hand wastes the step.
{feasibility_snapshot}
Season outlook: current season="{current_season}" (food/wood yield x{current_season_mult:.2f}); next step remains "{next_season}" (x{next_season_mult:.2f}). Plan to exploit high-multiplier seasons (e.g., summer) for production and stockpile before harsh seasons (e.g., winter at 0.40x).

Last self-set directive: "{prior_directive}"

Work points available this step: {work_points} (roughly 100 population per work point adjusted by morale).
Current diplomatic stance toward your neighbour: {relation} (score {relation_score}).

Recent history of your decisions in this run:
{history_text}
Recent diplomatic interactions with your neighbour:
{interaction_text}
Pay attention to moments where past actions failed (e.g., infrastructure attempts without wood) and adjust course proactively.

There is no hidden safety net; starvation or collapse are permanent. Take calculated risks when justified, but own the consequences.

Per-work yields with current infrastructure:
  - focus_food:  {yields.get('food_per_work', 0.0):.3f} food/work
  - focus_wood:  {yields.get('wood_per_work', 0.0):.3f} wood/work
  - focus_wealth: {yields.get('wealth_per_work', 0.0):.3f} wealth/work

Soft priority hint (you may override this):
  - food_safety_ratio (food vs. next {config.FOOD_SAFETY_HORIZON_STEPS} steps): {priority_hint.get('food_safety_ratio', 0.0):.3f}
  - suggested weights: {priority_hint.get('priorities', {})}

Before selecting an action, run this feasibility checklist:
  1. Ensure minimum food coverage for the next {config.FOOD_SAFETY_HORIZON_STEPS} steps (grow or trade if short).
  2. Confirm wood ≥ {config.INFRA_COST_WOOD} if you intend to build infrastructure; otherwise gather wood first.
  3. Confirm wealth ≥ {config.INFRA_COST_WEALTH} for infrastructure; if not, consider producing wealth.
  4. Cover this step's wage bill (~{wage_bill:.2f} wealth) or plan to raise wealth immediately to prevent morale collapse.
  5. Align choices with seasonal multipliers: push production during high-yield seasons and enter low-yield seasons with reserves ready.
  6. Use recent failures and relation shifts to avoid repeating mistakes.

Work allocation options:
  - "focus_food": grow food using the food_per_work yield.
  - "focus_wood": grow wood using the wood_per_work yield.
  - "focus_wealth": grow wealth using the wealth_per_work yield.
You may split work freely across these options (shares between 0.0 and 1.0 that sum to ≤ 1.0). Any unassigned share idles.
Infrastructure option:
  - set "build_infrastructure": true to spend {config.INFRA_COST_WOOD} wood and {config.INFRA_COST_WEALTH} wealth immediately if available (in addition to your work allocations).

Objectives (in soft order):
1. Avoid starvation in the short and medium term.
2. Build resilience by investing in infrastructure and spare supplies when food safety allows.
3. Develop prosperity (wealth/wood) to unlock future economic and diplomatic options.
4. Maintain workable relations with your neighbour where possible.
5. Improve population well-being (growth, quality of life) once survival is secured.

You can follow or override the hint freely. Consider trade-offs between immediate survival and long-term strength using both the history above and the current metrics.

Respond with a single JSON object of the form:
{{
  "allocations": {{"focus_food": <float>, "focus_wood": <float>, "focus_wealth": <float>}},
  "build_infrastructure": <true|false>,
  "reason": "<why this split makes sense now>",
  "next_prompt": "<a concise directive you want to remember for the next step>"
}}
Only include the keys you need inside "allocations"; shares must be between 0 and 1 and sum to at most 1. The optional "build_infrastructure" flag can be true even when you allocate work elsewhere, provided you can afford the cost. No extra text or keys outside this JSON.

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
        result = _try_load(candidate)
        if result is not None:
            return result

        # As a last resort, scan for balanced JSON objects (handles cases where the
        # model emits multiple JSON blobs sequentially). Return the last valid object.
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

    def compose_negotiation_context(self, state: Dict[str, Any]) -> str:
        """I outline the joint state so the LLM can improvise a mini dialogue and trade."""
        step = state.get("step", "unknown")
        east = state.get("east", {})
        west = state.get("west", {})
        last_actions = state.get("last_actions", {})
        east_history = state.get("east_history_text") or "No previous steps in this run."
        west_history = state.get("west_history_text") or "No previous steps in this run."
        east_interactions = state.get("east_interactions_text") or "No notable interactions recorded."
        west_interactions = state.get("west_interactions_text") or "No notable interactions recorded."
        context = f"""
Simulate a calm negotiation at step {step} between the leaders of East and West (this occurs before upkeep/starvation each tick).
Current relationship status: {east.get('relation_to_neighbor', 'neutral')} (score {east.get('relation_score', 0)}).
East -> food {east.get('food', 'unknown')}, wealth {east.get('wealth', 'unknown')}, population {east.get('population', 'unknown')}.
West -> food {west.get('food', 'unknown')}, wealth {west.get('wealth', 'unknown')}, population {west.get('population', 'unknown')}.
Last allocations -> East: {last_actions.get('east')}, West: {last_actions.get('west')}.

Recent history for East:
{east_history}

Recent history for West:
{west_history}

Recent interaction log for East:
{east_interactions}

Recent interaction log for West:
{west_interactions}

There is no safety net—if a side starves it may collapse. Reason carefully about whether to take risks, extend aid, or demand concessions.
"""
        return context

    def _negotiate_request(self, prompt: str, max_tokens: int) -> Dict[str, Any] | None:
        """I send a negotiation sub-task prompt and parse the JSON reply."""
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
                    "max_tokens": max_tokens,
                    "temperature": self.config.temperature,
                },
                timeout=10,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            log.warning("LLM negotiation request failed (%s).", exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("LLM negotiation response parsing failed (%s).", exc)
            return None
        return self._parse_response(raw)

    def negotiate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """I call LM Studio twice: once for dialogue, once for the final trade."""
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        context = self.compose_negotiation_context(state)
        dialogue_prompt = f"""{context}

Dialogue task: produce an alternating conversation (East speaks first, then West) with at least one full exchange and at most three exchanges per side. Each line must follow the numbers above and the speakers must strictly alternate (East, West, East, West, ...). Respond ONLY with JSON like {{"dialogue": [{{"speaker": "East", "line": "..."}}...]}}. Do not include trade information yet or any extra text.
"""
        dialogue_data = self._negotiate_request(dialogue_prompt, max(512, self.config.max_tokens))
        if dialogue_data is None:
            return self._fallback_negotiation()

        def _sanitise_dialogue(raw: Any) -> List[Dict[str, str]]:
            lines: List[Dict[str, str]] = []
            if isinstance(raw, list):
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    speaker = str(entry.get("speaker", "")).strip().lower()
                    text = str(entry.get("line", "")).strip()
                    if speaker not in ("east", "west") or not text:
                        continue
                    lines.append({"speaker": "East" if speaker == "east" else "West", "line": text})
                    if len(lines) >= 6:
                        break
            return lines

        dialogue = _sanitise_dialogue(dialogue_data.get("dialogue"))
        def _dialogue_valid(lines: List[Dict[str, str]]) -> bool:
            if len(lines) < 2:
                return False
            if lines[0]["speaker"] != "East":
                return False
            for idx, line in enumerate(lines):
                expected = "East" if idx % 2 == 0 else "West"
                if line["speaker"] != expected:
                    return False
                if not line["line"]:
                    return False
            return True

        attempts = 0
        while (not dialogue or not _dialogue_valid(dialogue)) and attempts < 2:
            log.warning("Negotiation dialogue invalid (attempt %s); retrying.", attempts + 1)
            dialogue_data = self._negotiate_request(dialogue_prompt, max(512, self.config.max_tokens))
            if dialogue_data is None:
                break
            dialogue = _sanitise_dialogue(dialogue_data.get("dialogue"))
            attempts += 1

        if not dialogue:
            log.warning("Negotiation dialogue invalid; using fallback trade.")
            return self._fallback_negotiation()

        transcript = "\n".join(f"{line['speaker']}: \"{line['line']}\"" for line in dialogue)
        trade_prompt = f"""{context}

Negotiation transcript:
{transcript}

Settlement task: determine the final deal reached in the transcript and respond ONLY with JSON like {{
  "trade": {{"food_from_east_to_west": 0, "wealth_from_west_to_east": 0, "reason": "..."}},
  "east_line": "...",
  "west_line": "..."
}}.
The trade values must match the final proposal, stay within each side's resources (positive food means East ships food to West; positive wealth means West sends wealth to East), and the summary lines should reflect each leader's closing statement.
"""
        trade_data = self._negotiate_request(trade_prompt, max(384, self.config.max_tokens))
        if trade_data is None:
            return self._fallback_negotiation()

        trade = trade_data.get("trade") or {}

        def _sanitise_flow(key: str) -> int:
            value = trade.get(key, 0)
            try:
                delta = int(value)
            except (ValueError, TypeError):
                delta = 0
            return max(-5, min(5, delta))

        reason_text = str(trade.get("reason", "LLM proposed this trade.")).strip() or "LLM proposed this trade."
        east_line = str(trade_data.get("east_line", "")).strip()
        west_line = str(trade_data.get("west_line", "")).strip()
        if not east_line and dialogue:
            east_line = dialogue[-2]["line"] if len(dialogue) >= 2 else dialogue[-1]["line"]
        if not west_line and dialogue:
            west_line = dialogue[-1]["line"]

        decision = {
            "dialogue": dialogue,
            "east_line": east_line or "Let's hold steady for now.",
            "west_line": west_line or "Agreed, we can revisit later.",
            "trade": {
                "food_from_east_to_west": _sanitise_flow("food_from_east_to_west"),
                "wealth_from_west_to_east": _sanitise_flow("wealth_from_west_to_east"),
                "reason": reason_text,
            },
        }
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

        allocations_raw = parsed.get("allocations", {})
        allocations: Dict[str, float] = {}
        if isinstance(allocations_raw, dict):
            for key, value in allocations_raw.items():
                if key not in _WORK_ACTIONS:
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

        build_flag = bool(parsed.get("build_infrastructure"))

        reason_value = str(parsed.get("reason", "")).strip() or "No reason provided by model."
        next_prompt = str(parsed.get("next_prompt", "")).strip() or "Maintain flexibility and reassess."

        action_raw = str(parsed.get("action", "")).strip().lower()
        legacy_action = action_raw if action_raw in _WORK_ACTIONS + ("build_infrastructure", "wait") else None

        if not allocations and not build_flag and not legacy_action:
            log.warning("LLM decision missing allocations and build flag; using fallback policy.")
            return self._fallback_decision("invalid JSON")

        return {
            "allocations": allocations,
            "build_infrastructure": build_flag,
            "legacy_action": legacy_action,
            "reason": reason_value,
            "next_prompt": next_prompt,
        }

    def _fallback_negotiation(self) -> Dict[str, Any]:
        """I return a safe no-trade negotiation result."""
        return {
            "dialogue": [
                {"speaker": "East", "line": "Let's hold steady for now."},
                {"speaker": "West", "line": "Agreed, we can revisit later."},
            ],
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
            "allocations": {},
            "build_infrastructure": False,
            "legacy_action": None,
            "reason": f"LLM decision fallback: {reason}",
            "next_prompt": "Rebuild buffers and gather wood.",
        }
