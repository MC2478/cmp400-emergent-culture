"""LM Studio OpenAI-style client for decisions and negotiations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

import config
from src.model.parsers import extract_action_hint, parse_json_response, sanitise_allocations
from src.model.prompt_builder import compose_negotiation_context, compose_prompt

log = logging.getLogger(__name__)


def summarise_memory_for_prompt(events: List[Dict[str, Any]], max_events: int = 10) -> str:
    """Condense recent in-run memory events into a readable history for the LLM."""
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


# Mirror allowed actions locally to avoid circular imports.
_WORK_ACTIONS: tuple[str, ...] = ("focus_food", "focus_wood", "focus_wealth")


@dataclass
class LLMConfig:
    """LM Studio connection details (URL, model, decoding params)."""

    base_url: str = "http://127.0.0.1:1234/v1/chat/completions"
    model: str = "meta-llama-3.1-8b-instruct"
    temperature: float = 0.1
    max_tokens: int = 128
    timeout: int = 30


class LLMDecisionClient:
    """Thin requests-powered helper that speaks OpenAI's schema to LM Studio."""

    def __init__(self, config: LLMConfig | None = None, enabled: bool = False) -> None:
        self.config = config or LLMConfig()
        self.enabled = enabled

    def _negotiate_request(self, prompt: str, max_tokens: int) -> Dict[str, Any] | None:
        """Send a negotiation sub-task prompt and parse the JSON reply."""
        try:
            response = requests.post(
                self.config.base_url,
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "You are simulating a calm negotiation between East and West."},
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
        return parse_json_response(raw)

    def negotiate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Call LM Studio twice: once for dialogue, once for the final trade."""
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        context = compose_negotiation_context(state)
        dialogue_prompt = f"""{context}

Dialogue task: produce an alternating conversation (East speaks first, then West) with at least one full exchange and at most three exchanges per side. Each line must follow the numbers above and the speakers must strictly alternate (East, West, East, West, ...). Respond ONLY with JSON like {{"dialogue": [{{"speaker": "East", "line": "..."}}...]}}. Do not include trade information yet or any extra text. It is acceptable to end with no deal (e.g., a polite \"hold steady\") if no agreement is reached.
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

        def _dialogue_valid(lines: List[Dict[str, str]]) -> bool:
            if len(lines) < 2:
                return False
            if lines[0]["speaker"] != "East":
                return False
            for idx, line in enumerate(lines):
                expected = "East" if idx % 2 == 0 else "West"
                if line["speaker"] != expected or not line["line"]:
                    return False
            return True

        dialogue = _sanitise_dialogue(dialogue_data.get("dialogue"))
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
Trade rules:
  - If no deal is reached, return zeros for both flows.
  - Flows must match the final proposal and stay within each side's resources (positive food means East ships food to West; positive wealth means West sends wealth to East).
  - Do not propose aid that would leave the sender below roughly 1.5x their immediate food requirement.
  - If one side is more food-insecure, food should flow to them unless the dialogue explicitly refuses; avoid gifts from the poorer side to the richer side unless the dialogue shows tribute/exploitation.
  - When it is reasonably safe, prefer a small non-zero trade (token gifts of ~0.1-0.2 or balanced swaps of similar value) instead of defaulting to zero. Balanced food-for-wealth trades are welcome when both sides remain safe.
  - Summary lines should reflect each leader's closing statement.
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
        """Send the composed prompt to LM Studio and interpret the JSON reply."""
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        def _request(extra_prompt: str | None = None) -> str | None:
            prompt = compose_prompt(state)
            if extra_prompt:
                prompt = f"{prompt}\n\nPrevious reply was invalid. {extra_prompt.strip()}"
            try:
                response = requests.post(
                    self.config.base_url,
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": "You are a careful decision-making ruler agent."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except requests.RequestException as exc:
                log.warning("LLM decision request failed (%s); using fallback policy.", exc)
                return None
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("LLM decision parsing failed (%s); using fallback policy.", exc)
                return None

        text = _request()
        if text is None:
            return self._fallback_decision("request failed")

        parsed = parse_json_response(text)
        if parsed is None:
            # One retry before heuristic; include a clear reminder.
            text_retry = _request("Respond with JSON only in the specified schema.")
            if text_retry:
                parsed = parse_json_response(text_retry)
        if parsed is None:
            # Salvage based on a hinted action if present.
            action_hint = extract_action_hint(text, _WORK_ACTIONS + ("build_infrastructure", "wait"))
            if action_hint is None:
                log.warning("LLM decision JSON invalid; using fallback policy.")
                return self._fallback_decision("invalid JSON")
            parsed = action_hint

        allocations = sanitise_allocations(parsed.get("allocations", {}), _WORK_ACTIONS)
        build_flag = bool(parsed.get("build_infrastructure"))
        reason_value = str(parsed.get("reason", "")).strip() or "No reason provided by model."
        next_prompt = str(parsed.get("next_prompt", "")).strip() or "Maintain flexibility and reassess."
        trait_adjustment = str(parsed.get("trait_adjustment", "")).strip() or "no change"

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
            "trait_adjustment": trait_adjustment,
        }

    def _fallback_negotiation(self) -> Dict[str, Any]:
        """Safe no-trade negotiation result."""
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
        """Neutral fallback decision when the LLM is unreachable."""
        return {
            "allocations": {},
            "build_infrastructure": False,
            "legacy_action": None,
            "reason": f"LLM decision fallback: {reason}",
            "next_prompt": "Rebuild buffers and gather wood.",
            "trait_adjustment": "no change",
        }
