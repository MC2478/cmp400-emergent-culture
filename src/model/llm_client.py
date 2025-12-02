"""Quick card: this module talks to LM Studio for decisions and trade chats."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

import config
from src.model.parsers import extract_action_hint, parse_json_response, sanitise_allocations
from src.model.prompt_builder import compose_negotiation_context, compose_negotiation_turn_prompt, compose_prompt

log = logging.getLogger(__name__)


def summarise_memory_for_prompt(events: List[Dict[str, Any]], max_events: int = 10) -> str:
    """Prompt prep note: condense the latest memory events into friendly one-liners."""
    if not events:
        return "No previous steps in this run."

    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    recent = events[-max_events:]
    lines: list[str] = []
    for event in recent:
        step = event.get("step", "?")
        action = event.get("action") or event.get("action_name") or "unknown_action"
        fb = _safe_float(event.get("food_before"))
        fa = _safe_float(event.get("food_after"))
        wb = _safe_float(event.get("wealth_before"))
        wa = _safe_float(event.get("wealth_after"))
        pb = _safe_float(event.get("pop_before"))
        pa = _safe_float(event.get("pop_after"))
        notes = event.get("notes") or event.get("note")

        segments: list[str] = [f"Step {step}: action={action}"]
        if fb is not None and fa is not None:
            segments.append(f"food {fb:.2f}->{fa:.2f}")
        if wb is not None and wa is not None:
            segments.append(f"wealth {wb:.2f}->{wa:.2f}")
        if pb is not None and pa is not None:
            segments.append(f"pop {pb:.0f}->{pa:.0f}")
        line = ", ".join(segments)
        if notes:
            trimmed = str(notes).strip()
            if trimmed:
                line = f"{line} ({trimmed})"
        lines.append(line)

    return "\n".join(lines)


# Cheat sheet: allowed work actions live here to dodge circular imports.
_WORK_ACTIONS: tuple[str, ...] = ("focus_food", "focus_wood", "focus_wealth", "focus_iron", "focus_gold")


@dataclass
class LLMConfig:
    """Config card: where the LM Studio endpoint, model, and decoding knobs live."""

    base_url: str = "http://127.0.0.1:1234/v1/chat/completions"
    model: str = "meta-llama-3.1-8b-instruct"
    temperature: float = 0.1
    max_tokens: int = 800
    timeout: int = 6


class LLMDecisionClient:
    """Helper blurb: wraps requests so LM Studio looks like the OpenAI API."""

    def __init__(self, config: LLMConfig | None = None, enabled: bool = False) -> None:
        self.config = config or LLMConfig()
        self.enabled = enabled
        self._network_available = True
        self._disable_reason: str | None = None

    def _disable_network_access(self, reason: str) -> None:
        """Flip the client into offline mode so future calls fall back instantly."""
        if not self._network_available:
            return
        self._network_available = False
        self._disable_reason = reason
        print(
            f"[LLM disabled] {reason}. Falling back to heuristic policy until you toggle LLM decisions off/on again."
        )

    def _negotiate_request(self, prompt: str, max_tokens: int) -> Dict[str, Any] | None:
        """Little sidekick: send a negotiation prompt and hand back parsed JSON."""
        if not self._network_available:
            return None
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
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            self._disable_network_access(f"LLM negotiation request failed ({exc})")
            return None
        except Exception as exc:  # pragma: no cover - defensive
            self._disable_network_access(f"LLM negotiation response parsing failed ({exc})")
            return None
        return parse_json_response(raw)

    def negotiate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Two-step dance: first get dialogue, then settle on the trade."""
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        context = compose_negotiation_context(state)
        initiator = state.get("dialogue_initiator") or "East"
        dialogue_prompt = f"""{context}

Dialogue task: produce an alternating conversation (the initiator speaks first: {initiator}) with at least one full exchange and continue until both sides either reach a concrete agreement, explicitly decline, or mutually decide to pause. Keep the dialogue concise but allow as many turns as needed. Each line must follow the numbers above and the speakers must strictly alternate (Initiator, Other, Initiator, Other, ...). Respond ONLY with JSON like {{"dialogue": [{{"speaker": "East", "line": "..."}}...]}}. Do not include trade information yet or any extra text. It is acceptable to end with no deal (e.g., a polite \"hold steady\") if no agreement is reached.
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
  "trade": {{"food_from_east_to_west": 0, "wealth_from_west_to_east": 0, "wood_from_east_to_west": 0, "wood_from_west_to_east": 0, "reason": "..."}},
  "east_line": "...",
  "west_line": "..."
}}.
Trade rules:
  - If no deal is reached or both parties explicitly decline, return zeros for all flows.
  - Flows must match the final proposal and stay within each side's resources (positive food/wood means East ships that resource to West; positive wealth means West sends wealth to East).
  - Do not propose aid that would leave the sender below roughly 1.5x their immediate food requirement.
  - Respect autonomy: if a party refused, keep flows at zero and summarize the refusal.
  - Token gifts are only acceptable when the donor remains clearly safer and the dialogue explicitly insists on a goodwill gesture.
  - Summary lines should reflect each leader's closing statement.
"""
        trade_data = self._negotiate_request(trade_prompt, max(384, self.config.max_tokens))
        if trade_data is None:
            return self._fallback_negotiation()

        trade = trade_data.get("trade") or {}

        def _sanitise_flow(key: str) -> float:
            value = trade.get(key, 0)
            try:
                delta = float(value)
            except (ValueError, TypeError):
                delta = 0.0
            return max(-5.0, min(5.0, delta))

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
                "wood_from_east_to_west": _sanitise_flow("wood_from_east_to_west"),
                "wood_from_west_to_east": _sanitise_flow("wood_from_west_to_east"),
                "reason": reason_text,
            },
        }
        return decision

    def negotiate_turn(self, *, side: str, session: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Turn-level helper: ask one leader to reply, accept, or counter a proposal."""
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        prompt = compose_negotiation_turn_prompt(side, session, state)
        payload = self._negotiate_request(prompt, max(512, self.config.max_tokens // 2))
        if payload is None:
            return self._fallback_turn(side, reason="request failed")

        parsed = payload if isinstance(payload, dict) else parse_json_response(str(payload))
        if parsed is None:
            return self._fallback_turn(side, reason="invalid JSON")

        decision_raw = str(parsed.get("decision", "")).strip().lower()
        decision_value = decision_raw if decision_raw in ("counter", "accept", "decline") else "counter"
        reply = str(parsed.get("reply", "")).strip()

        proposal_raw = parsed.get("proposal") or {}

        def _sanitise_flow(key: str) -> float:
            value = proposal_raw.get(key, 0)
            try:
                delta = float(value)
            except (ValueError, TypeError):
                delta = 0.0
            return max(-5.0, min(5.0, delta))

        proposal = {
            "food_from_east_to_west": _sanitise_flow("food_from_east_to_west"),
            "wealth_from_west_to_east": _sanitise_flow("wealth_from_west_to_east"),
            "wood_from_east_to_west": _sanitise_flow("wood_from_east_to_west"),
            "wood_from_west_to_east": _sanitise_flow("wood_from_west_to_east"),
            "iron_from_east_to_west": _sanitise_flow("iron_from_east_to_west"),
            "iron_from_west_to_east": _sanitise_flow("iron_from_west_to_east"),
            "gold_from_east_to_west": _sanitise_flow("gold_from_east_to_west"),
            "gold_from_west_to_east": _sanitise_flow("gold_from_west_to_east"),
            "reason": str(proposal_raw.get("reason", "")).strip() or "No justification provided.",
        }

        if not reply:
            reply = "Let's pause and rethink the trade."

        return {
            "reply": reply,
            "proposal": proposal,
            "decision": decision_value,
        }

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main loop: ship the decision prompt, wrangle JSON, and fall back gracefully."""
        if not self.enabled:
            raise RuntimeError("LLMDecisionClient is disabled.")

        def _request(extra_prompt: str | None = None) -> str | None:
            if not self._network_available:
                return None
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
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except requests.RequestException as exc:
                self._disable_network_access(f"LLM decision request failed ({exc})")
                return None
            except Exception as exc:  # pragma: no cover - defensive
                self._disable_network_access(f"LLM decision parsing failed ({exc})")
                return None

        text = _request()
        if text is None:
            return self._fallback_decision("request failed")

        parsed = parse_json_response(text)
        if parsed is None:
            # Quick reminder round if the first parse flopped.
            text_retry = _request("Respond with JSON only in the specified schema.")
            if text_retry:
                parsed = parse_json_response(text_retry)
        if parsed is None:
            # If JSON stays messy, peek for a hinted action to salvage intent.
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
        """Safety net: no-trade, polite dialogue when LM Studio flakes."""
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
                "wood_from_east_to_west": 0,
                "wood_from_west_to_east": 0,
                "reason": "fallback trade",
            },
        }

    def _fallback_decision(self, reason: str) -> Dict[str, Any]:
        """Backup plan: neutral allocations when the LLM can't be reached."""
        return {
            "allocations": {},
            "build_infrastructure": False,
            "legacy_action": None,
            "reason": f"LLM decision fallback: {reason}",
            "next_prompt": "Rebuild buffers and gather wood.",
            "trait_adjustment": "no change",
        }

    def _fallback_turn(self, side: str, reason: str) -> Dict[str, Any]:
        """Fallback for a single negotiation turn when the LLM response is unusable."""
        side_label = "East" if str(side).lower().startswith("e") else "West"
        other_side = "West" if side_label == "East" else "East"
        return {
            "reply": f"{other_side}, let's hold steady; I cannot commit without clearer terms. ({reason})",
            "proposal": {
                "food_from_east_to_west": 0.0,
                "wealth_from_west_to_east": 0.0,
                "wood_from_east_to_west": 0.0,
                "wood_from_west_to_east": 0.0,
                "iron_from_east_to_west": 0.0,
                "iron_from_west_to_east": 0.0,
                "gold_from_east_to_west": 0.0,
                "gold_from_west_to_east": 0.0,
                "reason": f"Fallback: {reason}",
            },
            "decision": "decline",
        }
