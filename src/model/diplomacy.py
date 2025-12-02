"""Quick card: diplomacy helpers for negotiations, trades, and relation scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import config
from src.model.llm_client import summarise_memory_for_prompt
from src.model.traits import neutral_personality_vector


@dataclass
class NegotiationSession:
    """Track ongoing negotiation state so each leader can act in turn."""

    initiator: str
    max_turns: int | None
    dialogue: list[Dict[str, Any]] = field(default_factory=list)
    current_proposal: Dict[str, Any] = field(default_factory=dict)
    current_proposer: str | None = None
    outcome: str = "pending"
    accepted_by: str | None = None
    turn_count: int = 0
    stall_counter: int = 0
    total_counters: int = 0

    def payload(self) -> Dict[str, Any]:
        """Lightweight view used by prompt builders."""
        return {
            "dialogue": self.dialogue,
            "current_proposal": self.current_proposal,
            "current_proposer": self.current_proposer,
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
        }


def relation_label(score: float) -> str:
    """Cue: turn a numeric relation score into a human-friendly label."""
    if score <= -1.5:
        return "hostile"
    if score <= -0.5:
        return "strained"
    if -0.5 < score < 0.5:
        return "neutral"
    if 0.5 <= score < 1.5:
        return "cordial"
    return "allied"


def _sanitise_flow(trade: Dict[str, Any], key: str) -> float:
    value = trade.get(key, 0)
    try:
        delta = float(value)
    except (ValueError, TypeError):
        delta = 0.0
    # Flow note: allow tiny fractional gifts, but clamp wild swings.
    return max(-5.0, min(5.0, delta))


_TOKEN_GIFT_KEYWORDS: tuple[str, ...] = ("token", "symbolic", "gesture", "gift", "small swap", "small trade")


def _dialogue_suggests_token(decision: Dict[str, Any], trade: Dict[str, Any]) -> bool:
    """Token cue: scan dialogue/closing lines/reason for explicit tiny-gift language."""
    bits: list[str] = []
    reason = trade.get("reason")
    if isinstance(reason, str):
        bits.append(reason.lower())
    for key in ("east_line", "west_line"):
        value = decision.get(key, "")
        if isinstance(value, str):
            bits.append(value.lower())
    for entry in decision.get("dialogue") or []:
        if isinstance(entry, dict):
            text = entry.get("line")
            if isinstance(text, str):
                bits.append(text.lower())
    if not bits:
        return False
    combined = " ".join(bits)
    return any(keyword in combined for keyword in _TOKEN_GIFT_KEYWORDS)


def _token_gift_flow(
    east_before: Dict[str, float],
    west_before: Dict[str, float],
    *,
    east_required: float,
    west_required: float,
    east_safety: float,
    west_safety: float,
) -> tuple[float, float]:
    """Token rule: decide if a tiny food/wealth gift is safe to inject."""
    token_size = 0.1
    min_ratio_cushion = 0.15
    food_gap_east = east_before["food"] - east_required
    food_gap_west = west_before["food"] - west_required

    if east_safety >= west_safety + min_ratio_cushion and food_gap_east >= token_size:
        return (token_size, 0.0)
    if west_safety >= east_safety + min_ratio_cushion and food_gap_west >= token_size:
        return (-token_size, 0.0)

    wealth_east = east_before.get("wealth", 0.0)
    wealth_west = west_before.get("wealth", 0.0)
    if wealth_east >= wealth_west + 0.5 and wealth_east >= token_size:
        return (0.0, -token_size)
    if wealth_west >= wealth_east + 0.5 and wealth_west >= token_size:
        return (0.0, token_size)
    return (0.0, 0.0)


def _safe_vec(vec: Dict[str, float] | None) -> Dict[str, float]:
    base = neutral_personality_vector()
    if not isinstance(vec, dict):
        return base
    for key, value in vec.items():
        if key in base:
            try:
                base[key] = float(value)
            except (TypeError, ValueError):
                continue
    return base


def _relation_delta_with_personality(delta: float, east_vec: Dict[str, float], west_vec: Dict[str, float]) -> float:
    if abs(delta) < 1e-9:
        return 0.0

    def _mult(vec: Dict[str, float]) -> float:
        coop = vec.get("cooperation", 0.5)
        trust = vec.get("trust_in_others", 0.5)
        aggression = vec.get("aggression", 0.5)
        if delta > 0:
            return max(0.5, min(1.3, 0.7 + 0.4 * coop + 0.2 * trust - 0.2 * aggression))
        return max(0.5, min(1.3, 0.7 + 0.3 * aggression + 0.2 * trust - 0.1 * coop))

    east_mult = _mult(_safe_vec(east_vec))
    west_mult = _mult(_safe_vec(west_vec))
    return delta * ((east_mult + west_mult) / 2.0)


def relation_stance(relation: str, personality: Dict[str, float]) -> str:
    """Stance cue: mix relation + personality into a quick logging label."""
    vec = _safe_vec(personality)
    coop = vec.get("cooperation", 0.5)
    trust = vec.get("trust_in_others", 0.5)
    aggression = vec.get("aggression", 0.5)
    risk = vec.get("risk_tolerance", 0.5)
    adaptability = vec.get("adaptability", 0.5)

    if relation == "neutral":
        if coop >= 0.6 and trust >= 0.55:
            return "optimistic neutral"
        if coop <= 0.4 and trust <= 0.4:
            return "wary neutral"
        return "guarded neutral"
    if relation == "cordial":
        if coop >= 0.6:
            return "supportive"
        if risk >= 0.6 or adaptability >= 0.6:
            return "pragmatic partner"
        return "cautiously cordial"
    if relation == "allied":
        if coop >= 0.6 and trust >= 0.55:
            return "close ally"
        if aggression >= 0.6:
            return "tense alliance"
        return "steady ally"
    if relation == "strained":
        return "uneasy"
    if relation == "hostile":
        return "openly hostile"
    return relation


def _is_tiny_gift(amount: float, donor_amount: float, donor_required: float) -> bool:
    """Tiny food rule: permit very small gifts only if the donor stays safe."""
    if amount <= 0 or donor_amount <= 0:
        return False
    tiny_absolute = amount <= 0.1
    tiny_relative = amount <= 0.05 * donor_amount
    if not (tiny_absolute or tiny_relative):
        return False
    post_amount = donor_amount - amount
    if donor_required <= 0:
        return post_amount >= 0
    post_ratio = post_amount / donor_required if donor_required > 0 else float("inf")
    return post_ratio >= 0.8 and post_amount >= 0


def _is_tiny_wealth_gift(amount: float, donor_amount: float) -> bool:
    """Tiny wealth rule: allow small coins to move without draining the giver."""
    if amount <= 0 or donor_amount <= 0:
        return False
    tiny_absolute = amount <= 0.1
    tiny_relative = amount <= 0.05 * donor_amount
    remaining = donor_amount - amount
    return (tiny_absolute or tiny_relative) and remaining >= 0.5


def _value_ratio_for_side(
    food_in: float,
    food_out: float,
    wealth_in: float,
    wealth_out: float,
    gold_in: float,
    gold_out: float,
    wood_in: float,
    wood_out: float,
) -> float:
    """Fairness cue: rough value ratio to sniff out exploitative deals."""
    value_in = food_in * 1.0 + wealth_in * 1.0 + gold_in * 1.0 + wood_in * 1.0
    value_out = food_out * 1.0 + wealth_out * 1.0 + gold_out * 1.0 + wood_out * 1.0
    if value_out < 1e-6:
        return float("inf")
    return value_in / value_out


def _update_exploitation_streak(model: Any, east_exploited: bool, west_exploited: bool) -> None:
    """Pressure cue: bump/reset exploitation streaks based on this trade."""
    if east_exploited:
        model.east.exploitation_streak += 1
    else:
        model.east.exploitation_streak = 0
    if west_exploited:
        model.west.exploitation_streak += 1
    else:
        model.west.exploitation_streak = 0


def run_negotiation(model: Any) -> None:
    """Negotiation card: orchestrate the LM Studio chat and apply the resulting flows."""
    # Presentation cue: this is the stand-in for the future council mechanics.
    llm_client = getattr(model, "llm_client", None)
    if llm_client is None or not llm_client.enabled:
        return

    east_intent = getattr(model.leader_east, "negotiation_intent", {}) or {"initiate": False}
    west_intent = getattr(model.leader_west, "negotiation_intent", {}) or {"initiate": False}
    if not (east_intent.get("initiate") or west_intent.get("initiate")):
        return
    if east_intent.get("initiate") and west_intent.get("initiate"):
        dialogue_initiator = "East"
        initiator_label = "both"
    elif east_intent.get("initiate"):
        dialogue_initiator = "East"
        initiator_label = "East"
    elif west_intent.get("initiate"):
        dialogue_initiator = "West"
        initiator_label = "West"
    else:
        dialogue_initiator = "East"
        initiator_label = "East"

    state = {
        "step": model.steps,
        "east": {
            "food": model.east.food,
            "wealth": model.east.wealth,
            "iron": model.east.iron,
            "gold": model.east.gold,
            "population": model.east.population,
            "relation_to_neighbor": model.east.relation_to_neighbor,
            "relation_score": model.east.relation_score,
            "personality_vector": dict(model.east.personality_vector),
            "active_traits": list(model.east.active_traits),
            "other_trait_notes": model.east.other_trait_notes,
            "trade_ledger": {
                "food_sent": model.east.food_sent,
                "food_received": model.east.food_received,
                "wealth_sent": model.east.wealth_sent,
                "wealth_received": model.east.wealth_received,
                "wood_sent": model.east.wood_sent,
                "wood_received": model.east.wood_received,
            },
            "resource_pressures": _resource_pressures_for_side(model.east),
        },
        "west": {
            "food": model.west.food,
            "wealth": model.west.wealth,
            "iron": model.west.iron,
            "gold": model.west.gold,
            "population": model.west.population,
            "relation_to_neighbor": model.west.relation_to_neighbor,
            "relation_score": model.west.relation_score,
            "personality_vector": dict(model.west.personality_vector),
            "active_traits": list(model.west.active_traits),
            "other_trait_notes": model.west.other_trait_notes,
            "trade_ledger": {
                "food_sent": model.west.food_sent,
                "food_received": model.west.food_received,
                "wealth_sent": model.west.wealth_sent,
                "wealth_received": model.west.wealth_received,
                "wood_sent": model.west.wood_sent,
                "wood_received": model.west.wood_received,
            },
            "resource_pressures": _resource_pressures_for_side(model.west),
        },
        "last_actions": {
            "east": getattr(model.leader_east, "last_action", None),
            "west": getattr(model.leader_west, "last_action", None),
        },
        "east_intent": east_intent,
        "west_intent": west_intent,
        "dialogue_initiator": dialogue_initiator,
        "east_history_text": summarise_memory_for_prompt(model.leader_east.memory_events),
        "west_history_text": summarise_memory_for_prompt(model.leader_west.memory_events),
        "east_interactions_text": "\n".join(model.leader_east.interaction_log[-5:]) or "No notable interactions recorded.",
        "west_interactions_text": "\n".join(model.leader_west.interaction_log[-5:]) or "No notable interactions recorded.",
    }

    def _normalise_trade(proposal: Dict[str, Any], fallback_reason: str) -> Dict[str, Any]:
        reason_val = str(proposal.get("reason", "")).strip() if isinstance(proposal, dict) else ""
        return {
            "food_from_east_to_west": _sanitise_flow(proposal, "food_from_east_to_west"),
            "wealth_from_west_to_east": _sanitise_flow(proposal, "wealth_from_west_to_east"),
            "wood_from_east_to_west": _sanitise_flow(proposal, "wood_from_east_to_west"),
            "wood_from_west_to_east": _sanitise_flow(proposal, "wood_from_west_to_east"),
            "iron_from_east_to_west": _sanitise_flow(proposal, "iron_from_east_to_west"),
            "iron_from_west_to_east": _sanitise_flow(proposal, "iron_from_west_to_east"),
            "gold_from_east_to_west": _sanitise_flow(proposal, "gold_from_east_to_west"),
            "gold_from_west_to_east": _sanitise_flow(proposal, "gold_from_west_to_east"),
            "reason": reason_val or fallback_reason,
        }

    max_turns = getattr(config, "NEGOTIATION_MAX_TURNS", None)
    session = NegotiationSession(initiator=dialogue_initiator, max_turns=max_turns)
    speaker = dialogue_initiator

    def _other_side(side: str) -> str:
        return "West" if side == "East" else "East"

    def _proposal_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        keys = (
            "food_from_east_to_west",
            "wealth_from_west_to_east",
            "wood_from_east_to_west",
            "wood_from_west_to_east",
            "iron_from_east_to_west",
            "iron_from_west_to_east",
            "gold_from_east_to_west",
            "gold_from_west_to_east",
        )
        for key in keys:
            try:
                aval = float(a.get(key, 0) or 0.0)
                bval = float(b.get(key, 0) or 0.0)
            except (TypeError, ValueError):
                aval = bval = 0.0
            if abs(aval - bval) > 1e-3:
                return False
        return True

    def _proposal_close(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        keys = (
            "food_from_east_to_west",
            "wealth_from_west_to_east",
            "wood_from_east_to_west",
            "wood_from_west_to_east",
            "iron_from_east_to_west",
            "iron_from_west_to_east",
            "gold_from_east_to_west",
            "gold_from_west_to_east",
        )
        for key in keys:
            try:
                aval = float(a.get(key, 0) or 0.0)
                bval = float(b.get(key, 0) or 0.0)
            except (TypeError, ValueError):
                aval = bval = 0.0
            if abs(aval - bval) > 0.15:
                return False
        return True

    def _proposal_magnitude(proposal: Dict[str, Any]) -> float:
        total = 0.0
        for key in (
            "food_from_east_to_west",
            "wealth_from_west_to_east",
            "wood_from_east_to_west",
            "wood_from_west_to_east",
            "iron_from_east_to_west",
            "iron_from_west_to_east",
            "gold_from_east_to_west",
            "gold_from_west_to_east",
        ):
            try:
                total += abs(float(proposal.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        return total

    while (session.max_turns is None or session.turn_count < session.max_turns) and session.outcome == "pending":
        turn = llm_client.negotiate_turn(side=speaker, session=session.payload(), state=state)
        decision_label = str(turn.get("decision", "counter")).strip().lower()
        if decision_label not in ("counter", "accept", "decline"):
            decision_label = "counter"
        reply = str(turn.get("reply", "")).strip() or "Let's hold steady for now."
        proposal = turn.get("proposal") or {}
        log_decision = decision_label
        if decision_label == "counter" and not session.current_proposal:
            log_decision = "offer"
        if decision_label == "counter":
            session.total_counters += 1
        proposal_matches_current = bool(session.current_proposal) and _proposal_equal(proposal, session.current_proposal)
        if decision_label == "accept":
            if session.current_proposal:
                proposal = dict(session.current_proposal)
            else:
                decision_label = "counter"
                log_decision = "offer"
        if not session.current_proposal and (decision_label in ("accept", "decline") or not proposal or _proposal_close(proposal, {})):
            # No proposal on the table: force an opening offer or end politely if nothing concrete.
            if decision_label == "accept":
                decision_label = "counter"
                log_decision = "offer"
                reply = reply + " (No proposal yet; stating an offer instead.)"
            elif decision_label == "decline":
                session.outcome = "declined"
                session.dialogue.append({"speaker": speaker, "line": reply, "decision": "decline", "proposal": {}})
                break
            if not proposal or _proposal_close(proposal, {}):
                # Still no numbers: end talks to avoid empty loops.
                session.outcome = "declined"
                session.dialogue.append({"speaker": speaker, "line": reply + " (No concrete proposal; ending talks.)", "decision": "decline", "proposal": {}})
                break
        if decision_label in ("counter", "offer") and _proposal_magnitude(proposal) < 0.05:
            # Zero-ish proposals are noise; end the chat.
            session.outcome = "declined"
            session.dialogue.append({"speaker": speaker, "line": reply + " (Proposal too small; ending talks.)", "decision": "decline", "proposal": {}})
            break
        if decision_label == "counter" and session.current_proposal and speaker != session.current_proposer and proposal_matches_current:
            # If you mirror the other side's offer without change, treat it as acceptance.
            decision_label = "accept"
            proposal = dict(session.current_proposal)
            log_decision = "accept"
        elif decision_label == "counter" and session.current_proposal and speaker == session.current_proposer and proposal_matches_current:
            # Repeating your own proposal stalls the loop; decline to end talks.
            decision_label = "decline"
            proposal = {}
            reply = reply + " (No new offer; ending talks.)"
            log_decision = "decline"

        if decision_label == "counter" and session.current_proposal and _proposal_close(proposal, session.current_proposal):
            session.stall_counter += 1
        else:
            session.stall_counter = 0
        if decision_label == "counter" and session.stall_counter >= 2:
            decision_label = "decline"
            log_decision = "decline"
            proposal = {}
            reply = reply + " (Talks stalled with minimal movement.)"
            session.stall_counter = 0
        if decision_label == "counter" and session.total_counters >= 6:
            decision_label = "decline"
            log_decision = "decline"
            proposal = {}
            reply = reply + " (Too many counters; ending talks.)"

        session.dialogue.append(
            {"speaker": speaker, "line": reply, "decision": log_decision, "proposal": dict(proposal)}
        )
        if decision_label == "accept":
            if session.current_proposal:
                session.accepted_by = speaker
                session.outcome = "accepted"
            else:
                decision_label = "counter"
        if decision_label == "decline":
            session.outcome = "declined"
            session.current_proposal = {}
            session.current_proposer = None
        if decision_label == "counter":
            session.current_proposal = dict(proposal)
            session.current_proposer = speaker
        session.turn_count += 1
        if session.outcome == "pending":
            speaker = _other_side(speaker)

    if session.outcome == "pending":
        session.outcome = "no_agreement"
    outcome_reason = "max turns reached without agreement" if session.outcome == "no_agreement" else session.outcome
    final_trade = _normalise_trade(session.current_proposal if session.outcome == "accepted" else {}, outcome_reason)
    outcome_label = "accepted" if session.outcome == "accepted" else ("declined" if session.outcome == "declined" else "no_agreement")
    east_line = next((d.get("line", "") for d in reversed(session.dialogue) if d.get("speaker") == "East"), "")
    west_line = next((d.get("line", "") for d in reversed(session.dialogue) if d.get("speaker") == "West"), "")
    decision = {
        "dialogue": session.dialogue,
        "east_line": east_line or "Let's hold steady for now.",
        "west_line": west_line or "Agreed, we can revisit later.",
        "trade": final_trade,
        "negotiation_outcome": outcome_label,
        "accepted_by": session.accepted_by,
        "turns": session.turn_count,
    }
    trade = decision.get("trade") or {}

    food_flow = _sanitise_flow(trade, "food_from_east_to_west")
    wealth_flow = _sanitise_flow(trade, "wealth_from_west_to_east")
    iron_e2w = max(0.0, _sanitise_flow(trade, "iron_from_east_to_west"))
    iron_w2e = max(0.0, _sanitise_flow(trade, "iron_from_west_to_east"))
    gold_e2w = max(0.0, _sanitise_flow(trade, "gold_from_east_to_west"))
    gold_w2e = max(0.0, _sanitise_flow(trade, "gold_from_west_to_east"))
    wood_e2w = max(0.0, _sanitise_flow(trade, "wood_from_east_to_west"))
    wood_w2e = max(0.0, _sanitise_flow(trade, "wood_from_west_to_east"))

    east_before = {
        "food": model.east.food,
        "wealth": model.east.wealth,
        "population": model.east.population,
        "iron": model.east.iron,
        "gold": model.east.gold,
        "wood": getattr(model.east, "wood", 0.0),
    }
    west_before = {
        "food": model.west.food,
        "wealth": model.west.wealth,
        "population": model.west.population,
        "iron": model.west.iron,
        "gold": model.west.gold,
        "wood": getattr(model.west, "wood", 0.0),
    }

    # Guardrail: cap flows to what each side actually holds (tiny fractions allowed).
    if food_flow > 0:
        food_flow = min(food_flow, float(model.east.food))
    elif food_flow < 0:
        food_flow = -min(abs(food_flow), float(model.west.food))

    if wealth_flow > 0:
        wealth_flow = min(wealth_flow, float(model.west.wealth))
        if wealth_flow < 0:
            wealth_flow = 0.0
    elif wealth_flow < 0:
        wealth_flow = -min(abs(wealth_flow), float(model.east.wealth))

    if iron_e2w > 0:
        iron_e2w = min(iron_e2w, float(model.east.iron))
    if iron_w2e > 0:
        iron_w2e = min(iron_w2e, float(model.west.iron))
    if gold_e2w > 0:
        gold_e2w = min(gold_e2w, float(model.east.gold))
    if gold_w2e > 0:
        gold_w2e = min(gold_w2e, float(model.west.gold))
    if wood_e2w > 0:
        wood_e2w = min(wood_e2w, float(model.east.wood))
    if wood_w2e > 0:
        wood_w2e = min(wood_w2e, float(model.west.wood))

    def _food_safety(before: Dict[str, float]) -> tuple[float, float]:
        required = (before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
        ratio = before.get("food", 0.0) / required if required > 0 else float("inf")
        return ratio, required

    east_safety, east_required = _food_safety(east_before)
    west_safety, west_required = _food_safety(west_before)
    allow_token_override = decision.get("negotiation_outcome") == "accepted"
    # Token override: if the chat begged for a token gift, inject one if it's safe.
    if allow_token_override and (
        abs(food_flow) < 1e-6
        and abs(wealth_flow) < 1e-6
        and _dialogue_suggests_token(decision, trade)
    ):
        token_food, token_wealth = _token_gift_flow(
            east_before,
            west_before,
            east_required=east_required,
            west_required=west_required,
            east_safety=east_safety,
            west_safety=west_safety,
        )
        if abs(token_food) > 0 or abs(token_wealth) > 0:
            food_flow = token_food
            wealth_flow = token_wealth
    east_tiny_food = food_flow > 0 and _is_tiny_gift(food_flow, east_before["food"], east_required)
    west_tiny_food = food_flow < 0 and _is_tiny_gift(-food_flow, west_before["food"], west_required)
    west_tiny_wealth = wealth_flow > 0 and _is_tiny_wealth_gift(wealth_flow, west_before["wealth"])
    east_tiny_wealth = wealth_flow < 0 and _is_tiny_wealth_gift(-wealth_flow, east_before["wealth"])

    # Safety first: only let substantive gifts flow downhill from the safer side.
    if food_flow > 0 and east_safety <= west_safety + 0.05 and not east_tiny_food:
        food_flow = 0
    if food_flow < 0 and west_safety <= east_safety + 0.05 and not west_tiny_food:
        food_flow = 0
    if wealth_flow > 0 and (model.west.wealth <= model.east.wealth or west_safety <= east_safety + 0.1) and not west_tiny_wealth:
        wealth_flow = 0
    if wealth_flow < 0 and (model.east.wealth <= model.west.wealth or east_safety <= west_safety + 0.1) and not east_tiny_wealth:
        wealth_flow = 0

    # Direction tweak: if something is flowing, steer it toward the hungrier side.
    if food_flow != 0 or wealth_flow != 0:
        if west_safety < 1.0 and east_safety >= 1.2 and food_flow <= 0:
            desired = max(0.1, round(max(0.0, west_required - west_before["food"]), 1))
            food_flow = min(desired, float(model.east.food))
        elif east_safety < 1.0 and west_safety >= 1.2 and food_flow >= 0:
            desired = max(0.1, round(max(0.0, east_required - east_before["food"]), 1))
            food_flow = -min(desired, float(model.west.food))

    # Fairness guard: block wealth leaving a poorer, hungrier side unless it's a tiny gift.
    if wealth_flow > 0 and (
        (west_safety < 1.1 and model.west.wealth < model.east.wealth)
        or model.west.wealth < 0.8 * model.east.wealth
    ) and not west_tiny_wealth:
        wealth_flow = 0
    if wealth_flow < 0 and (
        (east_safety < 1.1 and model.east.wealth < model.west.wealth)
        or model.east.wealth < 0.8 * model.west.wealth
    ) and not east_tiny_wealth:
        wealth_flow = 0

    east_sent_any = food_flow > 0 or wealth_flow < 0 or wood_e2w > 0 or iron_e2w > 0 or gold_e2w > 0
    west_sent_any = food_flow < 0 or wealth_flow > 0 or wood_w2e > 0 or iron_w2e > 0 or gold_w2e > 0
    gift_cap_triggered = False
    blocked_receiver = None
    max_unreciprocated = getattr(config, "MAX_UNRECIPROCATED_GIFTS", 3)
    if east_sent_any and not west_sent_any:
        streak = getattr(model.west, "gift_streak_received", 0)
        if streak >= max_unreciprocated:
            gift_cap_triggered = True
            blocked_receiver = "West"
    elif west_sent_any and not east_sent_any:
        streak = getattr(model.east, "gift_streak_received", 0)
        if streak >= max_unreciprocated:
            gift_cap_triggered = True
            blocked_receiver = "East"
    if gift_cap_triggered:
        food_flow = wealth_flow = 0.0
        iron_e2w = iron_w2e = 0.0
        gold_e2w = gold_w2e = 0.0
        wood_e2w = wood_w2e = 0.0
        east_sent_any = west_sent_any = False

    def _amount_from_other(resource: str, initiator_side: str) -> float:
        if resource == "food":
            return max(0.0, -food_flow) if initiator_side == "East" else max(0.0, food_flow)
        if resource == "wealth":
            return max(0.0, wealth_flow) if initiator_side == "East" else max(0.0, -wealth_flow)
        if resource == "wood":
            return wood_w2e if initiator_side == "East" else wood_e2w
        return 0.0

    def _amount_to_other(resource: str, initiator_side: str) -> float:
        if resource == "food":
            return max(0.0, food_flow) if initiator_side == "East" else max(0.0, -food_flow)
        if resource == "wealth":
            return max(0.0, -wealth_flow) if initiator_side == "East" else max(0.0, wealth_flow)
        if resource == "wood":
            return wood_e2w if initiator_side == "East" else wood_w2e
        return 0.0

    intent_failure_reason = ""
    intent_check_allowed = decision.get("negotiation_outcome") == "accepted" and (
        abs(food_flow)
        + abs(wealth_flow)
        + abs(iron_e2w)
        + abs(iron_w2e)
        + abs(gold_e2w)
        + abs(gold_w2e)
        + abs(wood_e2w)
        + abs(wood_w2e)
    ) > 1e-6
    if intent_check_allowed and initiator_label in ("East", "West"):
        intent_ref = east_intent if initiator_label == "East" else west_intent
        request_resource = intent_ref.get("request_resource")
        request_amount = float(intent_ref.get("request_amount") or 0.0)
        offer_resource = intent_ref.get("offer_resource")
        offer_amount = float(intent_ref.get("offer_amount") or 0.0)
        tolerance = 0.05
        if request_resource:
            delivered = _amount_from_other(request_resource, initiator_label)
            if request_amount > tolerance and delivered + 1e-6 < max(tolerance, request_amount * 0.8):
                intent_failure_reason = f"Trade cancelled: requested {request_resource} ({request_amount}) not received (delivered {delivered:.2f})."
        if not intent_failure_reason and offer_resource:
            paid = _amount_to_other(offer_resource, initiator_label)
            if offer_amount > tolerance and paid + 1e-6 < max(tolerance, offer_amount * 0.8):
                intent_failure_reason = f"Trade cancelled: promised {offer_resource} ({offer_amount}) not paid (sent {paid:.2f})."
        if intent_failure_reason:
            food_flow = wealth_flow = 0.0
            iron_e2w = iron_w2e = 0.0
            gold_e2w = gold_w2e = 0.0
            wood_e2w = wood_w2e = 0.0
            east_sent_any = west_sent_any = False

    if food_flow > 0:
        model.east.food -= food_flow
        model.west.food += food_flow
        model.east.food_sent += food_flow
        model.west.food_received += food_flow
    elif food_flow < 0:
        amount = -food_flow
        model.west.food -= amount
        model.east.food += amount
        model.west.food_sent += amount
        model.east.food_received += amount

    if wealth_flow > 0:
        model.west.wealth -= wealth_flow
        model.east.wealth += wealth_flow
        model.west.wealth_sent += wealth_flow
        model.east.wealth_received += wealth_flow
    elif wealth_flow < 0:
        amount = -wealth_flow
        model.east.wealth -= amount
        model.west.wealth += amount
        model.east.wealth_sent += amount
        model.west.wealth_received += amount

    if iron_e2w > 0:
        model.east.iron -= iron_e2w
        model.west.iron += iron_e2w
    if iron_w2e > 0:
        model.west.iron -= iron_w2e
        model.east.iron += iron_w2e
    if gold_e2w > 0:
        model.east.gold -= gold_e2w
        model.west.gold += gold_e2w
    if gold_w2e > 0:
        model.west.gold -= gold_w2e
        model.east.gold += gold_w2e

    model.east.food = max(0.0, model.east.food)
    model.west.food = max(0.0, model.west.food)
    model.east.wealth = max(0.0, model.east.wealth)
    model.west.wealth = max(0.0, model.west.wealth)
    model.east.iron = max(0.0, model.east.iron)
    model.west.iron = max(0.0, model.west.iron)
    model.east.gold = max(0.0, model.east.gold)
    model.west.gold = max(0.0, model.west.gold)

    east_after = {
        "food": model.east.food,
        "wealth": model.east.wealth,
        "population": model.east.population,
        "iron": model.east.iron,
        "gold": model.east.gold,
        "wood": getattr(model.east, "wood", 0.0),
    }
    west_after = {
        "food": model.west.food,
        "wealth": model.west.wealth,
        "population": model.west.population,
        "iron": model.west.iron,
        "gold": model.west.gold,
        "wood": getattr(model.west, "wood", 0.0),
    }

    if east_sent_any and not west_sent_any:
        model.west.gift_streak_received += 1
        model.east.gift_streak_received = 0
    elif west_sent_any and not east_sent_any:
        model.east.gift_streak_received += 1
        model.west.gift_streak_received = 0
    else:
        model.east.gift_streak_received = 0
        model.west.gift_streak_received = 0

    units_food = abs(food_flow)
    units_wood = wood_e2w + wood_w2e
    eps = 1e-6
    trade_type = "no_trade"
    effective_price = None
    east_value = (
        (east_after["food"] - east_before["food"])
        + (east_after["wealth"] - east_before["wealth"])
        + (east_after.get("gold", 0.0) - east_before.get("gold", 0.0))
        + (east_after.get("wood", 0.0) - east_before.get("wood", 0.0))
    )
    west_value = (
        (west_after["food"] - west_before["food"])
        + (west_after["wealth"] - west_before["wealth"])
        + (west_after.get("gold", 0.0) - west_before.get("gold", 0.0))
        + (west_after.get("wood", 0.0) - west_before.get("wood", 0.0))
    )
    gift_classified = False
    if east_sent_any and not west_sent_any:
        trade_type = "gift_from_east"
        gift_classified = True
    elif west_sent_any and not east_sent_any:
        trade_type = "gift_from_west"
        gift_classified = True
    if gift_cap_triggered:
        trade_type = "no_trade"
        gift_classified = False

    value_units = units_food + units_wood
    if not gift_classified and value_units > 0:
        if units_food > 0 and food_flow > 0:
            receiver_before = west_before
            receiver_required = (receiver_before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
            at_risk = receiver_before["food"] < 1.5 * receiver_required
        elif units_food > 0 and food_flow < 0:
            receiver_before = east_before
            receiver_required = (receiver_before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
            at_risk = receiver_before["food"] < 1.5 * receiver_required
        else:
            receiver_before = {}
            receiver_required = 0.0
            at_risk = False

        trade_units = units_food if units_food > eps else units_wood
        if abs(wealth_flow) > eps and trade_units > eps and ((food_flow * wealth_flow) < 0 or (units_food < eps and wealth_flow != 0)):
            effective_price = abs(wealth_flow) / trade_units
        else:
            effective_price = None

        if effective_price is None and units_food > 0:
            if at_risk:
                trade_type = "gift_from_east" if food_flow > 0 else "gift_from_west"
            else:
                trade_type = "balanced_trade"
        else:
            FAIR_LOW = 0.5
            FAIR_HIGH = 1.5
            STRONG_EXPLOIT = 3.0
            if FAIR_LOW <= effective_price <= FAIR_HIGH:
                trade_type = "balanced_trade"
            else:
                victim = None
                if east_value < 0 and west_value >= 0:
                    victim = "east"
                elif west_value < 0 and east_value >= 0:
                    victim = "west"
                elif east_value < 0 and west_value < 0:
                    victim = "east" if abs(east_value) >= abs(west_value) else "west"
                else:
                    if food_flow > 0:
                        victim = "west"
                    elif food_flow < 0:
                        victim = "east"
                    elif wood_e2w > 0 and wood_w2e == 0:
                        victim = "west"
                    elif wood_w2e > 0 and wood_e2w == 0:
                        victim = "east"
                    else:
                        victim = "east"
                if effective_price >= STRONG_EXPLOIT:
                    trade_type = f"strongly_exploitative_for_{victim}"
                else:
                    trade_type = f"mildly_exploitative_for_{victim}"
        if effective_price is None and units_food <= eps:
            trade_type = "balanced_trade"

    # Exploitation sniff: detect lopsided trades even when price math was fuzzy.
    east_exploited = False
    west_exploited = False
    if not gift_classified:
        east_food_in = max(-food_flow, 0.0)
        east_food_out = max(food_flow, 0.0)
        east_wealth_in = max(wealth_flow, 0.0)
        east_wealth_out = max(-wealth_flow, 0.0)
        east_gold_in = gold_w2e
        east_gold_out = gold_e2w
        east_wood_in = wood_w2e
        east_wood_out = wood_e2w
        west_food_in = max(food_flow, 0.0)
        west_food_out = max(-food_flow, 0.0)
        west_wealth_in = max(-wealth_flow, 0.0)
        west_wealth_out = max(wealth_flow, 0.0)
        west_gold_in = gold_e2w
        west_gold_out = gold_w2e
        west_wood_in = wood_e2w
        west_wood_out = wood_w2e
        east_ratio = _value_ratio_for_side(
            east_food_in, east_food_out, east_wealth_in, east_wealth_out, east_gold_in, east_gold_out, east_wood_in, east_wood_out
        )
        west_ratio = _value_ratio_for_side(
            west_food_in, west_food_out, west_wealth_in, west_wealth_out, west_gold_in, west_gold_out, west_wood_in, west_wood_out
        )
        east_exploited = east_ratio < 0.7 and (east_food_out + east_wealth_out + east_gold_out + east_wood_out) > eps
        west_exploited = west_ratio < 0.7 and (west_food_out + west_wealth_out + west_gold_out + west_wood_out) > eps

    if east_exploited and not trade_type.startswith("strongly_exploitative"):
        trade_type = "mildly_exploitative_for_east"
    if west_exploited and not trade_type.startswith("strongly_exploitative"):
        trade_type = "mildly_exploitative_for_west"

    _update_exploitation_streak(model, east_exploited, west_exploited)

    score_before = float(model.east.relation_score)
    base_delta = 0.0
    if trade_type in ("gift_from_east", "gift_from_west"):
        base_delta = 0.25
    elif trade_type == "balanced_trade" and score_before >= 0:
        base_delta = 0.25
    elif trade_type.startswith("mildly_exploitative"):
        base_delta = -0.25
    elif trade_type.startswith("strongly_exploitative"):
        base_delta = -0.5
    score = score_before + _relation_delta_with_personality(base_delta, model.east.personality_vector, model.west.personality_vector)
    score = max(-2.0, min(2.0, score))
    model.east.relation_score = score
    model.west.relation_score = score
    new_label = relation_label(score)
    model.east.relation_to_neighbor = new_label
    model.west.relation_to_neighbor = new_label

    if trade_type.startswith("strongly_exploitative_for_east") or trade_type.startswith("mildly_exploitative_for_east"):
        model.east.other_trait_notes = "They pushed exploitative trades recently."
        model.west.other_trait_notes = "You pressed a lopsided trade; expect caution from them."
    elif trade_type.startswith("strongly_exploitative_for_west") or trade_type.startswith("mildly_exploitative_for_west"):
        model.west.other_trait_notes = "They pushed exploitative trades recently."
        model.east.other_trait_notes = "You pressed a lopsided trade; expect caution from them."
    elif trade_type == "balanced_trade":
        model.east.other_trait_notes = "Recent talks landed on a balanced deal."
        model.west.other_trait_notes = "Recent talks landed on a balanced deal."
    elif trade_type in ("gift_from_east", "gift_from_west"):
        model.east.other_trait_notes = "Gifts were exchanged; goodwill may be building."
        model.west.other_trait_notes = "Gifts were exchanged; goodwill may be building."
    else:
        model.east.other_trait_notes = "No clear shift in style this step."
        model.west.other_trait_notes = "No clear shift in style this step."

    trade_reason = str(trade.get("reason", "no trade reason provided")).strip() or "no trade reason provided"
    if gift_cap_triggered and blocked_receiver:
        trade_reason = f"Gift blocked: {blocked_receiver} must reciprocate before accepting more aid."
    if intent_failure_reason:
        trade_reason = intent_failure_reason

    flows_total = (
        abs(food_flow)
        + abs(wealth_flow)
        + abs(iron_e2w)
        + abs(iron_w2e)
        + abs(gold_e2w)
        + abs(gold_w2e)
        + abs(wood_e2w)
        + abs(wood_w2e)
    )
    if decision.get("negotiation_outcome") == "accepted" and flows_total < 1e-6:
        decision["negotiation_outcome"] = "no_agreement"
        outcome_label = "no_agreement"
        decision["accepted_by"] = None
        trade_reason = trade_reason or "No agreement: all flows zero after safety checks."

    relation_delta_actual = score - score_before
    entry = {
        "event_type": "negotiation",
        "step": model.steps,
        "east_line": decision.get("east_line", ""),
        "west_line": decision.get("west_line", ""),
        "dialogue": decision.get("dialogue"),
        "negotiation_outcome": decision.get("negotiation_outcome"),
        "accepted_by": decision.get("accepted_by"),
        "turns": decision.get("turns"),
        "initiator": initiator_label,
        "trade": {
            "food_from_east_to_west": food_flow,
            "wealth_from_west_to_east": wealth_flow,
            "iron_from_east_to_west": iron_e2w,
            "iron_from_west_to_east": iron_w2e,
            "gold_from_east_to_west": gold_e2w,
            "gold_from_west_to_east": gold_w2e,
            "wood_from_east_to_west": wood_e2w,
            "wood_from_west_to_east": wood_w2e,
            "reason": trade_reason,
        },
        "food_east_before": east_before["food"],
        "food_east_after": east_after["food"],
        "wealth_east_before": east_before["wealth"],
        "wealth_east_after": east_after["wealth"],
        "iron_east_before": east_before.get("iron"),
        "iron_east_after": east_after.get("iron"),
        "gold_east_before": east_before.get("gold"),
        "gold_east_after": east_after.get("gold"),
        "wood_east_before": east_before.get("wood"),
        "wood_east_after": east_after.get("wood"),
        "food_west_before": west_before["food"],
        "food_west_after": west_after["food"],
        "wealth_west_before": west_before["wealth"],
        "wealth_west_after": west_after["wealth"],
        "iron_west_before": west_before.get("iron"),
        "iron_west_after": west_after.get("iron"),
        "gold_west_before": west_before.get("gold"),
        "gold_west_after": west_after.get("gold"),
        "wood_west_before": west_before.get("wood"),
        "wood_west_after": west_after.get("wood"),
        "population_east": model.east.population,
        "population_west": model.west.population,
        "trade_type": trade_type,
        "relation_score": score,
        "relation_label": new_label,
        "east_stance": relation_stance(new_label, model.east.personality_vector),
        "west_stance": relation_stance(new_label, model.west.personality_vector),
        "relation_delta": relation_delta_actual,
        "east_traits": list(model.east.active_traits),
        "west_traits": list(model.west.active_traits),
        "east_exploited": east_exploited,
        "west_exploited": west_exploited,
        "gift_streak_blocked": gift_cap_triggered,
        "gift_block_receiver": blocked_receiver,
    }
    model.chronicle.append(entry)
    model.current_step_log["negotiation"] = {
        "entry": entry,
        "east_before": east_before,
        "east_after": east_after,
        "west_before": west_before,
        "west_after": west_after,
    }
    flow_bits = [f"flows(E->W food {food_flow}, W->E wealth {wealth_flow})"]
    if iron_e2w or iron_w2e:
        flow_bits.append(f"E->W iron {iron_e2w}, W->E iron {iron_w2e}")
    if gold_e2w or gold_w2e:
        flow_bits.append(f"E->W gold {gold_e2w}, W->E gold {gold_w2e}")
    if wood_e2w or wood_w2e:
        flow_bits.append(f"E->W wood {wood_e2w}, W->E wood {wood_w2e}")
    summary_base = (
        f"Step {model.steps}: trade_type={trade_type}, negotiation={decision.get('negotiation_outcome')}, relation={new_label}, "
        + ", ".join(flow_bits)
    )
    model.leader_east.record_interaction(
        f'{summary_base}; West said "{entry.get("west_line", "")}".'
    )
    model.leader_west.record_interaction(
        f'{summary_base}; East said "{entry.get("east_line", "")}".'
    )
def _resource_pressures_for_side(territory: TerritoryState) -> Dict[str, float]:
    required = (territory.population / 10.0) * config.FOOD_PER_10_POP
    horizon_need = required * config.FOOD_SAFETY_HORIZON_STEPS
    food_ratio = territory.food / horizon_need if horizon_need > 0 else float("inf")
    wood_gap = max(0.0, config.INFRA_TIER_WOOD_WOOD_COST - territory.wood)
    wealth_gap = max(0.0, config.INFRA_TIER_WOOD_WEALTH_COST - territory.wealth)
    return {
        "food_ratio_horizon": food_ratio,
        "wood_gap": wood_gap,
        "wealth_gap": wealth_gap,
    }
