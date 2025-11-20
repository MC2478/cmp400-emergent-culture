"""Diplomacy helpers extracted from world_model.py for negotiations and relation scoring."""

from __future__ import annotations

from typing import Any, Dict

import config
from src.model.llm_client import summarise_memory_for_prompt
from src.model.traits import neutral_personality_vector


def relation_label(score: float) -> str:
    """Map relation score to a label."""
    if score <= -1.5:
        return "hostile"
    if score <= -0.5:
        return "strained"
    if -0.5 < score < 0.5:
        return "neutral"
    if 0.5 <= score < 1.5:
        return "cordial"
    return "allied"


def _sanitise_flow(trade: Dict[str, Any], key: str) -> int:
    value = trade.get(key, 0)
    try:
        delta = int(value)
    except (ValueError, TypeError):
        delta = 0
    return max(-5, min(5, delta))


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


def _update_exploitation_streak(trade_type: str, model: Any) -> None:
    if trade_type.endswith("for_east"):
        model.east.exploitation_streak += 1
        model.west.exploitation_streak = 0
    elif trade_type.endswith("for_west"):
        model.west.exploitation_streak += 1
        model.east.exploitation_streak = 0
    else:
        model.east.exploitation_streak = 0
        model.west.exploitation_streak = 0


def run_negotiation(model: Any) -> None:
    """Run negotiation between East and West if the LLM client is enabled."""
    llm_client = getattr(model, "llm_client", None)
    if llm_client is None or not llm_client.enabled:
        return

    state = {
        "step": model.steps,
        "east": {
            "food": model.east.food,
            "wealth": model.east.wealth,
            "population": model.east.population,
            "relation_to_neighbor": model.east.relation_to_neighbor,
            "relation_score": model.east.relation_score,
            "personality_vector": dict(model.east.personality_vector),
            "active_traits": list(model.east.active_traits),
            "other_trait_notes": model.east.other_trait_notes,
        },
        "west": {
            "food": model.west.food,
            "wealth": model.west.wealth,
            "population": model.west.population,
            "relation_to_neighbor": model.west.relation_to_neighbor,
            "relation_score": model.west.relation_score,
            "personality_vector": dict(model.west.personality_vector),
            "active_traits": list(model.west.active_traits),
            "other_trait_notes": model.west.other_trait_notes,
        },
        "last_actions": {
            "east": getattr(model.leader_east, "last_action", None),
            "west": getattr(model.leader_west, "last_action", None),
        },
        "east_history_text": summarise_memory_for_prompt(model.leader_east.memory_events),
        "west_history_text": summarise_memory_for_prompt(model.leader_west.memory_events),
        "east_interactions_text": "\n".join(model.leader_east.interaction_log[-5:]) or "No notable interactions recorded.",
        "west_interactions_text": "\n".join(model.leader_west.interaction_log[-5:]) or "No notable interactions recorded.",
    }
    decision = llm_client.negotiate(state)
    trade = decision.get("trade") or {}

    food_flow = _sanitise_flow(trade, "food_from_east_to_west")
    wealth_flow = _sanitise_flow(trade, "wealth_from_west_to_east")

    east_before = {
        "food": model.east.food,
        "wealth": model.east.wealth,
        "population": model.east.population,
    }
    west_before = {
        "food": model.west.food,
        "wealth": model.west.wealth,
        "population": model.west.population,
    }

    # Clamp flows to available resources.
    if food_flow > 0:
        food_flow = min(food_flow, int(model.east.food))
    elif food_flow < 0:
        food_flow = max(food_flow, -int(model.west.food))

    if wealth_flow > 0:
        wealth_flow = min(wealth_flow, int(model.west.wealth))
    elif wealth_flow < 0:
        wealth_flow = max(wealth_flow, -int(model.east.wealth))

    def _food_safety(before: Dict[str, float]) -> tuple[float, float]:
        required = (before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
        ratio = before.get("food", 0.0) / required if required > 0 else float("inf")
        return ratio, required

    east_safety, east_required = _food_safety(east_before)
    west_safety, west_required = _food_safety(west_before)

    # Block uphill gifts when donor is not safer.
    if food_flow > 0 and east_safety <= west_safety + 0.05:
        food_flow = 0
    if food_flow < 0 and west_safety <= east_safety + 0.05:
        food_flow = 0
    if wealth_flow > 0 and (model.west.wealth <= model.east.wealth or west_safety <= east_safety + 0.1):
        wealth_flow = 0
    if wealth_flow < 0 and (model.east.wealth <= model.west.wealth or east_safety <= west_safety + 0.1):
        wealth_flow = 0

    # Only adjust direction if a non-zero trade was proposed (and not blocked above).
    if food_flow != 0 or wealth_flow != 0:
        if west_safety < 1.0 and east_safety >= 1.2 and food_flow <= 0:
            desired = max(1, int(round(max(0.0, west_required - west_before["food"]))))
            food_flow = min(desired, int(model.east.food))
        elif east_safety < 1.0 and west_safety >= 1.2 and food_flow >= 0:
            desired = max(1, int(round(max(0.0, east_required - east_before["food"]))))
            food_flow = -min(desired, int(model.west.food))

    # Prevent wealth flowing from the poorer side to the richer one when the poorer side is food-insecure or clearly poorer.
    if wealth_flow > 0 and (
        (west_safety < 1.1 and model.west.wealth < model.east.wealth)
        or model.west.wealth < 0.8 * model.east.wealth
    ):
        wealth_flow = 0
    if wealth_flow < 0 and (
        (east_safety < 1.1 and model.east.wealth < model.west.wealth)
        or model.east.wealth < 0.8 * model.west.wealth
    ):
        wealth_flow = 0

    if food_flow > 0:
        model.east.food -= food_flow
        model.west.food += food_flow
    elif food_flow < 0:
        amount = -food_flow
        model.west.food -= amount
        model.east.food += amount

    if wealth_flow > 0:
        model.west.wealth -= wealth_flow
        model.east.wealth += wealth_flow
    elif wealth_flow < 0:
        amount = -wealth_flow
        model.east.wealth -= amount
        model.west.wealth += amount

    model.east.food = max(0.0, model.east.food)
    model.west.food = max(0.0, model.west.food)
    model.east.wealth = max(0.0, model.east.wealth)
    model.west.wealth = max(0.0, model.west.wealth)

    east_after = {
        "food": model.east.food,
        "wealth": model.east.wealth,
        "population": model.east.population,
    }
    west_after = {
        "food": model.west.food,
        "wealth": model.west.wealth,
        "population": model.west.population,
    }

    units_food = abs(food_flow)
    eps = 1e-6
    trade_type = "no_trade"
    effective_price = None
    east_value = (east_after["food"] - east_before["food"]) + (east_after["wealth"] - east_before["wealth"])
    west_value = (west_after["food"] - west_before["food"]) + (west_after["wealth"] - west_before["wealth"])
    gift_classified = False
    if east_value < 0 and west_value > 0 and wealth_flow <= 0:
        trade_type = "gift_from_east"
        gift_classified = True
    elif west_value < 0 and east_value > 0 and wealth_flow >= 0:
        trade_type = "gift_from_west"
        gift_classified = True

    if not gift_classified and units_food > 0:
        if food_flow > 0:
            receiver_before = west_before
            receiver_required = (receiver_before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
            at_risk = receiver_before["food"] < 1.5 * receiver_required
        else:
            receiver_before = east_before
            receiver_required = (receiver_before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
            at_risk = receiver_before["food"] < 1.5 * receiver_required

        if abs(wealth_flow) > eps and (food_flow * wealth_flow) < 0:
            effective_price = abs(wealth_flow) / units_food
        else:
            effective_price = None

        if effective_price is None:
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
                    victim = "west" if food_flow > 0 else "east"
                if effective_price >= STRONG_EXPLOIT:
                    trade_type = f"strongly_exploitative_for_{victim}"
                else:
                    trade_type = f"mildly_exploitative_for_{victim}"

    _update_exploitation_streak(trade_type, model)

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

    relation_delta_actual = score - score_before
    entry = {
        "event_type": "negotiation",
        "step": model.steps,
        "east_line": decision.get("east_line", ""),
        "west_line": decision.get("west_line", ""),
        "dialogue": decision.get("dialogue"),
        "trade": {
            "food_from_east_to_west": food_flow,
            "wealth_from_west_to_east": wealth_flow,
            "reason": trade_reason,
        },
        "food_east_before": east_before["food"],
        "food_east_after": east_after["food"],
        "wealth_east_before": east_before["wealth"],
        "wealth_east_after": east_after["wealth"],
        "food_west_before": west_before["food"],
        "food_west_after": west_after["food"],
        "wealth_west_before": west_before["wealth"],
        "wealth_west_after": west_after["wealth"],
        "population_east": model.east.population,
        "population_west": model.west.population,
        "trade_type": trade_type,
        "relation_score": score,
        "relation_label": new_label,
        "relation_delta": relation_delta_actual,
        "east_traits": list(model.east.active_traits),
        "west_traits": list(model.west.active_traits),
    }
    model.chronicle.append(entry)
    model.current_step_log["negotiation"] = {
        "entry": entry,
        "east_before": east_before,
        "east_after": east_after,
        "west_before": west_before,
        "west_after": west_after,
    }
    summary_base = (
        f"Step {model.steps}: trade_type={trade_type}, relation={new_label}, "
        f"flows(E->W food {food_flow}, W->E wealth {wealth_flow})"
    )
    model.leader_east.record_interaction(
        f'{summary_base}; West said "{entry.get("west_line", "")}".'
    )
    model.leader_west.record_interaction(
        f'{summary_base}; East said "{entry.get("east_line", "")}".'
    )
