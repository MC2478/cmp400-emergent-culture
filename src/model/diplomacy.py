"""Diplomacy helpers extracted from world_model.py for negotiations and relation scoring."""

from __future__ import annotations

from typing import Any, Dict

import config
from src.model.llm_client import summarise_memory_for_prompt


def relation_label(score: int) -> str:
    """Map relation score to a label."""
    if score <= -2:
        return "hostile"
    if score == -1:
        return "strained"
    if score == 0:
        return "neutral"
    if score == 1:
        return "cordial"
    return "allied"


def _sanitise_flow(trade: Dict[str, Any], key: str) -> int:
    value = trade.get(key, 0)
    try:
        delta = int(value)
    except (ValueError, TypeError):
        delta = 0
    return max(-5, min(5, delta))


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
        },
        "west": {
            "food": model.west.food,
            "wealth": model.west.wealth,
            "population": model.west.population,
            "relation_to_neighbor": model.west.relation_to_neighbor,
            "relation_score": model.west.relation_score,
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

    if food_flow > 0:
        food_flow = min(food_flow, int(model.east.food))
    elif food_flow < 0:
        food_flow = max(food_flow, -int(model.west.food))

    if wealth_flow > 0:
        wealth_flow = min(wealth_flow, int(model.west.wealth))
    elif wealth_flow < 0:
        wealth_flow = max(wealth_flow, -int(model.east.wealth))

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

    score = model.east.relation_score
    if trade_type in ("gift_from_east", "gift_from_west"):
        score += 1
    elif trade_type == "balanced_trade" and score >= 0:
        score += 1
    elif trade_type.startswith("mildly_exploitative"):
        score -= 1
    elif trade_type.startswith("strongly_exploitative"):
        score -= 2
    score = max(-2, min(2, score))
    model.east.relation_score = score
    model.west.relation_score = score
    new_label = relation_label(score)
    model.east.relation_to_neighbor = new_label
    model.west.relation_to_neighbor = new_label

    trade_reason = str(trade.get("reason", "no trade reason provided")).strip() or "no trade reason provided"

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
