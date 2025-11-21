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


def _sanitise_flow(trade: Dict[str, Any], key: str) -> float:
    value = trade.get(key, 0)
    try:
        delta = float(value)
    except (ValueError, TypeError):
        delta = 0.0
    # I allow fractional flows so tiny gifts make it through, but still clamp extremes.
    return max(-5.0, min(5.0, delta))


_TOKEN_GIFT_KEYWORDS: tuple[str, ...] = ("token", "symbolic", "gesture", "gift", "small swap", "small trade")


def _dialogue_suggests_token(decision: Dict[str, Any], trade: Dict[str, Any]) -> bool:
    """I scan the dialogue, closing lines, and trade reason for explicit token gift wording."""
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
    """I decide whether a tiny food or wealth gift is safe to inject when the dialogue asks for one."""
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
    """I turn relation + personality into a light stance label for logging."""
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
    """I allow very small gifts while keeping donors above a minimum safety floor."""
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
    """I flag tiny wealth gifts so I can loosen clamps slightly without emptying coffers."""
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
) -> float:
    """I approximate a value ratio so I can spot exploitative deals."""
    value_in = food_in * 1.0 + wealth_in * 1.0 + gold_in * 1.0
    value_out = food_out * 1.0 + wealth_out * 1.0 + gold_out * 1.0
    if value_out < 1e-6:
        return float("inf")
    return value_in / value_out


def _update_exploitation_streak(model: Any, east_exploited: bool, west_exploited: bool) -> None:
    """I increment or reset streaks based on per-side exploitation flags."""
    if east_exploited:
        model.east.exploitation_streak += 1
    else:
        model.east.exploitation_streak = 0
    if west_exploited:
        model.west.exploitation_streak += 1
    else:
        model.west.exploitation_streak = 0


def run_negotiation(model: Any) -> None:
    """Run negotiation between East and West if the LLM client is enabled."""
    # [PRESENTATION] I position this negotiation routine as the stand-in for the planned council,
    # emphasising how dialogue plus trade classification currently give me the diplomacy hooks I will later expand.
    llm_client = getattr(model, "llm_client", None)
    if llm_client is None or not llm_client.enabled:
        return

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
    iron_e2w = max(0.0, _sanitise_flow(trade, "iron_from_east_to_west"))
    iron_w2e = max(0.0, _sanitise_flow(trade, "iron_from_west_to_east"))
    gold_e2w = max(0.0, _sanitise_flow(trade, "gold_from_east_to_west"))
    gold_w2e = max(0.0, _sanitise_flow(trade, "gold_from_west_to_east"))

    east_before = {
        "food": model.east.food,
        "wealth": model.east.wealth,
        "population": model.east.population,
        "iron": model.east.iron,
        "gold": model.east.gold,
    }
    west_before = {
        "food": model.west.food,
        "wealth": model.west.wealth,
        "population": model.west.population,
        "iron": model.west.iron,
        "gold": model.west.gold,
    }

    # Clamp flows to available resources (allowing fractional gifts).
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

    def _food_safety(before: Dict[str, float]) -> tuple[float, float]:
        required = (before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
        ratio = before.get("food", 0.0) / required if required > 0 else float("inf")
        return ratio, required

    east_safety, east_required = _food_safety(east_before)
    west_safety, west_required = _food_safety(west_before)
    # --- Honour explicit token-gift language even if the flows were zeroed above. ---
    if (
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

    # --- Keep donors safer than the recipient before allowing substantive shipments. ---
    # Block uphill gifts when donor is not safer.
    if food_flow > 0 and east_safety <= west_safety + 0.05 and not east_tiny_food:
        food_flow = 0
    if food_flow < 0 and west_safety <= east_safety + 0.05 and not west_tiny_food:
        food_flow = 0
    if wealth_flow > 0 and (model.west.wealth <= model.east.wealth or west_safety <= east_safety + 0.1) and not west_tiny_wealth:
        wealth_flow = 0
    if wealth_flow < 0 and (model.east.wealth <= model.west.wealth or east_safety <= west_safety + 0.1) and not east_tiny_wealth:
        wealth_flow = 0

    # Only adjust direction if a non-zero trade was proposed (and not blocked above).
    if food_flow != 0 or wealth_flow != 0:
        if west_safety < 1.0 and east_safety >= 1.2 and food_flow <= 0:
            desired = max(0.1, round(max(0.0, west_required - west_before["food"]), 1))
            food_flow = min(desired, float(model.east.food))
        elif east_safety < 1.0 and west_safety >= 1.2 and food_flow >= 0:
            desired = max(0.1, round(max(0.0, east_required - east_before["food"]), 1))
            food_flow = -min(desired, float(model.west.food))

    # Prevent wealth flowing from the poorer side to the richer one when the poorer side is food-insecure or clearly poorer.
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
    }
    west_after = {
        "food": model.west.food,
        "wealth": model.west.wealth,
        "population": model.west.population,
        "iron": model.west.iron,
        "gold": model.west.gold,
    }

    units_food = abs(food_flow)
    eps = 1e-6
    trade_type = "no_trade"
    effective_price = None
    east_value = (
        (east_after["food"] - east_before["food"])
        + (east_after["wealth"] - east_before["wealth"])
        + (east_after.get("gold", 0.0) - east_before.get("gold", 0.0))
    )
    west_value = (
        (west_after["food"] - west_before["food"])
        + (west_after["wealth"] - west_before["wealth"])
        + (west_after.get("gold", 0.0) - west_before.get("gold", 0.0))
    )
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

    # --- Detect subtle exploitation so pressure counters and relation labels stay accurate. ---
    east_exploited = False
    west_exploited = False
    if not gift_classified:
        east_food_in = max(-food_flow, 0.0)
        east_food_out = max(food_flow, 0.0)
        east_wealth_in = max(wealth_flow, 0.0)
        east_wealth_out = max(-wealth_flow, 0.0)
        east_gold_in = gold_w2e
        east_gold_out = gold_e2w
        west_food_in = max(food_flow, 0.0)
        west_food_out = max(-food_flow, 0.0)
        west_wealth_in = max(-wealth_flow, 0.0)
        west_wealth_out = max(wealth_flow, 0.0)
        west_gold_in = gold_e2w
        west_gold_out = gold_w2e
        east_ratio = _value_ratio_for_side(
            east_food_in, east_food_out, east_wealth_in, east_wealth_out, east_gold_in, east_gold_out
        )
        west_ratio = _value_ratio_for_side(
            west_food_in, west_food_out, west_wealth_in, west_wealth_out, west_gold_in, west_gold_out
        )
        east_exploited = east_ratio < 0.7 and (east_food_out + east_wealth_out + east_gold_out) > eps
        west_exploited = west_ratio < 0.7 and (west_food_out + west_wealth_out + west_gold_out) > eps

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
            "iron_from_east_to_west": iron_e2w,
            "iron_from_west_to_east": iron_w2e,
            "gold_from_east_to_west": gold_e2w,
            "gold_from_west_to_east": gold_w2e,
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
        "food_west_before": west_before["food"],
        "food_west_after": west_after["food"],
        "wealth_west_before": west_before["wealth"],
        "wealth_west_after": west_after["wealth"],
        "iron_west_before": west_before.get("iron"),
        "iron_west_after": west_after.get("iron"),
        "gold_west_before": west_before.get("gold"),
        "gold_west_after": west_after.get("gold"),
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
    summary_base = (
        f"Step {model.steps}: trade_type={trade_type}, relation={new_label}, "
        + ", ".join(flow_bits)
    )
    model.leader_east.record_interaction(
        f'{summary_base}; West said "{entry.get("west_line", "")}".'
    )
    model.leader_west.record_interaction(
        f'{summary_base}; East said "{entry.get("east_line", "")}".'
    )
