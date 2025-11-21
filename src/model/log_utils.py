"""Logging helpers extracted from world_model.py for chronicle writing and console summaries."""

from __future__ import annotations

from typing import Any, Dict, List

import config
from src.model.traits import trait_gloss


def fmt_res(value: Any) -> str:
    """Format resource numbers with configured precision."""
    try:
        return f"{float(value):.{config.RESOURCE_DISPLAY_DECIMALS}f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_pop(value: Any) -> str:
    """Format population counts with configured precision."""
    try:
        decimals = max(0, int(config.POP_DISPLAY_DECIMALS))
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def append_chronicle_action(chronicle: List[Dict[str, Any]], step: int, territory_name: str, data: Dict[str, Any]) -> None:
    decision = data.get("decision", {})
    before = decision.get("before", {})
    after = decision.get("after", {})
    meta = decision.get("decision", {})
    trait_state = meta.get("trait_state", {})
    entry = {
        "event_type": "action",
        "step": step,
        "territory": territory_name,
        "actor": territory_name,
        "action": meta.get("action"),
        "target": meta.get("target"),
        "reason": meta.get("reason"),
        "allocations": meta.get("applied_allocations") or meta.get("allocations"),
        "build_infrastructure_requested": bool(meta.get("build_infrastructure")),
        "infrastructure_built": meta.get("infrastructure_built"),
        "trait_adjustment": meta.get("trait_adjustment"),
        "food_before": before.get("food"),
        "food_after": after.get("food"),
        "wealth_before": before.get("wealth"),
        "wealth_after": after.get("wealth"),
        "wood_before": before.get("wood"),
        "wood_after": after.get("wood"),
        "iron_before": before.get("iron"),
        "iron_after": after.get("iron"),
        "gold_before": before.get("gold"),
        "gold_after": after.get("gold"),
        "infrastructure_before": before.get("infrastructure_level"),
        "infrastructure_after": after.get("infrastructure_level"),
        "population_before": before.get("population"),
        "population_after": after.get("population"),
        "llm_used": decision.get("used_llm"),
        "relation_before": before.get("relation_to_neighbor"),
        "relation_after": after.get("relation_to_neighbor"),
        "neighbor_food_before": before.get("neighbor_food"),
        "neighbor_food_after": after.get("neighbor_food"),
        "neighbor_wealth_before": before.get("neighbor_wealth"),
        "neighbor_wealth_after": after.get("neighbor_wealth"),
        "neighbor_population_before": before.get("neighbor_population"),
        "neighbor_population_after": after.get("neighbor_population"),
        "neighbor_wood_before": before.get("neighbor_wood"),
        "neighbor_wood_after": after.get("neighbor_wood"),
        "neighbor_iron_before": before.get("neighbor_iron"),
        "neighbor_iron_after": after.get("neighbor_iron"),
        "neighbor_gold_before": before.get("neighbor_gold"),
        "neighbor_gold_after": after.get("neighbor_gold"),
        "active_traits": trait_state.get("active_traits"),
        "personality_vector": trait_state.get("personality_vector"),
        "trait_cooldown_steps": trait_state.get("trait_cooldown_steps"),
        "exploitation_streak": trait_state.get("exploitation_streak"),
        "starvation_streak": trait_state.get("starvation_streak"),
        "failed_strategy_streak": trait_state.get("failed_strategy_streak"),
        "other_trait_notes": trait_state.get("other_trait_notes"),
        "adaptation_pressure": trait_state.get("adaptation_pressure"),
        "trait_events": trait_state.get("trait_events") or meta.get("trait_events"),
    }
    chronicle.append(entry)


def append_chronicle_upkeep(
    chronicle: List[Dict[str, Any]],
    step: int,
    east_before: Dict[str, Any],
    east_after: Dict[str, Any],
    west_before: Dict[str, Any],
    west_after: Dict[str, Any],
) -> None:
    entry = {
        "event_type": "upkeep",
        "step": step,
        "east": {
            "food_before": east_before["food"],
            "food_after": east_after["food"],
            "population_before": east_before["population"],
            "population_after": east_after["population"],
            "wealth_before": east_before.get("wealth"),
            "wealth_after": east_after.get("wealth"),
            "wood_before": east_before.get("wood"),
            "wood_after": east_after.get("wood"),
            "iron_before": east_before.get("iron"),
            "iron_after": east_after.get("iron"),
            "gold_before": east_before.get("gold"),
            "gold_after": east_after.get("gold"),
            "infrastructure_before": east_before.get("infrastructure_level"),
            "infrastructure_after": east_after.get("infrastructure_level"),
        },
        "west": {
            "food_before": west_before["food"],
            "food_after": west_after["food"],
            "population_before": west_before["population"],
            "population_after": west_after["population"],
            "wealth_before": west_before.get("wealth"),
            "wealth_after": west_after.get("wealth"),
            "wood_before": west_before.get("wood"),
            "wood_after": west_after.get("wood"),
            "iron_before": west_before.get("iron"),
            "iron_after": west_after.get("iron"),
            "gold_before": west_before.get("gold"),
            "gold_after": west_after.get("gold"),
            "infrastructure_before": west_before.get("infrastructure_level"),
            "infrastructure_after": west_after.get("infrastructure_level"),
        },
    }
    chronicle.append(entry)


def print_step_summary(
    step: int,
    current_step_log: Dict[str, Any],
    chronicle: List[Dict[str, Any]],
    season_multipliers: Dict[str, float],
) -> None:
    """Print a readable summary for the step and append actions to the chronicle."""
    # [PRESENTATION] I highlight this summary when showing the assessor how every action, wage payment, and negotiation is mirrored into both the console and chronicle for later analysis.
    for idx, territory in enumerate(["West", "East"]):
        info = current_step_log.get(territory, {})
        decision = info.get("decision", {})
        if decision:
            if idx > 0:
                print("")
            before = decision.get("before", {})
            after = decision.get("after", {})
            used_llm = "yes" if decision.get("used_llm") else "no"
            meta = decision.get("decision", {})
            reason = meta.get("reason", "no reason provided")
            action = meta.get("action") or ("mixed_allocation" if meta.get("applied_allocations") else "wait")
            allocations = meta.get("applied_allocations") or meta.get("allocations") or {}
            focus_keys = ("focus_food", "focus_wood", "focus_wealth", "focus_iron", "focus_gold")
            formatted_allocs = ", ".join(
                f"{key}:{float(allocations.get(key, 0.0) or 0.0):.2f}" for key in focus_keys
            )
            allocation_line = f"\n    - Work allocation: {formatted_allocs}"
            infra_line = ""
            if meta.get("build_infrastructure"):
                status = "built" if meta.get("infrastructure_built") else "failed"
                infra_line = f"\n    - Infrastructure attempt: {status}"
            trait_state = meta.get("trait_state", {}) or {}
            trait_line = ""
            if trait_state:
                active = trait_state.get("active_traits") or []
                gloss = trait_gloss(trait_state.get("personality_vector") or {}, active)
                cooldown = trait_state.get("trait_cooldown_steps", 0)
                explo = trait_state.get("exploitation_streak", 0)
                starve = trait_state.get("starvation_streak", 0)
                stagnation = trait_state.get("failed_strategy_streak", 0)
                trait_label = ", ".join(active) if active else "none"
                trait_line = (
                    f"\n    - Traits: [{trait_label}] - {gloss} (cooldown {cooldown})"
                    f"\n      Pressures: exploitation={explo}, starvation={starve}, failed_strategy={stagnation}"
                )
                pressure = trait_state.get("adaptation_pressure")
                if pressure:
                    trait_line += f"\n      pressure: {pressure}"
                events = trait_state.get("trait_events") or meta.get("trait_events") or []
                if events:
                    summaries = "; ".join(
                        f"{e.get('event')}:{e.get('trait') or e.get('dimension')} ({e.get('reason', '').strip()})"
                        for e in events
                    )
                    trait_line += f"\n      trait_events: {summaries}"
            print(
                f"  {territory}: action={action} (LLM: {used_llm})\n"
                f"    - Reason:\n      {reason}{allocation_line}{infra_line}{trait_line}\n"
                f"    - Population: {fmt_pop(before.get('population'))} -> {fmt_pop(after.get('population'))}"
            )
            append_chronicle_action(chronicle, step, territory, info)

    # Compact resource table
    rows: list[str] = []
    header = "  Resources (before -> after)"
    for territory in ["West", "East"]:
        info = current_step_log.get(territory, {})
        decision = info.get("decision", {})
        if decision:
            before = decision.get("before", {})
            after = decision.get("after", {})
            row = (
                f"  {territory:<4}| food {fmt_res(before.get('food'))}->{fmt_res(after.get('food'))} | "
                f"wealth {fmt_res(before.get('wealth'))}->{fmt_res(after.get('wealth'))} | "
                f"wood {fmt_res(before.get('wood'))}->{fmt_res(after.get('wood'))} | "
                f"iron {fmt_res(before.get('iron'))}->{fmt_res(after.get('iron'))} | "
                f"gold {fmt_res(before.get('gold'))}->{fmt_res(after.get('gold'))} | "
                f"infra {before.get('infrastructure_level')}->{after.get('infrastructure_level')} | "
                f"pop {fmt_pop(before.get('population'))}->{fmt_pop(after.get('population'))}"
            )
            rows.append(row)
    if rows:
        print("\n" + header)
        for row in rows:
            print(row)

    negotiation_info = current_step_log.get("negotiation")
    if negotiation_info:
        entry = negotiation_info["entry"]
        east_before = negotiation_info["east_before"]
        east_after = negotiation_info["east_after"]
        west_before = negotiation_info["west_before"]
        west_after = negotiation_info["west_after"]
        dialogue_lines = entry.get("dialogue") or []
        if dialogue_lines:
            formatted = [f"      {line['speaker']}: \"{line['line']}\"" for line in dialogue_lines]
            dialogue_block = "\n" + "\n".join(formatted)
        else:
            dialogue_block = (
                f"    East: \"{entry['east_line']}\"\n"
                f"    West: \"{entry['west_line']}\""
            )
        trade = entry.get("trade", {})
        trade_line = (
            f"    Trade flows -> food East->West {trade.get('food_from_east_to_west', 0.0)}, "
            f"wealth West->East {trade.get('wealth_from_west_to_east', 0.0)}"
        )
        iron_e2w = trade.get("iron_from_east_to_west", 0.0)
        iron_w2e = trade.get("iron_from_west_to_east", 0.0)
        gold_e2w = trade.get("gold_from_east_to_west", 0.0)
        gold_w2e = trade.get("gold_from_west_to_east", 0.0)
        if iron_e2w or iron_w2e:
            trade_line += f", iron East->West {iron_e2w}, iron West->East {iron_w2e}"
        if gold_e2w or gold_w2e:
            trade_line += f", gold East->West {gold_e2w}, gold West->East {gold_w2e}"
        print(
            f"\n  Negotiation at step {step} ({entry.get('trade_type')}):\n"
            f"{dialogue_block}\n"
            f"{trade_line}\n"
            f"      East after trade: food {fmt_res(east_before['food'])}->{fmt_res(east_after['food'])}, "
            f"wealth {fmt_res(east_before['wealth'])}->{fmt_res(east_after['wealth'])}, "
            f"iron {fmt_res(east_before.get('iron'))}->{fmt_res(east_after.get('iron'))}, "
            f"gold {fmt_res(east_before.get('gold'))}->{fmt_res(east_after.get('gold'))}\n"
            f"      West after trade: food {fmt_res(west_before['food'])}->{fmt_res(west_after['food'])}, "
            f"wealth {fmt_res(west_before['wealth'])}->{fmt_res(west_after['wealth'])}, "
            f"iron {fmt_res(west_before.get('iron'))}->{fmt_res(west_after.get('iron'))}, "
            f"gold {fmt_res(west_before.get('gold'))}->{fmt_res(west_after.get('gold'))}\n"
            f"      Relation now: {entry.get('relation_label')} ({entry.get('east_stance') or entry.get('west_stance')}) "
            f"score={fmt_res(entry.get('relation_score'))}"
        )

    wage_rows: list[str] = []
    for territory in ["West", "East"]:
        wages = current_step_log.get(territory, {}).get("wages", {})
        if wages:
            before_w = wages.get("before", {})
            after_w = wages.get("after", {})
            wage_rows.append(
                f"  {territory:<4}| wealth {fmt_res(before_w.get('wealth'))}->{fmt_res(after_w.get('wealth'))} | "
                f"mult {fmt_res(before_w.get('multiplier'))}->{fmt_res(after_w.get('multiplier'))} | "
                f"strike {after_w.get('on_strike')} | unpaid {fmt_res(after_w.get('unpaid_steps'))}"
            )
    if wage_rows:
        print("\n  Wages:")
        for row in wage_rows:
            print(row)

    upkeep_rows: list[str] = []
    for territory in ["West", "East"]:
        info = current_step_log.get(territory, {})
        upkeep = info.get("upkeep", {})
        if upkeep:
            before_u = upkeep.get("before", {})
            after_u = upkeep.get("after", {})
            req = (before_u.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
            upkeep_rows.append(
                f"  {territory:<4}| food {fmt_res(before_u.get('food'))}->{fmt_res(after_u.get('food'))} | "
                f"wealth {fmt_res(before_u.get('wealth'))}->{fmt_res(after_u.get('wealth'))} | "
                f"wood {fmt_res(before_u.get('wood'))}->{fmt_res(after_u.get('wood'))} | "
                f"iron {fmt_res(before_u.get('iron'))}->{fmt_res(after_u.get('iron'))} | "
                f"gold {fmt_res(before_u.get('gold'))}->{fmt_res(after_u.get('gold'))} | "
                f"infra {before_u.get('infrastructure_level')}->{after_u.get('infrastructure_level')} | "
                f"pop {fmt_pop(before_u.get('population'))}->{fmt_pop(after_u.get('population'))} | req_food {fmt_res(req)}"
            )
    if upkeep_rows:
        print("\n  Upkeep:")
        for row in upkeep_rows:
            print(row)
    print("\n" + "-" * 60 + "\n")
