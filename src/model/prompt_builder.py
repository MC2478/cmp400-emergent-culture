"""Prompt composition utilities for decisions and negotiations."""

from __future__ import annotations

from typing import Any, Dict

import config
from src.model.traits import negotiation_style_line, personality_summary


def compose_prompt(state: Dict[str, Any]) -> str:
    """Describe the state, enumerate valid actions, and demand JSON output."""
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

    personality_vector = state.get("personality_vector") or {}
    active_traits = state.get("active_traits") or []
    other_trait_notes = state.get("other_trait_notes") or "No hypothesis about the neighbour yet."
    adaptation_pressure = state.get("adaptation_pressure") or ""
    personality_line = personality_summary(personality_vector, active_traits)

    def _gap_status(value: float) -> str:
        return "OK" if value >= 0 else "SHORTFALL"

    feasibility_snapshot = f"""
Resource feasibility snapshot (gap = have - need):
  - Food buffer (next {horizon_steps} steps): need {food_need_horizon:.2f}, have {food:.2f}, gap {food_gap:.2f} ({_gap_status(food_gap)}).
  - Wood for infrastructure: need {config.INFRA_COST_WOOD:.2f}, have {wood:.2f}, gap {wood_gap:.2f} ({_gap_status(wood_gap)}).
  - Wealth for infrastructure: need {config.INFRA_COST_WEALTH:.2f}, have {wealth:.2f}, gap {wealth_gap:.2f} ({_gap_status(wealth_gap)}).
  - Wage bill this step (~{wage_bill:.2f} wealth): gap {wage_gap:.2f} ({_gap_status(wage_gap)}).
"""
    infra_ready = (
        priority_hint.get("food_safety_ratio", 0.0) >= config.FOOD_SAFETY_GOOD_RATIO
        and wood >= config.INFRA_COST_WOOD
        and wealth >= config.INFRA_COST_WEALTH
        and infra < 2
    )
    infra_prompt = ""
    if infra_ready:
        infra_prompt = (
            f"You already meet the cost for an infrastructure upgrade (need wood {config.INFRA_COST_WOOD}, wealth {config.INFRA_COST_WEALTH}). "
            "If your food safety looks comfortably above 1.0 for the next few steps, setting build_infrastructure:true this step can boost long-term yields."
        )

    prompt = f"""
You are the autonomous leader of the territory "{name}" at simulation step {step}.
Population: {pop:.0f} people, requiring {required_food:.2f} food per step to avoid starvation.
Current resources: food={food:.2f}, wealth={wealth:.2f}, wood={wood:.2f}, infrastructure level={infra}.
Building infrastructure will immediately consume {config.INFRA_COST_WOOD:.0f} wood and {config.INFRA_COST_WEALTH:.0f} wealth; declaring this action without both on hand wastes the step.
{feasibility_snapshot}
Season outlook: current season="{current_season}" (food/wood yield x{current_season_mult:.2f}); next step remains "{next_season}" (x{next_season_mult:.2f}). Plan to exploit high-multiplier seasons for production and stockpile before harsh seasons.

Last self-set directive: "{prior_directive}"
{personality_line}
Belief about the other territory: {other_trait_notes}
{"Adaptation pressure: " + adaptation_pressure if adaptation_pressure else "No acute adaptation pressure; still stay alert to repeating mistakes."}
If adaptation pressure appears, treat it as a signal to adjust traits or strategy; explicitly propose a suitable "trait_adjustment" if you believe a mindset change would help.
{infra_prompt}

Work points available this step: {work_points} (roughly 100 population per work point adjusted by morale).
Current diplomatic stance toward your neighbour: {relation} (score {relation_score}). Your duty is to your citizens, but you should weigh short-term safety against long-term benefits; small trades and infra investments are acceptable when the buffer looks solid and the upside is clear.

Recent history of your decisions in this run:
{history_text}
Recent diplomatic interactions with your neighbour:
{interaction_text}
Pay attention to moments where past actions failed (e.g., infrastructure attempts without wood) and adjust course proactively.

Guiding prompts for priorities:
- Self-preservation first: if food_safety_ratio < 1, focus on survival and seek help; if > 1.5, consider cautious investments.
- Cooperation depends on relation + surplus: aid only when you retain a strong buffer and relations are cordial/allied, or when you receive clear benefit.
- Neutral/strained: prefer reciprocal trades; hostile: protect your position, but still consider fair exchanges if they improve resilience.
- If food safety looks comfortable for the next few steps and you have enough wood and wealth to build infrastructure, setting build_infrastructure:true is often worth it to raise future yields instead of idling.

Per-work yields with current infrastructure:
  - focus_food:  {yields.get('food_per_work', 0.0):.3f} food/work
  - focus_wood:  {yields.get('wood_per_work', 0.0):.3f} wood/work
  - focus_wealth: {yields.get('wealth_per_work', 0.0):.3f} wealth/work

Soft priority hint (you may override this):
  - food_safety_ratio (food vs. next {config.FOOD_SAFETY_HORIZON_STEPS} steps): {priority_hint.get('food_safety_ratio', 0.0):.3f}
  - suggested weights: {priority_hint.get('priorities', {})}

Before selecting an action, run this feasibility checklist:
  1. Ensure minimum food coverage for the next {config.FOOD_SAFETY_HORIZON_STEPS} steps (grow or trade if short).
  2. Confirm wood >= {config.INFRA_COST_WOOD} if you intend to build infrastructure; otherwise gather wood first.
  3. Confirm wealth >= {config.INFRA_COST_WEALTH} for infrastructure; if not, consider producing wealth.
  4. Cover this step's wage bill (~{wage_bill:.2f} wealth) or plan to raise wealth immediately to prevent morale collapse.
  5. Align choices with seasonal multipliers: push production during high-yield seasons and enter low-yield seasons with reserves ready.
  6. Use recent failures, relation shifts, and your stance toward the neighbour (hostile/strained/neutral/cordial/allied) to avoid repeating mistakes.

Work allocation options (splits can be uneven; tailor to current needs and season):
  - "focus_food": grow food using the food_per_work yield.
  - "focus_wood": grow wood using the wood_per_work yield.
  - "focus_wealth": grow wealth using the wealth_per_work yield.
You may split work freely across these options (shares between 0.0 and 1.0 that sum to <= 1.0). Uneven mixes like 0.6/0.3/0.1 are expected; avoid repeating identical 50/50 splits unless it truly fits the moment. Any unassigned share idles. If wood is scarce and infrastructure is still 0, consider a small wood share even when food is stable so you can build.
Infrastructure option:
  - set "build_infrastructure": true to spend {config.INFRA_COST_WOOD} wood and {config.INFRA_COST_WEALTH} wealth immediately if available (in addition to your work allocations).

Objectives (in soft order):
1. Avoid starvation in the short and medium term.
2. Build resilience by investing in infrastructure and spare supplies when food safety allows.
3. Develop prosperity (wealth/wood) to unlock future economic and diplomatic options.
4. Maintain workable relations where it serves your people; only extend aid when relations support it or you receive clear benefit.
5. Improve population well-being (growth, quality of life) once survival is secured.

You can follow or override the hint freely. Consider trade-offs between immediate survival and long-term strength using both the history above and the current metrics.

Respond with a single JSON object of the form:
{{
  "allocations": {{"focus_food": <float>, "focus_wood": <float>, "focus_wealth": <float>}},
  "build_infrastructure": <true|false>,
  "reason": "<why this split makes sense now>",
  "next_prompt": "<a concise directive you want to remember for the next step>",
  "trait_adjustment": "<short sentence proposing a trait or mindset change, or 'no change'>"
}}
Only include the keys you need inside "allocations"; shares must be between 0 and 1 and sum to at most 1. The optional "build_infrastructure" flag can be true even when you allocate work elsewhere, provided you can afford the cost. No extra text or keys outside this JSON.

"""
    return prompt


def compose_negotiation_context(state: Dict[str, Any]) -> str:
    """Outline the joint state so the LLM can improvise a mini dialogue and trade."""
    step = state.get("step", "unknown")
    east = state.get("east", {})
    west = state.get("west", {})
    last_actions = state.get("last_actions", {})
    east_history = state.get("east_history_text") or "No previous steps in this run."
    west_history = state.get("west_history_text") or "No previous steps in this run."
    east_interactions = state.get("east_interactions_text") or "No notable interactions recorded."
    west_interactions = state.get("west_interactions_text") or "No notable interactions recorded."
    east_personality = personality_summary(east.get("personality_vector") or {}, east.get("active_traits") or [])
    west_personality = personality_summary(west.get("personality_vector") or {}, west.get("active_traits") or [])
    east_style = negotiation_style_line(east.get("active_traits") or [])
    west_style = negotiation_style_line(west.get("active_traits") or [])

    def safety_ratio(side: Dict[str, Any]) -> float:
        pop = side.get("population", 0) or 0.0
        food = side.get("food", 0) or 0.0
        req = (pop / 10.0) * config.FOOD_PER_10_POP
        return food / req if req > 0 else float("inf")

    east_ratio = safety_ratio(east)
    west_ratio = safety_ratio(west)

    context = f"""
Simulate a calm negotiation at step {step} between the leaders of East and West (this occurs before upkeep/starvation each tick).
Current relationship status: {east.get('relation_to_neighbor', 'neutral')} (score {east.get('relation_score', 0)}). Each leader's primary duty is to their own citizens; cooperation or aid should depend on trust (cordial/allied) or clear reciprocity. Neutral/strained stances justify caution; hostile stances justify demands or refusals.
East -> food {east.get('food', 'unknown')} (safety ratio ~{east_ratio:.2f}), wealth {east.get('wealth', 'unknown')}, population {east.get('population', 'unknown')}.
West -> food {west.get('food', 'unknown')} (safety ratio ~{west_ratio:.2f}), wealth {west.get('wealth', 'unknown')}, population {west.get('population', 'unknown')}.
Last allocations -> East: {last_actions.get('east')}, West: {last_actions.get('west')}.

Recent history for East:
{east_history}

Recent history for West:
{west_history}

East personality: {east_personality}
Negotiation style hint for East: {east_style}
West personality: {west_personality}
Negotiation style hint for West: {west_style}

Recent interaction log for East:
{east_interactions}

Recent interaction log for West:
{west_interactions}

There is no safety net - if a side starves it may collapse. Weigh risk versus benefit: small non-zero trades are acceptable when buffers look safe (token gifts of 0.1-0.2 or balanced swaps like 0.1-0.2 food for similar wealth). Friendly leaders are happier to send tiny gifts or slight generosity; Wealth-hoarder only parts with wealth when buffers feel comfortable; Opportunistic may offer small trades to build leverage; Aggressive rarely gifts and prefers self-favouring deals; Isolationist often opts for no trade unless the benefit is clear. Balanced trades that exchange small food/wealth amounts are welcome when both sides stay safe.
"""
    return context
