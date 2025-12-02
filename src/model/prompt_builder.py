"""Quick card: prompt builders for decisions and negotiations."""

from __future__ import annotations

from typing import Any, Dict

import config
from src.model.traits import negotiation_style_line, personality_summary


def compose_prompt(state: Dict[str, Any]) -> str:
    """Cue card: spell out the state, options, and JSON shape for the decision call."""
    # Presentation cue: this builder shows how we steer the LLM into crisp, auditable JSON.
    name = state.get("territory", "Unknown")
    food = state.get("food", 0.0)
    wealth = state.get("wealth", 0.0)
    wood = state.get("wood", 0.0)
    pop = state.get("population", 0.0)
    iron = state.get("iron", 0.0)
    gold = state.get("gold", 0.0)
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
    pressures = state.get("resource_pressures") or {}
    current_season_mult = config.SEASON_MULTIPLIERS.get(current_season, 1.0)
    next_season_mult = config.SEASON_MULTIPLIERS.get(next_season, 1.0)

    horizon_steps = config.FOOD_SAFETY_HORIZON_STEPS
    food_need_horizon = required_food * horizon_steps
    food_gap = food - food_need_horizon
    workers = pop / config.PEOPLE_PER_WORK_POINT if config.PEOPLE_PER_WORK_POINT else 0.0
    wage_bill = workers * config.WAGE_PER_WORKER
    wage_gap = wealth - wage_bill

    personality_vector = state.get("personality_vector") or {}
    active_traits = state.get("active_traits") or []
    other_trait_notes = state.get("other_trait_notes") or "No hypothesis about the neighbour yet."
    adaptation_pressure = state.get("adaptation_pressure") or ""
    personality_line = personality_summary(personality_vector, active_traits)
    gift_note = state.get("gift_balance_note") or "No outstanding trade imbalance noted."

    def _gap_status(value: float) -> str:
        return "OK" if value >= 0 else "SHORTFALL"

    tier_costs = {
        "wood": {"wood": config.INFRA_TIER_WOOD_WOOD_COST, "wealth": config.INFRA_TIER_WOOD_WEALTH_COST},
        "iron": {"iron": config.INFRA_TIER_IRON_IRON_COST, "wealth": config.INFRA_TIER_IRON_WEALTH_COST},
        "gold": {"gold": config.INFRA_TIER_GOLD_GOLD_COST, "iron": config.INFRA_TIER_GOLD_IRON_COST},
    }
    tier_points = {
        "wood": config.INFRA_TIER_WOOD_POINTS,
        "iron": config.INFRA_TIER_IRON_POINTS,
        "gold": config.INFRA_TIER_GOLD_POINTS,
    }

    def _tier_line(label: str, costs: Dict[str, float], points: int) -> tuple[bool, str]:
        missing: list[str] = []
        for resource, amount in costs.items():
            have = float(state.get(resource, 0.0) or 0.0)
            if have < amount:
                missing.append(f"{resource} {have:.2f}/{amount:.2f}")
        ready = not missing
        detail = "ready" if ready else f"short {', '.join(missing)}"
        return ready, f"  - {label} tier (+{points * 10}% production): {detail}"

    wood_ready, wood_line = _tier_line("Wood", tier_costs["wood"], tier_points["wood"])
    iron_ready, iron_line = _tier_line("Iron", tier_costs["iron"], tier_points["iron"])
    gold_ready, gold_line = _tier_line("Gold", tier_costs["gold"], tier_points["gold"])
    infra_snapshot = (
        "Infrastructure tiers (auto-build picks the strongest you can afford):\n"
        f"{wood_line}\n{iron_line}\n{gold_line}\n"
    )
    exclusive_note = (
        "Only your territory mines either iron or gold (never both). Your neighbour has the other metal, "
        "so trade is the only way to reach the top tiers."
    )
    auto_infra_note = (
        "Setting build_infrastructure:true automatically spends the highest tier you can afford; "
        "each point adds +10% to every resource and points stack."
    )

    infra_ready = wood_ready or iron_ready or gold_ready
    highest_ready = None
    if gold_ready:
        highest_ready = "gold"
    elif iron_ready:
        highest_ready = "iron"
    elif wood_ready:
        highest_ready = "wood"
    infra_prompt = ""
    if infra_ready and highest_ready:
        bonus = tier_points[highest_ready] * 10
        infra_prompt = (
            f"You already meet the cost for a {highest_ready} tier upgrade (+{bonus}% production). "
            "If food safety looks secure for the next few steps, setting build_infrastructure:true will invest immediately."
        )

    ledger = state.get("trade_ledger") or {}
    ledger_line = (
        f"Food sent {ledger.get('food_sent', 0.0):.2f} / received {ledger.get('food_received', 0.0):.2f}; "
        f"Wealth sent {ledger.get('wealth_sent', 0.0):.2f} / received {ledger.get('wealth_received', 0.0):.2f}; "
        f"Wood sent {ledger.get('wood_sent', 0.0):.2f} / received {ledger.get('wood_received', 0.0):.2f}; "
        f"net food {ledger.get('net_food', 0.0):+.2f}, net wealth {ledger.get('net_wealth', 0.0):+.2f}, net wood {ledger.get('net_wood', 0.0):+.2f}"
    )
    milestones = state.get("long_term_notes") or []
    milestone_block = "\n".join(f"    * {note}" for note in milestones) if milestones else "    * None yet."

    prompt = f"""
You are the autonomous leader of "{name}" at step {step}. Population {pop:.0f} (requires {required_food:.2f} food/step). Your primary duty is to keep citizens fed, invest surpluses into infrastructure, and negotiate fair trades that respect reciprocity.

Status snapshot:
  - Season now {current_season} (food/wood x{current_season_mult:.2f}); next {next_season} (x{next_season_mult:.2f}).
  - Work points available: {work_points:.2f}. Wage bill ≈ {wage_bill:.2f} wealth ({_gap_status(wage_gap)} gap {wage_gap:.2f}).
  - Food horizon ({horizon_steps} steps): need {food_need_horizon:.2f}, have {food:.2f}, gap {food_gap:.2f} ({_gap_status(food_gap)}).
  - Diplomatic stance: {relation} (score {relation_score}).

Resources:
  Food {food:.2f} | Wealth {wealth:.2f} | Wood {wood:.2f} | Iron {iron:.2f} | Gold {gold:.2f} | Infra {infra} (+{infra*10:.0f}%).
Per-work yields (after infra): food {yields.get('food_per_work', 0.0):.3f}, wood {yields.get('wood_per_work', 0.0):.3f}, wealth {yields.get('wealth_per_work', 0.0):.3f}, iron {yields.get('iron_per_work', 0.0):.3f}, gold {yields.get('gold_per_work', 0.0):.3f}.

Resource pressures:
  - Food safety ratio (next {horizon_steps} steps): {pressures.get('food_ratio_horizon', 0.0):.2f}
  - Wood gap toward next tier: {pressures.get('wood_gap', 0.0):.2f}
  - Wealth gap toward next tier: {pressures.get('wealth_gap', 0.0):.2f}
  - Infra ready now? {pressures.get('infra_ready')}

Infrastructure readiness:
{infra_snapshot}{auto_infra_note}
{infra_prompt}

Trade + relations:
  - Belief about neighbour: {other_trait_notes}
  - Gift balance status: {gift_note}
  - Ledger: {ledger_line}
  - Long-term milestones:\n{milestone_block}

Guidance:
  1. Cover food/wages for the next {horizon_steps} steps before chasing prosperity.
  2. When buffers allow, gather the missing materials and set build_infrastructure:true; wood tier (5 wood + 2 wealth) should not be delayed once ready.
  3. Reciprocity matters: token gifts must include a matching wealth/metal concession unless your food safety ratio exceeds the neighbour's by ≥1.5×.
  4. Neutral/strained stances demand balanced trades. Only extend aid without payment when relations are cordial/allied and your buffer remains >1.5× safer; otherwise decline and explain.
  5. Remember you can trade food, wealth, wood, iron, or gold; match what the other side actually requested and demand compensation if you did not initiate.
  6. Use adaptation pressure notes to adjust traits or stance; log any deliberate changes in "trait_adjustment".

Recent directive: "{prior_directive}"
{personality_line}
{"Adaptation pressure: " + adaptation_pressure if adaptation_pressure else "No acute adaptation pressure; stay vigilant."}
Recent decisions:\n{history_text}
Recent interactions:\n{interaction_text}

Soft priority hint (override when appropriate):
  - food_safety_ratio: {priority_hint.get('food_safety_ratio', 0.0):.3f}
  - suggested weights: {priority_hint.get('priorities', {})}

Respond with JSON:
{{
  "allocations": {{"focus_food": <float>, "focus_wood": <float>, "focus_wealth": <float>, "focus_iron": <float>, "focus_gold": <float>}},
  "build_infrastructure": <true|false>,
  "reason": "<why this plan fits now>",
  "next_prompt": "<directive for your future self>",
  "trait_adjustment": "<trait guidance or 'no change'>"
}}
Shares must be between 0 and 1 and can sum to <= 1.0; leave unused capacity idle if helpful. The build flag buys the strongest tier you can pay for. No text outside that JSON object.
"""
    return prompt


def compose_negotiation_context(state: Dict[str, Any]) -> str:
    """Negotiation card: lay out both sides so the LLM can riff dialogue plus trade."""
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

    def _ledger_line(side: Dict[str, Any]) -> str:
        ledger = side.get("trade_ledger", {}) or {}
        return (
            f"food sent {ledger.get('food_sent', 0.0):.2f}, received {ledger.get('food_received', 0.0):.2f}; "
            f"wealth sent {ledger.get('wealth_sent', 0.0):.2f}, received {ledger.get('wealth_received', 0.0):.2f}"
        )

    east_intent = state.get("east_intent") or {}
    west_intent = state.get("west_intent") or {}

    def _intent_text(label: str, intent: Dict[str, Any]) -> str:
        if not intent or not intent.get("initiate"):
            return f"{label}: no request this step."
        offer = ""
        if intent.get("offer_resource"):
            offer = f" (can offer up to {intent.get('offer_amount', 0.0)} {intent.get('offer_resource')})"
        return (
            f"{label}: requests {intent.get('request_amount', 0.0)} {intent.get('request_resource')} "
            f"({intent.get('reason', '').strip()}){offer}; urgency={intent.get('urgency', 'normal')}"
        )

    context = f"""
Simulate a negotiation at step {step} between East and West (before upkeep). Current relationship: {east.get('relation_to_neighbor', 'neutral')} (score {east.get('relation_score', 0)}). Each leader must justify aid with clear reciprocity; neutral/strained stances default to balanced swaps.

State:
  - East -> food {east.get('food', 'unknown')} (safety ratio {east_ratio:.2f}), wealth {east.get('wealth', 'unknown')}, pop {east.get('population', 'unknown')}, ledger {_ledger_line(east)}.
  - West -> food {west.get('food', 'unknown')} (safety ratio {west_ratio:.2f}), wealth {west.get('wealth', 'unknown')}, pop {west.get('population', 'unknown')}, ledger {_ledger_line(west)}.
  - Last allocations -> East: {last_actions.get('east')}, West: {last_actions.get('west')}.
  - Dialogue initiator this step: {state.get('dialogue_initiator', 'East')}
  - Intent -> {_intent_text('East', east_intent)}
             {_intent_text('West', west_intent)}

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

Rules:
  - Resource pressures -> East: {east.get('resource_pressures', {})}; West: {west.get('resource_pressures', {})}.
  - Allowed trade fields: food_from_east_to_west, wealth_from_west_to_east, wood_from_east_to_west, wood_from_west_to_east, iron_from_east_to_west, iron_from_west_to_east, gold_from_east_to_west, gold_from_west_to_east.
  - If one side has received several gifts already, they must offer wealth/metal concessions before asking for more food; refusal is acceptable when it protects citizens.
  - Token gifts (0.1-0.2 food) are acceptable only when the donor remains ≥1.5x safer on food ratio and explicitly agrees.
  - Balanced trades (e.g., 0.2 food for 0.15 wealth) are preferred at neutral stance; strained relations require explicit justification for any aid.
  - You may decline a request; doing so may strain relations, especially if the requester is desperate, but autonomy takes precedence. If you did not initiate, treat the other side as petitioner and insist on fair compensation before agreeing.
  - Document the dialogue (initiator speaks first, alternate turns) and ensure the final trade matches the closing statements. If you refuse, clearly state the reason and return zero flows.
"""
    return context


def compose_negotiation_turn_prompt(side: str, session: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Turn prompt: ask one leader to reply/accept/counter with a concrete proposal."""
    side_label = "East" if str(side).lower().startswith("e") else "West"
    step = state.get("step", "unknown")
    east = state.get("east", {})
    west = state.get("west", {})
    east_intent = state.get("east_intent") or {}
    west_intent = state.get("west_intent") or {}
    relation = east.get("relation_to_neighbor", "neutral")
    initiator = state.get("dialogue_initiator", "East")
    turn_index = session.get("turn_count", 0)
    max_turns = session.get("max_turns")
    max_turns_text = str(max_turns) if max_turns is not None else "no cap"
    east_history = state.get("east_history_text") or "No previous steps in this run."
    west_history = state.get("west_history_text") or "No previous steps in this run."
    east_interactions = state.get("east_interactions_text") or "No notable interactions recorded."
    west_interactions = state.get("west_interactions_text") or "No notable interactions recorded."
    east_personality = personality_summary(east.get("personality_vector") or {}, east.get("active_traits") or [])
    west_personality = personality_summary(west.get("personality_vector") or {}, west.get("active_traits") or [])
    east_style = negotiation_style_line(east.get("active_traits") or [])
    west_style = negotiation_style_line(west.get("active_traits") or [])

    def _safety_ratio(side_state: Dict[str, Any]) -> float:
        pop = side_state.get("population", 0) or 0.0
        food = side_state.get("food", 0) or 0.0
        req = (pop / 10.0) * config.FOOD_PER_10_POP
        return food / req if req > 0 else float("inf")

    def _ledger_line(side_state: Dict[str, Any]) -> str:
        ledger = side_state.get("trade_ledger", {}) or {}
        return (
            f"food sent {ledger.get('food_sent', 0.0):.2f}, received {ledger.get('food_received', 0.0):.2f}; "
            f"wealth sent {ledger.get('wealth_sent', 0.0):.2f}, received {ledger.get('wealth_received', 0.0):.2f}; "
            f"wood sent {ledger.get('wood_sent', 0.0):.2f}, received {ledger.get('wood_received', 0.0):.2f}"
        )

    def _intent_text(label: str, intent: Dict[str, Any]) -> str:
        if not intent or not intent.get("initiate"):
            return f"{label}: no request this step."
        offer = ""
        if intent.get("offer_resource"):
            offer = f" (offers up to {intent.get('offer_amount', 0.0)} {intent.get('offer_resource')})"
        return (
            f"{label}: requests {intent.get('request_amount', 0.0)} {intent.get('request_resource')} "
            f"({intent.get('reason', '').strip()}){offer}; urgency={intent.get('urgency', 'normal')}"
        )

    def _pressure_line(side_state: Dict[str, Any]) -> str:
        pressures = side_state.get("resource_pressures", {}) or {}
        return (
            f"food_ratio_horizon {pressures.get('food_ratio_horizon', 0.0):.2f}, "
            f"wood_gap {pressures.get('wood_gap', 0.0):.2f}, "
            f"wealth_gap {pressures.get('wealth_gap', 0.0):.2f}"
        )

    def _proposal_line(proposal: Dict[str, Any]) -> str:
        if not proposal:
            return "none on the table yet."
        return (
            f"food E->W {proposal.get('food_from_east_to_west', 0.0)}, "
            f"wealth W->E {proposal.get('wealth_from_west_to_east', 0.0)}, "
            f"wood E->W {proposal.get('wood_from_east_to_west', 0.0)}, "
            f"wood W->E {proposal.get('wood_from_west_to_east', 0.0)}, "
            f"iron E->W {proposal.get('iron_from_east_to_west', 0.0)}, "
            f"iron W->E {proposal.get('iron_from_west_to_east', 0.0)}, "
            f"gold E->W {proposal.get('gold_from_east_to_west', 0.0)}, "
            f"gold W->E {proposal.get('gold_from_west_to_east', 0.0)}, "
            f"reason: {proposal.get('reason', '')}"
        )

    dialogue = session.get("dialogue") or []
    if dialogue:
        dialogue_lines = "\n".join(
            f"  {idx + 1}. {entry.get('speaker', '?')}: \"{entry.get('line', '')}\" "
            f"(decision {entry.get('decision', 'counter')})"
            for idx, entry in enumerate(dialogue)
        )
    else:
        dialogue_lines = "  (no dialogue yet; you open with a concrete proposal)."

    current_proposal = session.get("current_proposal") or {}
    current_proposer = session.get("current_proposer") or "none"
    side_safety = _safety_ratio(east if side_label == "East" else west)
    other_safety = _safety_ratio(west if side_label == "East" else east)

    def _tone_label(safety: float) -> str:
        if safety < 0.8:
            return "desperate"
        if safety < 1.0:
            return "anxious"
        if safety >= 1.3:
            return "confident"
        return "guarded"

    tone_self = _tone_label(side_safety)
    tone_other = _tone_label(other_safety)

    prompt = f"""
Negotiation turn at step {step} (turn {turn_index + 1} of {max_turns_text}). Relationship: {relation} (score {east.get('relation_score', 0)}). You are speaking as {side_label}; the other leader will answer separately.
Speak ONLY for {side_label}. Do not describe the other side's needs as your own.
Your tone should reflect your traits and safety: you feel {tone_self}; the other side likely feels {tone_other}. Let that shape word choice (e.g., terse if desperate, measured if confident). Avoid mirroring the other side's phrasing.

State snapshot:
  - East: food {east.get('food', 'unknown')} (safety ratio {_safety_ratio(east):.2f}), wealth {east.get('wealth', 'unknown')}, wood {east.get('wood', 'unknown')}, iron {east.get('iron', 'unknown')}, gold {east.get('gold', 'unknown')}, pop {east.get('population', 'unknown')}, ledger {_ledger_line(east)}, pressures {_pressure_line(east)}.
  - West: food {west.get('food', 'unknown')} (safety ratio {_safety_ratio(west):.2f}), wealth {west.get('wealth', 'unknown')}, wood {west.get('wood', 'unknown')}, iron {west.get('iron', 'unknown')}, gold {west.get('gold', 'unknown')}, pop {west.get('population', 'unknown')}, ledger {_ledger_line(west)}, pressures {_pressure_line(west)}.
  - Intent -> {_intent_text('East', east_intent)}
             {_intent_text('West', west_intent)}
  - Dialogue initiator: {initiator}

Current proposal from {current_proposer}: {_proposal_line(current_proposal)}
Dialogue so far:
{dialogue_lines}

Recent history and style notes:
  - East: {east_history}
  - West: {west_history}
  - East personality: {east_personality}; style hint: {east_style}
  - West personality: {west_personality}; style hint: {west_style}
  - East interactions: {east_interactions}
  - West interactions: {west_interactions}
Voice guide:
  - East should sound like: {east_style} ({east_personality})
  - West should sound like: {west_style} ({west_personality})
  - Do not copy the other side's phrasing; keep your own voice.

Task: write a single reply line for {side_label} and choose whether to counter, accept, or decline.
  - counter: change the proposal numbers to what you want (can be zeros) and explain briefly in your reply.
  - accept: explicitly accept the current proposal without changing its numbers.
  - decline: politely refuse and set all flows to zero.
  - If you counter, you MUST change at least one number from the current proposal. If you would repeat the same numbers, choose accept or decline instead.
  - Keep your wording distinct from prior lines; avoid repeating the same sentence structure.
  - If your food safety ratio is lower than the other side's, do NOT offer wealth payments; insist on compensation instead.
  - If no proposal is on the table (turn 1), do not accept or decline; put a concrete proposal with non-zero flows that fits your needs and constraints.
  - Include a brief change note like "change: +food, +wealth" describing how your numbers differ from the current proposal.
  - Let your traits color your phrasing (cautious adds caveats, opportunistic presses leverage, stoic stays terse). If you are desperate (low safety), be blunt; if confident, be measured.

Trading rules:
  - You may trade food, wealth, wood, iron, or gold. Positive food/wood/iron/gold numbers mean East ships that resource to West; positive wealth means West pays East.
  - Keep any offer within your side's resources and avoid leaving your own food safety below 1.0 unless desperate.
  - Reciprocity matters: repeated free gifts are discouraged; if you ask for aid, be ready to pay with wealth or metal.
  - Mention whether you are accepting, countering, or declining in the reply.
  - Speak in a style that fits your active traits: let a pragmatic trait keep wording concise and transactional; let a cautious trait add caveats; let an opportunistic trait press for leverage.
  - If the other side has already countered multiple times without moving, you may decline and end talks.

Respond ONLY with JSON:
{{
  "reply": "<one or two sentences from {side_label}>",
  "decision": "<counter|accept|decline>",
  "proposal": {{
    "food_from_east_to_west": <number>,
    "wealth_from_west_to_east": <number>,
    "wood_from_east_to_west": <number>,
    "wood_from_west_to_east": <number>,
    "iron_from_east_to_west": <number>,
    "iron_from_west_to_east": <number>,
    "gold_from_east_to_west": <number>,
    "gold_from_west_to_east": <number>,
    "reason": "<short justification>"
  }}
}}
"""
    return prompt
