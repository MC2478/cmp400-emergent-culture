"""Quick card: TerritoryState + LeaderAgent for the CMP400 demo mixing rules and LLM choices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mesa

import config
from config import (
    FOOD_PER_10_POP,
    INFRA_TIER_WOOD_WOOD_COST,
    INFRA_TIER_WOOD_WEALTH_COST,
    INFRA_TIER_WOOD_POINTS,
    INFRA_TIER_IRON_IRON_COST,
    INFRA_TIER_IRON_WEALTH_COST,
    INFRA_TIER_IRON_POINTS,
    INFRA_TIER_GOLD_GOLD_COST,
    INFRA_TIER_GOLD_IRON_COST,
    INFRA_TIER_GOLD_POINTS,
    FOOD_SAFETY_HORIZON_STEPS,
    FOOD_SAFETY_GOOD_RATIO,
    NON_FOOD_MIN_FRACTION,
    MAX_LEADER_MEMORY_EVENTS,
)
from src.agents.production import apply_allocations, effective_yields as compute_yields, work_points as compute_work_points
from src.model.llm_client import LLMDecisionClient, summarise_memory_for_prompt
from src.model.traits import (
    adaptation_pressure_text,
    apply_trait_actions,
    apply_pressure_adaptation,
    interpret_trait_adjustment,
    neutral_personality_vector,
    tick_trait_cooldown,
)

# Rulebook note: shared action vocabulary for both LLM and rule-based flows.
WORK_ACTIONS: set[str] = {"focus_food", "focus_wood", "focus_wealth", "focus_iron", "focus_gold"}
ALLOWED_ACTIONS: set[str] = set(WORK_ACTIONS) | {"build_infrastructure", "wait"}
INFRA_TIER_SEQUENCE: tuple[tuple[str, Dict[str, float], int], ...] = (
    (
        "gold",
        {"gold": INFRA_TIER_GOLD_GOLD_COST, "iron": INFRA_TIER_GOLD_IRON_COST},
        INFRA_TIER_GOLD_POINTS,
    ),
    (
        "iron",
        {"iron": INFRA_TIER_IRON_IRON_COST, "wealth": INFRA_TIER_IRON_WEALTH_COST},
        INFRA_TIER_IRON_POINTS,
    ),
    (
        "wood",
        {"wood": INFRA_TIER_WOOD_WOOD_COST, "wealth": INFRA_TIER_WOOD_WEALTH_COST},
        INFRA_TIER_WOOD_POINTS,
    ),
)


@dataclass
class TerritoryState:
    """State card: mutable stats per territory, ready to scale to more regions."""

    name: str
    food: float
    wealth: float
    relation_to_neighbor: str = "neutral"
    relation_score: float = 0.0
    population: int = 500
    food_yield: float = 1.0
    wealth_yield: float = 1.0
    wood: float = 0.0
    wood_yield: float = 1.0
    iron: float = 0.0
    iron_yield: float = 0.0
    gold: float = 0.0
    gold_yield: float = 0.0
    infrastructure_level: int = 0
    effective_work_multiplier: float = 1.0
    unpaid_steps: int = 0
    on_strike: bool = False
    required_food: float = 0.0
    personality_vector: Dict[str, float] = field(default_factory=neutral_personality_vector)
    active_traits: List[str] = field(default_factory=list)
    trait_cooldown_steps: int = 0
    trait_history: List[Dict[str, Any]] = field(default_factory=list)
    trait_events: List[Dict[str, Any]] = field(default_factory=list)
    other_trait_notes: str = "No clear read on the neighbour yet."
    exploitation_streak: int = 0
    starvation_streak: int = 0
    failed_strategy_streak: int = 0
    previous_allocations: Dict[str, float] = field(default_factory=dict)
    last_applied_allocations: Dict[str, float] = field(default_factory=dict)
    last_population_after: Optional[float] = None
    last_wealth_after: Optional[float] = None
    adaptation_pressure_note: str = ""
    gift_streak_received: int = 0
    food_sent: float = 0.0
    food_received: float = 0.0
    wealth_sent: float = 0.0
    wealth_received: float = 0.0
    wood_sent: float = 0.0
    wood_received: float = 0.0
    long_term_notes: List[str] = field(default_factory=list)


class LeaderAgent(mesa.Agent):
    """Leader card: single political leader toggling between rules and LLM help."""

    def __init__(
        self,
        model: "WorldModel",
        territory: TerritoryState,
        neighbor: TerritoryState | None = None,
        llm_client: Optional[LLMDecisionClient] = None,
    ) -> None:
        """Init cue: take the model, territory, optional neighbour, and LLM helper."""
        super().__init__(model=model)
        # Territory handle: keep the mutable state handy.
        self.territory = territory
        # Neighbour handle: track the other territory for support/exploit moves.
        self.neighbor = neighbor
        # LLM hook: optional client to override rules when enabled.
        self.llm_client = llm_client
        # Decision memo: remember the previous action/reason.
        self.last_action: Optional[str] = None
        self.last_reason: Optional[str] = None
        # Memory ledger: bounded in-run history so the LLM can adapt mid-simulation.
        self.memory_events: List[Dict[str, Any]] = []
        self.max_memory_events = MAX_LEADER_MEMORY_EVENTS
        self.next_directive: str = "Stabilise food and explore opportunities."
        self.interaction_log: List[str] = []
        self.max_interactions: int = 8
        self.last_trait_adjustment_text: str = "no change"
        self.negotiation_intent: Dict[str, Any] = {"initiate": False}

    def _required_food(self) -> float:
        """Food math cue: compute the granular food requirement using shared config."""
        return max(0.0, (self.territory.population / 10.0) * FOOD_PER_10_POP)

    def _priority_hint(self) -> Dict[str, Any]:
        """Priority hint: soft suggestions on where to focus (survive vs. invest)."""
        required_per_step = self._required_food()
        horizon = FOOD_SAFETY_HORIZON_STEPS
        required_for_horizon = required_per_step * horizon
        current_food = self.territory.food
        food_safety_ratio = (
            current_food / required_for_horizon if required_for_horizon > 0 else float("inf")
        )
        priorities = {"survive": 1.0, "resilience": 0.0, "prosperity": 0.0}
        if food_safety_ratio >= 1.0:
            priorities["resilience"] = max(priorities["resilience"], NON_FOOD_MIN_FRACTION)
        if food_safety_ratio >= FOOD_SAFETY_GOOD_RATIO:
            priorities["prosperity"] = max(priorities["prosperity"], NON_FOOD_MIN_FRACTION)
        return {
            "food_safety_ratio": food_safety_ratio,
            "priorities": priorities,
            "required_for_horizon": required_for_horizon,
        }

    def _gift_balance_note(self) -> str:
        """Gift balance cue: surface whether gifts have been one-sided."""
        received = getattr(self.territory, "gift_streak_received", 0)
        neighbor_received = getattr(self.neighbor, "gift_streak_received", 0) if self.neighbor else 0
        if received > 0 and neighbor_received <= 0:
            plural = "s" if received > 1 else ""
            return (
                f"You have accepted {received} consecutive gift{plural} without reciprocating; "
                "expect the neighbour to ask for repayment soon."
            )
        if neighbor_received > 0 and received <= 0:
            plural = "s" if neighbor_received > 1 else ""
            return (
                f"The neighbour has accepted {neighbor_received} consecutive gift{plural} from you; "
                "feel free to insist on reciprocity."
            )
        if received > 0 and neighbor_received > 0:
            # Should not happen, but keep the note useful.
            return "Gift exchanges have been rapid; make sure they stay balanced."
        return "No outstanding gift imbalance; recent trades look even."

    def _trade_ledger_note(self) -> Dict[str, float]:
        """Ledger cue: summarise cumulative trade flows."""
        territory = self.territory
        food_sent = getattr(territory, "food_sent", 0.0)
        food_received = getattr(territory, "food_received", 0.0)
        wealth_sent = getattr(territory, "wealth_sent", 0.0)
        wealth_received = getattr(territory, "wealth_received", 0.0)
        wood_sent = getattr(territory, "wood_sent", 0.0)
        wood_received = getattr(territory, "wood_received", 0.0)
        return {
            "food_sent": food_sent,
            "food_received": food_received,
            "wealth_sent": wealth_sent,
            "wealth_received": wealth_received,
            "wood_sent": wood_sent,
            "wood_received": wood_received,
            "net_food": food_sent - food_received,
            "net_wealth": wealth_sent - wealth_received,
            "net_wood": wood_sent - wood_received,
        }

    def _long_term_summary_text(self) -> str:
        notes = getattr(self.territory, "long_term_notes", [])
        if not notes:
            return "No milestone notes recorded yet."
        return "; ".join(notes)

    def _resource_pressures(self) -> Dict[str, float]:
        """Pressure cue: quantify food safety and infra gaps for prompts."""
        required_now = self._required_food()
        horizon_need = required_now * FOOD_SAFETY_HORIZON_STEPS
        food_ratio = (self.territory.food / horizon_need) if horizon_need > 0 else float("inf")
        wood_gap = max(0.0, INFRA_TIER_WOOD_WOOD_COST - self.territory.wood)
        wealth_gap = max(0.0, INFRA_TIER_WOOD_WEALTH_COST - self.territory.wealth)
        infra_ready = wood_gap <= 0.01 and wealth_gap <= 0.01
        return {
            "food_ratio_horizon": food_ratio,
            "wood_gap": wood_gap,
            "wealth_gap": wealth_gap,
            "infra_ready": infra_ready,
        }

    def _compute_negotiation_intent(self) -> Dict[str, Any]:
        """Intent cue: decide whether to initiate negotiation and what to request."""
        intent: Dict[str, Any] = {"initiate": False}
        territory = self.territory
        priority = self._priority_hint()
        ratio = priority.get("food_safety_ratio", 0.0)
        required = self._required_food()
        horizon_need = required * FOOD_SAFETY_HORIZON_STEPS
        shortfall_horizon = max(0.0, horizon_need - territory.food)
        reason = ""
        request_resource = None
        request_amount = 0.0
        offer_resource = None
        offer_amount = 0.0
        urgency = "low"

        if ratio < 1.05:
            # Use horizon shortfall so shallow buffers still trigger a request.
            if shortfall_horizon > 0.2 * required:
                request_resource = "food"
                # Spread the horizon gap over the next few steps to avoid giant asks.
                per_step_gap = shortfall_horizon / FOOD_SAFETY_HORIZON_STEPS
                request_amount = round(min(per_step_gap + 0.2, max(required, per_step_gap * 2)), 2)
                offer_resource = "wealth" if territory.wealth > 0.3 else None
                if offer_resource:
                    offer_amount = round(min(territory.wealth * 0.4, max(0.1, request_amount * 0.6)), 2)
                reason = f"Food safety ratio {ratio:.2f} over {FOOD_SAFETY_HORIZON_STEPS} steps; requesting aid."
                urgency = "critical_food" if ratio < 0.9 else "low_food"

        wood_cost = INFRA_TIER_WOOD_WOOD_COST
        wealth_cost = INFRA_TIER_WOOD_WEALTH_COST
        wood_missing = max(0.0, wood_cost - territory.wood)
        wealth_missing = max(0.0, wealth_cost - territory.wealth)
        close_to_wood = territory.wood >= 0.7 * wood_cost
        close_to_wealth = territory.wealth >= 0.7 * wealth_cost

        if request_resource is None and ratio >= 1.05 and close_to_wood and close_to_wealth:
            if wood_missing > 0.01:
                request_resource = "wood"
                request_amount = round(min(wood_missing, 1.0), 2)
                offer_resource = "wealth"
                offer_amount = round(min(territory.wealth * 0.2, max(0.1, request_amount * 0.6)), 2)
                reason = "Ready to upgrade infrastructure; missing final wood."
                urgency = "infrastructure"
            elif wealth_missing > 0.01:
                request_resource = "wealth"
                request_amount = round(min(wealth_missing, 0.6), 2)
                offer_resource = "food"
                offer_amount = round(min(territory.food * 0.2, max(0.1, request_amount * 0.5)), 2)
                reason = "Ready to upgrade infrastructure; missing final wealth."
                urgency = "infrastructure"

        if request_resource:
            intent.update(
                {
                    "initiate": True,
                    "request_resource": request_resource,
                    "request_amount": request_amount,
                    "offer_resource": offer_resource,
                    "offer_amount": offer_amount,
                    "reason": reason or "Resource request",
                    "urgency": urgency,
                }
            )
        else:
            intent.update({"initiate": False, "reason": "No pressing trade needs."})
        return intent

    def _refresh_negotiation_intent(self) -> None:
        self.negotiation_intent = self._compute_negotiation_intent()

    def _snapshot(self) -> Dict[str, Any]:
        """Snapshot card: grab a simple before/after view for chronicle logging."""
        neighbor = self.neighbor
        required_food = self._required_food()
        self.territory.required_food = required_food
        return {
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "wood": self.territory.wood,
            "iron": self.territory.iron,
            "gold": self.territory.gold,
            "relation_to_neighbor": self.territory.relation_to_neighbor,
            "relation_score": self.territory.relation_score,
            "population": self.territory.population,
            "required_food": required_food,
            "work_points": compute_work_points(self.territory),
            "infrastructure_level": self.territory.infrastructure_level,
            "effective_work_multiplier": self.territory.effective_work_multiplier,
            "unpaid_steps": self.territory.unpaid_steps,
            "on_strike": self.territory.on_strike,
            "failed_strategy_streak": self.territory.failed_strategy_streak,
            "neighbor_food": neighbor.food if neighbor else None,
            "neighbor_wealth": neighbor.wealth if neighbor else None,
            "neighbor_population": neighbor.population if neighbor else None,
            "neighbor_wood": neighbor.wood if neighbor else None,
            "neighbor_iron": neighbor.iron if neighbor else None,
            "neighbor_gold": neighbor.gold if neighbor else None,
            "personality_vector": dict(self.territory.personality_vector),
            "active_traits": list(self.territory.active_traits),
            "trait_cooldown_steps": self.territory.trait_cooldown_steps,
            "exploitation_streak": self.territory.exploitation_streak,
            "starvation_streak": self.territory.starvation_streak,
            "failed_strategy_streak": self.territory.failed_strategy_streak,
            "other_trait_notes": self.territory.other_trait_notes,
        }

    def _state_dict(self) -> Dict[str, Any]:
        """State payload: convert territory + timestep into a JSON-friendly dict."""
        work_points = compute_work_points(self.territory)
        required_food = self._required_food()
        yields = compute_yields(self.territory)
        priority_hint = self._priority_hint()
        adaptation_text = adaptation_pressure_text(self.territory)
        self.territory.adaptation_pressure_note = adaptation_text
        state = {
            "territory": self.territory.name,
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "wood": self.territory.wood,
            "iron": self.territory.iron,
            "gold": self.territory.gold,
            "relation_to_neighbor": self.territory.relation_to_neighbor,
            "relation_score": self.territory.relation_score,
            "population": self.territory.population,
            "required_food": required_food,
            "work_points": work_points,
            "infra": self.territory.infrastructure_level,
            "effective_work_multiplier": self.territory.effective_work_multiplier,
            "yields": yields,
            "priority_hint": priority_hint,
            "current_season": self.model.current_season(),
            "next_season": self.model.next_season(),
            "step": self.model.steps,
            "self_directive": self.next_directive,
            "interaction_text": "\n".join(self.interaction_log[-5:]) if self.interaction_log else "No notable interactions recorded.",
            "personality_vector": dict(self.territory.personality_vector),
            "active_traits": list(self.territory.active_traits),
            "trait_cooldown_steps": self.territory.trait_cooldown_steps,
            "other_trait_notes": self.territory.other_trait_notes,
            "failed_strategy_streak": self.territory.failed_strategy_streak,
            "adaptation_pressure": self.territory.adaptation_pressure_note,
            "gift_balance_note": self._gift_balance_note(),
            "trade_ledger": self._trade_ledger_note(),
            "long_term_notes": list(getattr(self.territory, "long_term_notes", [])),
            "long_term_summary": self._long_term_summary_text(),
            "resource_pressures": self._resource_pressures(),
        }
        if self.neighbor is not None:
            state.update(
                {
                    "neighbor_name": self.neighbor.name,
                    "neighbor_food": self.neighbor.food,
                    "neighbor_wealth": self.neighbor.wealth,
                    "neighbor_population": self.neighbor.population,
                    "neighbor_wood": self.neighbor.wood,
                    "neighbor_iron": self.neighbor.iron,
                    "neighbor_gold": self.neighbor.gold,
                }
            )
        else:
            state.update(
                {
                    "neighbor_name": None,
                    "neighbor_food": None,
                    "neighbor_wealth": None,
                    "neighbor_population": None,
                    "neighbor_wood": None,
                    "neighbor_iron": None,
                    "neighbor_gold": None,
                }
            )
        state["history_text"] = summarise_memory_for_prompt(self.memory_events)
        return state

    def record_step_outcome(
        self,
        *,
        step: int,
        action_name: str,
        food_before: float,
        food_after: float,
        wealth_before: float,
        wealth_after: float,
        pop_before: float,
        pop_after: float,
        notes: str | None = None,
    ) -> None:
        """Memory card: capture the latest step outcome and keep only the freshest events."""

        def _safe_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _safe_int(value: Any) -> int | None:
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return None

        trimmed_notes = (notes or "").strip()
        event: Dict[str, Any] = {
            "step": step,
            "action": action_name or "unknown_action",
            "food_before": _safe_float(food_before),
            "food_after": _safe_float(food_after),
            "wealth_before": _safe_float(wealth_before),
            "wealth_after": _safe_float(wealth_after),
            "pop_before": _safe_int(pop_before),
            "pop_after": _safe_int(pop_after),
        }
        if trimmed_notes:
            event["notes"] = trimmed_notes
        self.memory_events.append(event)
        if len(self.memory_events) > self.max_memory_events:
            self.memory_events.pop(0)

        interval = getattr(config, "LONG_TERM_SUMMARY_INTERVAL", 50)
        max_notes = getattr(config, "MAX_LONG_TERM_NOTES", 6)
        if interval > 0 and step % interval == 0:
            summary = (
                f"Step {step}: pop {int(pop_after)}, food {food_after:.1f}, "
                f"wealth {wealth_after:.1f}, infra {self.territory.infrastructure_level}"
            )
            notes = getattr(self.territory, "long_term_notes", [])
            notes.append(summary)
            if len(notes) > max_notes:
                notes.pop(0)
            self.territory.long_term_notes = notes

    def record_interaction(self, summary: str) -> None:
        """Interaction log cue: keep a short recap of recent diplomatic exchanges."""
        summary = (summary or "").strip()
        if not summary:
            return
        self.interaction_log.append(summary)
        if len(self.interaction_log) > self.max_interactions:
            self.interaction_log.pop(0)

    def set_next_directive(self, directive: str | None) -> None:
        """Directive note: store the LLM's latest self-authored reminder for next step."""
        text = (directive or "").strip()
        if not text:
            return
        self.next_directive = text

    def _best_affordable_infra_tier(self) -> tuple[str, Dict[str, float], int] | None:
        """Infra picker: choose the highest tier affordable this step."""
        for tier_name, costs, points in INFRA_TIER_SEQUENCE:
            if all(getattr(self.territory, resource) >= amount for resource, amount in costs.items()):
                return tier_name, costs, points
        return None

    def _infra_shortfall_text(self) -> str:
        """Infra gap note: explain why even the basic wood tier is out of reach."""
        lowest_costs = INFRA_TIER_SEQUENCE[-1][1]
        missing: list[str] = []
        for resource, amount in lowest_costs.items():
            current = getattr(self.territory, resource)
            if current < amount:
                missing.append(f"{resource} {current:.2f}/{amount:.2f}")
        if not missing:
            missing.append("metals unavailable")
        return ", ".join(missing)

    def _trait_snapshot(self) -> Dict[str, Any]:
        """Trait snapshot: expose trait state for logging."""
        return {
            "active_traits": list(self.territory.active_traits),
            "personality_vector": dict(self.territory.personality_vector),
            "trait_cooldown_steps": self.territory.trait_cooldown_steps,
            "exploitation_streak": self.territory.exploitation_streak,
            "starvation_streak": self.territory.starvation_streak,
            "failed_strategy_streak": self.territory.failed_strategy_streak,
            "other_trait_notes": self.territory.other_trait_notes,
            "trait_events": list(self.territory.trait_events),
            "adaptation_pressure": self.territory.adaptation_pressure_note,
        }

    def _prepare_allocations(self, decision: Dict[str, Any]) -> tuple[Dict[str, float], str]:
        """Allocation prep: filter requested allocations and fall back to legacy action when needed."""
        allocations_raw = decision.get("allocations") or {}
        sanitized: Dict[str, float] = {}
        if isinstance(allocations_raw, dict):
            for key, value in allocations_raw.items():
                if key not in WORK_ACTIONS:
                    continue
                try:
                    share = float(value)
                except (TypeError, ValueError):
                    continue
                if share <= 0:
                    continue
                sanitized[key] = share

        total_share = sum(sanitized.values())
        if total_share > 1.0 and total_share > 0:
            sanitized = {k: v / total_share for k, v in sanitized.items()}
        legacy_action = (decision.get("legacy_action") or decision.get("action") or "").lower()
        if not sanitized and legacy_action in WORK_ACTIONS:
            sanitized = {legacy_action: 1.0}
        return sanitized, legacy_action

    def _fallback_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback card: choose a simple action when the LLM response is unusable."""
        hint = state.get("priority_hint", {})
        ratio = hint.get("food_safety_ratio", 0.0)
        infra_level = state.get("infra", 0)
        plan: Dict[str, float] = {}
        build_flag = False
        tier_info = self._best_affordable_infra_tier()
        territory = self.territory
        food_share = 0.6 if ratio < 1.0 else 0.4
        plan["focus_food"] = food_share
        remaining = max(0.0, 1.0 - food_share)
        wood_need = max(0.0, INFRA_TIER_WOOD_WOOD_COST - territory.wood)
        wealth_need = max(0.0, INFRA_TIER_WOOD_WEALTH_COST - territory.wealth)
        if remaining > 0 and wood_need > 0:
            wood_share = min(0.5, remaining * 0.7)
            if wood_share > 0:
                plan["focus_wood"] = wood_share
                remaining -= wood_share
        if remaining > 0:
            plan["focus_wealth"] = remaining
        if ratio >= FOOD_SAFETY_GOOD_RATIO and tier_info:
            build_flag = True
        return {
            "allocations": plan,
            "build_infrastructure": build_flag,
            "reason": "Fallback heuristic decision.",
            "next_prompt": "Stabilise essentials and revisit infrastructure readiness.",
            "trait_adjustment": "no change",
        }

    def _maybe_force_infrastructure(self, decision: Dict[str, Any], state: Dict[str, Any]) -> None:
        """Auto-build cue: flip build_infrastructure on when buffers look healthy but the LLM demurs."""
        if decision.get("build_infrastructure"):
            return
        hint = state.get("priority_hint") or {}
        ratio = float(hint.get("food_safety_ratio", 0.0) or 0.0)
        required = state.get("required_food", 0.0) or self._required_food()
        if ratio < 1.0 and self.territory.food < required * FOOD_SAFETY_GOOD_RATIO:
            return
        level = self.territory.infrastructure_level
        if level >= 5:
            return
        tier_info = self._best_affordable_infra_tier()
        if tier_info is None:
            return
        tier_name, _, _ = tier_info
        if tier_name == "wood" and self.territory.food < required:
            # Still below the immediate requirement; hold off.
            return
        decision["build_infrastructure"] = True
        decision.setdefault(
            "reason",
            "Auto-build trigger: buffers are healthy so I am investing in infrastructure now.",
        )

    def decide_rule_based(self) -> Dict[str, Any]:
        """Rule path: use the heuristic action when the LLM output is invalid."""
        return self._fallback_action(self._state_dict())

    def apply_action(self, decision: Dict[str, Any]) -> None:
        """Action executor: mutate the territory state according to the chosen plan."""
        sanitized, legacy_action = self._prepare_allocations(decision)

        season = self.model.current_season()
        produced = apply_allocations(self.territory, sanitized, season, self.model.season_multipliers)
        # Strategy tracker: stash the allocation mix to spot repeated plans later.
        self.territory.previous_allocations = dict(self.territory.last_applied_allocations)
        self.territory.last_applied_allocations = dict(sanitized)

        decision["applied_allocations"] = {k: v for k, v in sanitized.items() if v > 0}

        wants_build = bool(decision.get("build_infrastructure"))
        action_label = "mixed_allocation" if sanitized else (legacy_action or "wait")
        decision["infrastructure_tier"] = None
        decision["infra_points_gained"] = 0
        if wants_build or action_label == "build_infrastructure":
            tier_info = self._best_affordable_infra_tier()
            if tier_info is None:
                decision["infrastructure_built"] = False
                shortage_text = self._infra_shortfall_text()
                decision.setdefault(
                    "reason",
                    f"Attempted build_infrastructure but lacked {shortage_text}; idling.",
                )
            else:
                tier_name, costs, points = tier_info
                for resource, amount in costs.items():
                    current = getattr(self.territory, resource)
                    setattr(self.territory, resource, current - amount)
                self.territory.infrastructure_level += points
                decision["infrastructure_built"] = True
                decision["infrastructure_tier"] = tier_name
                decision["infra_points_gained"] = points
        else:
            decision["infrastructure_built"] = False

        if not sanitized and not wants_build and action_label == "wait":
            # Explicit idle note when nothing productive happened.
            decision.setdefault("reason", "No productive allocation this step.")

        decision["action"] = action_label
        self.last_action = action_label
        self.last_reason = decision.get("reason", "no reason provided")

    def step(self) -> None:
        """Turn loop: pick an action (LLM preferred), execute it, and log everything."""
        tick_trait_cooldown(self.territory)
        pressure_events = apply_pressure_adaptation(self.territory)
        self.territory.trait_events = []
        if pressure_events:
            # Pressure cue: fold automatic adaptations into the trait events for logging.
            self.territory.trait_events.extend(pressure_events)
            for ev in pressure_events:
                self.model.chronicle.append(
                    {"event_type": "trait_event", "territory": self.territory.name, "step": self.model.steps, "details": ev}
                )
        if self.territory.population <= 0:
            decision = {
                "action": "wait",
                "target": "None",
                "reason": "No population remaining; the territory has collapsed.",
                "trait_adjustment": "no change",
            }
            state_before = self._snapshot()
            state_after = dict(state_before)
            self.last_action = decision["action"]
            self.last_reason = decision["reason"]
            self.model.record_decision(
                territory_name=self.territory.name,
                before=state_before,
                decision=decision,
                after=state_after,
                used_llm=False,
            )
            return

        decision: Optional[Dict[str, Any]] = None
        llm_used = False
        state_before = self._snapshot()
        state_payload = self._state_dict()

        if self.llm_client is not None and self.llm_client.enabled:
            try:
                decision = self.llm_client.decide(state_payload)
            except Exception as e:  # pragma: no cover - defensive log path
                print(f"[WARN] LLM decision failed ({e}), falling back to heuristic policy.")

        invalid_llm = True
        if isinstance(decision, dict):
            allocations = decision.get("allocations")
            has_allocations = isinstance(allocations, dict) and any(
                isinstance(v, (int, float)) and v > 0 for v in allocations.values()
            )
            build_flag = bool(decision.get("build_infrastructure"))
            legacy_action = decision.get("legacy_action")
            legacy_valid = isinstance(legacy_action, str) and legacy_action.lower() in ALLOWED_ACTIONS
            action_value = decision.get("action")
            action_valid = isinstance(action_value, str) and action_value.lower() in ALLOWED_ACTIONS
            if has_allocations or build_flag or legacy_valid or action_valid:
                invalid_llm = False
                llm_used = True

        if invalid_llm:
            if decision is not None:
                print("[INFO] LLM produced an invalid action, so I am using the heuristic policy instead.")
            decision = self._fallback_action(state_payload)
            llm_used = False

        self._maybe_force_infrastructure(decision, state_payload)
        directive = decision.get("next_prompt")
        if isinstance(directive, str):
            self.set_next_directive(directive)

        trait_adjust_text = str(decision.get("trait_adjustment", "")).strip()
        self.last_trait_adjustment_text = trait_adjust_text or "no change"
        self._apply_trait_adjustment_text(trait_adjust_text)
        decision["trait_events"] = list(self.territory.trait_events)
        decision["trait_state"] = self._trait_snapshot()

        # Execute + log: apply the decision and push it into the run log.
        self.apply_action(decision)
        state_after = self._snapshot()
        self.model.record_decision(
            territory_name=self.territory.name,
            before=state_before,
            decision=decision,
            after=state_after,
            used_llm=llm_used,
        )
        self.model.log_agent_state(self, decision, llm_used)
        self._refresh_negotiation_intent()

    def _apply_trait_adjustment_text(self, text: str) -> None:
        """Trait adjuster: interpret and apply trait changes when cooldown allows."""
        actions = interpret_trait_adjustment(text)
        if not actions:
            return
        events = apply_trait_actions(
            self.territory,
            actions,
            step=self.model.steps,
            reason_prefix=f"LLM prompt: {text}" if text else "LLM prompt",
        )
        if events:
            self.last_trait_adjustment_text = text or "applied actions"
            for ev in events:
                self.model.chronicle.append(
                    {
                        "event_type": "trait_event",
                        "territory": self.territory.name,
                        "step": ev.get("step", self.model.steps),
                        "details": ev,
                    }
                )
