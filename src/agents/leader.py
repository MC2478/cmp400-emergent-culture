"""I define ``TerritoryState`` and ``LeaderAgent`` for the CMP400 feasibility demo where a single
leader alternates between deterministic rules and LLM-backed decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mesa

from config import (
    FOOD_PER_10_POP,
    INFRA_COST_WOOD,
    INFRA_COST_WEALTH,
    FOOD_SAFETY_HORIZON_STEPS,
    FOOD_SAFETY_GOOD_RATIO,
    NON_FOOD_MIN_FRACTION,
    MAX_LEADER_MEMORY_EVENTS,
)
from src.agents.production import apply_allocations, effective_yields as compute_yields, work_points as compute_work_points
from src.model.llm_client import LLMDecisionClient, summarise_memory_for_prompt

# I centralise the rule set here so both the LLM and rule-based flows share the same vocabulary.
WORK_ACTIONS: set[str] = {"focus_food", "focus_wood", "focus_wealth"}
ALLOWED_ACTIONS: set[str] = set(WORK_ACTIONS) | {"build_infrastructure", "wait"}


@dataclass
class TerritoryState:
    """I keep the mutable stats for each territory so later I can scale to multiple regions."""

    name: str
    food: float
    wealth: float
    relation_to_neighbor: str = "neutral"
    relation_score: int = 0
    population: int = 500
    food_yield: float = 1.0
    wealth_yield: float = 1.0
    wood: float = 0.0
    wood_yield: float = 1.0
    infrastructure_level: int = 0
    effective_work_multiplier: float = 1.0
    unpaid_steps: int = 0
    on_strike: bool = False
    required_food: float = 0.0


class LeaderAgent(mesa.Agent):
    """I represent the single political leader in the prototype and switch between rule-based and
    LLM-assisted decision making."""

    def __init__(
        self,
        model: "WorldModel",
        territory: TerritoryState,
        neighbor: TerritoryState | None = None,
        llm_client: Optional[LLMDecisionClient] = None,
    ) -> None:
        """I receive the model, initial territory, and optional LLM helper."""
        super().__init__(model=model)
        # I keep a reference to the mutable territory state.
        self.territory = territory
        # I also track the neighbouring territory so I can model support/exploit moves.
        self.neighbor = neighbor
        # I store an optional LLM client so I can override the rules when enabled.
        self.llm_client = llm_client
        # I remember the previous decision for logging and inspection.
        self.last_action: Optional[str] = None
        self.last_reason: Optional[str] = None
        # I keep a bounded in-run memory of outcomes so the LLM can adapt mid-simulation.
        self.memory_events: List[Dict[str, Any]] = []
        self.max_memory_events = MAX_LEADER_MEMORY_EVENTS
        self.next_directive: str = "Stabilise food and explore opportunities."
        self.interaction_log: List[str] = []
        self.max_interactions: int = 8

    def _required_food(self) -> float:
        """I compute the granular food requirement using the shared config."""
        return max(0.0, (self.territory.population / 10.0) * FOOD_PER_10_POP)

    def _priority_hint(self) -> Dict[str, Any]:
        """I provide a soft suggestion about where to focus."""
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

    def _snapshot(self) -> Dict[str, Any]:
        """I grab a simple before/after snapshot for chronicle logging."""
        neighbor = self.neighbor
        required_food = self._required_food()
        self.territory.required_food = required_food
        return {
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "wood": self.territory.wood,
            "relation_to_neighbor": self.territory.relation_to_neighbor,
            "relation_score": self.territory.relation_score,
            "population": self.territory.population,
            "required_food": required_food,
            "work_points": compute_work_points(self.territory),
            "infrastructure_level": self.territory.infrastructure_level,
            "effective_work_multiplier": self.territory.effective_work_multiplier,
            "unpaid_steps": self.territory.unpaid_steps,
            "on_strike": self.territory.on_strike,
            "neighbor_food": neighbor.food if neighbor else None,
            "neighbor_wealth": neighbor.wealth if neighbor else None,
            "neighbor_population": neighbor.population if neighbor else None,
            "neighbor_wood": neighbor.wood if neighbor else None,
        }

    def _state_dict(self) -> Dict[str, Any]:
        """I convert the territory and timestep into a JSON-friendly dict."""
        work_points = compute_work_points(self.territory)
        required_food = self._required_food()
        yields = compute_yields(self.territory)
        priority_hint = self._priority_hint()
        state = {
            "territory": self.territory.name,
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "wood": self.territory.wood,
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
        }
        if self.neighbor is not None:
            state.update(
                {
                    "neighbor_name": self.neighbor.name,
                    "neighbor_food": self.neighbor.food,
                    "neighbor_wealth": self.neighbor.wealth,
                    "neighbor_population": self.neighbor.population,
                    "neighbor_wood": self.neighbor.wood,
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
                }
            )
        state["history_text"] = summarise_memory_for_prompt(self.memory_events)
        return state

    def record_step_outcome(
        self,
        *,
        step: int,
        action: str,
        food_before: float,
        food_after: float,
        wealth_before: float,
        wealth_after: float,
        pop_before: float,
        pop_after: float,
        starving: bool,
        strike: bool,
        note: str | None = None,
    ) -> None:
        """I store a structured in-run memory event for later prompt summaries."""
        event: Dict[str, Any] = {
            "step": step,
            "action": action,
            "food_before": food_before,
            "food_after": food_after,
            "wealth_before": wealth_before,
            "wealth_after": wealth_after,
            "pop_before": pop_before,
            "pop_after": pop_after,
            "starving": starving,
            "strike": strike,
        }
        if note:
            event["note"] = note

        self.memory_events.append(event)
        if len(self.memory_events) > self.max_memory_events:
            self.memory_events.pop(0)

    def record_interaction(self, summary: str) -> None:
        """I keep a short log of notable diplomatic interactions."""
        summary = (summary or "").strip()
        if not summary:
            return
        self.interaction_log.append(summary)
        if len(self.interaction_log) > self.max_interactions:
            self.interaction_log.pop(0)

    def _fallback_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """I choose a simple action when the LLM response is unusable."""
        hint = state.get("priority_hint", {})
        ratio = hint.get("food_safety_ratio", 0.0)
        infra_level = state.get("infra", 0)
        plan: Dict[str, float] = {}
        build_flag = False
        if ratio < 1.0:
            plan["focus_food"] = 1.0
        elif infra_level < 3 and self.territory.wood >= INFRA_COST_WOOD and self.territory.wealth >= INFRA_COST_WEALTH:
            build_flag = True
        elif ratio < FOOD_SAFETY_GOOD_RATIO:
            plan["focus_food"] = 1.0
        else:
            plan["focus_wealth"] = 1.0
        return {
            "allocations": plan,
            "build_infrastructure": build_flag,
            "reason": "Fallback heuristic decision.",
            "next_prompt": "Stabilise essentials and revisit infrastructure readiness.",
        }

    def decide_rule_based(self) -> Dict[str, Any]:
        """I fall back to the heuristic action when the LLM output is invalid."""
        return self._fallback_action(self._state_dict())

    def apply_action(self, decision: Dict[str, Any]) -> None:
        """I mutate the territory state according to the chosen action."""
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

        season = self.model.current_season()
        produced = apply_allocations(self.territory, sanitized, season, self.model.season_multipliers)

        decision["applied_allocations"] = {k: v for k, v in sanitized.items() if v > 0}

        wants_build = bool(decision.get("build_infrastructure"))
        action_label = "mixed_allocation" if sanitized else (legacy_action or "wait")
        if wants_build or action_label == "build_infrastructure":
            wood_available = self.territory.wood
            wealth_available = self.territory.wealth
            can_build = (
                wood_available >= INFRA_COST_WOOD
                and wealth_available >= INFRA_COST_WEALTH
            )
            if can_build:
                self.territory.wood -= INFRA_COST_WOOD
                self.territory.wealth -= INFRA_COST_WEALTH
                self.territory.infrastructure_level += 1
                decision["infrastructure_built"] = True
            else:
                shortfalls: list[str] = []
                if wood_available < INFRA_COST_WOOD:
                    shortfalls.append(f"wood {wood_available:.2f}/{INFRA_COST_WOOD:.2f}")
                if wealth_available < INFRA_COST_WEALTH:
                    shortfalls.append(f"wealth {wealth_available:.2f}/{INFRA_COST_WEALTH:.2f}")
                shortage_text = ", ".join(shortfalls) if shortfalls else "insufficient resources"
                decision["reason"] = (
                    f"Attempted build_infrastructure but lacked {shortage_text}; idling."
                )
                decision["infrastructure_built"] = False
        else:
            decision["infrastructure_built"] = False

        if not sanitized and not wants_build and action_label == "wait":
            # No production occurred; ensure we log explicitly.
            decision.setdefault("reason", "No productive allocation this step.")

        decision["action"] = action_label
        self.last_action = action_label
        self.last_reason = decision.get("reason", "no reason provided")

    def step(self) -> None:
        """I choose an action (LLM preferred) and then execute and log it."""
        if self.territory.population <= 0:
            decision = {
                "action": "wait",
                "target": "None",
                "reason": "No population remaining; the territory has collapsed.",
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
            except Exception as e:  # pragma: no cover - logging guard
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

        directive = decision.get("next_prompt")
        if isinstance(directive, str) and directive.strip():
            self.next_directive = directive.strip()

        # I enact the decision and ask the model to log the outcome.
        self.apply_action(decision)
        state_after = self._snapshot()
        self.model.record_decision(
            territory_name=self.territory.name,
            before=state_before,
            decision=decision,
            after=state_after,
            used_llm=llm_used,
        )
