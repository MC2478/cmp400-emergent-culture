"""I define ``TerritoryState`` and ``LeaderAgent`` for the CMP400 feasibility demo where a single
leader alternates between deterministic rules and LLM-backed decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mesa

from config import (
    PEOPLE_PER_WORK_POINT,
    FOOD_PER_10_POP,
    FOOD_PER_WORK_BASE,
    WOOD_PER_WORK_BASE,
    WEALTH_PER_WORK_BASE,
    INFRA_FOOD_YIELD_MULT_PER_LEVEL,
    INFRA_WOOD_YIELD_MULT_PER_LEVEL,
    INFRA_WEALTH_YIELD_MULT_PER_LEVEL,
    INFRA_COST_WOOD,
    INFRA_COST_WEALTH,
    FOOD_SAFETY_HORIZON_STEPS,
    FOOD_SAFETY_GOOD_RATIO,
    NON_FOOD_MIN_FRACTION,
)
from src.model.llm_client import LLMDecisionClient

# I centralise the rule set here so both the LLM and rule-based flows share the same vocabulary.
ALLOWED_ACTIONS: set[str] = {
    "focus_food",
    "focus_wood",
    "focus_wealth",
    "build_infrastructure",
    "wait",
}


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

    def _work_points(self) -> int:
        """I convert population into coarse work points (100 people = 1 work point)."""
        base = max(0, self.territory.population // PEOPLE_PER_WORK_POINT)
        return int(base * max(0.0, self.territory.effective_work_multiplier))

    def _required_food(self) -> float:
        """I compute the granular food requirement using the shared config."""
        return max(0.0, (self.territory.population / 10.0) * FOOD_PER_10_POP)

    def _effective_yields(self) -> Dict[str, float]:
        """I report yields per work point after infrastructure bonuses."""
        level = self.territory.infrastructure_level
        food_mult = 1.0 + level * INFRA_FOOD_YIELD_MULT_PER_LEVEL
        wood_mult = 1.0 + level * INFRA_WOOD_YIELD_MULT_PER_LEVEL
        wealth_mult = 1.0 + level * INFRA_WEALTH_YIELD_MULT_PER_LEVEL
        return {
            "food_per_work": self.territory.food_yield * FOOD_PER_WORK_BASE * food_mult,
            "wood_per_work": self.territory.wood_yield * WOOD_PER_WORK_BASE * wood_mult,
            "wealth_per_work": self.territory.wealth_yield * WEALTH_PER_WORK_BASE * wealth_mult,
        }

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
            "work_points": self._work_points(),
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
        work_points = self._work_points()
        required_food = self._required_food()
        yields = self._effective_yields()
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
        return state

    def _fallback_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """I choose a simple action when the LLM response is unusable."""
        hint = state.get("priority_hint", {})
        ratio = hint.get("food_safety_ratio", 0.0)
        infra_level = state.get("infra", 0)
        if ratio < 1.0:
            action = "focus_food"
        elif infra_level < 3 and self.territory.wood >= INFRA_COST_WOOD and self.territory.wealth >= INFRA_COST_WEALTH:
            action = "build_infrastructure"
        elif ratio < FOOD_SAFETY_GOOD_RATIO:
            action = "focus_food"
        else:
            action = "focus_wealth"
        return {"action": action, "target": "None", "reason": "Fallback heuristic decision."}

    def decide_rule_based(self) -> Dict[str, Any]:
        """I fall back to the heuristic action when the LLM output is invalid."""
        return self._fallback_action(self._state_dict())

    def apply_action(self, decision: Dict[str, Any]) -> None:
        """I mutate the territory state according to the chosen action."""
        action = (decision.get("action") or "wait").lower()
        if action not in ALLOWED_ACTIONS:
            decision = self._fallback_action(self._state_dict())
            action = decision["action"]

        work_points = self._work_points()
        yields = self._effective_yields()
        season = self.model.current_season()
        season_mult = self.model.season_multipliers.get(season, 1.0)
        food_yield = yields["food_per_work"] * season_mult
        wood_yield = yields["wood_per_work"] * season_mult
        wealth_yield = yields["wealth_per_work"]

        if action == "build_infrastructure":
            can_build = (
                self.territory.wood >= INFRA_COST_WOOD
                and self.territory.wealth >= INFRA_COST_WEALTH
            )
            if can_build:
                self.territory.wood -= INFRA_COST_WOOD
                self.territory.wealth -= INFRA_COST_WEALTH
                self.territory.infrastructure_level += 1
            else:
                action = "wait"
                decision["reason"] = "Could not afford infrastructure; idling."
        elif action == "focus_food":
            self.territory.food += work_points * food_yield
        elif action == "focus_wood":
            self.territory.wood += work_points * wood_yield
        elif action == "focus_wealth":
            self.territory.wealth += work_points * wealth_yield
        elif action == "wait":
            pass

        self.last_action = action
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
            action_value = decision.get("action")
            if isinstance(action_value, str):
                normalized = action_value.lower()
                if normalized in ALLOWED_ACTIONS:
                    decision["action"] = normalized
                    invalid_llm = False
                    llm_used = True

        if invalid_llm:
            if decision is not None:
                print("[INFO] LLM produced an invalid action, so I am using the heuristic policy instead.")
            decision = self._fallback_action(state_payload)
            llm_used = False

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
