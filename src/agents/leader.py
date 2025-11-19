"""I define ``TerritoryState`` and ``LeaderAgent`` for the CMP400 feasibility demo where a single
leader alternates between deterministic rules and LLM-backed decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

import mesa

from src.model.llm_client import LLMDecisionClient

# I centralise the rule set here so both the LLM and rule-based flows share the same vocabulary.
# I recently expanded the vocabulary with infrastructure building so the LLM can invest.
ALLOWED_ACTIONS: set[str] = {"focus_food", "focus_wealth", "balanced", "build_infrastructure", "wait"}


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
        base = max(0, self.territory.population // 100)
        return int(base * max(0.0, self.territory.effective_work_multiplier))

    def _snapshot(self) -> Dict[str, Any]:
        """I grab a simple before/after snapshot for chronicle logging."""
        neighbor = self.neighbor
        return {
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "wood": self.territory.wood,
            "relation_to_neighbor": self.territory.relation_to_neighbor,
            "relation_score": self.territory.relation_score,
            "population": self.territory.population,
            "required_food": max(1, self.territory.population // 100),
            "work_points": self._work_points(),
            "infrastructure_level": self.territory.infrastructure_level,
            "effective_work_multiplier": self.territory.effective_work_multiplier,
            "unpaid_steps": self.territory.unpaid_steps,
            "on_strike": self.territory.on_strike,
            "food_yield": self.territory.food_yield,
            "wealth_yield": self.territory.wealth_yield,
            "wood_yield": self.territory.wood_yield,
            "neighbor_food": neighbor.food if neighbor else None,
            "neighbor_wealth": neighbor.wealth if neighbor else None,
            "neighbor_population": neighbor.population if neighbor else None,
            "neighbor_wood": neighbor.wood if neighbor else None,
        }

    def _state_dict(self) -> Dict[str, Any]:
        """I convert the territory and timestep into a JSON-friendly dict."""
        # I include the current model step so the LLM knows where in the timeline we are.
        work_points = self._work_points()
        required_food = max(1, self.territory.population // 100)
        food_yield = max(0.0, self.territory.food_yield)
        wealth_yield = max(0.0, self.territory.wealth_yield)
        food_points_ff = math.ceil(0.75 * work_points)
        wealth_points_ff = work_points - food_points_ff
        max_food_ff = self.territory.food + food_points_ff * food_yield
        max_wealth_ff = self.territory.wealth + wealth_points_ff * wealth_yield
        wealth_points_fw = math.ceil(0.75 * work_points)
        food_points_fw = work_points - wealth_points_fw
        max_food_fw = self.territory.food + food_points_fw * food_yield
        max_wealth_fw = self.territory.wealth + wealth_points_fw * wealth_yield
        can_hit_ff = max_food_ff >= required_food
        can_hit_fw = max_food_fw >= required_food
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
            "effective_work_multiplier": self.territory.effective_work_multiplier,
            "max_food_if_focus_food": max_food_ff,
            "max_food_if_focus_wealth": max_food_fw,
            "max_wealth_if_focus_food": max_wealth_ff,
            "max_wealth_if_focus_wealth": max_wealth_fw,
            "can_meet_quota_if_focus_food": can_hit_ff,
            "can_meet_quota_if_focus_wealth": can_hit_fw,
            "food_yield": self.territory.food_yield,
            "wealth_yield": self.territory.wealth_yield,
            "wood_yield": self.territory.wood_yield,
            "infrastructure_level": self.territory.infrastructure_level,
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
        # I want the LLM to see both my own resources and a tiny snapshot of the neighbour.
        return state

    def decide_rule_based(self) -> Dict[str, Any]:
        """I provide the deterministic fallback decision policy."""
        required_food = max(1, self.territory.population // 100)
        work_points = self._work_points()
        food_yield = max(0.0, self.territory.food_yield)
        food_points_ff = math.ceil(0.75 * work_points)
        max_food_ff = self.territory.food + food_points_ff * food_yield
        can_hit_ff = max_food_ff >= required_food

        wood_ok = self.territory.wood >= 5.0
        wealth_ok = self.territory.wealth >= 3.0
        infra_room = self.territory.infrastructure_level < 5
        has_buffer = self.territory.food >= required_food * 1.5

        if wood_ok and wealth_ok and infra_room and has_buffer:
            action = "build_infrastructure"
            reason = "I can afford another infrastructure project to boost yields."
        elif not can_hit_ff:
            action = "focus_wealth"
            reason = "Even focusing on food I cannot reach my food requirement, so I focus on wealth to trade."
        elif self.territory.food < required_food:
            action = "focus_food"
            reason = "I can reach my food requirement by focusing on food, so I prioritise food."
        elif self.territory.wealth < 10.0:
            action = "focus_wealth"
            reason = "Food is safe and wealth is low, so I focus on wealth."
        else:
            action = "balanced"
            reason = "Food and wealth are both acceptable, so I balance them."

        # I always return a dict compatible with both logging and the LLM interface.
        return {"action": action, "target": "None", "reason": reason}

    def apply_action(self, decision: Dict[str, Any]) -> None:
        """I mutate the territory state according to the chosen action."""
        action = (decision.get("action") or "wait").lower()
        if action not in ALLOWED_ACTIONS:
            # I sanitize unknown actions to "wait" so a rogue response cannot break the feasibility demo.
            action = "wait"

        season = self.model.current_season()
        mult = self.model.season_multipliers.get(season, 1.0)
        infra_bonus = 1.0 + 0.1 * self.territory.infrastructure_level
        eff_food_yield = self.territory.food_yield * infra_bonus * mult
        eff_wealth_yield = self.territory.wealth_yield * infra_bonus
        eff_wood_yield = self.territory.wood_yield * infra_bonus * mult

        if action == "build_infrastructure":
            wood_cost = 5.0
            wealth_cost = 3.0
            can_build = self.territory.wood >= wood_cost and self.territory.wealth >= wealth_cost
            if can_build:
                self.territory.wood -= wood_cost
                self.territory.wealth -= wealth_cost
                self.territory.infrastructure_level += 1
            reason = decision.get("reason")
            if not reason:
                reason = (
                    "I invest wood and wealth to raise infrastructure yields."
                    if can_build
                    else "I attempted to build infrastructure but lacked resources."
                )
            decision["reason"] = reason
            self.last_action = action
            self.last_reason = reason
            return

        work_points = self._work_points()
        food_points = 0
        wealth_points = 0
        wood_points = 0

        if action == "focus_food":
            if work_points == 1:
                food_points = 1
            elif work_points >= 2:
                food_points = max(1, int(work_points * 0.7))
                wealth_points = int(work_points * 0.2)
                wood_points = work_points - food_points - wealth_points
                if wood_points < 0:
                    wealth_points = max(0, wealth_points + wood_points)
                    wood_points = 0
            # work_points == 0 leaves everything at zero
        elif action == "focus_wealth":
            if work_points == 1:
                wealth_points = 1
            elif work_points >= 2:
                wealth_points = max(1, int(work_points * 0.7))
                food_points = int(work_points * 0.2)
                wood_points = work_points - food_points - wealth_points
                if wood_points < 0:
                    food_points = max(0, food_points + wood_points)
                    wood_points = 0
        elif action == "balanced":
            if work_points > 0:
                base = work_points // 3
                remainder = work_points - base * 3
                food_points = base
                wealth_points = base
                wood_points = base
                if remainder > 0:
                    food_points += 1
                    remainder -= 1
                if remainder > 0:
                    wealth_points += 1
                    remainder -= 1
                if remainder > 0:
                    wood_points += 1
        elif action == "wait":
            food_points = 0
            wealth_points = 0
            wood_points = 0

        used_points = food_points + wealth_points + wood_points
        if used_points > work_points:
            wood_points = max(0, wood_points - (used_points - work_points))
        wood_points = max(0, min(wood_points, work_points))

        self.territory.food += food_points * eff_food_yield
        self.territory.wealth += wealth_points * eff_wealth_yield
        self.territory.wood += wood_points * eff_wood_yield

        # I record the action and reason for downstream inspection.
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

        if self.llm_client is not None and self.llm_client.enabled:
            try:
                # I attempt to get the richer LLM decision first.
                decision = self.llm_client.decide(self._state_dict())
            except Exception as e:  # pragma: no cover - logging guard
                # I want the Week 11 feasibility demo to keep running even if the LLM endpoint wobbles.
                print(f"[WARN] LLM decision failed ({e}), falling back to rule-based policy.")

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
            # I log this so I remember the sim intentionally dropped back to deterministic rules for robustness.
            if decision is not None:
                print("[INFO] LLM produced an invalid action, so I am using the rule-based policy instead.")
            decision = self.decide_rule_based()
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
