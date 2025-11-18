"""I define ``TerritoryState`` and ``LeaderAgent`` for the CMP400 feasibility demo where a single
leader alternates between deterministic rules and LLM-backed decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mesa

from src.model.llm_client import LLMDecisionClient

# I centralise the rule set here so both the LLM and rule-based flows share the same vocabulary.
ALLOWED_ACTIONS: set[str] = {"gather", "consume", "wait", "support_neighbor", "exploit_neighbor"}


@dataclass
class TerritoryState:
    """I keep the mutable stats for our lone territory so later I can swap in multiple regions."""

    name: str
    food: int
    wealth: int
    relation_to_neighbor: str = "neutral"


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

    def _snapshot(self) -> Dict[str, Any]:
        """I grab a simple before/after snapshot for chronicle logging."""
        neighbor = self.neighbor
        return {
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "relation": self.territory.relation_to_neighbor,
            "neighbor_food": neighbor.food if neighbor else None,
            "neighbor_wealth": neighbor.wealth if neighbor else None,
        }

    def _state_dict(self) -> Dict[str, Any]:
        """I convert the territory and timestep into a JSON-friendly dict."""
        # I include the current model step so the LLM knows where in the timeline we are.
        state = {
            "territory": self.territory.name,
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "relation_to_neighbor": self.territory.relation_to_neighbor,
            "step": self.model.steps,
        }
        if self.neighbor is not None:
            state.update(
                {
                    "neighbor_name": self.neighbor.name,
                    "neighbor_food": self.neighbor.food,
                    "neighbor_wealth": self.neighbor.wealth,
                }
            )
        else:
            state.update(
                {"neighbor_name": None, "neighbor_food": None, "neighbor_wealth": None}
            )
        # I want the LLM to see both my own resources and a tiny snapshot of the neighbour.
        return state

    def decide_rule_based(self) -> Dict[str, Any]:
        """I provide the deterministic fallback decision policy."""
        neighbor = self.neighbor
        # I gather resources whenever food is critically low.
        if self.territory.food < 5:
            action = "gather"
            reason = "food is low so I gather more"
        # I support the neighbor when I'm friendly, richer, and they are struggling.
        elif (
            neighbor
            and self.territory.relation_to_neighbor == "friendly"
            and neighbor.food < self.territory.food - 5
        ):
            action = "support_neighbor"
            reason = "I'm friendly and have surplus so I support the neighbor"
        # I exploit when relations are tense and they clearly have more food.
        elif (
            neighbor
            and self.territory.relation_to_neighbor != "friendly"
            and neighbor.food > self.territory.food + 5
        ):
            action = "exploit_neighbor"
            reason = "relations are tense and they are richer so I exploit them"
        # I consume food if stockpiles are abundant.
        elif self.territory.food > 15:
            action = "consume"
            reason = "food is high so I consume some"
        # Otherwise I hold steady to maintain balance.
        else:
            action = "wait"
            reason = "resources look stable so I wait"

        # I always return a dict compatible with both logging and the LLM interface.
        return {"action": action, "target": "None", "reason": reason}

    def apply_action(self, decision: Dict[str, Any]) -> None:
        """I mutate the territory state according to the chosen action."""
        action = (decision.get("action") or "wait").lower()
        if action not in ALLOWED_ACTIONS:
            # I sanitize unknown actions to "wait" so a rogue response cannot break the feasibility demo.
            action = "wait"

        neighbor = self.neighbor

        # I increase food supply when gathering.
        if action == "gather":
            self.territory.food += 3
        # I consume food carefully and never drop below zero.
        elif action == "consume":
            self.territory.food = max(0, self.territory.food - 2)
        elif action == "support_neighbor" and neighbor is not None:
            # I treat support as gifting one unit and nudging the relation friendlier.
            if self.territory.food > 0:
                self.territory.food -= 1
                neighbor.food += 1
            if self.territory.relation_to_neighbor == "hostile":
                self.territory.relation_to_neighbor = "neutral"
            elif self.territory.relation_to_neighbor == "neutral":
                self.territory.relation_to_neighbor = "friendly"
        elif action == "exploit_neighbor" and neighbor is not None:
            # I treat exploitation as siphoning one unit and worsening relations.
            if neighbor.food > 0:
                neighbor.food -= 1
                self.territory.food += 1
            if self.territory.relation_to_neighbor == "friendly":
                self.territory.relation_to_neighbor = "neutral"
            elif self.territory.relation_to_neighbor == "neutral":
                self.territory.relation_to_neighbor = "hostile"
        # I intentionally do nothing for "wait" so resources remain steady.
        # I record the action and reason for downstream inspection.
        self.last_action = action
        self.last_reason = decision.get("reason", "no reason provided")

    def step(self) -> None:
        """I choose an action (LLM preferred) and then execute and log it."""
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
        # I pass both states so the chronicle captures before/after for the feasibility artifact.
        self.model.log_step(self, decision, state_before, state_after, llm_used)
