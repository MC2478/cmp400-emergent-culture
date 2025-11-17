"""I define ``TerritoryState`` and ``LeaderAgent`` with rule-based and LLM decision paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import mesa

from src.model.llm_client import LLMDecisionClient


@dataclass
class TerritoryState:
    """I use this dataclass to hold a territory's name, food, wealth, and neighbor relation."""
    name: str
    food: int
    wealth: int
    relation_to_neighbor: str = "neutral"


class LeaderAgent(mesa.Agent):
    """I represent the political leader who can rely on rules or an ``LLMDecisionClient``."""

    def __init__(
        self,
        model: "WorldModel",
        territory: TerritoryState,
        llm_client: Optional[LLMDecisionClient] = None,
    ):
        """I receive the model, initial territory, and optional LLM helper."""
        super().__init__(model=model)
        # I keep a reference to the mutable territory state.
        self.territory = territory
        # I store an optional LLM client so I can override the rules when enabled.
        self.llm_client = llm_client
        # I remember the previous decision for logging and inspection.
        self.last_action: Optional[str] = None
        self.last_reason: Optional[str] = None

    def _state_dict(self) -> Dict[str, Any]:
        """I convert the territory and timestep into a JSON-friendly dict."""
        # I include the current model step so the LLM knows where in the timeline we are.
        return {
            "territory": self.territory.name,
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "relation_to_neighbor": self.territory.relation_to_neighbor,
            "step": self.model.steps,
        }

    def decide_rule_based(self) -> Dict[str, Any]:
        """I provide the deterministic fallback decision policy."""
        # I gather resources whenever food is critically low.
        if self.territory.food < 5:
            action = "gather"
            reason = "food is low so I gather more"
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
        action = decision.get("action", "wait")

        # I increase food supply when gathering.
        if action == "gather":
            self.territory.food += 3
        # I consume food carefully and never drop below zero.
        elif action == "consume":
            self.territory.food = max(0, self.territory.food - 2)
        # I intentionally do nothing for "wait" so resources remain steady.
        # I record the action and reason for downstream inspection.
        self.last_action = action
        self.last_reason = decision.get("reason", "no reason provided")

    def step(self) -> None:
        """I choose an action (LLM preferred) and then execute and log it."""
        if self.llm_client is not None and self.llm_client.enabled:
            try:
                # I attempt to get the richer LLM decision first.
                decision = self.llm_client.decide(self._state_dict())
            except Exception as e:  # pragma: no cover - logging guard
                # I warn when the LLM path fails so I know to investigate.
                print(f"[WARN] LLM decision failed ({e}), falling back to rules.")
                decision = self.decide_rule_based()
        else:
            # I default to the deterministic policy when LLM support is absent.
            decision = self.decide_rule_based()

        # I enact the decision and ask the model to log the outcome.
        self.apply_action(decision)
        self.model.log_step(self, decision)
