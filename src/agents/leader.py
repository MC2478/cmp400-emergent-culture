# src/agents/leader.py
from __future__ import annotations
from typing import Dict, Any, Optional

from dataclasses import dataclass
from typing import Dict, Any, Optional

import mesa

from src.model.llm_client import LLMDecisionClient


@dataclass
class TerritoryState:
    name: str
    food: int
    wealth: int
    relation_to_neighbor: str = "neutral"


class LeaderAgent(mesa.Agent):
    """Single territory leader; rule-based with optional LLM override."""

    def __init__(
        self,
        model: "WorldModel",
        territory: TerritoryState,
        llm_client: Optional[LLMDecisionClient] = None,
    ):
        super().__init__(model=model)
        self.territory = territory
        self.llm_client = llm_client
        self.last_action: Optional[str] = None
        self.last_reason: Optional[str] = None

    def _state_dict(self) -> Dict[str, Any]:
        return {
            "territory": self.territory.name,
            "food": self.territory.food,
            "wealth": self.territory.wealth,
            "relation_to_neighbor": self.territory.relation_to_neighbor,
            "step": self.model.steps,
        }

    def decide_rule_based(self) -> Dict[str, Any]:
        if self.territory.food < 5:
            action = "gather"
            reason = "food is low so I gather more"
        elif self.territory.food > 15:
            action = "consume"
            reason = "food is high so I consume some"
        else:
            action = "wait"
            reason = "resources look stable so I wait"

        return {"action": action, "target": "None", "reason": reason}

    def apply_action(self, decision: Dict[str, Any]) -> None:
        action = decision.get("action", "wait")

        if action == "gather":
            self.territory.food += 3
        elif action == "consume":
            self.territory.food = max(0, self.territory.food - 2)

        self.last_action = action
        self.last_reason = decision.get("reason")

    def step(self) -> None:
        decision: Dict[str, Any]

        if self.llm_client is not None and self.llm_client.enabled:
            try:
                # I ask the LLM to choose an action based on current state.
                decision = self.llm_client.decide(self._state_dict())
            except Exception as e:
                print(f"[WARN] LLM decision failed ({e}), so I fall back to rules.")
                decision = self.decide_rule_based()
        else:
            decision = self.decide_rule_based()

        self.apply_action(decision)
        self.model.log_step(self, decision)

