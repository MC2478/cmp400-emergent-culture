# src/model/world_model.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import json

import mesa

from src.agents.leader import LeaderAgent, TerritoryState
from src.model.llm_client import LLMDecisionClient


class WorldModel(mesa.Model):
    """Single-territory world for the feasibility demo."""

    def __init__(
        self,
        random_seed: int | None = None,
        initial_food: int = 8,
        use_llm: bool = False,
    ):
        super().__init__(seed=random_seed)

        self.chronicle: List[Dict[str, Any]] = []

        territory = TerritoryState(name="East", food=initial_food, wealth=5)

        llm_client = None
        if use_llm:
            # I only try to construct the LLM client if explicitly requested.
            llm_client = LLMDecisionClient(enabled=True)

        self.leader = LeaderAgent(model=self, territory=territory, llm_client=llm_client)

    def log_step(self, agent: LeaderAgent, decision: Dict[str, Any]) -> None:
        entry = {
            "step": self.steps,
            "territory": agent.territory.name,
            "action": decision.get("action"),
            "target": decision.get("target"),
            "reason": decision.get("reason"),
            "food": agent.territory.food,
            "wealth": agent.territory.wealth,
        }
        self.chronicle.append(entry)
        print(
            f"Step {entry['step']}: {entry['territory']} -> "
            f"{entry['action']} (food={entry['food']}, reason={entry['reason']})"
        )

    def step(self) -> None:
        # Mesa 3.x: super().step() increments self.steps
        super().step()
        self.agents.shuffle_do("step")

    def save_chronicle(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.chronicle, f, indent=2)
