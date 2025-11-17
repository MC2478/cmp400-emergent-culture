"""I define the Mesa ``WorldModel`` with one territory, one leader, and a running chronicle."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import json

import mesa

from src.agents.leader import LeaderAgent, TerritoryState
from src.model.llm_client import LLMDecisionClient, LLMConfig


class WorldModel(mesa.Model):
    """I subclass ``mesa.Model`` to hold the global state, chronicle, and optional LLM client."""

    def __init__(self, random_seed: int | None = None, initial_food: int = 8, use_llm: bool = False):
        """I accept ``random_seed``, ``initial_food``, and ``use_llm`` to configure the single-territory world."""
        # I let Mesa set up its internal agent containers while honoring the random seed.
        super().__init__(seed=random_seed)
        self.chronicle: List[Dict[str, Any]] = []

        # I start with a single East territory to keep the world small and controlled.
        territory = TerritoryState(name="East", food=initial_food, wealth=5)

        # I enable the HTTP LLM client only when requested so rules remain the default path.
        llm_client = None
        if use_llm:
            llm_client = LLMDecisionClient(config=LLMConfig(), enabled=True)

        # I register the sole leader agent with the Mesa scheduler
        self.leader = LeaderAgent(model=self, territory=territory, llm_client=llm_client)
        self.agents.add(self.leader)

    def log_step(self, agent: LeaderAgent, decision: Dict[str, Any]) -> None:
        """I capture each decision in a serialisable dict for later inspection."""
        # I build a concise record from the agent's territory state and the chosen decision
        entry = {
            "step": self.steps,
            "territory": agent.territory.name,
            "action": decision.get("action"),
            "target": decision.get("target"),
            "reason": decision.get("reason"),
            "food": agent.territory.food,
            "wealth": agent.territory.wealth,
        }
        # I append the record to the running chronicle.
        self.chronicle.append(entry)
        # I echo the decision for quick debugging while the sim runs.
        print(
            f"Step {entry['step']}: {entry['territory']} -> "
            f"{entry['action']} (food={entry['food']}, reason={entry['reason']})"
        )

    def step(self) -> None:
        """I advance the Mesa scheduler one tick so the leader agent can act."""
        # Mesa 3.x: super().step() increments self.steps
        super().step()
        # I still ask Mesa to shuffle agents even though there is currently only one.
        self.agents.shuffle_do("step")

    def save_chronicle(self, path: Path) -> None:
        """I persist the chronicle as JSON to ``path`` so I can analyze runs later."""
        # I ensure the log directory exists before writing.
        path.parent.mkdir(parents=True, exist_ok=True)
        # I dump the structured chronicle to disk in a readable format.
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.chronicle, f, indent=2)
