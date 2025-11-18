"""I hold the single-territory Mesa ``WorldModel`` used in the CMP400 feasibility demo where
one ``LeaderAgent`` (optionally LLM-driven) makes decisions that I log to a chronicle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import mesa

from src.agents.leader import LeaderAgent, TerritoryState
from src.model.llm_client import LLMDecisionClient, LLMConfig


class WorldModel(mesa.Model):
    """I track the single territory, plug in the leader agent, and collect every decision for
    the feasibility write-up."""

    def __init__(self, random_seed: int | None = None, initial_food: int = 8, use_llm: bool = False) -> None:
        """I accept ``random_seed``, ``initial_food``, and ``use_llm`` so I can reproduce runs and
        decide whether the leader should query the local LLM."""
        # I let Mesa set up its internal agent containers while honoring the random seed.
        super().__init__(seed=random_seed)
        # I track a stable chronicle schema here so I can reuse it as an artifact in the feasibility report.
        self.chronicle: List[Dict[str, Any]] = []

        # I document the initial territory stats so the feasibility demo stays simple to explain.
        # I start with a single East territory to keep the world small and controlled.
        territory = TerritoryState(name="East", food=initial_food, wealth=5)

        # I enable the HTTP LLM client only when requested so rules remain the default path.
        llm_client: LLMDecisionClient | None = None
        if use_llm:
            llm_client = LLMDecisionClient(config=LLMConfig(), enabled=True)

        # I register the sole leader agent with Mesa's ``AgentSet`` so upgrading to multi-agent later is easy.
        self.leader = LeaderAgent(model=self, territory=territory, llm_client=llm_client)
        self.agents.add(self.leader)

    def log_step(
        self,
        agent: LeaderAgent,
        decision: Dict[str, Any],
        state_before: Dict[str, int],
        state_after: Dict[str, int],
        llm_used: bool,
    ) -> None:
        """I capture each decision in a serialisable dict for later inspection."""
        # I keep a simple chronicle list here so I can export a readable history for the feasibility demo.
        # I build a concise record from the territory state before and after the chosen decision
        entry = {
            "step": self.steps,
            "territory": agent.territory.name,
            "action": decision.get("action"),
            "target": decision.get("target"),
            "reason": decision.get("reason"),
            "food_before": state_before["food"],
            "food_after": state_after["food"],
            "wealth_before": state_before["wealth"],
            "wealth_after": state_after["wealth"],
            "llm_used": llm_used,
        }
        # I keep this structure stable now so I can extend it later for councils without rewriting the analysis tools.
        # I append the record to the running chronicle so I can later show the structured trace.
        self.chronicle.append(entry)
        # I echo the decision for quick debugging while the sim runs.
        llm_label = "yes" if llm_used else "no"
        reason_text = entry["reason"] or "no reason provided"
        print(
            f"Step {entry['step']}: {entry['territory']} -> {entry['action']} "
            f"(food {entry['food_before']}→{entry['food_after']}; LLM: {llm_label}; reason: {reason_text})"
        )

    def step(self) -> None:
        """I advance the Mesa scheduler one tick so the leader agent can act."""
        # Mesa 3.x: super().step() increments self.steps
        super().step()
        # I still ask Mesa to shuffle agents even though there is currently only one so the logic scales to councils later.
        self.agents.shuffle_do("step")

    def save_chronicle(self, path: Path) -> None:
        """I persist the chronicle as JSON to ``path`` so I can analyze runs later."""
        # I ensure the log directory exists before writing.
        path.parent.mkdir(parents=True, exist_ok=True)
        # I dump the structured chronicle to disk in a readable format.
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.chronicle, f, indent=2)
