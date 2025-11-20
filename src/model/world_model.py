"""Mesa world model wiring East/West leaders, now using extracted helpers for economy, diplomacy, and logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import mesa

import config
from src.agents.leader import LeaderAgent, TerritoryState
from src.model.diplomacy import relation_label, run_negotiation
from src.model.economy import apply_population_dynamics, apply_wages
from src.model.llm_client import LLMDecisionClient, LLMConfig
from src.model.log_utils import append_chronicle_upkeep, print_step_summary


class WorldModel(mesa.Model):
    """Track two territories, plug in leader agents, and collect decisions/negotiations for analysis."""

    def __init__(self, random_seed: int | None = None, initial_food: float = config.STARTING_FOOD, use_llm: bool = False) -> None:
        """Accept ``random_seed``, ``initial_food``, and ``use_llm`` so runs are reproducible."""
        super().__init__(seed=random_seed)
        self.chronicle: List[Dict[str, Any]] = []

        # Initialise East and West with asymmetric yields.
        self.east = TerritoryState(
            name="East",
            food=float(initial_food),
            wealth=config.STARTING_WEALTH,
            relation_to_neighbor="neutral",
            population=config.STARTING_POPULATION,
            food_yield=config.EAST_FOOD_YIELD,
            wealth_yield=config.EAST_WEALTH_YIELD,
            wood=config.STARTING_WOOD,
            wood_yield=config.EAST_WOOD_YIELD,
            infrastructure_level=config.STARTING_INFRASTRUCTURE_LEVEL,
        )
        self.west = TerritoryState(
            name="West",
            food=float(initial_food),
            wealth=config.STARTING_WEALTH,
            relation_to_neighbor="neutral",
            population=config.STARTING_POPULATION,
            food_yield=config.WEST_FOOD_YIELD,
            wealth_yield=config.WEST_WEALTH_YIELD,
            wood=config.STARTING_WOOD,
            wood_yield=config.WEST_WOOD_YIELD,
            infrastructure_level=config.STARTING_INFRASTRUCTURE_LEVEL,
        )
        self.seasons = list(config.SEASONS)
        self.season_multipliers = dict(config.SEASON_MULTIPLIERS)
        initial_label = relation_label(0)
        self.east.relation_score = 0
        self.west.relation_score = 0
        self.east.relation_to_neighbor = initial_label
        self.west.relation_to_neighbor = initial_label

        llm_client: LLMDecisionClient | None = None
        if use_llm:
            llm_client = LLMDecisionClient(config=LLMConfig(), enabled=True)
        self.llm_client = llm_client

        self.leader_east = LeaderAgent(model=self, territory=self.east, neighbor=self.west, llm_client=llm_client)
        self.leader_west = LeaderAgent(model=self, territory=self.west, neighbor=self.east, llm_client=llm_client)
        self.current_step_log: Dict[str, Dict[str, Any]] = {}
        self.agents.add(self.leader_east)
        self.agents.add(self.leader_west)

    def _season_for_step(self, step_number: int) -> str:
        if not self.seasons:
            return "unknown"
        if step_number <= 0:
            return self.seasons[0]
        idx = ((step_number - 1) // 2) % len(self.seasons)
        return self.seasons[idx]

    def current_season(self) -> str:
        """Expose the current season derived from the simulation step counter."""
        return self._season_for_step(self.steps)

    def next_season(self) -> str:
        """Expose the upcoming season so the LLM can plan ahead."""
        return self._season_for_step(self.steps + 1)

    def get_config_summary(self) -> dict:
        """Snapshot the core knobs so they can be serialised alongside a run."""
        return {
            "territories": {
                "East": {
                    "initial_food": self.east.food,
                    "initial_wealth": self.east.wealth,
                    "initial_population": self.east.population,
                    "food_yield": self.east.food_yield,
                    "wealth_yield": self.east.wealth_yield,
                    "wood_yield": self.east.wood_yield,
                    "infrastructure_level": self.east.infrastructure_level,
                },
                "West": {
                    "initial_food": self.west.food,
                    "initial_wealth": self.west.wealth,
                    "initial_population": self.west.population,
                    "food_yield": self.west.food_yield,
                    "wealth_yield": self.west.wealth_yield,
                    "wood_yield": self.west.wood_yield,
                    "infrastructure_level": self.west.infrastructure_level,
                },
            },
            "actions": [
                "focus_food",
                "focus_wealth",
                "focus_wood",
                "wait",
                "build_infrastructure",
            ],
            "wages": {
                "wage_per_worker": config.WAGE_PER_WORKER,
            },
            "seasons": {
                "order": list(self.seasons),
                "multipliers": dict(self.season_multipliers),
            },
            "infrastructure": {
                "wood_cost": config.INFRA_COST_WOOD,
                "wealth_cost": config.INFRA_COST_WEALTH,
                "yield_bonus_per_level": config.INFRA_FOOD_YIELD_MULT_PER_LEVEL,
            },
        }

    def save_config_summary(self, path: str) -> None:
        """Persist a human-readable description of the configuration to ``path``."""
        cfg = self.get_config_summary()
        lines: list[str] = []
        lines.append("Simulation configuration summary")
        lines.append("================================")
        lines.append("")
        lines.append("Territories:")
        for name, tcfg in cfg["territories"].items():
            lines.append(f"  {name}:")
            lines.append(f"    initial_food: {tcfg['initial_food']}")
            lines.append(f"    initial_wealth: {tcfg['initial_wealth']}")
            lines.append(f"    initial_population: {tcfg['initial_population']}")
            lines.append(f"    food_yield: {tcfg['food_yield']}")
            lines.append(f"    wealth_yield: {tcfg['wealth_yield']}")
            lines.append(f"    wood_yield: {tcfg['wood_yield']}")
            lines.append(f"    infrastructure_level: {tcfg['infrastructure_level']}")
            lines.append("")
        lines.append("Actions:")
        for action in cfg["actions"]:
            lines.append(f"  - {action}")
        lines.append("")
        lines.append("Wages:")
        lines.append(f"  wage_per_worker: {cfg['wages']['wage_per_worker']}")
        lines.append("")
        lines.append("Seasons:")
        lines.append(f"  order: {', '.join(cfg['seasons']['order'])}")
        lines.append("  multipliers:")
        for season, mult in cfg["seasons"]["multipliers"].items():
            lines.append(f"    {season}: {mult}")
        lines.append("")
        lines.append("Infrastructure:")
        lines.append(f"  wood_cost: {cfg['infrastructure']['wood_cost']}")
        lines.append(f"  wealth_cost: {cfg['infrastructure']['wealth_cost']}")
        lines.append(f"  yield_bonus_per_level: {cfg['infrastructure']['yield_bonus_per_level']}")
        lines.append("")
        text = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def record_decision(
        self,
        territory_name: str,
        before: Dict[str, Any],
        decision: Dict[str, Any],
        after: Dict[str, Any],
        used_llm: bool,
    ) -> None:
        """Buffer decision details so grouped logging can use them."""
        entry = self.current_step_log.setdefault(territory_name, {})
        entry["decision"] = {
            "before": dict(before),
            "after": dict(after),
            "decision": dict(decision),
            "used_llm": used_llm,
        }

    def record_upkeep(
        self,
        territory_name: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        """Buffer upkeep outcome per territory for grouped summary."""
        entry = self.current_step_log.setdefault(territory_name, {})
        entry["upkeep"] = {"before": dict(before), "after": dict(after)}

    def record_wages(
        self,
        territory_name: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        """Capture wage payments so the summary can reveal morale issues."""
        entry = self.current_step_log.setdefault(territory_name, {})
        entry["wages"] = {"before": dict(before), "after": dict(after)}

    def log_upkeep(
        self,
        east_before: Dict[str, Any],
        east_after: Dict[str, Any],
        west_before: Dict[str, Any],
        west_after: Dict[str, Any],
    ) -> None:
        """Capture upkeep snapshots for both logging and the chronicle."""
        self.record_upkeep("East", east_before, east_after)
        self.record_upkeep("West", west_before, west_after)
        append_chronicle_upkeep(self.chronicle, self.steps, east_before, east_after, west_before, west_after)

    def _record_leader_memories(
        self,
        *,
        east_before: Dict[str, Any],
        east_after: Dict[str, Any],
        west_before: Dict[str, Any],
        west_after: Dict[str, Any],
    ) -> None:
        """Capture before/after stats for each leader so prompts can include memory."""

        def _starving_flag(before_state: Dict[str, Any]) -> bool:
            required = (before_state.get("population", 0.0) / 10.0) * config.FOOD_PER_10_POP
            return before_state.get("food", 0.0) < required

        def _record(leader: LeaderAgent, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> None:
            action = leader.last_action or "wait"
            note_bits: list[str] = []
            if leader.last_reason:
                note_bits.append(str(leader.last_reason))
            morale_mult = getattr(leader.territory, "effective_work_multiplier", 1.0)
            unpaid = getattr(leader.territory, "unpaid_steps", 0)
            if getattr(leader.territory, "on_strike", False):
                note_bits.append(f"workers on strike (mult {morale_mult:.2f}, unpaid debt {unpaid:.1f})")
            elif morale_mult < 0.999:
                note_bits.append(f"low morale mult {morale_mult:.2f} (unpaid debt {unpaid:.1f})")
            note_text = " | ".join(note_bits) if note_bits else None
            leader.record_step_outcome(
                step=self.steps,
                action=action,
                food_before=float(before_state.get("food", 0.0)),
                food_after=float(after_state.get("food", 0.0)),
                wealth_before=float(before_state.get("wealth", 0.0)),
                wealth_after=float(after_state.get("wealth", 0.0)),
                pop_before=float(before_state.get("population", 0.0)),
                pop_after=float(after_state.get("population", 0.0)),
                starving=_starving_flag(before_state),
                strike=getattr(leader.territory, "on_strike", False),
                note=note_text,
            )

        _record(self.leader_east, east_before, east_after)
        _record(self.leader_west, west_before, west_after)

    def step(self) -> None:
        """Advance the Mesa scheduler one tick so leader agents can act."""
        super().step()
        self.current_step_log = {}
        print(f"Step {self.steps}:")
        season = self.current_season()
        next_season = self.next_season()
        season_mult = self.season_multipliers.get(season, 1.0)
        print(f"  World: season={season} (food/wood yield x{season_mult:.2f}), next={next_season}")
        self.agents.shuffle_do("step")

        if self.llm_client is not None and self.llm_client.enabled:
            run_negotiation(self)

        # Wages/morale after production so next turn reflects morale.
        for name, territory in (("East", self.east), ("West", self.west)):
            before_wages = {
                "wealth": territory.wealth,
                "multiplier": territory.effective_work_multiplier,
                "on_strike": territory.on_strike,
                "unpaid_steps": territory.unpaid_steps,
            }
            apply_wages(territory)
            after_wages = {
                "wealth": territory.wealth,
                "multiplier": territory.effective_work_multiplier,
                "on_strike": territory.on_strike,
                "unpaid_steps": territory.unpaid_steps,
            }
            self.record_wages(name, before_wages, after_wages)

        east_before = {
            "food": self.east.food,
            "population": self.east.population,
            "wealth": self.east.wealth,
            "wood": self.east.wood,
            "infrastructure_level": self.east.infrastructure_level,
        }
        west_before = {
            "food": self.west.food,
            "population": self.west.population,
            "wealth": self.west.wealth,
            "wood": self.west.wood,
            "infrastructure_level": self.west.infrastructure_level,
        }
        apply_population_dynamics(self.east)
        apply_population_dynamics(self.west)
        east_after = {
            "food": self.east.food,
            "population": self.east.population,
            "wealth": self.east.wealth,
            "wood": self.east.wood,
            "infrastructure_level": self.east.infrastructure_level,
        }
        west_after = {
            "food": self.west.food,
            "population": self.west.population,
            "wealth": self.west.wealth,
            "wood": self.west.wood,
            "infrastructure_level": self.west.infrastructure_level,
        }
        self.log_upkeep(east_before, east_after, west_before, west_after)
        self._record_leader_memories(
            east_before=east_before,
            east_after=east_after,
            west_before=west_before,
            west_after=west_after,
        )
        print_step_summary(self.steps, self.current_step_log, self.chronicle, self.season_multipliers)

    def all_territories_dead(self) -> bool:
        """Report whether every territory has collapsed so callers can stop early."""
        return self.east.population <= 0 and self.west.population <= 0

    def save_chronicle(self, path: Path) -> None:
        """Persist the chronicle as JSON to ``path`` so it can be analysed later."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.chronicle, f, indent=2)
