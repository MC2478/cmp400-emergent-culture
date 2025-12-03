"""Quick card: Mesa world wiring for East/West leaders plus economy, diplomacy, and logging helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TextIO

import mesa
from mesa.datacollection import DataCollector

import config
from src.agents.leader import LeaderAgent, TerritoryState
from src.model.diplomacy import relation_label, run_negotiation
from src.model.environment import EnvironmentSnapshot, generate_environment
from src.model.economy import apply_population_dynamics, apply_wages
from src.model.llm_client import LLMDecisionClient, LLMConfig
from src.model.log_utils import append_chronicle_upkeep, print_step_summary
from src.model.traits import adaptation_pressure_text, add_trait_to_state, sample_starting_trait


class WorldModel(mesa.Model):
    """Model card: track two territories, plug in leaders, and collect decisions/negotiations."""

    def __init__(
        self,
        random_seed: int | None = None,
        initial_wealth: float | None = None,
        initial_food: float | None = None,
        initial_wood: float | None = None,
        initial_iron: float | None = None,
        initial_gold: float | None = None,
        population_E: float | None = None,
        population_W: float | None = None,
        use_llm: bool = False,
    ) -> None:
        """Init cue: accept seed/resource overrides so runs stay reproducible and tweakable."""
        super().__init__(seed=random_seed)
        self.chronicle: List[Dict[str, Any]] = []
        self.environment: EnvironmentSnapshot = generate_environment(self.random)

        # Setup note: initialise East and West with asymmetric yields.
        east_env = self.environment.east
        west_env = self.environment.west
        starting_food_east = float(initial_food) if initial_food is not None else east_env.starting_food
        starting_food_west = float(initial_food) if initial_food is not None else west_env.starting_food
        starting_wealth_east = float(initial_wealth) if initial_wealth is not None else east_env.starting_wealth
        starting_wealth_west = float(initial_wealth) if initial_wealth is not None else west_env.starting_wealth
        starting_wood_east = float(initial_wood) if initial_wood is not None else east_env.starting_wood
        starting_wood_west = float(initial_wood) if initial_wood is not None else west_env.starting_wood
        starting_iron_east = float(initial_iron) if initial_iron is not None else east_env.starting_iron
        starting_iron_west = float(initial_iron) if initial_iron is not None else west_env.starting_iron
        starting_gold_east = float(initial_gold) if initial_gold is not None else east_env.starting_gold
        starting_gold_west = float(initial_gold) if initial_gold is not None else west_env.starting_gold
        population_east = float(population_E) if population_E is not None else float(config.STARTING_POPULATION)
        population_west = float(population_W) if population_W is not None else float(config.STARTING_POPULATION)
        self.east = TerritoryState(
            name="East",
            food=starting_food_east,
            wealth=starting_wealth_east,
            iron=starting_iron_east,
            gold=starting_gold_east,
            relation_to_neighbor="neutral",
            population=population_east,
            food_yield=east_env.food_yield,
            wealth_yield=east_env.wealth_yield,
            wood=starting_wood_east,
            wood_yield=east_env.wood_yield,
            iron_yield=east_env.iron_yield,
            gold_yield=east_env.gold_yield,
            infrastructure_level=config.STARTING_INFRASTRUCTURE_LEVEL,
        )
        self.west = TerritoryState(
            name="West",
            food=starting_food_west,
            wealth=starting_wealth_west,
            iron=starting_iron_west,
            gold=starting_gold_west,
            relation_to_neighbor="neutral",
            population=population_west,
            food_yield=west_env.food_yield,
            wealth_yield=west_env.wealth_yield,
            wood=starting_wood_west,
            wood_yield=west_env.wood_yield,
            iron_yield=west_env.iron_yield,
            gold_yield=west_env.gold_yield,
            infrastructure_level=config.STARTING_INFRASTRUCTURE_LEVEL,
        )
        self.seasons = list(config.SEASONS)
        self.season_multipliers = dict(config.SEASON_MULTIPLIERS)
        initial_label = relation_label(0)
        self.east.relation_score = 0
        self.west.relation_score = 0
        self.east.relation_to_neighbor = initial_label
        self.west.relation_to_neighbor = initial_label

        east_start_trait = sample_starting_trait(east_env.category, self.random)
        west_start_trait = sample_starting_trait(west_env.category, self.random)
        if east_start_trait:
            add_trait_to_state(self.east, east_start_trait, step=0, reason=f"environment:{east_env.category}")
        if west_start_trait:
            add_trait_to_state(self.west, west_start_trait, step=0, reason=f"environment:{west_env.category}")

        llm_client: LLMDecisionClient | None = None
        if use_llm:
            llm_client = LLMDecisionClient(config=LLMConfig(), enabled=True)
        self.llm_client = llm_client

        self.leader_east = LeaderAgent(model=self, territory=self.east, neighbor=self.west, llm_client=llm_client)
        self.leader_west = LeaderAgent(model=self, territory=self.west, neighbor=self.east, llm_client=llm_client)
        self.current_step_log: Dict[str, Dict[str, Any]] = {}
        self.agent_state_logs: Dict[str, TextIO] = {}
        self.agents.add(self.leader_east)
        self.agents.add(self.leader_west)
        self.datacollector = DataCollector(
            model_reporters={
                "wealth_E": lambda m: m.east.wealth,
                "wealth_W": lambda m: m.west.wealth,
                "food_E": lambda m: m.east.food,
                "food_W": lambda m: m.west.food,
            }
        )
        self.datacollector.collect(self)

    def _season_for_step(self, step_number: int) -> str:
        if not self.seasons:
            return "unknown"
        if step_number <= 0:
            return self.seasons[0]
        idx = ((step_number - 1) // 2) % len(self.seasons)
        return self.seasons[idx]

    def current_season(self) -> str:
        """Season cue: expose the current season from the step counter."""
        return self._season_for_step(self.steps)

    def next_season(self) -> str:
        """Season cue: show the upcoming season so prompts can plan ahead."""
        return self._season_for_step(self.steps + 1)

    def _ensure_diplomacy_client(self) -> None:
        """Guarantee that the world model keeps a live LLM client for negotiations.

        The dashboard can rebuild models or toggle the LLM flag, which occasionally leaves
        ``self.llm_client`` as ``None`` even though each leader still holds a reference.
        This helper synchronises the shared pointer so diplomacy always has a client when
        at least one leader is still LLM-enabled.
        """
        if self.llm_client is not None and self.llm_client.enabled:
            return
        for leader in (self.leader_east, self.leader_west):
            leader_client = getattr(leader, "llm_client", None)
            if leader_client is not None and leader_client.enabled:
                self.llm_client = leader_client
                return

    def get_config_summary(self) -> dict:
        """Config card: snapshot the key knobs to serialize alongside a run."""
        return {
            "environment": {
                "east": {
                    "category": self.environment.east.category,
                    **self.environment.east.metrics,
                },
                "west": {
                    "category": self.environment.west.category,
                    **self.environment.west.metrics,
                },
                "metals": {
                    "iron_holder": self.environment.iron_holder,
                    "gold_holder": self.environment.gold_holder,
                },
            },
            "territories": {
                "East": {
                    "initial_food": self.east.food,
                    "initial_wealth": self.east.wealth,
                    "initial_iron": self.east.iron,
                    "initial_gold": self.east.gold,
                    "initial_population": self.east.population,
                    "food_yield": self.east.food_yield,
                    "wealth_yield": self.east.wealth_yield,
                    "wood_yield": self.east.wood_yield,
                    "iron_yield": self.east.iron_yield,
                    "gold_yield": self.east.gold_yield,
                    "infrastructure_level": self.east.infrastructure_level,
                },
                "West": {
                    "initial_food": self.west.food,
                    "initial_wealth": self.west.wealth,
                    "initial_iron": self.west.iron,
                    "initial_gold": self.west.gold,
                    "initial_population": self.west.population,
                    "food_yield": self.west.food_yield,
                    "wealth_yield": self.west.wealth_yield,
                    "wood_yield": self.west.wood_yield,
                    "iron_yield": self.west.iron_yield,
                    "gold_yield": self.west.gold_yield,
                    "infrastructure_level": self.west.infrastructure_level,
                },
            },
            "actions": [
                "focus_food",
                "focus_wealth",
                "focus_wood",
                "focus_iron",
                "focus_gold",
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
                "point_bonus": config.INFRA_FOOD_YIELD_MULT_PER_LEVEL,
                "tiers": {
                    "wood": {
                        "costs": {
                            "wood": config.INFRA_TIER_WOOD_WOOD_COST,
                            "wealth": config.INFRA_TIER_WOOD_WEALTH_COST,
                        },
                        "points": config.INFRA_TIER_WOOD_POINTS,
                    },
                    "iron": {
                        "costs": {
                            "iron": config.INFRA_TIER_IRON_IRON_COST,
                            "wealth": config.INFRA_TIER_IRON_WEALTH_COST,
                        },
                        "points": config.INFRA_TIER_IRON_POINTS,
                    },
                    "gold": {
                        "costs": {
                            "gold": config.INFRA_TIER_GOLD_GOLD_COST,
                            "iron": config.INFRA_TIER_GOLD_IRON_COST,
                        },
                        "points": config.INFRA_TIER_GOLD_POINTS,
                    },
                },
            },
        }

    def save_config_summary(self, path: str) -> None:
        """Export cue: write a human-readable config summary to a given path."""
        cfg = self.get_config_summary()
        lines: list[str] = []
        lines.append("Simulation configuration summary")
        lines.append("================================")
        lines.append("")
        lines.append("Environment:")
        lines.append(f"  East category: {self.environment.east.category}")
        for key, value in self.environment.east.metrics.items():
            lines.append(f"    east_{key}: {value:.3f}")
        lines.append(f"  West category: {self.environment.west.category}")
        for key, value in self.environment.west.metrics.items():
            lines.append(f"    west_{key}: {value:.3f}")
        lines.append(f"  Iron holder: {self.environment.iron_holder}")
        lines.append(f"  Gold holder: {self.environment.gold_holder}")
        lines.append("")
        east_yields = {
            "food": self.environment.east.food_yield,
            "wealth": self.environment.east.wealth_yield,
            "wood": self.environment.east.wood_yield,
            "iron": self.environment.east.iron_yield,
            "gold": self.environment.east.gold_yield,
        }
        west_yields = {
            "food": self.environment.west.food_yield,
            "wealth": self.environment.west.wealth_yield,
            "wood": self.environment.west.wood_yield,
            "iron": self.environment.west.iron_yield,
            "gold": self.environment.west.gold_yield,
        }
        lines.append("Resource yields:")
        lines.append(
            "  East: "
            + ", ".join(f"{key} {value:.3f}" for key, value in east_yields.items())
            + f" (total {sum(east_yields.values()):.3f})"
        )
        lines.append(
            "  West: "
            + ", ".join(f"{key} {value:.3f}" for key, value in west_yields.items())
            + f" (total {sum(west_yields.values()):.3f})"
        )
        lines.append(
            "  Totals: "
            + ", ".join(
                f"{key} {(east_yields[key] + west_yields[key]):.3f}/{getattr(config, f'WORLD_MAX_{key.upper()}_YIELD'):.3f}"
                for key in east_yields
            )
        )
        lines.append("")
        lines.append("Territories:")
        for name, tcfg in cfg["territories"].items():
            lines.append(f"  {name}:")
            lines.append(f"    initial_food: {tcfg['initial_food']}")
            lines.append(f"    initial_wealth: {tcfg['initial_wealth']}")
            lines.append(f"    initial_iron: {tcfg['initial_iron']}")
            lines.append(f"    initial_gold: {tcfg['initial_gold']}")
            lines.append(f"    initial_population: {tcfg['initial_population']}")
            lines.append(f"    food_yield: {tcfg['food_yield']}")
            lines.append(f"    wealth_yield: {tcfg['wealth_yield']}")
            lines.append(f"    wood_yield: {tcfg['wood_yield']}")
            lines.append(f"    iron_yield: {tcfg['iron_yield']}")
            lines.append(f"    gold_yield: {tcfg['gold_yield']}")
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
        lines.append(f"  point_bonus: {cfg['infrastructure']['point_bonus']}")
        lines.append("  tiers:")
        for tier_name, tier_data in cfg["infrastructure"]["tiers"].items():
            cost_desc = ", ".join(f"{resource}:{amount}" for resource, amount in tier_data["costs"].items())
            lines.append(f"    {tier_name}: costs({cost_desc}), points={tier_data['points']}")
        lines.append("")
        text = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def enable_agent_state_logging(self, run_dir: Path) -> None:
        """Logging setup: open per-agent JSONL files for mental-state snapshots."""
        run_dir.mkdir(parents=True, exist_ok=True)
        self.close_agent_state_logs()
        for territory in (self.east, self.west):
            path = run_dir / f"{territory.name.lower()}_agent_state.jsonl"
            handle = path.open("w", encoding="utf-8")
            self.agent_state_logs[territory.name] = handle

    def log_agent_state(self, leader: LeaderAgent, decision: Dict[str, Any], llm_used: bool) -> None:
        """State tap: emit a JSON line capturing the leader's internal state for this step."""
        handle = self.agent_state_logs.get(leader.territory.name)
        if handle is None:
            return
        territory = leader.territory
        allocations = decision.get("applied_allocations") or decision.get("allocations") or {}
        memory_event = leader.memory_events[-1] if leader.memory_events else None
        snapshot = {
            "step": self.steps,
            "name": territory.name,
            "resources": {
                "food": territory.food,
                "wealth": territory.wealth,
                "wood": territory.wood,
                "iron": territory.iron,
                "gold": territory.gold,
                "infrastructure_level": territory.infrastructure_level,
            },
            "traits": {
                "active_traits": list(territory.active_traits),
                "personality_vector": dict(territory.personality_vector),
                "trait_cooldown_steps": territory.trait_cooldown_steps,
            },
            "pressures": {
                "exploitation_streak": territory.exploitation_streak,
                "starvation_streak": territory.starvation_streak,
                "failed_strategy_streak": territory.failed_strategy_streak,
                "adaptation_pressure": territory.adaptation_pressure_note,
            },
            "decision": {
                "action": decision.get("action"),
                "allocations": allocations,
                "build_infrastructure": bool(decision.get("build_infrastructure")),
                "infrastructure_built": decision.get("infrastructure_built"),
                "reason": decision.get("reason"),
                "trait_adjustment": decision.get("trait_adjustment"),
                "used_llm": llm_used,
            },
            "meta": {
                "next_directive": leader.next_directive,
                "last_trait_adjustment_text": getattr(leader, "last_trait_adjustment_text", None),
                "last_memory_event": memory_event,
                "recent_interactions": leader.interaction_log[-3:],
            },
        }
        handle.write(json.dumps(snapshot) + "\n")
        handle.flush()

    def record_decision(
        self,
        territory_name: str,
        before: Dict[str, Any],
        decision: Dict[str, Any],
        after: Dict[str, Any],
        used_llm: bool,
    ) -> None:
        """Decision buffer: stash before/after/metadata so grouped logging can use it."""
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
        """Upkeep buffer: store pre/post upkeep snapshots for summaries."""
        entry = self.current_step_log.setdefault(territory_name, {})
        entry["upkeep"] = {"before": dict(before), "after": dict(after)}

    def record_wages(
        self,
        territory_name: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        """Wage buffer: capture wage payments so summaries reveal morale issues."""
        entry = self.current_step_log.setdefault(territory_name, {})
        entry["wages"] = {"before": dict(before), "after": dict(after)}

    def log_upkeep(
        self,
        east_before: Dict[str, Any],
        east_after: Dict[str, Any],
        west_before: Dict[str, Any],
        west_after: Dict[str, Any],
    ) -> None:
        """Upkeep card: capture upkeep snapshots for logging and the chronicle."""
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
        """Memory card: capture before/after stats so prompts can include recent context."""

        def _starving_flag(before_state: Dict[str, Any]) -> bool:
            required = (before_state.get("population", 0.0) / 10.0) * config.FOOD_PER_10_POP
            return required > 0 and before_state.get("food", 0.0) < required

        def _decision_meta(name: str) -> Dict[str, Any]:
            wrapper = self.current_step_log.get(name, {}).get("decision") or {}
            return wrapper.get("decision", {}) or {}

        def _record(name: str, leader: LeaderAgent, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> None:
            action = leader.last_action or "wait"
            note_bits: list[str] = []
            starving_event = _starving_flag(before_state)
            if starving_event:
                note_bits.append("starvation")
            if after_state.get("population", 0.0) <= 0.0:
                note_bits.append("collapse")
            strike_active = bool(getattr(leader.territory, "on_strike", False))
            morale_mult = getattr(leader.territory, "effective_work_multiplier", 1.0)
            unpaid = getattr(leader.territory, "unpaid_steps", 0.0)
            if strike_active:
                note_bits.append("strike_active")
            elif morale_mult < 0.999:
                note_bits.append(f"low_morale x{morale_mult:.2f} (debt {unpaid:.1f})")
            decision_meta = _decision_meta(name)
            attempted_build = bool(decision_meta.get("build_infrastructure")) or str(
                decision_meta.get("action", "")
            ).lower() == "build_infrastructure"
            if not attempted_build:
                legacy = str(decision_meta.get("legacy_action", "")).lower()
                attempted_build = legacy == "build_infrastructure"
            if attempted_build and decision_meta.get("infrastructure_built") is False:
                note_bits.append("infra_failed_missing_resources")
            last_reason = (leader.last_reason or "").strip()
            if last_reason:
                note_bits.append(f"reason: {last_reason}")
            notes_text = "; ".join(note_bits) if note_bits else None
            leader.record_step_outcome(
                step=self.steps,
                action_name=action,
                food_before=before_state.get("food", 0.0),
                food_after=after_state.get("food", 0.0),
                wealth_before=before_state.get("wealth", 0.0),
                wealth_after=after_state.get("wealth", 0.0),
                pop_before=before_state.get("population", 0.0),
                pop_after=after_state.get("population", 0.0),
                notes=notes_text,
            )

        _record("East", self.leader_east, east_before, east_after)
        _record("West", self.leader_west, west_before, west_after)

    def _refresh_trait_state_for_logging(self) -> None:
        """Trait cache: refresh trait state so summaries show the latest pressures."""
        for name, territory in (("East", self.east), ("West", self.west)):
            decision_wrap = self.current_step_log.get(name, {}).get("decision")
            if not decision_wrap:
                continue
            meta = decision_wrap.get("decision", {})
            trait_state = meta.get("trait_state", {}) or {}
            territory.adaptation_pressure_note = adaptation_pressure_text(territory)
            trait_state.update(
                {
                    "active_traits": list(territory.active_traits),
                    "personality_vector": dict(territory.personality_vector),
                    "trait_cooldown_steps": territory.trait_cooldown_steps,
                    "exploitation_streak": territory.exploitation_streak,
                    "starvation_streak": territory.starvation_streak,
                    "failed_strategy_streak": territory.failed_strategy_streak,
                    "adaptation_pressure": territory.adaptation_pressure_note,
                    "trait_events": list(territory.trait_events),
                }
            )
            meta["trait_state"] = trait_state
            decision_wrap["decision"] = meta

    def step(self) -> None:
        """Loop card: advance one tick to run actions, negotiation, upkeep, and logging."""
        # Presentation cue: this loop is the whole turn laid bare for walkthroughs.
        super().step()
        self.current_step_log = {}
        print(f"Step {self.steps}:")
        season = self.current_season()
        next_season = self.next_season()
        season_mult = self.season_multipliers.get(season, 1.0)
        print(f"  World: season={season} (food/wood yield x{season_mult:.2f}), next={next_season}")
        self.agents.shuffle_do("step")

        self._ensure_diplomacy_client()
        if self.llm_client is not None and self.llm_client.enabled:
            run_negotiation(self)

        # Morale timing: pay wages after production so next turn reflects morale.
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

        east_before = self._resource_state(self.east)
        west_before = self._resource_state(self.west)
        self._update_starvation_pressure(east_before, self.east)
        self._update_starvation_pressure(west_before, self.west)
        apply_population_dynamics(self.east)
        apply_population_dynamics(self.west)
        east_after = self._resource_state(self.east)
        west_after = self._resource_state(self.west)
        self._update_strategy_pressure(self.east, pop_after=east_after["population"], wealth_after=east_after["wealth"])
        self._update_strategy_pressure(self.west, pop_after=west_after["population"], wealth_after=west_after["wealth"])
        self.log_upkeep(east_before, east_after, west_before, west_after)
        self._record_leader_memories(
            east_before=east_before,
            east_after=east_after,
            west_before=west_before,
            west_after=west_after,
        )
        self._refresh_trait_state_for_logging()
        print_step_summary(self.steps, self.current_step_log, self.chronicle, self.season_multipliers)
        self.datacollector.collect(self)

    def all_territories_dead(self) -> bool:
        """End check: report whether both territories have collapsed."""
        return self.east.population <= 0 and self.west.population <= 0

    def save_chronicle(self, path: Path) -> None:
        """Export cue: save the chronicle as JSON so it can be analysed later."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.chronicle, f, indent=2)
        self.close_agent_state_logs()

    def close_agent_state_logs(self) -> None:
        """Cleanup note: close any open state log handles to avoid leaks."""
        for handle in self.agent_state_logs.values():
            try:
                handle.close()
            except Exception:
                pass
        self.agent_state_logs.clear()

    def _resource_state(self, territory: TerritoryState) -> Dict[str, Any]:
        """Snapshot cue: shallow copy of resources/infrastructure for logging."""
        return {
            "food": territory.food,
            "population": territory.population,
            "wealth": territory.wealth,
            "wood": territory.wood,
            "iron": territory.iron,
            "gold": territory.gold,
            "infrastructure_level": territory.infrastructure_level,
        }

    def _update_starvation_pressure(self, before_state: Dict[str, Any], territory: TerritoryState) -> None:
        """Starvation tracker: adjust streaks based on pre-upkeep safety."""
        required = (before_state.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
        food = before_state.get("food", 0.0)
        ratio = food / required if required > 0 else float("inf")
        if ratio < 1.0:
            territory.starvation_streak += 1
        else:
            territory.starvation_streak = 0

    def _update_strategy_pressure(self, territory: TerritoryState, *, pop_after: float, wealth_after: float) -> None:
        """Stagnation tracker: bump pressure when similar plans fail to deliver gains."""
        prev_pop = territory.last_population_after
        prev_wealth = territory.last_wealth_after
        prev_allocs = getattr(territory, "previous_allocations", {}) or {}
        current_allocs = getattr(territory, "last_applied_allocations", {}) or {}
        diff_sum = 0.0
        for key in ("focus_food", "focus_wood", "focus_wealth"):
            diff_sum += abs(current_allocs.get(key, 0.0) - prev_allocs.get(key, 0.0))
        similar_plan = bool(prev_allocs) and diff_sum < 0.1

        if prev_pop is None or prev_wealth is None:
            territory.failed_strategy_streak = 0
        else:
            no_gain = wealth_after <= prev_wealth + 1e-6 and pop_after <= prev_pop + 1e-6
            if no_gain and similar_plan:
                territory.failed_strategy_streak += 1
            else:
                territory.failed_strategy_streak = 0

        territory.last_population_after = pop_after
        territory.last_wealth_after = wealth_after
