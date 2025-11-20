"""I now model two symmetric leaders (East and West) with different resource yields so the LLM can
contrast their strategic choices while I capture both actions and negotiations for the demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import mesa

import config


def _fmt_res(value: Any) -> str:
    """I format resource numbers with the configured precision."""
    try:
        return f"{float(value):.{config.RESOURCE_DISPLAY_DECIMALS}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pop(value: Any) -> str:
    """I format population counts with the configured precision (typically whole numbers)."""
    try:
        decimals = max(0, int(config.POP_DISPLAY_DECIMALS))
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"
from src.agents.leader import LeaderAgent, TerritoryState
from src.model.llm_client import LLMDecisionClient, LLMConfig, summarise_memory_for_prompt


class WorldModel(mesa.Model):
    """I track the single territory, plug in the leader agent, and collect every decision for
    the feasibility write-up."""

    def __init__(self, random_seed: int | None = None, initial_food: float = config.STARTING_FOOD, use_llm: bool = False) -> None:
        """I accept ``random_seed``, ``initial_food``, and ``use_llm`` so I can reproduce runs and
        decide whether the leader should query the local LLM."""
        # I let Mesa set up its internal agent containers while honoring the random seed.
        super().__init__(seed=random_seed)
        # I track a stable chronicle schema here so I can reuse it as an artifact in the feasibility report.
        self.chronicle: List[Dict[str, Any]] = []

        # I document the initial territory stats so the feasibility demo stays simple to explain.
        # I now set up East and West with different yields so they feel like distinct actors.
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
        initial_label = self._relation_label(0)
        self.east.relation_score = 0
        self.west.relation_score = 0
        self.east.relation_to_neighbor = initial_label
        self.west.relation_to_neighbor = initial_label

        # I enable the HTTP LLM client only when requested so rules remain the default path.
        llm_client: LLMDecisionClient | None = None
        if use_llm:
            llm_client = LLMDecisionClient(config=LLMConfig(), enabled=True)

        self.llm_client = llm_client
        # I register the sole leader agent with Mesa's ``AgentSet`` so upgrading to multi-agent later is easy.
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
        """I expose the current season derived from the simulation step counter."""
        return self._season_for_step(self.steps)

    def next_season(self) -> str:
        """I expose the upcoming season so the LLM can plan ahead."""
        return self._season_for_step(self.steps + 1)

    def get_config_summary(self) -> dict:
        """I expose a snapshot of the core knobs so I can serialise them elsewhere."""
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
                "balanced",
                "wait",
                "build_infrastructure",
            ],
            "wages": {
                "wage_per_worker": 0.2,
            },
            "seasons": {
                "order": list(self.seasons),
                "multipliers": dict(self.season_multipliers),
            },
            "infrastructure": {
                "wood_cost": 5.0,
                "wealth_cost": 3.0,
                "yield_bonus_per_level": 0.1,
            },
        }

    def save_config_summary(self, path: str) -> None:
        """I persist a human-readable description of the configuration to ``path``."""
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
        """I buffer the decision details so I can later log and print them in a grouped block."""
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
        """I buffer the upkeep outcome per territory for the grouped summary."""
        entry = self.current_step_log.setdefault(territory_name, {})
        entry["upkeep"] = {"before": dict(before), "after": dict(after)}

    def record_wages(
        self,
        territory_name: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        """I capture wage payments so the summary can reveal morale issues."""
        entry = self.current_step_log.setdefault(territory_name, {})
        entry["wages"] = {"before": dict(before), "after": dict(after)}

    def log_upkeep(
        self,
        east_before: Dict[str, Any],
        east_after: Dict[str, Any],
        west_before: Dict[str, Any],
        west_after: Dict[str, Any],
    ) -> None:
        """I capture upkeep snapshots so the step summary and chronicle use consistent data."""
        self.record_upkeep("East", east_before, east_after)
        self.record_upkeep("West", west_before, west_after)
        self._append_chronicle_upkeep(east_before, east_after, west_before, west_after)

    def _append_chronicle_action(self, territory_name: str, data: Dict[str, Any]) -> None:
        decision = data.get("decision", {})
        before = decision.get("before", {})
        after = decision.get("after", {})
        meta = decision.get("decision", {})
        entry = {
            "event_type": "action",
            "step": self.steps,
            "territory": territory_name,
            "actor": territory_name,
            "action": meta.get("action"),
            "target": meta.get("target"),
            "reason": meta.get("reason"),
            "allocations": meta.get("applied_allocations") or meta.get("allocations"),
            "build_infrastructure_requested": bool(meta.get("build_infrastructure")),
            "infrastructure_built": meta.get("infrastructure_built"),
            "food_before": before.get("food"),
            "food_after": after.get("food"),
            "wealth_before": before.get("wealth"),
            "wealth_after": after.get("wealth"),
            "wood_before": before.get("wood"),
            "wood_after": after.get("wood"),
            "infrastructure_before": before.get("infrastructure_level"),
            "infrastructure_after": after.get("infrastructure_level"),
            "population_before": before.get("population"),
            "population_after": after.get("population"),
            "llm_used": decision.get("used_llm"),
            "relation_before": before.get("relation_to_neighbor"),
            "relation_after": after.get("relation_to_neighbor"),
            "neighbor_food_before": before.get("neighbor_food"),
            "neighbor_food_after": after.get("neighbor_food"),
            "neighbor_wealth_before": before.get("neighbor_wealth"),
            "neighbor_wealth_after": after.get("neighbor_wealth"),
            "neighbor_population_before": before.get("neighbor_population"),
            "neighbor_population_after": after.get("neighbor_population"),
            "neighbor_wood_before": before.get("neighbor_wood"),
            "neighbor_wood_after": after.get("neighbor_wood"),
        }
        self.chronicle.append(entry)

    def _record_leader_memories(
        self,
        *,
        east_before: Dict[str, Any],
        east_after: Dict[str, Any],
        west_before: Dict[str, Any],
        west_after: Dict[str, Any],
    ) -> None:
        """I capture before/after stats for each leader so their prompts can include memory."""

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

    def _append_chronicle_upkeep(self, east_before, east_after, west_before, west_after) -> None:
        entry = {
            "event_type": "upkeep",
            "step": self.steps,
            "east": {
                "food_before": east_before["food"],
                "food_after": east_after["food"],
                "population_before": east_before["population"],
                "population_after": east_after["population"],
                "wealth_before": east_before.get("wealth"),
                "wealth_after": east_after.get("wealth"),
                "wood_before": east_before.get("wood"),
                "wood_after": east_after.get("wood"),
                "infrastructure_before": east_before.get("infrastructure_level"),
                "infrastructure_after": east_after.get("infrastructure_level"),
            },
            "west": {
                "food_before": west_before["food"],
                "food_after": west_after["food"],
                "population_before": west_before["population"],
                "population_after": west_after["population"],
                "wealth_before": west_before.get("wealth"),
                "wealth_after": west_after.get("wealth"),
                "wood_before": west_before.get("wood"),
                "wood_after": west_after.get("wood"),
                "infrastructure_before": west_before.get("infrastructure_level"),
                "infrastructure_after": west_after.get("infrastructure_level"),
            },
        }
        self.chronicle.append(entry)

    def _print_step_summary(self) -> None:
        for territory in ["West", "East"]:
            info = self.current_step_log.get(territory, {})
            decision = info.get("decision", {})
            if decision:
                before = decision.get("before", {})
                after = decision.get("after", {})
                used_llm = "yes" if decision.get("used_llm") else "no"
                meta = decision.get("decision", {})
                reason = meta.get("reason", "no reason provided")
                action = meta.get("action") or ("mixed_allocation" if meta.get("applied_allocations") else "wait")
                allocations = meta.get("applied_allocations") or meta.get("allocations") or {}
                allocation_line = ""
                if allocations:
                    formatted_allocs = ", ".join(f"{k}:{float(v):.2f}" for k, v in allocations.items())
                    allocation_line = f"\n    ⋅ Work allocation: {formatted_allocs}"
                infra_line = ""
                if meta.get("build_infrastructure"):
                    status = "built" if meta.get("infrastructure_built") else "failed"
                    infra_line = f"\n    ⋅ Infrastructure attempt: {status}"
                print(
                    f"  {territory}: action={action} (LLM: {used_llm})\n"
                    f"    ⋅ Reason: {reason}{allocation_line}{infra_line}\n"
                    f"    ⋅ Resources: food {_fmt_res(before.get('food'))}->{_fmt_res(after.get('food'))}, "
                    f"wealth {_fmt_res(before.get('wealth'))}->{_fmt_res(after.get('wealth'))}, "
                    f"wood {_fmt_res(before.get('wood'))}->{_fmt_res(after.get('wood'))}, "
                    f"infra {before.get('infrastructure_level')}->{after.get('infrastructure_level')}\n"
                    f"    ⋅ Population: {_fmt_pop(before.get('population'))}"
                )
                self._append_chronicle_action(territory, info)

        negotiation_info = self.current_step_log.get("negotiation")
        if negotiation_info:
            entry = negotiation_info["entry"]
            east_before = negotiation_info["east_before"]
            east_after = negotiation_info["east_after"]
            west_before = negotiation_info["west_before"]
            west_after = negotiation_info["west_after"]
            dialogue_lines = entry.get("dialogue") or []
            if dialogue_lines:
                formatted = [f"      {line['speaker']}: \"{line['line']}\"" for line in dialogue_lines]
                dialogue_block = "\n" + "\n".join(formatted)
            else:
                dialogue_block = (
                    f"    East: \"{entry['east_line']}\"\n"
                    f"    West: \"{entry['west_line']}\""
                )
            print(
                f"\n  Negotiation at step {self.steps} ({entry.get('trade_type')}):\n"
                f"{dialogue_block}\n"
                f"    Trade flows -> food East->West {entry['trade']['food_from_east_to_west']}, "
                f"wealth West->East {entry['trade']['wealth_from_west_to_east']}\n"
                f"      East after trade: food {_fmt_res(east_before['food'])}->{_fmt_res(east_after['food'])}, "
                f"wealth {_fmt_res(east_before['wealth'])}->{_fmt_res(east_after['wealth'])}\n"
                f"      West after trade: food {_fmt_res(west_before['food'])}->{_fmt_res(west_after['food'])}, "
                f"wealth {_fmt_res(west_before['wealth'])}->{_fmt_res(west_after['wealth'])}\n"
                f"      Relation now: {entry.get('relation_label')}"
            )

        wage_lines: list[str] = []
        for territory in ["West", "East"]:
            wages = self.current_step_log.get(territory, {}).get("wages", {})
            if wages:
                before_w = wages.get("before", {})
                after_w = wages.get("after", {})
                wage_lines.append(
                    f"  Wages {territory}: wealth {_fmt_res(before_w.get('wealth'))}->{_fmt_res(after_w.get('wealth'))}, "
                    f"mult {_fmt_res(before_w.get('multiplier'))}->{_fmt_res(after_w.get('multiplier'))}, "
                    f"on_strike={after_w.get('on_strike')}, unpaid_steps={after_w.get('unpaid_steps')}"
                )
        if wage_lines:
            print("")
            print("\n".join(wage_lines))

        print("")
        for idx, territory in enumerate(["West", "East"]):
            info = self.current_step_log.get(territory, {})
            upkeep = info.get("upkeep", {})
            if upkeep:
                before_u = upkeep.get("before", {})
                after_u = upkeep.get("after", {})
                req = (before_u.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
                prefix = "\n" if idx == 0 else ""
                print(
                    f"{prefix}  {territory} upkeep: food {_fmt_res(before_u.get('food'))}->{_fmt_res(after_u.get('food'))}, "
                    f"wealth {_fmt_res(before_u.get('wealth'))}->{_fmt_res(after_u.get('wealth'))}, "
                    f"wood {_fmt_res(before_u.get('wood'))}->{_fmt_res(after_u.get('wood'))}, "
                    f"infra {before_u.get('infrastructure_level')}->{after_u.get('infrastructure_level')}, "
                    f"pop {_fmt_pop(before_u.get('population'))}->{_fmt_pop(after_u.get('population'))}, "
                    f"required_food {_fmt_res(req)}"
                )
        print("\n" + "-" * 60 + "\n")

    def apply_wages(self, territory: TerritoryState) -> None:
        """I deduct wages and model morale strikes when wealth runs dry."""
        workers = max(0.0, territory.population / config.PEOPLE_PER_WORK_POINT)
        wage_per_worker = config.WAGE_PER_WORKER
        wage_bill = workers * wage_per_worker
        if wage_bill <= 0:
            territory.effective_work_multiplier = 1.0
            territory.unpaid_steps = 0
            territory.on_strike = False
            return

        if territory.wealth >= wage_bill:
            territory.wealth -= wage_bill
            territory.effective_work_multiplier = 1.0
            territory.unpaid_steps = 0.0
            territory.on_strike = False
            return

        amount_paid = max(0.0, min(territory.wealth, wage_bill))
        coverage = amount_paid / wage_bill if wage_bill > 0 else 0.0
        territory.wealth -= amount_paid
        territory.wealth = max(0.0, territory.wealth)

        shortfall = 1.0 - coverage
        territory.unpaid_steps = max(0.0, territory.unpaid_steps + shortfall)
        if coverage > 0.0:
            territory.unpaid_steps = max(
                0.0, territory.unpaid_steps - coverage * config.PARTIAL_PAY_RECOVERY
            )

        territory.on_strike = territory.unpaid_steps >= config.STRIKE_THRESHOLD_STEPS
        if territory.on_strike:
            territory.effective_work_multiplier = config.STRIKE_MULTIPLIER
        else:
            morale = config.LOW_MORALE_MULTIPLIER + coverage * (1.0 - config.LOW_MORALE_MULTIPLIER)
            territory.effective_work_multiplier = max(config.LOW_MORALE_MULTIPLIER, morale)

    def _apply_population_dynamics(self, territory: TerritoryState) -> None:
        """I now focus this on upkeep: food consumption, starvation, and simple growth."""
        required_food = (territory.population / 10.0) * config.FOOD_PER_10_POP
        territory.required_food = required_food
        if required_food <= 0:
            territory.food = max(0.0, territory.food)
            territory.wealth = max(0.0, territory.wealth)
            return

        if territory.food >= required_food:
            territory.food -= required_food
            growth = territory.population * config.POP_GROWTH_RATE
            territory.population += growth
        else:
            deficit = required_food - territory.food
            territory.food = 0.0
            loss_fraction = deficit * config.POP_LOSS_RATE_PER_MISSING_FOOD
            loss_fraction = max(0.0, min(loss_fraction, 0.9))
            if loss_fraction > 0.0 and territory.population > 0.0:
                territory.population *= (1.0 - loss_fraction)

        territory.food = max(0.0, territory.food)
        territory.wealth = max(0.0, territory.wealth)
        if territory.population < 0.0:
            territory.population = 0.0

    def run_negotiation(self) -> None:
        """I let the two leaders negotiate every couple of steps and capture the dialogue plus trade."""
        if self.llm_client is None or not self.llm_client.enabled:
            return

        state = {
            "step": self.steps,
            "east": {
                "food": self.east.food,
                "wealth": self.east.wealth,
                "population": self.east.population,
                "relation_to_neighbor": self.east.relation_to_neighbor,
                "relation_score": self.east.relation_score,
            },
            "west": {
                "food": self.west.food,
                "wealth": self.west.wealth,
                "population": self.west.population,
                "relation_to_neighbor": self.west.relation_to_neighbor,
                "relation_score": self.west.relation_score,
            },
            "last_actions": {
                "east": getattr(self.leader_east, "last_action", None),
                "west": getattr(self.leader_west, "last_action", None),
            },
            "east_history_text": summarise_memory_for_prompt(self.leader_east.memory_events),
            "west_history_text": summarise_memory_for_prompt(self.leader_west.memory_events),
            "east_interactions_text": "\n".join(self.leader_east.interaction_log[-5:]) or "No notable interactions recorded.",
            "west_interactions_text": "\n".join(self.leader_west.interaction_log[-5:]) or "No notable interactions recorded.",
        }
        decision = self.llm_client.negotiate(state)

        trade = decision.get("trade") or {}

        def _sanitise_flow(key: str) -> int:
            value = trade.get(key, 0)
            try:
                delta = int(value)
            except (ValueError, TypeError):
                delta = 0
            return max(-5, min(5, delta))

        food_flow = _sanitise_flow("food_from_east_to_west")
        wealth_flow = _sanitise_flow("wealth_from_west_to_east")

        east_before = {
            "food": self.east.food,
            "wealth": self.east.wealth,
            "population": self.east.population,
        }
        west_before = {
            "food": self.west.food,
            "wealth": self.west.wealth,
            "population": self.west.population,
        }

        if food_flow > 0:
            food_flow = min(food_flow, int(self.east.food))
        elif food_flow < 0:
            food_flow = max(food_flow, -int(self.west.food))

        if wealth_flow > 0:
            wealth_flow = min(wealth_flow, int(self.west.wealth))
        elif wealth_flow < 0:
            wealth_flow = max(wealth_flow, -int(self.east.wealth))

        if food_flow > 0:
            self.east.food -= food_flow
            self.west.food += food_flow
        elif food_flow < 0:
            amount = -food_flow
            self.west.food -= amount
            self.east.food += amount

        if wealth_flow > 0:
            self.west.wealth -= wealth_flow
            self.east.wealth += wealth_flow
        elif wealth_flow < 0:
            amount = -wealth_flow
            self.east.wealth -= amount
            self.west.wealth += amount

        self.east.food = max(0.0, self.east.food)
        self.west.food = max(0.0, self.west.food)
        self.east.wealth = max(0.0, self.east.wealth)
        self.west.wealth = max(0.0, self.west.wealth)

        east_after = {
            "food": self.east.food,
            "wealth": self.east.wealth,
            "population": self.east.population,
        }
        west_after = {
            "food": self.west.food,
            "wealth": self.west.wealth,
            "population": self.west.population,
        }

        units_food = abs(food_flow)
        eps = 1e-6
        trade_type = "no_trade"
        effective_price = None
        east_value = (east_after["food"] - east_before["food"]) + (east_after["wealth"] - east_before["wealth"])
        west_value = (west_after["food"] - west_before["food"]) + (west_after["wealth"] - west_before["wealth"])
        gift_classified = False
        if east_value < 0 and west_value > 0 and wealth_flow <= 0:
            trade_type = "gift_from_east"
            gift_classified = True
        elif west_value < 0 and east_value > 0 and wealth_flow >= 0:
            trade_type = "gift_from_west"
            gift_classified = True

        if not gift_classified and units_food > 0:
            if food_flow > 0:
                receiver_before = west_before
                receiver_required = (receiver_before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
                at_risk = receiver_before["food"] < 1.5 * receiver_required
            else:
                receiver_before = east_before
                receiver_required = (receiver_before.get("population", 0) / 10.0) * config.FOOD_PER_10_POP
                at_risk = receiver_before["food"] < 1.5 * receiver_required

            if abs(wealth_flow) > eps and (food_flow * wealth_flow) < 0:
                effective_price = abs(wealth_flow) / units_food
            else:
                effective_price = None

            if effective_price is None:
                if at_risk:
                    trade_type = "gift_from_east" if food_flow > 0 else "gift_from_west"
                else:
                    trade_type = "balanced_trade"
            else:
                FAIR_LOW = 0.5
                FAIR_HIGH = 1.5
                STRONG_EXPLOIT = 3.0
                if FAIR_LOW <= effective_price <= FAIR_HIGH:
                    trade_type = "balanced_trade"
                else:
                    victim = None
                    if east_value < 0 and west_value >= 0:
                        victim = "east"
                    elif west_value < 0 and east_value >= 0:
                        victim = "west"
                    elif east_value < 0 and west_value < 0:
                        victim = "east" if abs(east_value) >= abs(west_value) else "west"
                    else:
                        victim = "west" if food_flow > 0 else "east"
                    if effective_price >= STRONG_EXPLOIT:
                        trade_type = (
                            f"strongly_exploitative_for_{victim}"
                        )
                    else:
                        trade_type = f"mildly_exploitative_for_{victim}"

        score = self.east.relation_score
        if trade_type in ("gift_from_east", "gift_from_west"):
            score += 1
        elif trade_type == "balanced_trade" and score >= 0:
            score += 1
        elif trade_type.startswith("mildly_exploitative"):
            score -= 1
        elif trade_type.startswith("strongly_exploitative"):
            score -= 2
        score = max(-2, min(2, score))
        self.east.relation_score = score
        self.west.relation_score = score
        new_label = self._relation_label(score)
        self.east.relation_to_neighbor = new_label
        self.west.relation_to_neighbor = new_label

        trade_reason = str(trade.get("reason", "no trade reason provided")).strip() or "no trade reason provided"

        entry = {
            "event_type": "negotiation",
            "step": self.steps,
            "east_line": decision.get("east_line", ""),
            "west_line": decision.get("west_line", ""),
            "dialogue": decision.get("dialogue"),
            "trade": {
                "food_from_east_to_west": food_flow,
                "wealth_from_west_to_east": wealth_flow,
                "reason": trade_reason,
            },
            "food_east_before": east_before["food"],
            "food_east_after": east_after["food"],
            "wealth_east_before": east_before["wealth"],
            "wealth_east_after": east_after["wealth"],
            "food_west_before": west_before["food"],
            "food_west_after": west_after["food"],
            "wealth_west_before": west_before["wealth"],
            "wealth_west_after": west_after["wealth"],
            "population_east": self.east.population,
            "population_west": self.west.population,
            "trade_type": trade_type,
            "relation_score": score,
            "relation_label": new_label,
        }
        self.chronicle.append(entry)
        self.current_step_log["negotiation"] = {
            "entry": entry,
            "east_before": east_before,
            "east_after": east_after,
            "west_before": west_before,
            "west_after": west_after,
        }
        summary_base = f"Step {self.steps}: trade_type={trade_type}, relation={new_label}, flows(E→W food {food_flow}, W→E wealth {wealth_flow})"
        self.leader_east.record_interaction(
            f"{summary_base}; West said \"{entry.get('west_line', '')}\"."
        )
        self.leader_west.record_interaction(
            f"{summary_base}; East said \"{entry.get('east_line', '')}\"."
        )

    def step(self) -> None:
        """I advance the Mesa scheduler one tick so the leader agent can act."""
        # Mesa 3.x: super().step() increments self.steps
        super().step()
        self.current_step_log = {}
        print(f"Step {self.steps}:")
        season = self.current_season()
        next_season = self.next_season()
        season_mult = self.season_multipliers.get(season, 1.0)
        print(
            f"  World: season={season} (food/wood yield x{season_mult:.2f}), next={next_season}"
        )
        # I still ask Mesa to shuffle agents even though there is currently only one so the logic scales to councils later.
        self.agents.shuffle_do("step")
        if self.llm_client is not None and self.llm_client.enabled:
            self.run_negotiation()
        # I settle wages after production so the next turn's work multiplier reflects morale.
        for name, territory in (("East", self.east), ("West", self.west)):
            before_wages = {
                "wealth": territory.wealth,
                "multiplier": territory.effective_work_multiplier,
                "on_strike": territory.on_strike,
                "unpaid_steps": territory.unpaid_steps,
            }
            self.apply_wages(territory)
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
        self._apply_population_dynamics(self.east)
        self._apply_population_dynamics(self.west)
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
        self._print_step_summary()

    def all_territories_dead(self) -> bool:
        """I report whether every territory has collapsed so callers can stop the run early."""
        return self.east.population <= 0 and self.west.population <= 0

    def save_chronicle(self, path: Path) -> None:
        """I persist the chronicle as JSON to ``path`` so I can analyze runs later."""
        # I ensure the log directory exists before writing.
        path.parent.mkdir(parents=True, exist_ok=True)
        # I dump the structured chronicle to disk in a readable format.
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.chronicle, f, indent=2)
    def _relation_label(self, score: int) -> str:
        if score <= -2:
            return "hostile"
        if score == -1:
            return "strained"
        if score == 0:
            return "neutral"
        if score == 1:
            return "cordial"
        return "allied"
