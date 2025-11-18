"""I now model two symmetric leaders (East and West) with different resource yields so the LLM can
contrast their strategic choices while I capture both actions and negotiations for the demo."""

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
        # I now set up East and West with different yields so they feel like distinct actors.
        self.east = TerritoryState(
            name="East",
            food=float(initial_food),
            wealth=5.0,
            relation_to_neighbor="neutral",
            population=100,
            food_yield=1.5,
            wealth_yield=0.5,
        )
        self.west = TerritoryState(
            name="West",
            food=float(initial_food),
            wealth=5.0,
            relation_to_neighbor="neutral",
            population=100,
            food_yield=0.5,
            wealth_yield=1.5,
        )
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
            "food_before": before.get("food"),
            "food_after": after.get("food"),
            "wealth_before": before.get("wealth"),
            "wealth_after": after.get("wealth"),
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
        }
        self.chronicle.append(entry)

    def _append_chronicle_upkeep(self, east_before, east_after, west_before, west_after) -> None:
        entry = {
            "event_type": "upkeep",
            "step": self.steps,
            "east": {
                "food_before": east_before["food"],
                "food_after": east_after["food"],
                "population_before": east_before["population"],
                "population_after": east_after["population"],
            },
            "west": {
                "food_before": west_before["food"],
                "food_after": west_after["food"],
                "population_before": west_before["population"],
                "population_after": west_after["population"],
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
                reason = decision.get("decision", {}).get("reason", "no reason provided")
                action = decision.get("decision", {}).get("action")
                print(
                    f"  {territory}: action={action} (LLM: {used_llm})\n"
                    f"    Resources: food {before.get('food')}->{after.get('food')}, wealth {before.get('wealth')}->{after.get('wealth')}\n"
                    f"    Pop: {before.get('population')}\n"
                    f"    Reason: {reason}"
                )
                self._append_chronicle_action(territory, info)

        negotiation_info = self.current_step_log.get("negotiation")
        if negotiation_info:
            entry = negotiation_info["entry"]
            east_before = negotiation_info["east_before"]
            east_after = negotiation_info["east_after"]
            west_before = negotiation_info["west_before"]
            west_after = negotiation_info["west_after"]
            print(
                f"\n  Negotiation at step {self.steps} ({entry.get('trade_type')}):\n"
                f"    East: \"{entry['east_line']}\"\n"
                f"    West: \"{entry['west_line']}\"\n"
                f"    Trade flows -> food East->West {entry['trade']['food_from_east_to_west']}, "
                f"wealth West->East {entry['trade']['wealth_from_west_to_east']}\n"
                f"      East after trade: food {east_before['food']}->{east_after['food']}, "
                f"wealth {east_before['wealth']}->{east_after['wealth']}\n"
                f"      West after trade: food {west_before['food']}->{west_after['food']}, "
                f"wealth {west_before['wealth']}->{west_after['wealth']}\n"
                f"      Relation now: {entry.get('relation_label')}"
            )

        for idx, territory in enumerate(["West", "East"]):
            info = self.current_step_log.get(territory, {})
            upkeep = info.get("upkeep", {})
            if upkeep:
                before_u = upkeep.get("before", {})
                after_u = upkeep.get("after", {})
                req = max(1, int(before_u.get("population", 0)) // 100)
                prefix = "\n" if idx == 0 else ""
                print(
                    f"{prefix}  {territory} upkeep: food {before_u.get('food')}->{after_u.get('food')}, "
                    f"pop {before_u.get('population')}->{after_u.get('population')}, required_food {req}"
                )
        print("\n" + "-" * 60 + "\n")

    def _apply_population_dynamics(self, territory: TerritoryState) -> None:
        """I now focus this on upkeep: food consumption, starvation, and simple growth."""
        required_food = max(1, territory.population // 100)
        starvation = 0
        if territory.food >= required_food:
            territory.food -= required_food
        else:
            deficit = required_food - territory.food
            territory.food = 0
            starvation = deficit * 100
            territory.population = max(0, territory.population - starvation)
        territory.food = max(0, territory.food)
        territory.wealth = max(0, territory.wealth)
        if starvation > 0:
            return
        food_after = territory.food
        big_surplus_threshold = 2 * required_food
        if food_after >= big_surplus_threshold:
            territory.population += 100
        elif self.random.random() < 0.5:
            territory.population += 100

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

        east_before = {"food": self.east.food, "wealth": self.east.wealth}
        west_before = {"food": self.west.food, "wealth": self.west.wealth}

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

        east_after = {"food": self.east.food, "wealth": self.east.wealth}
        west_after = {"food": self.west.food, "wealth": self.west.wealth}

        units_food = abs(food_flow)
        eps = 1e-6
        trade_type = "no_trade"
        effective_price = None

        if units_food > 0:
            if food_flow > 0:
                receiver_before = west_before
                receiver_required = max(1, int(west_before.get("population", 0)) // 100)
                at_risk = west_before["food"] < 1.5 * receiver_required
            else:
                receiver_before = east_before
                receiver_required = max(1, int(east_before.get("population", 0)) // 100)
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
                    if effective_price >= STRONG_EXPLOIT:
                        trade_type = (
                            "strongly_exploitative_for_west"
                            if food_flow > 0
                            else "strongly_exploitative_for_east"
                        )
                    else:
                        trade_type = (
                            "mildly_exploitative_for_west"
                            if food_flow > 0
                            else "mildly_exploitative_for_east"
                        )

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

    def step(self) -> None:
        """I advance the Mesa scheduler one tick so the leader agent can act."""
        # Mesa 3.x: super().step() increments self.steps
        super().step()
        self.current_step_log = {}
        print(f"Step {self.steps}:")
        # I still ask Mesa to shuffle agents even though there is currently only one so the logic scales to councils later.
        self.agents.shuffle_do("step")
        if self.llm_client is not None and self.llm_client.enabled:
            self.run_negotiation()
        east_before = {"food": self.east.food, "population": self.east.population}
        west_before = {"food": self.west.food, "population": self.west.population}
        self._apply_population_dynamics(self.east)
        self._apply_population_dynamics(self.west)
        east_after = {"food": self.east.food, "population": self.east.population}
        west_after = {"food": self.west.food, "population": self.west.population}
        self.log_upkeep(east_before, east_after, west_before, west_after)
        self._print_step_summary()

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
