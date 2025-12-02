"""Quick card: production helpers to keep LeaderAgent lean."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import config

if TYPE_CHECKING:
    from src.agents.leader import TerritoryState


def work_points(territory: TerritoryState) -> float:
    """Work points cue: turn population into work capacity, factoring morale."""
    base = max(0.0, territory.population / config.PEOPLE_PER_WORK_POINT)
    return base * max(0.0, territory.effective_work_multiplier)


def effective_yields(territory: TerritoryState) -> Dict[str, float]:
    """Yield cue: per-work outputs after infrastructure bonuses."""
    level = territory.infrastructure_level
    food_mult = 1.0 + level * config.INFRA_FOOD_YIELD_MULT_PER_LEVEL
    wood_mult = 1.0 + level * config.INFRA_WOOD_YIELD_MULT_PER_LEVEL
    wealth_mult = 1.0 + level * config.INFRA_WEALTH_YIELD_MULT_PER_LEVEL
    iron_mult = 1.0 + level * config.INFRA_IRON_YIELD_MULT_PER_LEVEL
    gold_mult = 1.0 + level * config.INFRA_GOLD_YIELD_MULT_PER_LEVEL
    return {
        "food_per_work": territory.food_yield * config.FOOD_PER_WORK_BASE * food_mult,
        "wood_per_work": territory.wood_yield * config.WOOD_PER_WORK_BASE * wood_mult,
        "wealth_per_work": territory.wealth_yield * config.WEALTH_PER_WORK_BASE * wealth_mult,
        "iron_per_work": territory.iron_yield * config.IRON_PER_WORK_BASE * iron_mult,
        "gold_per_work": territory.gold_yield * config.GOLD_PER_WORK_BASE * gold_mult,
    }


def apply_allocations(
    territory: TerritoryState,
    allocations: Dict[str, float],
    season: str,
    season_multipliers: Dict[str, float],
) -> Dict[str, float]:
    """Production card: apply allocations for the season and return produced amounts."""
    wp = work_points(territory)
    yields = effective_yields(territory)
    season_mult = season_multipliers.get(season, 1.0)
    food_yield = yields["food_per_work"] * season_mult
    wood_yield = yields["wood_per_work"] * season_mult
    wealth_yield = yields["wealth_per_work"]
    iron_yield = yields["iron_per_work"]
    gold_yield = yields["gold_per_work"]

    produced: Dict[str, float] = {
        "focus_food": 0.0,
        "focus_wood": 0.0,
        "focus_wealth": 0.0,
        "focus_iron": 0.0,
        "focus_gold": 0.0,
    }
    for key, share in allocations.items():
        share = max(0.0, min(1.0, share))
        if share == 0.0:
            continue
        if key == "focus_food":
            delta = wp * share * food_yield
            territory.food += delta
            produced[key] += delta
        elif key == "focus_wood":
            delta = wp * share * wood_yield
            territory.wood += delta
            produced[key] += delta
        elif key == "focus_wealth":
            delta = wp * share * wealth_yield
            territory.wealth += delta
            produced[key] += delta
        elif key == "focus_iron":
            delta = wp * share * iron_yield
            territory.iron += delta
            produced[key] += delta
        elif key == "focus_gold":
            delta = wp * share * gold_yield
            territory.gold += delta
            produced[key] += delta
    return produced
