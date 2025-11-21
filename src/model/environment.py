"""Seed-driven environment generation for asymmetric territories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import config
from src.model.traits import classify_environment


@dataclass
class TerritoryEnvironment:
    """I bundle environment-derived yields, starting resources, and metrics for one territory."""

    food_yield: float
    wealth_yield: float
    wood_yield: float
    iron_yield: float
    gold_yield: float
    starting_food: float
    starting_wealth: float
    starting_wood: float
    starting_iron: float
    starting_gold: float
    metrics: Dict[str, float]
    category: str


@dataclass
class EnvironmentSnapshot:
    """I hold both sides' generated environments so I can trace deterministic seeds."""

    east: TerritoryEnvironment
    west: TerritoryEnvironment
    iron_holder: str
    gold_holder: str


def _starting_amount(range_pair: tuple[float, float], rng: Any) -> float:
    low, high = range_pair
    return rng.uniform(low, high)


def _metrics(food_yield: float, wealth_yield: float, wood_yield: float, iron_yield: float, gold_yield: float) -> Dict[str, float]:
    """I convert raw yields into relative richness scores."""
    baseline_food = config.WORLD_MAX_FOOD_YIELD / 2.0
    baseline_wealth = config.WORLD_MAX_WEALTH_YIELD / 2.0
    baseline_wood = config.WORLD_MAX_WOOD_YIELD / 2.0
    food_richness = food_yield / baseline_food if baseline_food else 1.0
    wealth_richness = wealth_yield / baseline_wealth if baseline_wealth else 1.0
    wood_richness = wood_yield / baseline_wood if baseline_wood else 1.0
    scarcity = food_richness < 0.9 and wealth_richness < 0.9
    metrics = {
        "env_food_richness": food_richness,
        "env_wealth_richness": wealth_richness,
        "env_wood_richness": wood_richness,
        "env_scarcity": scarcity,
    }
    metrics["env_iron_yield"] = iron_yield
    metrics["env_gold_yield"] = gold_yield
    return metrics


def _split_yield(total: float, rng: Any) -> tuple[float, float]:
    """Split a total yield between East/West while avoiding extreme imbalance."""
    split = rng.uniform(0.1, 0.9)
    east_value = total * split
    west_value = total - east_value
    return east_value, west_value


def generate_environment(rng: Any) -> EnvironmentSnapshot:
    """I produce deterministic yields/resources per seed within the configured world caps."""
    east_food_yield, west_food_yield = _split_yield(config.WORLD_MAX_FOOD_YIELD, rng)
    east_wealth_yield, west_wealth_yield = _split_yield(config.WORLD_MAX_WEALTH_YIELD, rng)
    east_wood_yield, west_wood_yield = _split_yield(config.WORLD_MAX_WOOD_YIELD, rng)
    east_iron_yield, west_iron_yield = _split_yield(config.WORLD_MAX_IRON_YIELD, rng)
    east_gold_yield, west_gold_yield = _split_yield(config.WORLD_MAX_GOLD_YIELD, rng)

    iron_holder = "East" if east_iron_yield >= west_iron_yield else "West"
    gold_holder = "East" if east_gold_yield >= west_gold_yield else "West"

    east_metrics = _metrics(east_food_yield, east_wealth_yield, east_wood_yield, east_iron_yield, east_gold_yield)
    west_metrics = _metrics(west_food_yield, west_wealth_yield, west_wood_yield, west_iron_yield, west_gold_yield)

    east_starting_iron = _starting_amount(config.ENV_STARTING_IRON_RANGE, rng) if east_iron_yield > 0 else 0.0
    west_starting_iron = _starting_amount(config.ENV_STARTING_IRON_RANGE, rng) if west_iron_yield > 0 else 0.0
    east_starting_gold = _starting_amount(config.ENV_STARTING_GOLD_RANGE, rng) if east_gold_yield > 0 else 0.0
    west_starting_gold = _starting_amount(config.ENV_STARTING_GOLD_RANGE, rng) if west_gold_yield > 0 else 0.0

    east = TerritoryEnvironment(
        food_yield=east_food_yield,
        wealth_yield=east_wealth_yield,
        wood_yield=east_wood_yield,
        iron_yield=east_iron_yield,
        gold_yield=east_gold_yield,
        starting_food=_starting_amount(config.ENV_STARTING_FOOD_RANGE, rng),
        starting_wealth=_starting_amount(config.ENV_STARTING_WEALTH_RANGE, rng),
        starting_wood=_starting_amount(config.ENV_STARTING_WOOD_RANGE, rng),
        starting_iron=east_starting_iron,
        starting_gold=east_starting_gold,
        metrics=east_metrics,
        category=classify_environment(east_metrics),
    )
    west = TerritoryEnvironment(
        food_yield=west_food_yield,
        wealth_yield=west_wealth_yield,
        wood_yield=west_wood_yield,
        iron_yield=west_iron_yield,
        gold_yield=west_gold_yield,
        starting_food=_starting_amount(config.ENV_STARTING_FOOD_RANGE, rng),
        starting_wealth=_starting_amount(config.ENV_STARTING_WEALTH_RANGE, rng),
        starting_wood=_starting_amount(config.ENV_STARTING_WOOD_RANGE, rng),
        starting_iron=west_starting_iron,
        starting_gold=west_starting_gold,
        metrics=west_metrics,
        category=classify_environment(west_metrics),
    )
    return EnvironmentSnapshot(east=east, west=west, iron_holder=iron_holder, gold_holder=gold_holder)
