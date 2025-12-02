"""Quick card: wage, morale, and population upkeep helpers."""

from __future__ import annotations

import config
from src.agents.leader import TerritoryState


def apply_wages(territory: TerritoryState) -> None:
    """Wages cue: pay workers, then set morale/strike based on how much you covered."""
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


def apply_population_dynamics(territory: TerritoryState) -> None:
    """Upkeep cue: consume food, grow if fed, cut population on starvation, clamp resources."""
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
