"""I centralise the main numeric knobs for the CMP400 simulation so I can tweak balance easily."""

# Territory starting resources
STARTING_FOOD: float = 3.0
STARTING_WEALTH: float = 5.0
STARTING_WOOD: float = 0.0
STARTING_POPULATION: int = 100
STARTING_INFRASTRUCTURE_LEVEL: int = 0

# Territory-specific yields per work point
EAST_FOOD_YIELD: float = 2.0
EAST_WEALTH_YIELD: float = 0.5
EAST_WOOD_YIELD: float = 1.0

WEST_FOOD_YIELD: float = 1.0
WEST_WEALTH_YIELD: float = 1.5
WEST_WOOD_YIELD: float = 1.0

# --- Infrastructure & yields ---

# Food/wood/wealth/iron/gold yield per work point before infra bonuses
FOOD_PER_WORK_BASE: float = 1.0
WOOD_PER_WORK_BASE: float = 1.1
WEALTH_PER_WORK_BASE: float = 1.0
IRON_PER_WORK_BASE: float = 1.0
GOLD_PER_WORK_BASE: float = 0.7

# Multiplicative infra bonus per level (each point adds +10%)
INFRA_FOOD_YIELD_MULT_PER_LEVEL: float = 0.10
INFRA_WOOD_YIELD_MULT_PER_LEVEL: float = 0.10
INFRA_WEALTH_YIELD_MULT_PER_LEVEL: float = 0.10
INFRA_IRON_YIELD_MULT_PER_LEVEL: float = 0.10
INFRA_GOLD_YIELD_MULT_PER_LEVEL: float = 0.10

# Tiered infrastructure costs (+10% per point)
INFRA_TIER_WOOD_WOOD_COST: float = 5.0
INFRA_TIER_WOOD_WEALTH_COST: float = 2.0
INFRA_TIER_WOOD_POINTS: int = 1

INFRA_TIER_IRON_IRON_COST: float = 5.0
INFRA_TIER_IRON_WEALTH_COST: float = 5.0
INFRA_TIER_IRON_POINTS: int = 2

INFRA_TIER_GOLD_GOLD_COST: float = 5.0
INFRA_TIER_GOLD_IRON_COST: float = 5.0
INFRA_TIER_GOLD_POINTS: int = 3

# Population and upkeep
PEOPLE_PER_WORK_POINT: int = 200
# New percentage-based knobs
FOOD_PER_10_POP: float = 0.05
POP_GROWTH_RATE: float = 0.10
POP_LOSS_RATE_PER_MISSING_FOOD: float = 0.05

# --- Priority heuristics ---
FOOD_SAFETY_HORIZON_STEPS: int = 3
FOOD_SAFETY_GOOD_RATIO: float = 1.2
NON_FOOD_MIN_FRACTION: float = 0.2

# --- Leader memory / history ---
# I cap within-run memory per leader so prompts stay short but still reflect recent outcomes.
MAX_LEADER_MEMORY_EVENTS: int = 30

# Wages and morale
WAGE_PER_WORKER: float = 0.1
STRIKE_THRESHOLD_STEPS: int = 4
STRIKE_MULTIPLIER: float = 0.5
LOW_MORALE_MULTIPLIER: float = 0.9
PARTIAL_PAY_RECOVERY: float = 0.5

# --- Rounding / display ---
POP_DISPLAY_DECIMALS: int = 0
RESOURCE_DISPLAY_DECIMALS: int = 2

# Seasons
SEASONS = ["spring", "summer", "autumn", "winter"]
SEASON_MULTIPLIERS = {
    "spring": 1.0,
    "summer": 1.2,
    "autumn": 0.8,
    "winter": 0.4,
}

# Trait and personality tuning
TRAIT_MAX_ACTIVE: int = 2
TRAIT_ADAPTATION_ALPHA: float = 0.7
TRAIT_COOLDOWN_STEPS: int = 5
TRAIT_CHANGE_PRESSURE_THRESHOLD: int = 2
TRAIT_SOFT_ADJUST_DELTA: float = 0.1
TRAIT_NEUTRAL_VALUE: float = 0.5

# Environment generation bounds (seed-driven, within modest caps)
ENV_TOTAL_FOOD_YIELD_MIN: float = 2.5
ENV_TOTAL_FOOD_YIELD_MAX: float = 3.5
ENV_TOTAL_WEALTH_YIELD_MIN: float = 1.7
ENV_TOTAL_WEALTH_YIELD_MAX: float = 2.3
ENV_TOTAL_WOOD_YIELD_MIN: float = 1.7
ENV_TOTAL_WOOD_YIELD_MAX: float = 2.3
ENV_TOTAL_IRON_YIELD_MIN: float = 0.5
ENV_TOTAL_IRON_YIELD_MAX: float = 2.0
ENV_TOTAL_GOLD_YIELD_MIN: float = 0.3
ENV_TOTAL_GOLD_YIELD_MAX: float = 1.5
ENV_STARTING_FOOD_RANGE: tuple[float, float] = (2.0, 4.5)
ENV_STARTING_WEALTH_RANGE: tuple[float, float] = (4.0, 6.5)
ENV_STARTING_WOOD_RANGE: tuple[float, float] = (0.0, 2.0)
ENV_STARTING_IRON_RANGE: tuple[float, float] = (0.0, 1.5)
ENV_STARTING_GOLD_RANGE: tuple[float, float] = (0.0, 1.0)
