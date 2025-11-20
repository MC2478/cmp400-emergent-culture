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

# Food/wood/wealth yield per work point before infra bonuses
FOOD_PER_WORK_BASE: float = 1.0
WOOD_PER_WORK_BASE: float = 1.0
WEALTH_PER_WORK_BASE: float = 1.0

# Multiplicative infra bonus per level
INFRA_FOOD_YIELD_MULT_PER_LEVEL: float = 0.10
INFRA_WOOD_YIELD_MULT_PER_LEVEL: float = 0.10
INFRA_WEALTH_YIELD_MULT_PER_LEVEL: float = 0.05

# Infrastructure build costs
INFRA_COST_WOOD: float = 5.0
INFRA_COST_WEALTH: float = 2.0

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
