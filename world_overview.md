# Purpose

I run a tiny two-territory world (East and West) so I can explore emergent dynamics with an optional LM Studio-backed leader. Each territory tracks food, wealth, wood, population, infrastructure, and a diplomatic relationship score. The leaders either follow rule-based fallbacks or call the LLM to pick one of the allowed actions, and I log every step as JSON for later analysis.

# Modules and Responsibilities

- **main.py** – I act as the CLI entry point. I build `WorldModel`, advance it for the configured number of steps (or until everyone collapses), trigger logging, and print summaries.
- **src/model/world_model.py** – I hold the Mesa model, territory state, and world rules (production, wages, upkeep, relations, negotiations). I also expose helpers to save chronicles and emit configuration summaries.
- **src/agents/leader.py** – I define `TerritoryState` and `LeaderAgent`. A leader inspects its state, optionally calls the LLM, and applies the resulting action using the effective yields and heuristic hint.
- **src/model/llm_client.py** – I wrap LM Studio's OpenAI-style HTTP API, turning state dicts into prompts and parsing the JSON-only responses (both decisions and optional negotiation snippets).
- **config.py** – I centralise all the numeric knobs (starting resources, yields, infrastructure multipliers, wages, population rules, rounding, seasons) so I can tweak balance without digging through gameplay logic.

# Resources and State

- **Food** – Consumed every upkeep step; produced via work allocations and season-adjusted yields.
- **Wealth** – Earned via work allocations and trade; spent on wages, infrastructure, or resilience buffers.
- **Wood** – Gathered via `focus_wood`, stored for future use, and consumed (with wealth) to raise infrastructure.
- **Population** – Drives work point availability. When food meets the quota, population grows by `POP_GROWTH_RATE`; deficits shrink it by `POP_LOSS_RATE_PER_MISSING_FOOD` per missing food unit (capped at 90% loss per tick).
- **Infrastructure level (`infra`)** – A non-consumed state variable that multiplies food, wood, and wealth yields per work point. Raising `infra` costs `INFRA_COST_WOOD` wood and `INFRA_COST_WEALTH` wealth.
- **Relation score/label** – Shared diplomatic status between East and West, influenced by negotiation outcomes.
- **Seasons** – Cycle spring → summer → autumn → winter, modifying food/wood yields via multipliers.

# Actions

- **`focus_food`** – Spend all work points on food using the infra- and season-adjusted `food_per_work` yield.
- **`focus_wood`** – Spend all work points on wood using the infra- and season-adjusted `wood_per_work` yield.
- **`focus_wealth`** – Spend all work points on wealth using the infra-adjusted `wealth_per_work` yield (wealth isn’t season-sensitive).
- **`build_infrastructure`** – If both `INFRA_COST_WOOD` wood and `INFRA_COST_WEALTH` wealth are available, consume them to raise `infra` by 1 (otherwise the action degenerates into `wait`).
- **`wait`** – Do nothing; useful when pooling resources or after a failed infra attempt.

Infrastructure bonuses are multiplicative, so each level adds `INFRA_*_YIELD_MULT_PER_LEVEL` to the respective yields.

# Turn Structure (per step)

1. **Action selection** – Each leader snapshots its state, builds a payload for the LLM (including effective yields and hints), and either uses the LLM decision or a fallback action.
2. **Negotiation (LLM only)** – If the LLM client is enabled, I simulate a dialogue/trade. The JSON response drives resource transfers, trade classification, and relation adjustments.
3. **Wages** – After production, I deduct wages per worker (`WAGE_PER_WORKER`). Paying resets morale; skipping two steps in a row puts workers on strike and drops the effective work multiplier.
4. **Upkeep** – Each step requires `(population / 10) * FOOD_PER_10_POP` food. Meeting the quota consumes that food and grows population by `POP_GROWTH_RATE`. Falling short consumes the remaining food, computes a deficit, and reduces population by `deficit * POP_LOSS_RATE_PER_MISSING_FOOD`, capped at 90% loss in a single tick.
5. **Logging** – I append action, negotiation, wage, and upkeep entries to the chronicle and print a matching step summary (rounded using `POP_DISPLAY_DECIMALS` and `RESOURCE_DISPLAY_DECIMALS`).
6. **Termination** – After each step, `main.py` checks `WorldModel.all_territories_dead()`, stopping early once both populations hit zero.

# Decision & Priority Heuristics

Before querying the LLM I compute a `priority_hint` that includes:

- `food_safety_ratio`: current food divided by the requirement over the next `FOOD_SAFETY_HORIZON_STEPS`.
- Suggested weights over `{survive, resilience, prosperity}` that encourage at least `NON_FOOD_MIN_FRACTION` of attention to non-food goals once food is safe, and more prosperity once the ratio exceeds `FOOD_SAFETY_GOOD_RATIO`.

The hint is included in the LLM payload as guidance, not as a mandate. If the LLM returns malformed JSON or an unknown action, I fall back to a simple heuristic: focus on food if the ratio < 1, otherwise build infrastructure when affordable and infra < 3, otherwise focus on wealth. Valid LLM decisions are never overridden.

# Key Tunable Parameters (config.py)

- `STARTING_FOOD`, `STARTING_WEALTH`, `STARTING_WOOD`, `STARTING_POPULATION`, `STARTING_INFRASTRUCTURE_LEVEL` – Initial conditions for both territories.
- `EAST_*_YIELD` / `WEST_*_YIELD` – Base per-territory multipliers applied to the generic yields.
- `PEOPLE_PER_WORK_POINT` – How many people contribute a single work point; currently 100.
- `FOOD_PER_10_POP`, `POP_GROWTH_RATE`, `POP_LOSS_RATE_PER_MISSING_FOOD` – Food requirement and the percentage-based growth/loss when supply is above/below the requirement.
- `FOOD_PER_WORK_BASE`, `WOOD_PER_WORK_BASE`, `WEALTH_PER_WORK_BASE`, `INFRA_*_YIELD_MULT_PER_LEVEL` – Base yields per work point and how much infrastructure amplifies them.
- `INFRA_COST_WOOD`, `INFRA_COST_WEALTH` – Resource costs for each infrastructure upgrade.
- `FOOD_SAFETY_HORIZON_STEPS`, `FOOD_SAFETY_GOOD_RATIO`, `NON_FOOD_MIN_FRACTION` – Parameters that drive the priority hint exposed to (and optionally ignored by) the LLM.
- `WAGE_PER_WORKER`, `STRIKE_THRESHOLD_STEPS`, `LOW_MORALE_MULTIPLIER`, `STRIKE_MULTIPLIER` – Wage bill per worker, how many missed payments trigger a strike, and the resulting effect on work output.
- `POP_DISPLAY_DECIMALS`, `RESOURCE_DISPLAY_DECIMALS` – How console logs round population and resource values.
- `SEASONS`, `SEASON_MULTIPLIERS` – Order of the seasons and yield multipliers applied to food/wood each turn.
