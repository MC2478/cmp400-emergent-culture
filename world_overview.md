# Purpose

I simulate two asymmetric territories (East and West) to study emergent dynamics with an optional LM Studio-backed leader. Each territory tracks food, wealth, wood, population, infrastructure, and a shared diplomatic relationship score. Leaders either follow a rule-based fallback or query the LLM to pick actions, and every step is logged as JSON for later analysis.

# Modules and Responsibilities

- **main.py** — CLI entry point that builds `WorldModel`, advances it for the requested steps (or until collapse), and saves/prints summaries.
- **src/model/world_model.py** — Mesa model wiring that orchestrates phases (actions → negotiation → wages → upkeep → logging) using helper modules.
- **src/model/economy.py** — Wages, morale/strike handling, and population/upkeep helpers.
- **src/model/diplomacy.py** — Negotiation orchestration, trade clamping/classification, relation scoring/labels, interaction logging.
- **src/model/log_utils.py** — Chronicle append helpers and console summary formatting (ASCII bullets).
- **src/agents/leader.py** — `TerritoryState` and `LeaderAgent`. A leader inspects state, optionally calls the LLM, applies the action (allocations + build flag), and records memory.
- **src/agents/production.py** — Work point/yield calculation and allocation application.
- **src/model/llm_client.py** — LM Studio HTTP wrapper that calls prompt builders, parses JSON responses for both decisions and negotiations, and falls back safely.
- **src/model/prompt_builder.py** / **parsers.py** — Prompt assembly and JSON coercion/validation utilities shared by the LLM client.
- **config.py** — Central numeric knobs (starting resources, yields, infrastructure multipliers/costs, wages, population rules, rounding, seasons).

# Resources and State

- **Food** — Produced via work allocations and season-adjusted yields; consumed every upkeep step.
- **Wealth** — Earned via work allocations and trade; spent on wages, infrastructure, and buffers.
- **Wood** — Gathered via `focus_wood`; consumed (with wealth) to raise infrastructure.
- **Population** — Drives work points. When food covers the quota, population grows by `POP_GROWTH_RATE`; deficits shrink it by `POP_LOSS_RATE_PER_MISSING_FOOD` per missing food unit (capped at 90% loss per tick).
- **Infrastructure level (`infra`)** — Multiplies food/wood/wealth yields per work point. Raising infra costs `INFRA_COST_WOOD` wood and `INFRA_COST_WEALTH` wealth.
- **Relation score/label** — Shared diplomatic status between East and West, adjusted after each negotiation.
- **Seasons** — Spring → summer → autumn → winter; each season lasts two steps and modifies food/wood yields via `SEASON_MULTIPLIERS`.

# Actions

- **Work allocations (`focus_food`, `focus_wood`, `focus_wealth`)** — Leaders split fractional work points across these options in a single step. Shares are clamped to [0, 1]; if they sum above 1.0 they are normalised to 1.0. Unused share idles. Each share uses infra- and season-adjusted yields, so mixes are allowed (e.g., 0.6 food + 0.4 wealth).
- **`build_infrastructure` flag** — Independent of the work split. If `INFRA_COST_WOOD` wood and `INFRA_COST_WEALTH` wealth are on hand, the upgrade happens immediately; otherwise the log records the shortfall and only production applies.
- **`wait`** — Implicit when no allocations/build are provided.

Infrastructure bonuses are multiplicative: each level adds `INFRA_*_YIELD_MULT_PER_LEVEL` to the respective yields.

# Turn Structure (per step)

1. **Actions** — Both leaders act (LLM-first, heuristic fallback) using the current season multipliers, yields, morale, and priority hint.
2. **Negotiation (LLM only)** — Runs every step when the LLM is enabled. A short dialogue (up to three exchanges per side) plus trade flows are generated, clamped to available resources, classified (gift/balanced/exploitative), and used to update the shared relation score/label.
3. **Wages & morale** — Wage bill per worker (`WAGE_PER_WORKER`) is deducted. Partial payment accumulates `unpaid_steps`; reaching `STRIKE_THRESHOLD_STEPS` puts workers on strike with `STRIKE_MULTIPLIER` output. Partial payments soften debt and morale uses `LOW_MORALE_MULTIPLIER` when underpaid.
4. **Upkeep & population** — Food requirement is `(population / 10) * FOOD_PER_10_POP`. Paying it consumes food and grows population by `POP_GROWTH_RATE`; deficits consume all food and cut population by `deficit * POP_LOSS_RATE_PER_MISSING_FOOD` (max 90% per step). Wealth/wood are floored at zero.
5. **Logging** — Action (allocations, infra attempt), negotiation, wages, and upkeep are appended to the chronicle and printed with `POP_DISPLAY_DECIMALS` / `RESOURCE_DISPLAY_DECIMALS` rounding.
6. **Termination** — `main.py` stops early if `WorldModel.all_territories_dead()` reports both populations at zero.

# Leader Memory & Adaptation

- Each leader keeps a capped in-run memory of recent steps (action, food/wealth/pop deltas, starvation, strikes, notes) plus a small interaction log of past negotiations for prompts.
- After every action the LLM writes a short directive for its future self; that text seeds the next prompt so the leader can build on its own plan.
- Decision and negotiation prompts embed the summarised history, latest directive, recent interactions, and the priority hint.
- Nothing persists between runs; memory resets when a new `WorldModel` is constructed. Survival depends on on-run history plus current metrics.

# Decision & Priority Heuristics

Before querying the LLM, the leader computes a `priority_hint`:

- `food_safety_ratio`: current food divided by the requirement over the next `FOOD_SAFETY_HORIZON_STEPS`.
- Suggested weights over `{survive, resilience, prosperity}` encourage at least `NON_FOOD_MIN_FRACTION` toward non-food goals once food is safe, and tilt toward prosperity after `FOOD_SAFETY_GOOD_RATIO`.

If the LLM returns malformed JSON or an unknown action, I fall back to a simple heuristic: focus on food if `food_safety_ratio < 1`, otherwise build infrastructure when affordable and infra < 3, otherwise focus on wealth. Valid LLM decisions are never overridden.

# LLM vs. Heuristic Control

- The LLM is the primary decision-maker for actions and negotiations. Prompts include territory history, current metrics, relation status, the last self-authored directive, interaction log, and reminders that collapse is final.
- Negotiation uses two calls: first to draft an alternating dialogue, then to settle a trade consistent with that transcript. Failed or malformed responses fall back to a no-trade default.
- If HTTP fails or the LLM output cannot be parsed, warnings are logged and the rule-based action/negotiation fallbacks take over for that step.

# Key Tunable Parameters (config.py)

- `STARTING_FOOD`, `STARTING_WEALTH`, `STARTING_WOOD`, `STARTING_POPULATION`, `STARTING_INFRASTRUCTURE_LEVEL` — Initial conditions (defaults: food 3, wealth 5, wood 0, population 100, infra 0).
- `EAST_*_YIELD` / `WEST_*_YIELD` — Per-territory yield multipliers (East is food-heavy at 2.0/0.5/1.0; West is wealth-heavy at 1.0/1.5/1.0).
- `PEOPLE_PER_WORK_POINT` — People per work point (200). Work points are fractional and scaled by morale.
- `FOOD_PER_10_POP`, `POP_GROWTH_RATE`, `POP_LOSS_RATE_PER_MISSING_FOOD` — Food requirement and percentage growth/loss rules (0.05 food per 10 pop; +/-10% growth/loss factor per unit deficit, capped at 90% loss per step).
- `FOOD_PER_WORK_BASE`, `WOOD_PER_WORK_BASE`, `WEALTH_PER_WORK_BASE`, `INFRA_*_YIELD_MULT_PER_LEVEL` — Base yields and infra bonuses per level (food/wood +0.10, wealth +0.05).
- `INFRA_COST_WOOD`, `INFRA_COST_WEALTH` — Build costs (5 wood, 2 wealth).
- `FOOD_SAFETY_HORIZON_STEPS`, `FOOD_SAFETY_GOOD_RATIO`, `NON_FOOD_MIN_FRACTION` — Parameters driving the priority hint shared with the LLM.
- `WAGE_PER_WORKER`, `STRIKE_THRESHOLD_STEPS`, `LOW_MORALE_MULTIPLIER`, `STRIKE_MULTIPLIER`, `PARTIAL_PAY_RECOVERY` — Wage per worker (0.1), unpaid debt threshold (4 steps) for strikes, and morale multipliers.
- `POP_DISPLAY_DECIMALS`, `RESOURCE_DISPLAY_DECIMALS` — Console rounding for population/resources.
- `SEASONS`, `SEASON_MULTIPLIERS` — Order of seasons with two-step duration and per-resource multipliers (spring 1.0, summer 1.2, autumn 0.8, winter 0.4 for food/wood).
- `MAX_LEADER_MEMORY_EVENTS` — Maximum within-run events each leader retains for prompts.
