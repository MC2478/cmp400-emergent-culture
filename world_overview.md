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

- **Work allocations (`focus_food`, `focus_wood`, `focus_wealth`)** — The leader now splits fractional work points across these options in a single step. Shares are between 0.0 and 1.0 and need not sum to 1.0; any remainder idles. Each share taps the infra- and season-adjusted yields, so a territory can, for example, devote 0.6 work points to food and 0.4 to wealth simultaneously.
- **`build_infrastructure` flag** — Independently of the work split, the leader may request an infrastructure build. If `INFRA_COST_WOOD` wood and `INFRA_COST_WEALTH` wealth are on hand, the upgrade happens immediately; otherwise the log records the shortfall and the turn continues with just the production split.
- **`wait`** — Implicit when no allocations are provided and no infrastructure attempt is made.

Infrastructure bonuses are multiplicative, so each level adds `INFRA_*_YIELD_MULT_PER_LEVEL` to the respective yields.

# Turn Structure (per step)

1. **Action selection** – Each leader snapshots its state, builds a payload for the LLM (including effective yields and hints), and either uses the LLM decision or a fallback action.
2. **Negotiation (LLM only)** — If the LLM client is enabled, I simulate a multi-turn dialogue/trade (up to three exchanges per side). The JSON transcript drives resource transfers, trade classification, and relation adjustments.
3. **Wages** – After production, I deduct wages per worker (`WAGE_PER_WORKER`). Paying resets morale; skipping two steps in a row puts workers on strike and drops the effective work multiplier.
4. **Upkeep** – Each step requires `(population / 10) * FOOD_PER_10_POP` food. Meeting the quota consumes that food and grows population by `POP_GROWTH_RATE`. Falling short consumes the remaining food, computes a deficit, and reduces population by `deficit * POP_LOSS_RATE_PER_MISSING_FOOD`, capped at 90% loss in a single tick.
5. **Logging** — I append action (now including allocation splits), negotiation, wage, and upkeep entries to the chronicle and print a matching step summary (rounded using `POP_DISPLAY_DECIMALS` and `RESOURCE_DISPLAY_DECIMALS`).
6. **Termination** – After each step, `main.py` checks `WorldModel.all_territories_dead()`, stopping early once both populations hit zero.

# Leader Memory & Adaptation

- Each leader maintains a capped, in-run memory of recent steps (action taken, food/wealth/population changes, starvation, strikes, and notes).
- After every action the LLM also writes a short directive for its future self; that text is stored and prepended to the next prompt so the leader can build on its own plan rather than relying solely on external hints.
- A short diplomatic interaction log captures recent negotiation outcomes (quotes plus resource flows) so each LLM remembers how the other side behaved; the latest entries are injected into subsequent prompts.
- The summarised history plus the last directive feed into every LLM decision and negotiation prompt so the model can reason about consequences and adjust strategy mid-run.
- Nothing is persisted between runs—memory resets when a new `WorldModel` is constructed.
- There is no global safety net; the LLM must weigh survival against growth using only the on-run history plus current metrics.

# Decision & Priority Heuristics

Before querying the LLM I compute a `priority_hint` that includes:

- `food_safety_ratio`: current food divided by the requirement over the next `FOOD_SAFETY_HORIZON_STEPS`.
- Suggested weights over `{survive, resilience, prosperity}` that encourage at least `NON_FOOD_MIN_FRACTION` of attention to non-food goals once food is safe, and more prosperity once the ratio exceeds `FOOD_SAFETY_GOOD_RATIO`.

The hint is included in the LLM payload as guidance, not as a mandate. If the LLM returns malformed JSON or an unknown action, I fall back to a simple heuristic: focus on food if the ratio < 1, otherwise build infrastructure when affordable and infra < 3, otherwise focus on wealth. Valid LLM decisions are never overridden.

# LLM vs. Heuristic Control

- The LLM is the primary decision-maker for both regular actions and negotiations. Prompts now include territory history, current metrics, relation status, the leader’s previous self-authored directive, and explicit reminders that collapse is final so that the agent can take calculated risks when needed.
- Negotiation prompts request a short transcript plus the trade, so the model can probe and counter within a single response; the entire dialogue is logged for review.
- If the HTTP call fails or the LLM output cannot be parsed, I log a warning and fall back to the simple heuristic action logic described above.
- The heuristic never fires when the LLM responds with a valid action, and it includes no hidden long-horizon rescue rules—leaders live or die by their choices.

# Key Tunable Parameters (config.py)

- `STARTING_FOOD`, `STARTING_WEALTH`, `STARTING_WOOD`, `STARTING_POPULATION`, `STARTING_INFRASTRUCTURE_LEVEL` – Initial conditions for both territories.
- `EAST_*_YIELD` / `WEST_*_YIELD` – Base per-territory multipliers applied to the generic yields.
- `PEOPLE_PER_WORK_POINT` — How many people contribute a single work point; currently 200, and work points are fractional (each person contributes 0.005).
- `FOOD_PER_10_POP`, `POP_GROWTH_RATE`, `POP_LOSS_RATE_PER_MISSING_FOOD` – Food requirement and the percentage-based growth/loss when supply is above/below the requirement.
- `FOOD_PER_WORK_BASE`, `WOOD_PER_WORK_BASE`, `WEALTH_PER_WORK_BASE`, `INFRA_*_YIELD_MULT_PER_LEVEL` – Base yields per work point and how much infrastructure amplifies them.
- `INFRA_COST_WOOD`, `INFRA_COST_WEALTH` – Resource costs for each infrastructure upgrade.
- `FOOD_SAFETY_HORIZON_STEPS`, `FOOD_SAFETY_GOOD_RATIO`, `NON_FOOD_MIN_FRACTION` – Parameters that drive the priority hint exposed to (and optionally ignored by) the LLM.
- `WAGE_PER_WORKER`, `STRIKE_THRESHOLD_STEPS`, `LOW_MORALE_MULTIPLIER`, `STRIKE_MULTIPLIER` – Wage bill per worker, how many missed payments trigger a strike, and the resulting effect on work output.
- `POP_DISPLAY_DECIMALS`, `RESOURCE_DISPLAY_DECIMALS` – How console logs round population and resource values.
- `SEASONS`, `SEASON_MULTIPLIERS` — Order of the seasons and yield multipliers applied to food/wood; each season now lasts two steps before advancing.
- `MAX_LEADER_MEMORY_EVENTS` – Maximum within-run events each leader keeps for LLM prompts (older entries drop first).
