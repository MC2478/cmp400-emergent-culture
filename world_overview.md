# Purpose

I simulate two asymmetric territories (East and West) to study emergent diplomatic dynamics with an optional LM Studio-backed leader. Each run wires Mesa-based agents into the shared `WorldModel`, and I can drive it through the CLI (`main.py`) or an interactive Solara dashboard (`run.py`). Both surfaces expose the same knobs (steps, seed, starting stocks, LLM toggle) and stream every tick into structured logs so I can audit behaviour later.

# Modules and Responsibilities

- **main.py** - CLI entry point wrapped around a `RunConfig`. It prompts for steps/seed, tees stdout/stderr into `logs/run_*`, builds `WorldModel`, opens per-leader JSONL logs plus a config summary, advances until the requested steps (or collapse), saves the chronicle, and calls `summarize_chronicle()` to report event counts and min/mean/max food/population stats.
- **run.py** - Solara dashboard with sliders for initial food/wealth/wood/iron/gold, population, LLM toggle, and seed. `RunArtifactManager` mirrors the CLI artefacts under `logs/webui_*` (chronicle, config summary, params, agent-state logs, `simulation_output.log`, and `resource_history.csv` exported from the Mesa `DataCollector`), while the UI can queue multiple steps, stop mid-run, or reset the model without restarting the server.
- **src/model/world_model.py** - Mesa model that sequences leader actions, diplomacy, wages, upkeep, logging, and datacollection. It tracks `current_step_log`, exposes helpers such as `enable_agent_state_logging()`, `log_agent_state()`, the various `record_*()` buffers, `save_config_summary()`, and `save_chronicle()`, and maintains a `DataCollector` that captures wealth/food for both sides so dashboard charts stay in sync.
- **src/model/environment.py** - Seed-driven generator that splits each world-level yield between East/West (bounded to 10-90% of the cap), applies jittered starting stocks, records territory richness metrics, labels each side (`food_rich`, `wealth_rich`, `scarce`, `balanced`), and tags which territory holds the larger iron/gold share for logging.
- **src/model/economy.py** - Implements `apply_wages()` (wage coverage, morale multiplier, strikes, `unpaid_steps`) and `apply_population_dynamics()` (food consumption, growth/loss, clamping) so upkeep rules stay isolated.
- **src/model/diplomacy.py** - Negotiation orchestration. Builds a `NegotiationSession`, synthesises/overrides intents, auto-triggers emergency asks when food horizons fall, enforces safety (token gifts, fairness classification, exploitation streaks, `MAX_UNRECIPROCATED_GIFTS`, intent compliance), runs multi-turn LLM dialogues, applies flows, updates relation score/labels, and logs detailed chronicle entries plus leader interaction notes.
- **src/model/log_utils.py** - Chronicle helpers (`append_chronicle_action`, `append_chronicle_upkeep`) and the verbose console summary (per-territory tables, wages/upkeep, negotiation transcript) used for both CLI and dashboard runs.
- **src/agents/leader.py** - `TerritoryState` and `LeaderAgent`. Each leader keeps memory events, long-term notes, interaction logs, trade ledgers, gift streak counters, negotiation intent, directives, and trait state. `step()` ticks trait cooldowns, applies pressure adaptations, builds the prompt payload, optionally calls the LLM, applies fallback heuristics, executes allocations/builds, records memories, and exposes negotiation intent overrides for diplomacy.
- **src/agents/production.py** - Pure helpers for work points (population ÷ `PEOPLE_PER_WORK_POINT` scaled by morale), infra-adjusted per-work yields, and applying allocations per current season multipliers.
- **src/model/llm_client.py** - LM Studio HTTP wrapper (`LLMConfig` defaults: `http://127.0.0.1:1234/v1/chat/completions`, `meta-llama-3.1-8b-instruct`, temperature 0.1, `max_tokens=800`, timeout 6s). Provides decision (`decide`), multi-turn negotiation (`negotiate_turn`), and fallback helpers with request retries, JSON sanitising, and graceful degradation (once a request fails the client disables network access and immediately returns heuristic fallbacks).
- **src/model/prompt_builder.py** / **parsers.py** - Prompt assembly for decisions plus both the negotiation context and per-turn prompts, along with JSON coercion utilities (`parse_json_response`, `sanitise_allocations`, `extract_action_hint`). Prompts include trade ledgers, gift balance notes, long-term milestones, neighbour directives/reasons, trait pressure cues, resource pressures, and explicit negotiation intent slots.
- **src/model/traits.py** - Trait catalogue (Aggressive, Friendly, Wealth-hoarder, Food-secure, Opportunistic, Isolationist) plus blending, cooldowns, adaptation pressure, JSONL logging, environment classification, and text interpreters for LLM trait adjustments. Supplies helpers such as `personality_summary()`, `trait_gloss()`, and `negotiation_style_line()`.
- **config.py** - Central place for numeric knobs: starting resources, world yield caps, base yields, infra multipliers/costs, heuristics, wages/morale, rounding, seasons, memory caps, trade guardrails, trait tuning, environment ranges, and optional diplomacy settings (e.g., `DIPLOMACY_BALANCE_INTERVAL`).

# Run Artefacts & Logging

- The CLI uses `_choose_run_artifact_path()` to create timestamped folders under `logs/run_*`. Each contains the chronicle JSON, config summary, tee'd console output (`output_log_seed...`), per-territory agent-state JSONL files, and additional chronicle copies if the same seed/steps are rerun.
- The Solara dashboard mirrors those artefacts under `logs/webui_*` via `RunArtifactManager`: for every run it stores `chronicle.json`, `config_summary.txt`, `params.json`, `starting_settings.txt`, both agent-state logs, `simulation_output.log`, and `resource_history.csv` (exported from `model.datacollector` for Altair charts).
- `WorldModel.save_config_summary()` captures environment richness, iron/gold holders, per-territory starting metrics, action set, wage settings, season multipliers, and infra tier costs/bonuses so each artefact bundles behaviour plus knobs.
- `enable_agent_state_logging()` opens `<territory>_agent_state.jsonl` handles and `log_agent_state()` writes one JSON line per step containing resources, infra level, trait/personality state, streak counters, decision metadata (allocations, infra attempts, trait adjustments, LLM flag), and meta info (next directive, last trait text, recent interactions, last memory entry).
- `current_step_log` buffers per-step decisions, wage/upkeep snapshots, and negotiation info; `log_utils.print_step_summary()` prints the recap and calls `append_chronicle_action()` / `append_chronicle_upkeep()` / negotiation logging so the chronicle records actions, negotiations, wage/upkeep outcomes, trait events, relation deltas, and stance notes.
- Every step the Mesa `DataCollector` grabs `wealth_E/W` and `food_E/W`, which the dashboard writes to CSV for the trend charts. The CLI's `summarize_chronicle()` reloads the chronicle at the end of a run to report total events plus min/mean/max population and food per territory.

# Resources and State

- **Food** - Produced via seasonal work allocations and consumed during upkeep. `TerritoryState.required_food` stores the current per-step need for summaries and prompts.
- **Wealth** - Generated via work/trade. Pays wages, infra upgrades, and token gifts; morale drops and strikes occur when wages go unpaid.
- **Wood** - Produced via `focus_wood`, consumed (with wealth) for the wood-tier infra upgrade, and included in trade ledgers plus negotiation reasoning.
- **Iron & Gold** - Yield shares come from the seed split (both territories retain a non-zero share, but the larger holder is flagged in config summaries). Higher tiers consume these metals, and they can be traded during diplomacy.
- **Population** - Drives work points. Upkeep uses `POP_GROWTH_RATE` on surplus and removes up to 90% per tick using `POP_LOSS_RATE_PER_MISSING_FOOD` when food deficits occur.
- **Infrastructure level** - Adds +10% per point to every per-work yield. Leaders always buy the highest affordable tier (wood +1, iron +2, gold +3 points).
- **Relation score/label** - Shared float [-2,2] plus a label (`hostile`...`allied`). Diplomacy recalculates it each negotiation based on trade classification, a logistic acceptance probability, and both personality vectors.
- **Seasons** - Deterministic two-step cycle (`spring`, `summer`, `autumn`, `winter`) with configurable multipliers that only affect food/wood yields.
- **Gift streaks & trade ledgers** - `TerritoryState` stores cumulative food/wealth/wood sent/received plus how many consecutive gifts each side accepted without reciprocating; diplomacy blocks additional aid past `MAX_UNRECIPROCATED_GIFTS` until balance is restored.
- **Morale/strikes** - Wage coverage updates `effective_work_multiplier`, `unpaid_steps`, and `on_strike`; these stats are logged and surfaced in memory notes and prompts.
- **Long-term notes** - Every `LONG_TERM_SUMMARY_INTERVAL` steps each territory appends a milestone summary so prompts reference longer-term outcomes.

# Actions

- **Work allocations (`focus_food`, `focus_wood`, `focus_wealth`, `focus_iron`, `focus_gold`)** - Leaders supply fractional shares (clamped to [0,1]; any sum >1 is normalised). Mesa production converts shares into resource deltas using current infrastructure, morale, and the current season multiplier.
- **`build_infrastructure` flag** - Independent of allocations. When true `LeaderAgent` buys the highest affordable tier (gold -> iron -> wood). Costs are wood tier (5 wood + 2 wealth, +1 infra), iron tier (5 iron + 5 wealth, +2), gold tier (5 gold + 5 iron, +3). `_maybe_force_infrastructure()` can flip the flag on automatically when buffers look healthy even if the LLM demurs.
- **`negotiation_intent` payload** - Each decision includes (or inherits) an intent dict describing whether to initiate dialogue, which resource/amount to request, any reciprocal offer, urgency, and reason. Rule-based intents are recomputed every step from food safety and infra gaps; the LLM can override them by returning its own intent JSON, and diplomacy falls back to exploratory chats when both sides stay silent.
- **`wait`** - Implicit when no allocations are provided and infra is not attempted; the logger still records the rationale and trait context so stalls are explainable.

# Turn Structure (per step)

1. **Leader decisions & trait pressure** - `LeaderAgent.step()` ticks trait cooldowns, runs `apply_pressure_adaptation()` when streak thresholds fire, updates `adaptation_pressure_note`, builds the prompt payload (state snapshot, priority hint, trait info, gift ledger, neighbour summary, long-term notes, resource pressures), queries the LLM when enabled, or falls back to the heuristic plan (prioritise food, gather wood/wealth, attempt infra when safe). Decisions set allocations, build flag, trait adjustments, directives, and negotiation intent overrides.
2. **Negotiation (LLM-only)** - `run_negotiation()` guarantees a live LLM client, synthesises intents (auto-triggering emergency food requests and periodic balance checks via `DIPLOMACY_BALANCE_INTERVAL`), and schedules exploratory dialogue when nobody initiates. Negotiations run as multi-turn sessions (`NEGOTIATION_MAX_TURNS`) using `LLMDecisionClient.negotiate_turn()` with alternating speakers, explicit accept/counter/decline moves, stall detection, and fallback replies when proposals repeat. Accepted proposals pass through safety clamps (respect available stock, enforce food horizon safety, detect token gift language, block further gifts past `MAX_UNRECIPROCATED_GIFTS`, and verify promised compensation). Each trade is classified (`gift`, `balanced`, `mildly_exploitative`, `strongly_exploitative`), exploitation streaks update trait pressures, relation deltas are modulated by personality vectors, and leaders' interaction logs capture the recap.
3. **Wages & morale** - After production, `apply_wages()` deducts the wage bill, updates morale multipliers, increments `unpaid_steps`, toggles strikes, and writes the before/after snapshot for the wage table.
4. **Upkeep & population** - `apply_population_dynamics()` consumes food, grows population on surplus, applies loss fractions when deficits occur, and clamps resources at zero. `WorldModel` records pre/post states, updates starvation streaks, and calls `_update_strategy_pressure()` so repeated allocation mixes without gains push the stagnation streak.
5. **Logging & metrics** - `log_upkeep()` and `_record_leader_memories()` add chronicle entries and per-leader memory events (notes highlight starvation flags, strikes, failed infra attempts, last reasons). `_refresh_trait_state_for_logging()` refreshes trait context before `print_step_summary()` emits resource tables, negotiation transcripts (dialogue lines, proposals, acceptance rolls, relation updates), wage/morale rows, and upkeep tables. The datacollector grabs the new point for charts.
6. **Termination** - `WorldModel.all_territories_dead()` stops the run once both populations hit zero; the CLI and dashboard detect this and prompt for reset.

# Seeded Environment & Starting Traits

- `generate_environment()` uses the Mesa RNG to split every world-max yield with a 10-90% bound so both territories have some of each resource but asymmetric strengths. Starting food/wealth/wood/iron/gold stocks jitter within the configured ranges; CLI/dashboard overrides (e.g., `initial_food`) replace both sides' starting values for quick experiments.
- Each territory gets a `TerritoryEnvironment` snapshot (yields, starting stocks, richness metrics, category, iron/gold share holder). When `WorldModel` builds `TerritoryState`, it copies these values, sets relation labels to neutral, and records iron/gold holders in the config summary.
- `sample_starting_trait()` consumes the shared Mesa RNG to pick an initial trait consistent with the environment label (e.g., wealth-rich leans `Wealth-hoarder`, scarce worlds can roll `Aggressive`/`Isolationist`). Traits added at step 0 are logged so chronicle readers know each leader's initial style.

# Personality & Trait System

- Personality vectors span seven dimensions in [0,1] with a neutral baseline of 0.5 (`aggression`, `cooperation`, `wealth_focus`, `food_focus`, `risk_tolerance`, `trust_in_others`, `adaptability`). Traits blend into the current vector via `TRAIT_ADAPTATION_ALPHA`, and incompatibilities (`Friendly` vs `Aggressive`/`Isolationist`) are enforced.
- `TRAIT_MAX_ACTIVE` caps active traits, `TRAIT_COOLDOWN_STEPS` prevents rapid flip-flops, and `apply_pressure_adaptation()` can auto-remove/add traits and nudge dimensions when exploitation/starvation/stagnation streaks match `TRAIT_CHANGE_PRESSURE_THRESHOLD`.
- Trait events are mirrored into the chronicle, stored on the territory, and surfaced in console summaries. `adaptation_pressure_text()` explains why the next prompt might push for a trait change.
- LLM decisions return `"trait_adjustment"` text that `interpret_trait_adjustment()` parses into add/remove/nudge actions; `LeaderAgent` applies them (respecting cooldown) and logs each event.

# Leader Memory, Logs & Negotiation Intent

- `record_step_outcome()` builds per-step memory events (before/after food, wealth, pop, action, notes), keeps only `MAX_LEADER_MEMORY_EVENTS`, and every `LONG_TERM_SUMMARY_INTERVAL` steps appends a milestone summary to `territory.long_term_notes` (capped by `MAX_LONG_TERM_NOTES`). These notes feed into prompts so the LLM remembers longer arcs.
- `interaction_log` stores the last few negotiation recaps per leader; prompts include both the local history and the neighbour's recent interactions.
- The trade ledger (`food_sent/received`, `wealth_sent/received`, `wood_sent/received`, net deltas) plus `gift_streak_received` feed into prompts via `_gift_balance_note()` so the LLM sees reciprocity pressure.
- `LeaderAgent._compute_negotiation_intent()` inspects food horizons, infra gaps, neighbour ratios, and wealth gaps to propose concrete trade asks; `_apply_negotiation_override()` swaps in any LLM-specified intent. Diplomacy also injects intents when food horizons fall below 1.0 or when periodic balance checks demand it.
- `log_agent_state()` captures resources, infra, trait state, pressure counters, decision metadata, and meta notes (next directive, last trait text, recent interactions) so every step has a replayable snapshot.

# Decision & Priority Heuristics

- The `priority_hint` object (food safety ratio across `FOOD_SAFETY_HORIZON_STEPS`, suggested weights across survive/resilience/prosperity) is computed each step and included in prompts so the LLM gets a quick summary. It also drives the heuristic fallback plan.
- The heuristic decision focuses on food when safety <1, gathers wood/wealth otherwise, and attempts infrastructure when buffers look good. `_maybe_force_infrastructure()` ensures obvious upgrades happen even when the LLM forgets.
- `_update_starvation_pressure()` increments starvation streaks before upkeep when food coverage falls short; `_update_strategy_pressure()` compares current vs previous allocations/pop/wealth to detect stagnation (difference sum <0.1 and no gains) and increments `failed_strategy_streak`, feeding trait pressure text.

# LLM vs. Heuristic Control

- `LLMDecisionClient.decide()` builds the prompt via `compose_prompt()`, calls LM Studio (OpenAI-compatible API), retries once with stricter instructions when JSON fails, and, if parsing still fails, looks for action hints before falling back to the heuristic plan. Once a network error occurs the client disables further requests until the run restarts.
- Negotiations use two prompt builders: `compose_negotiation_context()` for the overall summary and `compose_negotiation_turn_prompt()` for each alternating reply. `negotiate_turn()` returns JSON with `reply`, `decision` (`counter|accept|decline`), and a concrete proposal (food/wealth/wood/iron/gold flows plus reason).
- `run_negotiation()` loops through turns (respecting `NEGOTIATION_MAX_TURNS`, stall counters, and counter limits), enforces proposal changes, and, when no deal emerges, records `no_agreement` / `declined` outcomes with reasoning. Acceptance probability blends fairness ratios, food safety, relation label, personality, and trade magnitude; the resulting relation delta is logged.
- When the LLM is disabled or produces invalid output, both decisions and negotiations fall back to deterministic heuristics so the simulation always continues.

# Key Tunable Parameters (config.py)

- **Starting state** - `STARTING_FOOD`, `STARTING_WEALTH`, `STARTING_WOOD`, `STARTING_POPULATION`, `STARTING_INFRASTRUCTURE_LEVEL`.
- **World yield caps** - `WORLD_MAX_*_YIELD` per resource; the environment generator splits these between East/West.
- **Production & infrastructure** - `FOOD_PER_WORK_BASE`, `WOOD_PER_WORK_BASE`, `WEALTH_PER_WORK_BASE`, `IRON_PER_WORK_BASE`, `GOLD_PER_WORK_BASE`, infra multipliers (`INFRA_*_YIELD_MULT_PER_LEVEL`), and tier costs/points (`INFRA_TIER_*` constants).
- **Population & food** - `PEOPLE_PER_WORK_POINT`, `FOOD_PER_10_POP`, `POP_GROWTH_RATE`, `POP_LOSS_RATE_PER_MISSING_FOOD`.
- **Priority heuristics** - `FOOD_SAFETY_HORIZON_STEPS`, `FOOD_SAFETY_GOOD_RATIO`, `NON_FOOD_MIN_FRACTION`, `MAX_LEADER_MEMORY_EVENTS`.
- **Wages & morale** - `WAGE_PER_WORKER`, `STRIKE_THRESHOLD_STEPS`, `LOW_MORALE_MULTIPLIER`, `STRIKE_MULTIPLIER`, `PARTIAL_PAY_RECOVERY`.
- **Display/logging** - `POP_DISPLAY_DECIMALS`, `RESOURCE_DISPLAY_DECIMALS`, `LONG_TERM_SUMMARY_INTERVAL`, `MAX_LONG_TERM_NOTES`.
- **Diplomacy guardrails** - `MAX_UNRECIPROCATED_GIFTS`, `NEGOTIATION_MAX_TURNS`, optional `DIPLOMACY_BALANCE_INTERVAL` (defaults to 4 if omitted).
- **Seasons** - `SEASONS` order and `SEASON_MULTIPLIERS`.
- **Trait tuning** - `TRAIT_MAX_ACTIVE`, `TRAIT_ADAPTATION_ALPHA`, `TRAIT_COOLDOWN_STEPS`, `TRAIT_CHANGE_PRESSURE_THRESHOLD`, `TRAIT_SOFT_ADJUST_DELTA`, `TRAIT_NEUTRAL_VALUE`.
- **Environment ranges** - `ENV_STARTING_*_RANGE` for food/wealth/wood/iron/gold jitter envelopes used by the environment generator.

Together these descriptions match the current code, so readers can jump from the overview to the implementation without surprises.
