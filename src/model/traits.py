"""Trait catalogue and helpers for personality-driven leader behaviour."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import config

PERSONALITY_DIMENSIONS: Sequence[str] = (
    "aggression",
    "cooperation",
    "wealth_focus",
    "food_focus",
    "risk_tolerance",
    "trust_in_others",
    "adaptability",
)


@dataclass(frozen=True)
class Trait:
    """I keep a personality vector, prompt snippet, and incompatibility list together."""

    name: str
    vector: Dict[str, float]
    description: str
    incompatible_with: set[str]


def neutral_personality_vector() -> Dict[str, float]:
    """I return a fresh neutral personality vector so callers avoid mutable defaults."""
    return {dim: float(config.TRAIT_NEUTRAL_VALUE) for dim in PERSONALITY_DIMENSIONS}


TRAIT_CATALOGUE: Dict[str, Trait] = {
    "Aggressive": Trait(
        name="Aggressive",
        vector={
            "aggression": 0.85,
            "cooperation": 0.25,
            "trust_in_others": 0.25,
            "risk_tolerance": 0.7,
        },
        description="You push for self-favouring deals and accept damaged relations for near-term gain.",
        incompatible_with={"Friendly"},
    ),
    "Friendly": Trait(
        name="Friendly",
        vector={
            "aggression": 0.2,
            "cooperation": 0.8,
            "trust_in_others": 0.7,
            "risk_tolerance": 0.45,
        },
        description="You prefer balanced, mutually beneficial outcomes and avoid extreme exploitation.",
        incompatible_with={"Aggressive", "Isolationist"},
    ),
    "Wealth-hoarder": Trait(
        name="Wealth-hoarder",
        vector={
            "wealth_focus": 0.85,
            "risk_tolerance": 0.35,
            "trust_in_others": 0.45,
        },
        description="You want to maintain a strong wealth buffer and dislike spending or gifting it.",
        incompatible_with=set(),
    ),
    "Food-secure": Trait(
        name="Food-secure",
        vector={
            "food_focus": 0.85,
            "risk_tolerance": 0.3,
            "cooperation": 0.55,
        },
        description="You guard food above all else and avoid risky expansion that endangers supplies.",
        incompatible_with=set(),
    ),
    "Opportunistic": Trait(
        name="Opportunistic",
        vector={
            "risk_tolerance": 0.8,
            "adaptability": 0.8,
            "cooperation": 0.5,
            "trust_in_others": 0.5,
        },
        description="You adjust quickly based on recent outcomes and exploit openings when they appear.",
        incompatible_with=set(),
    ),
    "Isolationist": Trait(
        name="Isolationist",
        vector={
            "cooperation": 0.25,
            "trust_in_others": 0.25,
            "aggression": 0.45,
        },
        description="You prefer minimal trade and only engage when it is clearly beneficial or desperate.",
        incompatible_with={"Friendly"},
    ),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _with_defaults(vector: Dict[str, float] | None) -> Dict[str, float]:
    current = neutral_personality_vector()
    if not vector:
        return current
    for dim, value in vector.items():
        if dim in current:
            try:
                current[dim] = _clamp01(float(value))
            except (TypeError, ValueError):
                continue
    return current


def blend_personality_vector(current: Dict[str, float], trait_vector: Dict[str, float], alpha: float | None = None) -> Dict[str, float]:
    """I blend the given trait vector into the current map with the provided alpha."""
    alpha = config.TRAIT_ADAPTATION_ALPHA if alpha is None else alpha
    existing = _with_defaults(current)
    trait_values = _with_defaults(trait_vector)
    blended: Dict[str, float] = {}
    for dim in PERSONALITY_DIMENSIONS:
        blended[dim] = _clamp01(existing.get(dim, config.TRAIT_NEUTRAL_VALUE) * (1 - alpha) + trait_values.get(dim, config.TRAIT_NEUTRAL_VALUE) * alpha)
    return blended


def tick_trait_cooldown(state: Any) -> None:
    """I decrement cooldown timers safely."""
    if getattr(state, "trait_cooldown_steps", 0) > 0:
        state.trait_cooldown_steps = max(0, int(getattr(state, "trait_cooldown_steps")) - 1)


def _record_event(state: Any, event: Dict[str, Any]) -> Dict[str, Any]:
    """I stash trait events both for per-step logging and long-run history."""
    history: List[Dict[str, Any]] = getattr(state, "trait_history", [])
    history.append(event)
    if len(history) > 50:
        history.pop(0)
    state.trait_history = history
    step_events: List[Dict[str, Any]] = getattr(state, "trait_events", [])
    step_events.append(event)
    state.trait_events = step_events
    return event


def add_trait_to_state(state: Any, trait_name: str, *, step: int | None = None, reason: str | None = None) -> Dict[str, Any] | None:
    """I add a trait if allowed, resolving incompatibilities and enforcing the cap."""
    trait = TRAIT_CATALOGUE.get(trait_name)
    if trait is None:
        return None

    active: List[str] = list(getattr(state, "active_traits", []))
    removed: list[str] = []
    for existing in list(active):
        if existing in trait.incompatible_with:
            active.remove(existing)
            removed.append(existing)

    if len(active) >= config.TRAIT_MAX_ACTIVE:
        removed.append(active.pop(0))

    active.append(trait.name)
    state.active_traits = active[: config.TRAIT_MAX_ACTIVE]
    state.personality_vector = blend_personality_vector(getattr(state, "personality_vector", None), trait.vector)
    state.trait_cooldown_steps = config.TRAIT_COOLDOWN_STEPS
    event = {
        "step": step,
        "event": "added_trait",
        "trait": trait.name,
        "reason": reason or "",
        "removed": removed,
    }
    return _record_event(state, event)


def remove_trait_from_state(state: Any, trait_name: str, *, step: int | None = None, reason: str | None = None) -> Dict[str, Any] | None:
    """I remove a trait and ease relevant dimensions back toward neutral."""
    active: List[str] = list(getattr(state, "active_traits", []))
    if trait_name not in active:
        return None
    active = [t for t in active if t != trait_name]
    state.active_traits = active
    trait = TRAIT_CATALOGUE.get(trait_name)
    if trait:
        state.personality_vector = blend_personality_vector(
            getattr(state, "personality_vector", None),
            {dim: config.TRAIT_NEUTRAL_VALUE for dim in trait.vector},
            alpha=config.TRAIT_ADAPTATION_ALPHA * 0.5,
        )
    state.trait_cooldown_steps = config.TRAIT_COOLDOWN_STEPS
    event = {
        "step": step,
        "event": "removed_trait",
        "trait": trait_name,
        "reason": reason or "",
    }
    return _record_event(state, event)


def nudge_personality(state: Any, dimension: str, delta: float, *, step: int | None = None, reason: str | None = None) -> Dict[str, Any] | None:
    """I nudge a specific dimension directly for softer adjustments."""
    if dimension not in PERSONALITY_DIMENSIONS:
        return None
    current = _with_defaults(getattr(state, "personality_vector", None))
    current[dimension] = _clamp01(current.get(dimension, config.TRAIT_NEUTRAL_VALUE) + delta)
    state.personality_vector = current
    state.trait_cooldown_steps = config.TRAIT_COOLDOWN_STEPS
    event = {
        "step": step,
        "event": "nudge_dimension",
        "dimension": dimension,
        "delta": delta,
        "reason": reason or "",
    }
    return _record_event(state, event)


def apply_trait_actions(state: Any, actions: Iterable[Dict[str, Any]], *, step: int | None = None, reason_prefix: str = "") -> List[Dict[str, Any]]:
    """I apply parsed trait actions, respecting cooldown and caps."""
    events: list[Dict[str, Any]] = []
    if getattr(state, "trait_cooldown_steps", 0) > 0:
        return events

    for action in actions:
        action_type = action.get("type")
        if action_type == "add":
            trait_name = action.get("name")
            if trait_name:
                ev = add_trait_to_state(state, trait_name, step=step, reason=reason_prefix)
                if ev:
                    events.append(ev)
        elif action_type == "remove":
            trait_name = action.get("name")
            if trait_name:
                ev = remove_trait_from_state(state, trait_name, step=step, reason=reason_prefix)
                if ev:
                    events.append(ev)
        elif action_type == "nudge":
            dim = action.get("dimension")
            delta = float(action.get("delta", 0.0))
            ev = nudge_personality(state, dim, delta, step=step, reason=reason_prefix)
            if ev:
                events.append(ev)
    return events


def interpret_trait_adjustment(text: str) -> List[Dict[str, Any]]:
    """I parse LLM free-text into structured trait/nudge actions."""
    lower = (text or "").strip().lower()
    if not lower or "no change" in lower:
        return []

    actions: list[Dict[str, Any]] = []
    for trait_name in TRAIT_CATALOGUE:
        key = trait_name.lower()
        if key in lower:
            if any(marker in lower for marker in (f"less {key}", f"not so {key}", f"drop {key}", f"without {key}", f"reduce {key}")):
                actions.append({"type": "remove", "name": trait_name})
            else:
                actions.append({"type": "add", "name": trait_name})

    dimension_hints = {
        "risk_tolerance": (("more risk", "bolder", "riskier", "take risks"), ("less risk", "cautious", "safer", "avoid risk")),
        "food_focus": (("food first", "focus food", "secure food", "protect food"), ("ignore food", "less food")),
        "wealth_focus": (("wealth", "money", "gold", "hoard wealth"), ("spend wealth", "less wealth")),
        "cooperation": (("cooperate", "cooperation", "work together", "ally", "friendly"), ("less cooperation", "stop cooperating", "stand alone")),
        "trust_in_others": (("trust", "rely on them"), ("distrust", "suspicious", "less trusting")),
        "aggression": (("aggressive", "tough", "forceful"), ("less aggressive", "calm down")),
        "adaptability": (("adapt", "flexible", "adjust"), ("rigid", "less flexible")),
    }

    for dimension, (increase_terms, decrease_terms) in dimension_hints.items():
        if any(term in lower for term in increase_terms):
            actions.append({"type": "nudge", "dimension": dimension, "delta": config.TRAIT_SOFT_ADJUST_DELTA})
        elif any(term in lower for term in decrease_terms):
            actions.append({"type": "nudge", "dimension": dimension, "delta": -config.TRAIT_SOFT_ADJUST_DELTA})

    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    deduped: list[Dict[str, Any]] = []
    for action in actions:
        key = (action.get("type"), str(action.get("name") or action.get("dimension")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def describe_personality_vector(personality: Dict[str, float]) -> str:
    """I build a short human-readable description of numeric dimensions."""
    vec = _with_defaults(personality)

    def bucket(value: float) -> str:
        if value >= 0.75:
            return "high"
        if value <= 0.25:
            return "low"
        if value >= 0.6:
            return "elevated"
        if value <= 0.4:
            return "muted"
        return "balanced"

    bits = [
        f"aggression {bucket(vec['aggression'])}",
        f"cooperation {bucket(vec['cooperation'])}",
        f"trust {bucket(vec['trust_in_others'])}",
        f"risk tolerance {bucket(vec['risk_tolerance'])}",
    ]
    focus = []
    if vec["food_focus"] >= 0.65:
        focus.append("food-focused")
    if vec["wealth_focus"] >= 0.65:
        focus.append("wealth-focused")
    if vec["adaptability"] >= 0.65:
        focus.append("highly adaptive")
    if focus:
        bits.append(", ".join(focus))
    return "; ".join(bits)


def personality_summary(personality: Dict[str, float], active_traits: Sequence[str]) -> str:
    """I summarise current personality and trait labels."""
    trait_label = ", ".join(active_traits) if active_traits else "no named traits"
    vector_text = describe_personality_vector(personality)
    return f"Personality: {vector_text}. Active traits: {trait_label}."


NEGOTIATION_STYLE_HINTS: Dict[str, str] = {
    "Aggressive": "You are comfortable proposing one-sided deals but remember pushing too hard can sour relations.",
    "Friendly": "You prefer balanced trades and may offer slight generosity to build long-term cooperation.",
    "Wealth-hoarder": "Avoid offers that cut deep into your wealth buffer unless the return is clear.",
    "Food-secure": "Protect your food reserves; only export food when you remain safely above your needs.",
    "Opportunistic": "Feel free to pivot quickly if a new advantage appears during the dialogue.",
    "Isolationist": "Keep offers minimal and only engage when the benefit is obvious.",
}


def negotiation_style_line(active_traits: Sequence[str]) -> str:
    """I pick a short style note for negotiation prompts."""
    for trait in active_traits:
        if trait in NEGOTIATION_STYLE_HINTS:
            return NEGOTIATION_STYLE_HINTS[trait]
    return "You can improvise deals but ensure they make sense for your citizens."


def adaptation_pressure_text(state: Any) -> str:
    """I surface adaptation pressure when streaks are high and cooldown is open."""
    pressure_parts: list[str] = []
    if getattr(state, "exploitation_streak", 0) >= config.TRAIT_CHANGE_PRESSURE_THRESHOLD:
        pressure_parts.append(f"Repeated exploitative trades ({state.exploitation_streak}) are hurting you.")
    if getattr(state, "starvation_streak", 0) >= config.TRAIT_CHANGE_PRESSURE_THRESHOLD:
        pressure_parts.append(f"Food has been unsafe for {state.starvation_streak} steps.")
    if pressure_parts and getattr(state, "trait_cooldown_steps", 0) == 0:
        return "Recent outcomes suggest your approach is faltering: " + " ".join(pressure_parts)
    return ""


def classify_environment(env_metrics: Dict[str, float]) -> str:
    """I map environment richness into a coarse category."""
    food_rich = env_metrics.get("env_food_richness", 1.0)
    wealth_rich = env_metrics.get("env_wealth_richness", 1.0)
    scarce = food_rich < 0.9 and wealth_rich < 0.9
    if scarce:
        return "scarce"
    if food_rich >= wealth_rich and food_rich >= 1.05 and wealth_rich < 1.15:
        return "food_rich"
    if wealth_rich > food_rich and wealth_rich >= 1.05 and food_rich < 1.15:
        return "wealth_rich"
    return "balanced"


def sample_starting_trait(category: str, rng: Any) -> str | None:
    """I deterministically sample a starting trait based on environment category."""
    roll = rng.random()
    if category == "wealth_rich":
        if roll < 0.6:
            return "Wealth-hoarder"
        if roll < 0.8:
            return "Friendly"
        return "Opportunistic"
    if category == "food_rich":
        if roll < 0.5:
            return "Friendly"
        if roll < 0.8:
            return "Opportunistic"
        return "Food-secure"
    if category == "scarce":
        if roll < 0.4:
            return "Food-secure"
        if roll < 0.8:
            return "Aggressive"
        return "Isolationist"
    # balanced
    if roll < 0.4:
        return "Friendly"
    if roll < 0.7:
        return "Opportunistic"
    return "Wealth-hoarder"
