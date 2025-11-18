"""I keep this as the CLI entry point for the CMP400 feasibility demo so I can launch the
single-agent world, flip the LLM override on or off, and capture the chronicle for write-up."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

from src.model.world_model import WorldModel


@dataclass
class RunConfig:
    """I centralise the key knobs (steps, seed, initial food, LLM flag) so future multi-agent demos
    stay consistent."""

    steps: int = 10
    seed: int = 42
    initial_food: int = 3
    use_llm: bool = True

    def chronicle_path(self) -> Path:
        """I standardise the log file name so I can reference runs in my feasibility write-up."""
        return Path("logs") / f"demo_seed{self.seed}_steps{self.steps}.json"


def run_demo(
    steps: int = 10,
    seed: int = 42,
    initial_food: Optional[int] = None,
    save_log: bool = True,
    use_llm: bool = False,
    config: Optional[RunConfig] = None,
) -> None:
    """I drive the feasibility demo loop: build ``WorldModel``, advance configured ticks, and
    optionally persist the chronicle artifact for later reflection and reporting.

    :param steps: I control how many simulation steps to run.
    :param seed: I pass this seed into the model for deterministic runs.
    :param initial_food: I override the config default when I want quick tweaks from the CLI.
    :param save_log: I set this to False if I do not want to write the chronicle to disk.
    :param use_llm: I flip this flag to switch between the LLM override and rule-based logic.
    :param config: I pass a ``RunConfig`` when I want to manage all knobs from one object.
    """
    if config is None:
        defaults = RunConfig()
        config = RunConfig(
            steps=steps,
            seed=seed,
            initial_food=initial_food if initial_food is not None else defaults.initial_food,
            use_llm=use_llm,
        )

    # I construct the single-territory world with a deterministic seed so I can narrate runs.
    model = WorldModel(
        random_seed=config.seed,
        initial_food=config.initial_food,
        use_llm=config.use_llm,
    )

    for _ in range(config.steps):
        # I step the Mesa model manually because I want to print each tick in real time.
        model.step()

    if save_log:
        # I persist the chronicle to share the structured trace in the feasibility report.
        out_path = config.chronicle_path()
        model.save_chronicle(out_path)
        print(f"Saved chronicle to {out_path.resolve()}")
        summarize_chronicle(out_path)


def summarize_chronicle(path: Path) -> None:
    """I load the chronicle JSON and print quick stats so I can sanity-check the run."""
    if not path.exists():
        print(f"No chronicle found at {path}")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        print("Chronicle is empty.")
        return

    actions = Counter(entry.get("action", "unknown") for entry in data)
    food_values = [entry.get("food_after", entry.get("food_before", 0)) for entry in data]
    food_min = min(food_values)
    food_max = max(food_values)
    food_mean = mean(food_values)
    total_steps = len(data)

    # I skim these stats so I can confirm the log artifact looks sensible before adding it to the feasibility demo.
    print("Chronicle summary:")
    print(f"  Steps logged: {total_steps}")
    print(f"  Action counts: {dict(actions)}")
    print(f"  Food stats -> min: {food_min}, mean: {food_mean:.2f}, max: {food_max}")


if __name__ == "__main__":
    # I run a simple demo here and explicitly enable the LLM for decisions.
    run_demo(use_llm=True)
