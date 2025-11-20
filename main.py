"""I keep this as the CLI entry point for the CMP400 feasibility demo so I can launch the
single-agent world, flip the LLM override on or off, and capture the chronicle for write-up."""

from __future__ import annotations

import json
from datetime import datetime
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional
import contextlib
import sys

import config
from src.model.world_model import WorldModel

_RUN_DIR: Path | None = None
_RUN_KEY: tuple[int | None, int | None] = (None, None)


@dataclass
class RunConfig:
    """I centralise the key knobs (steps, seed, initial food, LLM flag) so future multi-agent demos
    stay consistent."""

    steps: int = 25
    seed: int = 42
    initial_food: Optional[float] = None
    use_llm: bool = True

    def chronicle_path(self) -> Path:
        """I standardise the log file name so I can reference runs in my feasibility write-up."""
        return _choose_run_artifact_path("demo", self.seed, self.steps, ".json")


def run_demo(
    steps: int = 25,
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
    :param initial_food: I override the seed-derived starting food when I want quick tweaks from the CLI.
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
    # I snapshot the current configuration so I can review the key knobs alongside each run.
    config_path = _choose_run_artifact_path("config_summary", config.seed, config.steps, ".txt")
    model.save_config_summary(str(config_path))

    for _ in range(config.steps):
        # I step the Mesa model manually because I want to print each tick in real time.
        model.step()
        if model.all_territories_dead():
            print(f"All territories have collapsed by step {model.steps}. Ending simulation early.")
            break

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

    actions = Counter(entry.get("action", "unknown") for entry in data if entry.get("action"))
    total_events = len(data)

    def describe(series: list[float]) -> tuple[float | None, float | None, float | None]:
        if not series:
            return (None, None, None)
        return (min(series), mean(series), max(series))

    east_pop = []
    west_pop = []
    east_food = []
    west_food = []

    for entry in data:
        if entry.get("event_type") == "upkeep":
            east = entry.get("east", {})
            west = entry.get("west", {})
            if east:
                if "population_after" in east:
                    east_pop.append(east["population_after"])
                if "food_after" in east:
                    east_food.append(east["food_after"])
            if west:
                if "population_after" in west:
                    west_pop.append(west["population_after"])
                if "food_after" in west:
                    west_food.append(west["food_after"])

    east_pop_stats = describe(east_pop)
    west_pop_stats = describe(west_pop)
    east_food_stats = describe(east_food)
    west_food_stats = describe(west_food)

    def print_stats(label: str, stats: tuple[float | None, float | None, float | None]) -> None:
        lo, avg, hi = stats
        if lo is None:
            print(f"  {label} -> n/a")
        else:
            print(f"  {label} -> min {lo:.1f}, mean {avg:.1f}, max {hi:.1f}")

    # I skim these stats so I can confirm the log artifact looks sensible before adding it to the feasibility demo.
    print("Chronicle summary:")
    print(f"  Events logged: {total_events}")
    print(f"  Action counts: {dict(actions)}")
    print_stats("East pop", east_pop_stats)
    print_stats("West pop", west_pop_stats)
    print_stats("East food", east_food_stats)
    print_stats("West food", west_food_stats)


class Tee:
    """Simple tee to duplicate stdout/stderr to a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _choose_run_artifact_path(prefix: str, seed: int, steps: int, ext: str) -> Path:
    """I save run artifacts under a dated logs/ subfolder so the newest run is obvious."""
    global _RUN_DIR, _RUN_KEY
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    if _RUN_DIR is None or _RUN_KEY != (seed, steps):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate = logs_dir / f"run_{stamp}_seed{seed}_steps{steps}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = logs_dir / f"run_{stamp}_seed{seed}_steps{steps}_v{suffix}"
        candidate.mkdir(parents=True, exist_ok=True)
        _RUN_DIR = candidate
        _RUN_KEY = (seed, steps)
        print(f"[info] Logging run artifacts under {_RUN_DIR}")
    base = _RUN_DIR / f"{prefix}_seed{seed}_steps{steps}{ext}"
    if not base.exists():
        return base
    idx = 1
    while True:
        candidate = _RUN_DIR / f"{prefix}_seed{seed}_steps{steps}_run{idx}{ext}"
        if not candidate.exists():
            return candidate
        idx += 1


def _choose_output_log_path(seed: int, steps: int) -> Path:
    """I save output logs under logs/ and avoid clobbering previous runs."""
    return _choose_run_artifact_path("output_log", seed, steps, ".txt")


if __name__ == "__main__":
    raw = input("How many steps should the simulation run for? [default: 25] ")
    try:
        steps = int(raw.strip()) if raw.strip() else 25
        if steps <= 0:
            print("Steps must be positive, defaulting to 25.")
            steps = 25
    except ValueError:
        print("Invalid input, defaulting to 25 steps.")
        steps = 25

    seed_raw = input("Which seed should I use? [default: 42] ")
    try:
        seed_value = int(seed_raw.strip()) if seed_raw.strip() else 42
    except ValueError:
        print("Invalid seed, defaulting to 42.")
        seed_value = 42

    # I run a simple demo here and explicitly enable the LLM for decisions. I also tee console output to output_log.txt.
    log_path = _choose_output_log_path(seed_value, steps)
    print(f"Saving console output to {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            run_demo(steps=steps, seed=seed_value, use_llm=True)
