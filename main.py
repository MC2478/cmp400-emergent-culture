"""I use this file as the CLI entry point where I can toggle LLM or rule-based mode."""

from pathlib import Path

from src.model.world_model import WorldModel


def run_demo(
    steps: int = 10,
    seed: int = 42,
    save_log: bool = True,
    use_llm: bool = False,
) -> None:
    """I build a ``WorldModel``, advance it for ``steps`` and optionally save the chronicle.

    :param steps: I control how many simulation steps to run.
    :param seed: I pass this seed into the model for deterministic runs.
    :param save_log: I set this to False if I do not want to write the chronicle to disk.
    :param use_llm: I flip this flag to switch between the LLM override and rule-based logic.
    """
    model = WorldModel(random_seed=seed, initial_food=3, use_llm=use_llm)

    for _ in range(steps):
        model.step()

    if save_log:
        out_path = Path("logs") / f"demo_seed{seed}_steps{steps}.json"
        model.save_chronicle(out_path)
        print(f"Saved chronicle to {out_path.resolve()}")

if __name__ == "__main__":
    # I run a simple demo here and explicitly enable the LLM for decisions.
    run_demo(use_llm=True)
