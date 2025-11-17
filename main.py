# main.py
from pathlib import Path

from src.model.world_model import WorldModel


def run_demo(
    steps: int = 10,
    seed: int = 42,
    save_log: bool = True,
    use_llm: bool = False,
) -> None:
    model = WorldModel(random_seed=seed, initial_food=3, use_llm=use_llm)

    for _ in range(steps):
        model.step()

    if save_log:
        out_path = Path("logs") / f"demo_seed{seed}_steps{steps}.json"
        model.save_chronicle(out_path)
        print(f"Saved chronicle to {out_path.resolve()}")


if __name__ == "__main__":
    run_demo(use_llm=False)
