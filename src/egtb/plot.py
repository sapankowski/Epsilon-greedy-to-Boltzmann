"""Plot evaluation returns from one or more experiment run directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_eval_csv(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    returns: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(float(row["step"])))
            returns.append(float(row["mean_return"]))
    return steps, returns


def plot_runs(run_dirs: list[Path], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install with `pip install -e '.[plots]'`."
        ) from exc

    fig, ax = plt.subplots(figsize=(9, 5))
    for run_dir in run_dirs:
        eval_path = run_dir / "eval.csv"
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing {eval_path}")
        steps, returns = read_eval_csv(eval_path)
        ax.plot(steps, returns, marker="o", linewidth=1.8, label=run_dir.name)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean evaluation return")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"saved plot: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/comparison.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_runs(args.run_dirs, args.output)


if __name__ == "__main__":
    main()
