"""Aggregate experiment runs by strategy and seed."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EvalPoint:
    step: int
    mean_return: float
    std_return: float


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    strategy: str
    seed: int
    final_return: float
    best_return: float
    best_step: int
    late_return: float
    auc_return: float
    eval_points: tuple[EvalPoint, ...]


def read_eval_points(path: Path) -> tuple[EvalPoint, ...]:
    """Read eval.csv and remove duplicate step rows caused by final evaluation."""

    by_step: dict[int, EvalPoint] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["step"]))
            by_step[step] = EvalPoint(
                step=step,
                mean_return=float(row["mean_return"]),
                std_return=float(row["std_return"]),
            )
    return tuple(by_step[step] for step in sorted(by_step))


def load_run(run_dir: Path, late_window: int) -> RunResult:
    summary_path = run_dir / "summary.json"
    eval_path = run_dir / "eval.csv"
    if not summary_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Run directory is missing summary.json or eval.csv: {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary["config"]
    strategy = config["exploration"]["name"]
    seed = int(config["seed"])
    points = read_eval_points(eval_path)
    if not points:
        raise ValueError(f"No evaluation rows found in {eval_path}")

    returns = np.asarray([point.mean_return for point in points], dtype=np.float64)
    steps = np.asarray([point.step for point in points], dtype=np.float64)
    best_index = int(np.argmax(returns))
    late_count = min(max(1, late_window), len(points))
    auc = float(np.trapezoid(returns, steps) / max(1.0, steps[-1] - steps[0]))
    return RunResult(
        run_dir=run_dir,
        strategy=strategy,
        seed=seed,
        final_return=float(summary["final_eval"]["mean_return"]),
        best_return=float(returns[best_index]),
        best_step=int(points[best_index].step),
        late_return=float(np.mean(returns[-late_count:])),
        auc_return=auc,
        eval_points=points,
    )


def discover_runs(root: Path, late_window: int) -> list[RunResult]:
    run_dirs = sorted(path.parent for path in root.glob("*/summary.json"))
    if not run_dirs and (root / "summary.json").exists():
        run_dirs = [root]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return [load_run(run_dir, late_window) for run_dir in run_dirs]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def grouped(results: Iterable[RunResult]) -> dict[str, list[RunResult]]:
    groups: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        groups[result.strategy].append(result)
    return dict(sorted(groups.items()))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def strategy_summary_rows(results: list[RunResult], solved_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy, group in grouped(results).items():
        final_values = [item.final_return for item in group]
        best_values = [item.best_return for item in group]
        late_values = [item.late_return for item in group]
        auc_values = [item.auc_return for item in group]
        final_mean, final_std = mean_std(final_values)
        best_mean, best_std = mean_std(best_values)
        late_mean, late_std = mean_std(late_values)
        auc_mean, auc_std = mean_std(auc_values)
        rows.append(
            {
                "strategy": strategy,
                "n": len(group),
                "seeds": " ".join(str(item.seed) for item in sorted(group, key=lambda x: x.seed)),
                "final_mean": round(final_mean, 4),
                "final_std": round(final_std, 4),
                "final_min": round(float(min(final_values)), 4),
                "final_max": round(float(max(final_values)), 4),
                "best_mean": round(best_mean, 4),
                "best_std": round(best_std, 4),
                "late_mean": round(late_mean, 4),
                "late_std": round(late_std, 4),
                "auc_mean": round(auc_mean, 4),
                "auc_std": round(auc_std, 4),
                "final_solved_rate": round(
                    sum(value >= solved_threshold for value in final_values) / len(final_values), 4
                ),
                "best_solved_rate": round(
                    sum(value >= solved_threshold for value in best_values) / len(best_values), 4
                ),
            }
        )
    return rows


def per_seed_rows(results: list[RunResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in sorted(results, key=lambda x: (x.strategy, x.seed, x.run_dir.name)):
        rows.append(
            {
                "strategy": result.strategy,
                "seed": result.seed,
                "final_return": round(result.final_return, 4),
                "best_return": round(result.best_return, 4),
                "best_step": result.best_step,
                "late_return": round(result.late_return, 4),
                "auc_return": round(result.auc_return, 4),
                "eval_points": len(result.eval_points),
                "run_dir": str(result.run_dir),
            }
        )
    return rows


def aggregate_eval_rows(results: list[RunResult], band: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy, group in grouped(results).items():
        values_by_step: dict[int, list[float]] = defaultdict(list)
        for result in group:
            for point in result.eval_points:
                values_by_step[point.step].append(point.mean_return)
        for step in sorted(values_by_step):
            values = values_by_step[step]
            value_mean, value_std = mean_std(values)
            error = value_std / np.sqrt(len(values)) if band == "sem" else value_std
            rows.append(
                {
                    "strategy": strategy,
                    "step": step,
                    "n": len(values),
                    "mean_return": round(value_mean, 4),
                    "std_return": round(value_std, 4),
                    "error_return": round(float(error), 4),
                }
            )
    return rows


def configure_matplotlib(output_dir: Path):
    cache_dir = output_dir / ".mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install with `pip install -e '.[plots]'`."
        ) from exc
    return plt


def plot_learning_curves(
    results: list[RunResult], output: Path, band: str, solved_threshold: float
) -> None:
    plt = configure_matplotlib(output.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, group in grouped(results).items():
        values_by_step: dict[int, list[float]] = defaultdict(list)
        for result in group:
            for point in result.eval_points:
                values_by_step[point.step].append(point.mean_return)
        steps = np.asarray(sorted(values_by_step), dtype=np.float64)
        means = np.asarray([mean(values_by_step[int(step)]) for step in steps], dtype=np.float64)
        stds = np.asarray([pstdev(values_by_step[int(step)]) for step in steps], dtype=np.float64)
        ns = np.asarray([len(values_by_step[int(step)]) for step in steps], dtype=np.float64)
        errors = stds / np.sqrt(ns) if band == "sem" else stds
        line = ax.plot(steps, means, linewidth=2.2, label=strategy)[0]
        if band != "none":
            ax.fill_between(
                steps,
                means - errors,
                means + errors,
                color=line.get_color(),
                alpha=0.16,
                linewidth=0,
            )
    ax.axhline(solved_threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean evaluation return")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_final_returns(results: list[RunResult], output: Path) -> None:
    plt = configure_matplotlib(output.parent)
    groups = grouped(results)
    strategies = list(groups)
    final_means = [mean([item.final_return for item in groups[strategy]]) for strategy in strategies]
    final_stds = [pstdev([item.final_return for item in groups[strategy]]) for strategy in strategies]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(strategies), dtype=np.float64)
    bars = ax.bar(x, final_means, yerr=final_stds, capsize=5, alpha=0.78)
    for index, strategy in enumerate(strategies):
        seed_values = sorted((item.seed, item.final_return) for item in groups[strategy])
        offsets = np.linspace(-0.18, 0.18, num=len(seed_values)) if seed_values else []
        for offset, (_, value) in zip(offsets, seed_values):
            ax.scatter(index + offset, value, color="black", s=20, alpha=0.72, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=18, ha="right")
    ax.set_ylabel("final evaluation return")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, final_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 8,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze(
    runs_dir: Path,
    output_dir: Path,
    late_window: int,
    band: str,
    solved_threshold: float,
) -> None:
    results = discover_runs(runs_dir, late_window)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "strategy_summary.csv",
        [
            "strategy",
            "n",
            "seeds",
            "final_mean",
            "final_std",
            "final_min",
            "final_max",
            "best_mean",
            "best_std",
            "late_mean",
            "late_std",
            "auc_mean",
            "auc_std",
            "final_solved_rate",
            "best_solved_rate",
        ],
        strategy_summary_rows(results, solved_threshold),
    )
    write_csv(
        output_dir / "per_seed_summary.csv",
        [
            "strategy",
            "seed",
            "final_return",
            "best_return",
            "best_step",
            "late_return",
            "auc_return",
            "eval_points",
            "run_dir",
        ],
        per_seed_rows(results),
    )
    write_csv(
        output_dir / "aggregate_eval.csv",
        ["strategy", "step", "n", "mean_return", "std_return", "error_return"],
        aggregate_eval_rows(results, band),
    )
    plot_learning_curves(
        results, output_dir / "learning_curves.png", band, solved_threshold
    )
    plot_final_returns(results, output_dir / "final_returns.png")
    print(f"analyzed {len(results)} runs from {runs_dir}")
    print(f"wrote: {output_dir / 'strategy_summary.csv'}")
    print(f"wrote: {output_dir / 'per_seed_summary.csv'}")
    print(f"wrote: {output_dir / 'aggregate_eval.csv'}")
    print(f"wrote: {output_dir / 'learning_curves.png'}")
    print(f"wrote: {output_dir / 'final_returns.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, help="Directory containing run folders.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregate CSVs and plots.",
    )
    parser.add_argument(
        "--late-window",
        type=int,
        default=3,
        help="Number of final evaluation points averaged into late_return.",
    )
    parser.add_argument(
        "--band",
        choices=["std", "sem", "none"],
        default="std",
        help="Uncertainty band for aggregate learning curves.",
    )
    parser.add_argument(
        "--solved-threshold",
        type=float,
        default=475.0,
        help="Return threshold counted as solved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path(f"{args.runs_dir}_analysis")
    analyze(args.runs_dir, output_dir, args.late_window, args.band, args.solved_threshold)


if __name__ == "__main__":
    main()
