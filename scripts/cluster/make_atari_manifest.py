#!/usr/bin/env python3
"""Create a TSV manifest for Atari cluster array jobs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CONFIGS = (
    Path("configs/atari/breakout_epsilon_greedy.yaml"),
    Path("configs/atari/breakout_annealed.yaml"),
    Path("configs/atari/breakout_topk_boltzmann.yaml"),
    Path("configs/atari/breakout_uncertainty_switch.yaml"),
)


def strategy_from_config(path: Path) -> str:
    names = {
        "breakout_epsilon_greedy": "epsilon_greedy",
        "breakout_annealed": "annealed_epsilon_boltzmann",
        "breakout_topk_boltzmann": "topk_boltzmann",
        "breakout_uncertainty_switch": "uncertainty_switch",
    }
    return names[path.stem]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cluster_manifests/breakout_1m.tsv"),
        help="Manifest TSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/breakout_1m"),
        help="Training output directory used by each task.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    task_id = 0
    for seed in args.seeds:
        for config in CONFIGS:
            strategy = strategy_from_config(config)
            run_name = f"breakout_{strategy}_seed{seed}"
            rows.append(
                {
                    "task_id": task_id,
                    "config": str(config),
                    "seed": seed,
                    "strategy": strategy,
                    "run_name": run_name,
                    "output_dir": str(args.output_dir),
                    "total_steps": args.total_steps,
                    "eval_episodes": "default" if args.eval_episodes is None else args.eval_episodes,
                    "checkpoint_interval": args.checkpoint_interval,
                }
            )
            task_id += 1

    fieldnames = [
        "task_id",
        "config",
        "seed",
        "strategy",
        "run_name",
        "output_dir",
        "total_steps",
        "eval_episodes",
        "checkpoint_interval",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    last_index = len(rows) - 1
    print(f"wrote {len(rows)} tasks: {args.output}")
    print(f"submit with: sbatch --array=0-{last_index} scripts/cluster/slurm_atari_array.sbatch")


if __name__ == "__main__":
    main()
