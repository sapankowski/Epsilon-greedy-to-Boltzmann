"""Command line training entry point."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig, apply_overrides, load_config, save_config
from .dqn import DQNAgent, OptimizeResult
from .envs import describe_spaces, make_env, seed_everything
from .logging import CSVLogger, write_json


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def default_run_name(config: ExperimentConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env = config.env.env_id.replace("/", "_")
    return f"{timestamp}_{env}_{config.exploration.name}_seed{config.seed}"


def maybe_clip_reward(reward: float, clip_value: float | None) -> float:
    if clip_value is None:
        return float(reward)
    return float(np.clip(reward, -abs(clip_value), abs(clip_value)))


def evaluate(agent: DQNAgent, config: ExperimentConfig, step: int) -> dict[str, float]:
    returns: list[float] = []
    lengths: list[int] = []
    env = make_env(config.env, seed=config.seed + 50_000 + step)
    try:
        for episode in range(config.eval_episodes):
            obs, _ = env.reset(seed=config.seed + 50_000 + step + episode)
            episode_return = 0.0
            episode_length = 0
            done = False
            while not done:
                action = agent.act(obs, step, explore=False).action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                obs = next_obs
                done = bool(terminated or truncated)
            returns.append(episode_return)
            lengths.append(episode_length)
    finally:
        env.close()

    return {
        "step": float(step),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes": float(config.eval_episodes),
    }


def save_agent_checkpoint(agent: DQNAgent, run_dir: Path, step: int) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), checkpoint_dir / f"agent_step_{step}.pt")


def train(
    config: ExperimentConfig,
    *,
    run_name_override: str | None = None,
    skip_existing: bool = False,
    checkpoint_interval: int = 0,
) -> Path:
    seed_everything(config.seed)
    device = choose_device(config.device)
    run_dir = Path(config.output_dir) / (run_name_override or default_run_name(config))
    if skip_existing and (run_dir / "summary.json").exists():
        print(f"skipping completed run: {run_dir}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    env = make_env(config.env, seed=config.seed, run_dir=run_dir)
    space_info = describe_spaces(env)
    agent = DQNAgent(
        space_info.observation_shape,
        space_info.observation_dtype,
        space_info.action_count,
        config.dqn,
        config.exploration,
        device,
        config.seed,
    )

    train_logger = CSVLogger(
        run_dir / "train.csv",
        [
            "step",
            "episode",
            "episode_return",
            "episode_length",
            "epsilon",
            "temperature",
            "boltzmann_weight",
            "uncertainty",
            "max_q",
            "selected_q",
            "loss",
            "mean_q",
            "mean_target",
        ],
    )
    eval_logger = CSVLogger(
        run_dir / "eval.csv",
        ["step", "mean_return", "std_return", "mean_length", "episodes"],
    )

    obs, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    episode = 0
    recent_returns: list[float] = []
    last_optimization: OptimizeResult | None = None

    try:
        for step in range(config.total_steps):
            action_result = agent.act(obs, step, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action_result.action)
            train_reward = maybe_clip_reward(reward, config.env.reward_clip)

            # Only true environment termination blocks bootstrapping. Time-limit
            # truncation still has a valid next observation for the Bellman target.
            agent.observe(obs, action_result.action, train_reward, next_obs, bool(terminated))

            if agent.ready_to_train(step):
                for _ in range(config.dqn.gradient_steps):
                    last_optimization = agent.optimize()

            if config.dqn.tau >= 1.0 and (step + 1) % config.dqn.target_update_interval == 0:
                agent.hard_update_target()

            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if terminated or truncated:
                recent_returns.append(episode_return)
                recent_returns = recent_returns[-100:]
                train_row: dict[str, Any] = {
                    "step": step + 1,
                    "episode": episode,
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    **action_result.diagnostics,
                }
                if last_optimization is not None:
                    train_row.update(asdict(last_optimization))
                train_logger.write(train_row)
                episode += 1
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            if config.eval_interval > 0 and (step + 1) % config.eval_interval == 0:
                eval_result = evaluate(agent, config, step + 1)
                eval_logger.write(eval_result)
                print(
                    "step={step} eval_return={ret:.2f} recent_train_return={train:.2f}".format(
                        step=step + 1,
                        ret=eval_result["mean_return"],
                        train=float(np.mean(recent_returns)) if recent_returns else math.nan,
                    )
                )

            if config.log_interval > 0 and (step + 1) % config.log_interval == 0:
                print(
                    "step={step} episodes={episodes} buffer={buffer} recent_return={ret:.2f}".format(
                        step=step + 1,
                        episodes=episode,
                        buffer=len(agent.replay),
                        ret=float(np.mean(recent_returns)) if recent_returns else math.nan,
                    )
                )

            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                save_agent_checkpoint(agent, run_dir, step + 1)

        final_eval = evaluate(agent, config, config.total_steps)
        eval_logger.write(final_eval)
        torch.save(agent.state_dict(), run_dir / "agent.pt")
        write_json(
            run_dir / "summary.json",
            {
                "run_dir": str(run_dir),
                "final_eval": final_eval,
                "episodes": episode,
                "recent_train_return": float(np.mean(recent_returns))
                if recent_returns
                else None,
                "config": asdict(config),
            },
        )
    finally:
        train_logger.close()
        eval_logger.close()
        env.close()

    print(f"finished run: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--env-id", type=str, default=None, help="Gymnasium environment id.")
    parser.add_argument("--strategy", type=str, default=None, help="Exploration strategy name.")
    parser.add_argument("--total-steps", type=int, default=None, help="Training steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, mps, or auto.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluation interval.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes.")
    parser.add_argument("--log-interval", type=int, default=None, help="Console log interval.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="Replay warmup steps before optimization.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="DQN batch size.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Stable output subdirectory name. Useful for cluster arrays.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when run_dir already contains summary.json.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save lightweight model checkpoints every N environment steps.",
    )
    parser.add_argument("--atari", action="store_true", help="Enable Atari preprocessing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else ExperimentConfig()
    overrides = {
        "env.env_id": args.env_id,
        "exploration.name": args.strategy,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "device": args.device,
        "output_dir": args.output_dir,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "log_interval": args.log_interval,
        "dqn.learning_starts": args.learning_starts,
        "dqn.batch_size": args.batch_size,
        "env.atari": True if args.atari else None,
    }
    train(
        apply_overrides(config, overrides),
        run_name_override=args.run_name,
        skip_existing=args.skip_existing,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
