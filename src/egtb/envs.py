"""Gymnasium environment construction and validation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import EnvConfig


@dataclass(frozen=True)
class SpaceInfo:
    observation_shape: tuple[int, ...]
    observation_dtype: np.dtype
    action_count: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _import_gymnasium():
    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gymnasium is required to create environments. Install with "
            "`pip install -e '.[classic]'`, `pip install -e '.[box2d]'`, or "
            "`pip install -e '.[atari]'` depending on the experiment."
        ) from exc
    return gym


def make_env(config: EnvConfig, seed: int, run_dir: Path | None = None):
    gym = _import_gymnasium()
    if config.atari:
        try:
            import ale_py
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Atari experiments require ale-py and ROMs. Install with "
                "`pip install -e '.[atari]'` and ensure the ROM license is accepted."
            ) from exc
        gym.register_envs(ale_py)

    if config.atari:
        # AtariPreprocessing performs frame skipping itself; ALE envs should not
        # also repeat actions internally.
        env = gym.make(config.env_id, frameskip=1)
    else:
        env = gym.make(config.env_id)
    if config.max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)

    if config.atari:
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=config.noop_max,
            frame_skip=config.frame_skip,
            screen_size=config.screen_size,
            terminal_on_life_loss=config.terminal_on_life_loss,
            grayscale_obs=True,
            scale_obs=config.scale_obs,
        )
        frame_stack_wrapper = getattr(gym.wrappers, "FrameStackObservation", None)
        if frame_stack_wrapper is None:
            frame_stack_wrapper = getattr(gym.wrappers, "FrameStack")
        env = frame_stack_wrapper(env, config.frame_stack)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def describe_spaces(env: Any) -> SpaceInfo:
    gym = _import_gymnasium()
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(
            f"This DQN implementation requires a discrete action space, got {env.action_space}."
        )
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError(
            f"This DQN implementation requires a Box observation space, got {env.observation_space}."
        )
    return SpaceInfo(
        observation_shape=tuple(int(dim) for dim in env.observation_space.shape),
        observation_dtype=np.dtype(env.observation_space.dtype),
        action_count=int(env.action_space.n),
    )
