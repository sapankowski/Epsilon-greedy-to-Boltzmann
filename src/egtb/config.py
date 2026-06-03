"""Configuration dataclasses and YAML loading helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_size: int = 100_000
    learning_starts: int = 1_000
    train_every: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    tau: float = 1.0
    double_dqn: bool = True
    hidden_sizes: tuple[int, ...] = (128, 128)
    max_grad_norm: float = 10.0
    huber_delta: float = 1.0


@dataclass
class ExplorationConfig:
    name: str = "epsilon_greedy"
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    temperature_start: float = 2.0
    temperature_end: float = 0.15
    temperature_decay_steps: int = 50_000
    switch_start_step: int = 10_000
    switch_end_step: int = 50_000
    top_k: int = 2
    low_uncertainty: float = 0.25
    high_uncertainty: float = 0.85
    uncertainty_temperature: float = 1.0
    smooth_switch: bool = True


@dataclass
class EnvConfig:
    env_id: str = "CartPole-v1"
    max_episode_steps: int | None = None
    reward_clip: float | None = None
    atari: bool = False
    frame_stack: int = 4
    screen_size: int = 84
    terminal_on_life_loss: bool = False
    noop_max: int = 30
    frame_skip: int = 4
    scale_obs: bool = True


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    total_steps: int = 50_000
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs"
    eval_interval: int = 5_000
    eval_episodes: int = 5
    log_interval: int = 1_000


def _known_field_names(cls: type) -> set[str]:
    return {field_.name for field_ in fields(cls)}


def _construct_dataclass(cls: type, data: dict[str, Any]):
    known = _known_field_names(cls)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {', '.join(unknown)}")
    return cls(**data)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` and reject misspelled keys early."""

    data = dict(raw)
    env = _construct_dataclass(EnvConfig, data.pop("env", {}))
    dqn_data = data.pop("dqn", {})
    if "hidden_sizes" in dqn_data:
        dqn_data["hidden_sizes"] = tuple(dqn_data["hidden_sizes"])
    dqn = _construct_dataclass(DQNConfig, dqn_data)
    exploration = _construct_dataclass(ExplorationConfig, data.pop("exploration", {}))
    config = _construct_dataclass(ExperimentConfig, data)
    config.env = env
    config.dqn = dqn
    config.exploration = exploration
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return config_from_dict(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)


def apply_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply CLI overrides with dotted keys such as ``env.env_id``."""

    for dotted_key, value in overrides.items():
        if value is None:
            continue
        target: Any = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return config
