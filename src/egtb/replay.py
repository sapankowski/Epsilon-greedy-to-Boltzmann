"""Replay memory for off-policy DQN training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-size ring buffer with vectorized random sampling."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        observation_dtype: np.dtype,
        seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.observations = np.zeros(
            (self.capacity, *observation_shape), dtype=observation_dtype
        )
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample batch of {batch_size} transitions from {self.size} items."
            )
        indices = self.rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
        )
