"""DQN agent with pluggable exploration strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import DQNConfig, ExplorationConfig
from .exploration import ExplorationStrategy, build_strategy, entropy_uncertainty
from .networks import make_q_network, observations_to_tensor
from .replay import ReplayBuffer


@dataclass(frozen=True)
class ActionResult:
    action: int
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class OptimizeResult:
    loss: float
    mean_q: float
    mean_target: float


class DQNAgent:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        observation_dtype: np.dtype,
        action_count: int,
        dqn_config: DQNConfig,
        exploration_config: ExplorationConfig,
        device: torch.device,
        seed: int,
    ) -> None:
        self.observation_shape = observation_shape
        self.action_count = action_count
        self.config = dqn_config
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.strategy: ExplorationStrategy = build_strategy(exploration_config)
        self.policy_net = make_q_network(
            observation_shape, action_count, dqn_config.hidden_sizes
        ).to(device)
        self.target_net = make_q_network(
            observation_shape, action_count, dqn_config.hidden_sizes
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=dqn_config.learning_rate, amsgrad=True
        )
        self.replay = ReplayBuffer(
            dqn_config.replay_size, observation_shape, observation_dtype, seed + 10_007
        )
        self.optimization_steps = 0

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        with torch.no_grad():
            obs_tensor = observations_to_tensor(
                observation, self.observation_shape, self.device
            )
            q_values = self.policy_net(obs_tensor).squeeze(0).cpu().numpy()
        self.policy_net.train()
        return q_values

    def act(self, observation: np.ndarray, step: int, *, explore: bool) -> ActionResult:
        q_values = self.q_values(observation)
        uncertainty = entropy_uncertainty(
            q_values, self.strategy.config.uncertainty_temperature
        )
        action = self.strategy.select_action(
            q_values,
            step,
            self.rng,
            explore=explore,
            uncertainty=uncertainty,
        )
        diagnostics = self.strategy.diagnostics(q_values, step, uncertainty)
        diagnostics["max_q"] = float(np.max(q_values))
        diagnostics["selected_q"] = float(q_values[action])
        return ActionResult(action, diagnostics)

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(observation, action, reward, next_observation, done)

    def ready_to_train(self, step: int) -> bool:
        return (
            step >= self.config.learning_starts
            and len(self.replay) >= self.config.batch_size
            and step % self.config.train_every == 0
        )

    def optimize(self) -> OptimizeResult:
        batch = self.replay.sample(self.config.batch_size)
        observations = observations_to_tensor(
            batch.observations, self.observation_shape, self.device
        )
        next_observations = observations_to_tensor(
            batch.next_observations, self.observation_shape, self.device
        )
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        predicted_q = self.policy_net(observations).gather(1, actions)
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = torch.argmax(self.policy_net(next_observations), dim=1, keepdim=True)
                next_q = self.target_net(next_observations).gather(1, next_actions)
            else:
                next_q = self.target_net(next_observations).max(dim=1, keepdim=True).values
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = nn.functional.smooth_l1_loss(predicted_q, target_q, beta=self.config.huber_delta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimization_steps += 1

        if self.config.tau < 1.0:
            self.soft_update_target()

        return OptimizeResult(
            loss=float(loss.item()),
            mean_q=float(predicted_q.detach().mean().item()),
            mean_target=float(target_q.detach().mean().item()),
        )

    def hard_update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * policy_param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "optimization_steps": self.optimization_steps,
        }
