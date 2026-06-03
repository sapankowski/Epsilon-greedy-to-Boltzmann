"""Exploration strategies for discrete-action DQN agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import ExplorationConfig
from .schedules import LinearSchedule, SwitchSchedule, clamp

EPS = 1e-8


def _one_dimensional(q_values: np.ndarray) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.ndim != 1:
        raise ValueError(f"Expected one-dimensional q-values, got shape {q_values.shape}")
    return q_values


def epsilon_greedy_distribution(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    """Probability distribution that follows argmax with uniform random exploration."""

    q_values = _one_dimensional(q_values)
    action_count = q_values.shape[0]
    epsilon = clamp(float(epsilon))
    probabilities = np.full(action_count, epsilon / action_count, dtype=np.float64)
    probabilities[int(np.argmax(q_values))] += 1.0 - epsilon
    return probabilities


def boltzmann_distribution(q_values: np.ndarray, temperature: float) -> np.ndarray:
    """Stable softmax over Q-values with a positive temperature."""

    q_values = _one_dimensional(q_values)
    temperature = max(float(temperature), EPS)
    logits = (q_values - np.max(q_values)) / temperature
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(q_values.shape[0], 1.0 / q_values.shape[0], dtype=np.float64)
    return weights / total


def topk_boltzmann_distribution(q_values: np.ndarray, temperature: float, top_k: int) -> np.ndarray:
    """Softmax only over the highest-value actions."""

    q_values = _one_dimensional(q_values)
    action_count = q_values.shape[0]
    k = min(max(1, int(top_k)), action_count)
    top_indices = np.argpartition(q_values, -k)[-k:]
    probabilities = np.zeros(action_count, dtype=np.float64)
    probabilities[top_indices] = boltzmann_distribution(q_values[top_indices], temperature)
    return probabilities


def entropy_uncertainty(q_values: np.ndarray, temperature: float = 1.0) -> float:
    """Use normalized softmax entropy as a single-network uncertainty proxy.

    A flat softmax means the network cannot distinguish actions and returns a value
    near 1. A peaked softmax means the action preference is clear and returns near 0.
    """

    q_values = _one_dimensional(q_values)
    if q_values.shape[0] <= 1:
        return 0.0
    probabilities = boltzmann_distribution(q_values, temperature)
    entropy = -float(np.sum(probabilities * np.log(probabilities + EPS)))
    return clamp(entropy / np.log(q_values.shape[0]))


class ExplorationStrategy(Protocol):
    config: ExplorationConfig
    name: str

    def distribution(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        ...

    def diagnostics(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> dict[str, float]:
        ...

    def select_action(
        self,
        q_values: np.ndarray,
        step: int,
        rng: np.random.Generator,
        *,
        explore: bool = True,
        uncertainty: float | None = None,
    ) -> int:
        ...


@dataclass
class BaseStrategy:
    config: ExplorationConfig
    name: str

    @property
    def epsilon(self) -> LinearSchedule:
        return LinearSchedule(
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.config.epsilon_decay_steps,
        )

    @property
    def temperature(self) -> LinearSchedule:
        return LinearSchedule(
            self.config.temperature_start,
            self.config.temperature_end,
            self.config.temperature_decay_steps,
        )

    @property
    def switch(self) -> SwitchSchedule:
        return SwitchSchedule(
            self.config.switch_start_step,
            self.config.switch_end_step,
            self.config.smooth_switch,
        )

    def select_action(
        self,
        q_values: np.ndarray,
        step: int,
        rng: np.random.Generator,
        *,
        explore: bool = True,
        uncertainty: float | None = None,
    ) -> int:
        q_values = _one_dimensional(q_values)
        if not explore:
            return int(np.argmax(q_values))
        probabilities = self.distribution(q_values, step, uncertainty)
        return int(rng.choice(q_values.shape[0], p=probabilities))

    def diagnostics(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> dict[str, float]:
        return {
            "epsilon": self.epsilon.value(step),
            "temperature": self.temperature.value(step),
            "boltzmann_weight": 0.0,
            "uncertainty": entropy_uncertainty(
                q_values, self.config.uncertainty_temperature
            )
            if uncertainty is None
            else float(uncertainty),
        }


class EpsilonGreedyStrategy(BaseStrategy):
    def __init__(self, config: ExplorationConfig):
        super().__init__(config, "epsilon_greedy")

    def distribution(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        return epsilon_greedy_distribution(q_values, self.epsilon.value(step))


class BoltzmannStrategy(BaseStrategy):
    def __init__(self, config: ExplorationConfig):
        super().__init__(config, "boltzmann")

    def distribution(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        return boltzmann_distribution(q_values, self.temperature.value(step))

    def diagnostics(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> dict[str, float]:
        diagnostics = super().diagnostics(q_values, step, uncertainty)
        diagnostics["boltzmann_weight"] = 1.0
        return diagnostics


class AnnealedEpsilonBoltzmannStrategy(BaseStrategy):
    """Gradually mix epsilon-greedy behavior into Boltzmann behavior."""

    def __init__(self, config: ExplorationConfig):
        super().__init__(config, "annealed_epsilon_boltzmann")

    def distribution(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        weight = self.switch.value(step)
        epsilon_part = epsilon_greedy_distribution(q_values, self.epsilon.value(step))
        boltzmann_part = boltzmann_distribution(q_values, self.temperature.value(step))
        return (1.0 - weight) * epsilon_part + weight * boltzmann_part

    def diagnostics(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> dict[str, float]:
        diagnostics = super().diagnostics(q_values, step, uncertainty)
        diagnostics["boltzmann_weight"] = self.switch.value(step)
        return diagnostics


class TopKBoltzmannStrategy(BaseStrategy):
    """Use random exploration, then sample among the top-k non-random actions."""

    def __init__(self, config: ExplorationConfig):
        super().__init__(config, "topk_boltzmann")

    def distribution(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        epsilon = self.epsilon.value(step)
        random_part = np.full(q_values.shape[0], 1.0 / q_values.shape[0], dtype=np.float64)
        topk_part = topk_boltzmann_distribution(
            q_values, self.temperature.value(step), self.config.top_k
        )
        return epsilon * random_part + (1.0 - epsilon) * topk_part

    def diagnostics(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> dict[str, float]:
        diagnostics = super().diagnostics(q_values, step, uncertainty)
        diagnostics["boltzmann_weight"] = 1.0 - self.epsilon.value(step)
        return diagnostics


class UncertaintyAwareSwitchStrategy(BaseStrategy):
    """Switch only when training is late enough and Q preferences are confident."""

    def __init__(self, config: ExplorationConfig):
        super().__init__(config, "uncertainty_switch")

    def _boltzmann_weight(
        self, q_values: np.ndarray, step: int, uncertainty: float | None
    ) -> tuple[float, float]:
        measured_uncertainty = (
            entropy_uncertainty(q_values, self.config.uncertainty_temperature)
            if uncertainty is None
            else float(uncertainty)
        )
        low = self.config.low_uncertainty
        high = max(self.config.high_uncertainty, low + EPS)
        confidence = 1.0 - clamp((measured_uncertainty - low) / (high - low))
        return self.switch.value(step) * confidence, measured_uncertainty

    def distribution(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        weight, _ = self._boltzmann_weight(q_values, step, uncertainty)
        epsilon_part = epsilon_greedy_distribution(q_values, self.epsilon.value(step))
        boltzmann_part = boltzmann_distribution(q_values, self.temperature.value(step))
        return (1.0 - weight) * epsilon_part + weight * boltzmann_part

    def diagnostics(
        self, q_values: np.ndarray, step: int, uncertainty: float | None = None
    ) -> dict[str, float]:
        weight, measured_uncertainty = self._boltzmann_weight(q_values, step, uncertainty)
        diagnostics = super().diagnostics(q_values, step, measured_uncertainty)
        diagnostics["boltzmann_weight"] = weight
        diagnostics["uncertainty"] = measured_uncertainty
        return diagnostics


def build_strategy(config: ExplorationConfig) -> ExplorationStrategy:
    names = {
        "epsilon_greedy": EpsilonGreedyStrategy,
        "boltzmann": BoltzmannStrategy,
        "annealed_epsilon_boltzmann": AnnealedEpsilonBoltzmannStrategy,
        "topk_boltzmann": TopKBoltzmannStrategy,
        "uncertainty_switch": UncertaintyAwareSwitchStrategy,
    }
    normalized = config.name.lower().replace("-", "_")
    try:
        return names[normalized](config)
    except KeyError as exc:
        valid = ", ".join(sorted(names))
        raise ValueError(f"Unknown exploration strategy '{config.name}'. Valid: {valid}") from exc
