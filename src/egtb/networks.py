"""Q-network definitions and observation tensor conversion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def is_image_shape(observation_shape: Sequence[int]) -> bool:
    return len(tuple(observation_shape)) >= 2


def canonical_image_shape(observation_shape: Sequence[int]) -> tuple[int, int, int]:
    """Return image observations as ``channels, height, width``."""

    shape = tuple(int(dim) for dim in observation_shape)
    if len(shape) == 2:
        height, width = shape
        return 1, height, width
    if len(shape) == 3:
        if shape[-1] in (1, 3) and shape[0] not in (1, 3, 4):
            height, width, channels = shape
            return channels, height, width
        channels, height, width = shape
        return channels, height, width
    if len(shape) == 4:
        stack, height, width, channels = shape
        if channels in (1, 3):
            return stack * channels, height, width
        channels, stack_height, width, extra = shape
        return channels * extra, stack_height, width
    raise ValueError(f"Unsupported image observation shape: {shape}")


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_count: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        input_size = int(np.prod(tuple(observation_shape)))
        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, int(hidden_size)))
            layers.append(nn.ReLU())
            previous_size = int(hidden_size)
        layers.append(nn.Linear(previous_size, action_count))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations.flatten(start_dim=1))


class NatureCNNQNetwork(nn.Module):
    """DQN convolutional torso for Atari-sized image observations."""

    def __init__(self, observation_shape: Sequence[int], action_count: int) -> None:
        super().__init__()
        channels, height, width = canonical_image_shape(observation_shape)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_size = int(self.conv(dummy).shape[1])
        self.head = nn.Sequential(
            nn.Linear(conv_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(observations))


def make_q_network(
    observation_shape: Sequence[int],
    action_count: int,
    hidden_sizes: Sequence[int],
) -> nn.Module:
    if is_image_shape(observation_shape):
        return NatureCNNQNetwork(observation_shape, action_count)
    return MLPQNetwork(observation_shape, action_count, hidden_sizes)


def _to_channel_first(
    observations: torch.Tensor, observation_shape: Sequence[int]
) -> torch.Tensor:
    shape = tuple(observation_shape)
    if len(shape) == 2:
        return observations.unsqueeze(1)
    if len(shape) == 3:
        if shape[-1] in (1, 3) and shape[0] not in (1, 3, 4):
            return observations.permute(0, 3, 1, 2)
        return observations
    if len(shape) == 4 and shape[-1] in (1, 3):
        batch, stack, height, width, channels = observations.shape
        return observations.permute(0, 1, 4, 2, 3).reshape(
            batch, stack * channels, height, width
        )
    return observations


def observations_to_tensor(
    observations: np.ndarray | torch.Tensor,
    observation_shape: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Convert single or batched observations into model-ready tensors."""

    array = observations
    if not torch.is_tensor(array):
        array = torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)
    else:
        array = array.to(device=device, dtype=torch.float32)

    if tuple(array.shape) == tuple(observation_shape):
        array = array.unsqueeze(0)

    if is_image_shape(observation_shape):
        array = _to_channel_first(array, observation_shape)
        if torch.max(array).item() > 1.5:
            array = array / 255.0
    return array
