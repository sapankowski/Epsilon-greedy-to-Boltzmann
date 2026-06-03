"""Small deterministic schedules used by exploration strategies."""

from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a scalar into a closed interval."""

    return max(lower, min(upper, value))


def smoothstep(value: float) -> float:
    """Smoothly interpolate from 0 to 1 for values already normalized to [0, 1]."""

    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class LinearSchedule:
    """Linear interpolation between two values after an optional delay."""

    start: float
    end: float
    duration: int
    start_step: int = 0

    def value(self, step: int) -> float:
        if self.duration <= 0:
            return self.end
        progress = clamp((step - self.start_step) / float(self.duration))
        return self.start + progress * (self.end - self.start)


@dataclass(frozen=True)
class SwitchSchedule:
    """A named schedule for moving from one exploration regime to another."""

    start_step: int
    end_step: int
    smooth: bool = True

    def value(self, step: int) -> float:
        duration = max(1, self.end_step - self.start_step)
        progress = clamp((step - self.start_step) / float(duration))
        return smoothstep(progress) if self.smooth else progress
