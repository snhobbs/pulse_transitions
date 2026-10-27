"""Result records, polarity and the exception hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

__all__ = [
    "Aberration",
    "EdgeMeasurement",
    "EdgeNotFoundError",
    "InputError",
    "LevelError",
    "Polarity",
    "Pulse",
    "PulseTransitionsError",
    "StateLevels",
    "Transition",
    "TransitionMetrics",
]


class Polarity(IntEnum):
    """Direction of a transition or pulse: FALLING is -1, RISING is 1."""

    FALLING = -1
    RISING = 1


class PulseTransitionsError(ValueError):
    """Base of every library error."""


class InputError(PulseTransitionsError):
    """Invalid argument; the message names it."""


class LevelError(PulseTransitionsError):
    """Two state levels cannot be determined."""


class EdgeNotFoundError(PulseTransitionsError):
    """An edge that must exist doesn't; the message says what is missing."""


@dataclass(frozen=True)
class StateLevels:
    """The low and high state levels of a signal, and how they were found."""

    low: float
    high: float
    method: str
    histogram: np.ndarray | None = field(default=None, compare=False, repr=False)
    bin_edges: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def levels(self) -> tuple[float, float]:
        """Return the state levels as a (low, high) tuple."""
        return (self.low, self.high)

    @property
    def amplitude(self) -> float:
        """Return the difference between the high and low state levels."""
        return self.high - self.low


@dataclass(frozen=True)
class Transition:
    """A single edge: its polarity, reference crossing instants and levels, and sample indices."""

    polarity: Polarity
    lower_instant: float
    mid_instant: float
    upper_instant: float
    lower_level: float
    mid_level: float
    upper_level: float
    start_index: int
    end_index: int
    mid_method: str = "median"

    @property
    def start(self) -> float:
        """Return the earlier of the lower and upper reference instants."""
        return min(self.lower_instant, self.upper_instant)

    @property
    def end(self) -> float:
        """Return the later of the lower and upper reference instants."""
        return max(self.lower_instant, self.upper_instant)

    @property
    def duration(self) -> float:
        """Return the edge duration, always positive regardless of polarity."""
        return self.end - self.start

    @property
    def slew_rate(self) -> float:
        """Return the signed slew rate between the lower and upper reference crossings."""
        return (self.upper_level - self.lower_level) / (
            self.upper_instant - self.lower_instant
        )


@dataclass(frozen=True)
class Aberration:
    """An overshoot or undershoot extreme: its percentage of amplitude, instant and value."""

    percent: float
    instant: float
    value: float


@dataclass(frozen=True)
class TransitionMetrics:
    """Full measurement of one transition: levels, aberrations and settling time."""

    transition: Transition
    levels: tuple[float, float]
    pre_overshoot: Aberration
    pre_undershoot: Aberration
    post_overshoot: Aberration
    post_undershoot: Aberration
    settling_time: float


@dataclass(frozen=True)
class Pulse:
    """A pulse formed by a first and second transition, with its timing and duty cycle."""

    polarity: Polarity
    first: Transition
    second: Transition
    width: float
    period: float
    separation: float
    duty_cycle: float


@dataclass(frozen=True)
class EdgeMeasurement:
    """A transition measured between a pair of levels, whatever its direction."""

    transition: Transition
    levels: tuple[float, float]

    @property
    def duration(self) -> float:
        """Return the duration of the underlying transition."""
        return self.transition.duration
