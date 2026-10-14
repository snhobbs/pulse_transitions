from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional


class EdgeNotFound(IndexError, ValueError):
    """There is no measurable edge where one was asked for; the message says what was missing.
    An IndexError and a ValueError both, as the code it replaces raised either."""


class EdgeSign(Enum):
    falling = -1
    rising = 1
    none = 0


class GroupStrategy(Enum):
    median = "median"
    mode = "mode"
    first = "first"
    last = "last"


@dataclass
class CrossingDetectionSettings:
    """
    window (float): Area around an edge to isolate for threshold crossing
    filter_order (int): Butterworth filter order for smoothing before edge finding
    filter_cutoff(float): Fraction of Nyquist frequency to filter off
    """

    window: float = 0
    filter_order: int = 3
    filter_cutoff: float = 0.9


@dataclass
class Edge:
    """
    Represents an edge transition in a signal.

    Attributes:
        start (float): Time of the low threshold crossing.
        end (float): Time of the high threshold crossing.
        dx (float): Duration between low and high threshold crossings.
        type (str): 'rise' or 'fall'.
    """

    start: float
    end: float
    ymin: float
    ymax: float
    thresholds: Tuple[float, float]
    sign: EdgeSign
    fractional_thresholds: Tuple[float, float] = (0.1, 0.9)

    def __post_init__(self) -> None:
        self.thresholds = tuple(self.thresholds)
        self.fractional_thresholds = tuple(self.fractional_thresholds)
        if self.start >= self.end:
            msg = "Edge start must be before end"
            raise ValueError(msg)

    @property
    def dx(self) -> float:
        return self.end - self.start

    @property
    def bound_low(self) -> Tuple[float, float]:
        return (min([self.end, self.start]), min(self.thresholds))

    @property
    def bound_high(self) -> Tuple[float, float]:
        return (max([self.end, self.start]), max(self.thresholds))


@dataclass
class EdgeMeasurement:
    """One edge and the levels it was measured between: `levels` is (bottom, top) whichever way the edge goes.
    `edge.start` and `edge.end` are the times of the first and the last threshold crossing, so a falling edge
    crosses its high threshold at `start`."""

    edge: Edge
    levels: Tuple[float, float]

    @property
    def duration(self) -> float:
        return self.edge.dx


@dataclass
class PairedEdge:
    rise: Edge
    fall: Edge

    @property
    def pulse_width(self) -> float:
        return self.fall.end - self.rise.start

    @property
    def amplitude(self) -> float:
        return self.rise.ymax - self.rise.ymin

    @property
    def is_valid(self) -> bool:
        return (
            (self.rise.sign == EdgeSign.rising)
            and (self.fall.sign == EdgeSign.falling)
            and (self.fall.start >= self.rise.end)
        )


@dataclass
class Peak:
    sign: EdgeSign
    start: float
    end: float  # x value


@dataclass
class EdgeMetrics:
    fractional_thresholds: Tuple[float, float]
    settling_time_fraction: float
    thresholds: Tuple[float, float]
    levels: Tuple[float, float]
    midcross: float
    risetime: Optional[Tuple[float, Edge]]
    falltime: Optional[Tuple[float, Edge]]
    slewrate: float
    overshoot: Tuple[int, float]
    undershoot: Tuple[int, float]
    settling_time: float
