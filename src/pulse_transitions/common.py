from enum import Enum
from enum import StrEnum
from typing import Tuple

from pydantic import BaseModel
from pydantic import Extra
from pydantic import model_validator


class EdgeSign(Enum):
    falling = -1
    rising = 1
    none = 0

class StrictBaseModel(BaseModel):
    class Config:
        extra = Extra.forbid  # No extra fields allowed

class GroupStrategy(StrEnum):
    median = "median"
    mode = "mode"
    first = "first"
    last = "last"

class CrossingDetectionSettings(StrictBaseModel):
    """
    window (float): Area around an edge to isolate for threshold crossing
    filter_order (int): Butterworth filter order for smoothing before edge finding
    filter_cutoff(float): Fraction of Nyquist frequency to filter off
    """
    window: float = 0
    filter_order: int = 3
    filter_cutoff: float = 0.9

class Edge(BaseModel):
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

    @model_validator(mode="after")
    def validate_edge(self) -> "Edge":
        if self.start >= self.end:
            raise ValueError("Edge start must be before end")
        return self

    @property
    def dx(self) -> float:
        return self.end - self.start

    @property
    def bound_low(self) -> Tuple[float, float]:
        return (min([self.end, self.start]), min(self.thresholds))

    @property
    def bound_high(self) -> Tuple[float, float]:
        return (max([self.end, self.start]), max(self.thresholds))

class PairedEdge(BaseModel):
    rise: Edge
    fall: Edge

    @property
    def pulse_width(self) -> float:
        return self.fall.end - self.rise.start

    @property
    def amplitude(self) -> float:
        return self.rise.ymax - self.fall.ymin

    @property
    def is_valid(self) -> bool:
        return ((self.rise.sign == EdgeSign.rising) and
                (self.fall.sign == EdgeSign.falling) and
                (self.fall.start >= self.rise.end))


class Peak(BaseModel):
    sign: EdgeSign
    start: float
    end: float # x value
