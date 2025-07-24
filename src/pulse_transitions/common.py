from enum import Enum, StrEnum
from dataclasses import dataclass
from typing import Tuple
from pydantic import BaseModel, model_validator, Extra
import numpy as np

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
    '''
    window (float): Area around an edge to isolate for threshold crossing
    filter_order (int): Butterworth filter order for smoothing before edge finding
    filter_cutoff(float): Fraction of Nyquist frequency to filter off
    '''
    window: float = 0
    filter_order: int = 3
    filter_cutoff: float = 0.9

class Edge(BaseModel):
    '''
    Represents an edge transition in a signal.

    Attributes:
        start (float): Time of the low threshold crossing.
        end (float): Time of the high threshold crossing.
        dx (float): Duration between low and high threshold crossings.
        type (str): 'rise' or 'fall'.
    '''
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
    def dx(self):
        return abs(self.end - self.start)

    @property
    def bound_low(self):
        return [min([self.end, self.start]), min(self.thresholds)]

    @property
    def bound_high(self):
        return [max([self.end, self.start]), max(self.thresholds)]

def closest_index(arr: np.ndarray, value: float) -> int:
    '''
    Return the index of the point closest to the given value
    '''
    return np.abs(arr - value).argmin()

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
