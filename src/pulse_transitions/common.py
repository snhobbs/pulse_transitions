from pydantic import BaseModel, model_validator, Extra
from enum import StrEnum
from dataclasses import dataclass
from typing import Tuple
import numpy as np

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
    window (int): Max width (samples) for expanding peak.
    min_separation (float): Minimum time between edges.
    group_strategy (str): 'first', 'last', 'mean', or 'median'.
    derivative_threshold (float): Minimum |dydx| to be part of peak width.
    hysteresis_window (float): Crossing point hysteresis
    '''
    window: int = 1
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
    sign: int

    @property
    def type(self):
        if self.sign > 0:
            return 'rise'
        return 'fall'

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
        return (self.rise.type == 'rise') and (self.fall.type == 'fall') and (self.fall.start >= self.rise.end)

@dataclass
class Peak:
    sign: int # +1 for rising, -1 for falling
    start: float
    end: float # x value
