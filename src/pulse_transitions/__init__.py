"""Pulse and transition metrics (IEEE Std 181-2003, MATLAB-compatible)."""
from . import matlab
from ._api import (
    flatness,
    measure_edge,
    pulses,
    reference_levels,
    state_levels,
    step_metrics,
    transition_metrics,
    transitions,
)
from ._types import (
    Aberration,
    EdgeMeasurement,
    EdgeNotFoundError,
    InputError,
    LevelError,
    Polarity,
    Pulse,
    PulseTransitionsError,
    StateLevels,
    Transition,
    TransitionMetrics,
)

__version__ = "0.3.0"

__all__ = [
    "Aberration", "EdgeMeasurement", "EdgeNotFoundError", "InputError", "LevelError", "Polarity",
    "Pulse", "PulseTransitionsError", "StateLevels", "Transition", "TransitionMetrics",
    "flatness", "matlab", "measure_edge", "pulses", "reference_levels", "state_levels",
    "step_metrics", "transition_metrics", "transitions",
]
