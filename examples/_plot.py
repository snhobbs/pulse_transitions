"""Shared figure style, test signals and SVG export for the example scripts.

Each example saves one figure, examples/figures/<script name>.svg, which the Makefile
uses to rebuild only the figures whose script or library code changed. second_order_step and
second_order_exact give the closed-form response and its exact metrics for the analytic_* examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# PT_FIGURES redirects the figures, so the smoke test (tests/test_examples.py) leaves examples/figures alone.
FIGURES = Path(
    os.environ.get("PT_FIGURES", Path(__file__).resolve().parent / "figures")
)
DATA = Path(__file__).resolve().parent / "data"

SIGNAL = "#4a5568"  # the waveform
RISING = "#2b6cb0"
FALLING = "#c05621"
ACCENT = "#2f855a"
REFERENCE = "#a0aec0"

plt.rcParams.update(
    {
        "figure.figsize": (9, 4.5),
        "figure.dpi": 100,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.4,
        "svg.hashsalt": "pulse_transitions",  # stable element ids, so unchanged figures don't diff
    }
)

FloatArray = npt.NDArray[np.float64]


def second_order_step(t: FloatArray, wn: float, zeta: float) -> FloatArray:
    """Unit step response of an underdamped second-order system, zero for t < 0."""
    wd = wn * np.sqrt(1.0 - zeta**2)
    tp = np.clip(t, 0.0, None)
    y = 1.0 - np.exp(-zeta * wn * tp) / np.sqrt(1.0 - zeta**2) * np.sin(
        wd * tp + np.arccos(zeta)
    )
    return np.where(t >= 0.0, y, 0.0)


@dataclass(frozen=True)
class SecondOrderExact:
    """Step-response metrics of `second_order_step`, from its closed form rather than from samples."""

    overshoot: float  # percent: the first peak, 100 exp(-pi zeta / sqrt(1 - zeta^2))
    undershoot: float  # percent: the first trough below the final level, the overshoot squared / 100
    rise_time: float  # 10-90 %
    mid_instant: float  # the 50 % crossing
    settling_time: float  # from the mid instant to the last entry into the tolerance band, as the library measures it


def second_order_exact(
    wn: float, zeta: float, state_tolerance: float = 2.0
) -> SecondOrderExact:
    """The exact metrics of `second_order_step`: the peaks in closed form, the crossings by root finding."""
    from scipy.optimize import brentq

    def crossing(level: float, after: float = 0.0) -> float:
        """The first instant after `after` where the response crosses `level`."""
        grid = np.linspace(after, after + 200.0 / (zeta * wn), 20_001)
        above = second_order_step(grid, wn, zeta) >= level
        k = int(np.flatnonzero(above != above[0])[0])
        return brentq(
            lambda t: second_order_step(np.array([t]), wn, zeta)[0] - level,
            grid[k - 1],
            grid[k],
        )

    peak = np.exp(-np.pi * zeta / np.sqrt(1.0 - zeta**2))
    tol = state_tolerance / 100
    # The response leaves the band for the last time at an extremum whose envelope still exceeds the
    # tolerance; the extrema sit at multiples of pi / wd, and the n-th has excursion peak**n.
    wd = wn * np.sqrt(1.0 - zeta**2)
    n = int(np.floor(np.log(tol) / np.log(peak))) if peak > tol else 0
    if n == 0:
        last = crossing(1.0 - tol)
    else:
        level = 1.0 + tol if n % 2 else 1.0 - tol
        last = crossing(level, after=n * np.pi / wd)
    mid = crossing(0.5)
    return SecondOrderExact(
        100 * peak, 100 * peak**2, crossing(0.9) - crossing(0.1), mid, last - mid
    )


def lowpass(y: FloatArray, fs: float, cutoff: float) -> FloatArray:
    """Zero-phase third-order Butterworth low-pass (scipy) with its -3 dB point at `cutoff` (same units as fs)."""
    from scipy.signal import butter, filtfilt

    b, a = butter(3, cutoff, fs=fs)
    return filtfilt(b, a, y)


def ramp_edges(t: FloatArray, edges: list[tuple[float, float, float]]) -> FloatArray:
    """Sum of linear ramps (start instant, duration, step height) starting from zero."""
    y = np.zeros_like(t)
    for start, duration, height in edges:
        y += height * np.clip((t - start) / duration, 0.0, 1.0)
    return y


def post_region(
    x: FloatArray,
    y: FloatArray,
    metrics,
    aberration_factor: float = 3.0,
    state_tolerance: float = 2.0,
) -> tuple[float, float]:
    """The post-transition aberration region the library measured: from the first sample after the
    edge that has reached the final level (else its band), for aberration_factor edge durations (§C14)."""
    edge = metrics.transition
    low, high = metrics.levels
    final = high if edge.polarity > 0 else low
    tol = state_tolerance / 100 * (high - low)
    distance = int(edge.polarity) * (y[edge.end_index :] - final)
    reached = (
        np.flatnonzero(distance >= 0)
        if np.any(distance >= 0)
        else np.flatnonzero(distance >= -tol)
    )
    start = x[edge.end_index + reached[0]]
    return float(start), float(start + aberration_factor * edge.duration)


def load_csv(name: str) -> np.ndarray:
    """A measured trace from examples/data/<name>, as a structured array keyed by column name."""
    return np.genfromtxt(DATA / name, delimiter=",", names=True)


def save(fig: Figure, script: str) -> Path:
    """Save `fig` as examples/figures/<stem of script>.svg and close it."""
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{Path(script).stem}.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    print(f"Saved {path}")
    return path
