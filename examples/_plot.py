"""Shared figure style, test signals and SVG export for the example scripts.

Each example saves one figure, examples/figures/<script name>.svg, which the Makefile
uses to rebuild only the figures whose script or library code changed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

FIGURES = Path(__file__).resolve().parent / "figures"

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
    y = 1.0 - np.exp(-zeta * wn * tp) / np.sqrt(1.0 - zeta**2) * np.sin(wd * tp + np.arccos(zeta))
    return np.where(t >= 0.0, y, 0.0)


def ramp_edges(t: FloatArray, edges: list[tuple[float, float, float]]) -> FloatArray:
    """Sum of linear ramps (start instant, duration, step height) starting from zero."""
    y = np.zeros_like(t)
    for start, duration, height in edges:
        y += height * np.clip((t - start) / duration, 0.0, 1.0)
    return y


def save(fig: Figure, script: str) -> Path:
    """Save `fig` as examples/figures/<stem of script>.svg and close it."""
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{Path(script).stem}.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    print(f"Saved {path}")
    return path
