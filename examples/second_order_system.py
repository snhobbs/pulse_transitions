"""Measure a noisy, underdamped second-order step response with pulse_transitions.

Run from the project root:

    PYTHONPATH=src python examples/second_order_system.py

The script synthesises the closed-form step response of a second-order system,
adds a little measurement noise, and optionally low-pass filters it with scipy
before measuring. The library itself needs only numpy (decision D5); the scipy
pre-filter is shown for users who want it and is skipped when scipy is missing.
It then prints the rise time, mid instant, overshoot, undershoot and settling
time, and saves an annotated plot to examples/second-order-annotations.png.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import pulse_transitions as pt  # noqa: E402

WN = 30.0  # natural frequency (rad/s); the step has settled well before t = 1
ZETA = 0.2  # damping ratio
NOISE_SIGMA = 0.005  # standard deviation of the added noise (V)
REFERENCE = (10.0, 50.0, 90.0)  # percent levels used for the rise time
SAVE_PATH = Path(__file__).resolve().with_name("second-order-annotations.png")


def second_order_step(
    t: npt.NDArray[np.float64], wn: float = WN, zeta: float = ZETA
) -> npt.NDArray[np.float64]:
    """Closed-form unit step response of an underdamped second-order system, zero for t < 0."""
    wd = wn * np.sqrt(1.0 - zeta**2)
    phi = np.arccos(zeta)
    t_pos = np.clip(t, 0.0, None)
    y = 1.0 - np.exp(-zeta * wn * t_pos) / np.sqrt(1.0 - zeta**2) * np.sin(wd * t_pos + phi)
    return np.where(t >= 0.0, y, 0.0)


def lowpass_if_available(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Zero-phase Butterworth low-pass (order 3, cutoff 0.2 of Nyquist), or `y` if scipy is absent."""
    try:
        from scipy.signal import butter
        from scipy.signal import filtfilt
    except ImportError:
        print("scipy not installed: measuring the unfiltered signal")
        return y
    b, a = butter(3, 0.2)
    print("scipy found: low-pass filtered (butter order 3, cutoff 0.2) before measuring")
    return filtfilt(b, a, y)


def make_annotations_plot(
    t: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    levels: tuple[float, float],
    transition: pt.Transition,
    settling_time: float,
) -> tuple[Figure, Axes]:
    """Plot the signal with its reference levels, mid instant and settling instant."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, y, color="C0", label="Signal")

    lower, mid, upper = pt.reference_levels(levels, REFERENCE)
    for percent, value, color in zip(REFERENCE, (lower, mid, upper), ("C1", "C2", "C3"), strict=True):
        ax.axhline(value, linestyle="--", color=color, label=f"{percent:.0f} % level")

    ax.axvline(transition.mid_instant, linestyle=":", color="black", label="Mid instant")
    if isfinite(settling_time):
        settle_instant = transition.mid_instant + settling_time
        ax.axvline(settle_instant, linestyle="-.", color="C4", label="Settling instant")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("Second-order step response with transition metrics")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.5, color="lightgray")  # noqa: FBT003
    fig.tight_layout()
    return fig, ax


def main() -> None:
    t = np.linspace(-0.25, 1.0, 1000)
    clean = second_order_step(t)
    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(0.0, NOISE_SIGMA, t.size)
    y = lowpass_if_available(noisy)

    levels = pt.state_levels(y, x=t)
    print(f"State levels: low {levels.low:.4f} V, high {levels.high:.4f} V")

    step = pt.step_metrics(y, x=t, levels=levels.levels)
    transition = step.transition
    print(f"Rise time ({REFERENCE[0]:.0f}-{REFERENCE[2]:.0f} %): {transition.duration:.4f} s")
    print(f"Mid instant: {transition.mid_instant:.4f} s")
    print(f"Post-overshoot: {step.post_overshoot.percent:.2f} %")
    print(f"Post-undershoot: {step.post_undershoot.percent:.2f} %")
    print(f"Settling time: {step.settling_time:.4f} s")

    fig, _ = make_annotations_plot(t, y, levels.levels, transition, step.settling_time)
    fig.savefig(SAVE_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {SAVE_PATH}")


if __name__ == "__main__":
    main()
