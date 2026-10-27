"""step_metrics: measure a noisy, underdamped second-order step response.

    PYTHONPATH=src python examples/step_metrics.py

The script synthesises the closed-form step response of a second-order system, adds a
little measurement noise, and optionally low-pass filters it with scipy before measuring.
The library itself needs only numpy; the scipy pre-filter is shown for users who want it
and is skipped when scipy is missing. step_metrics requires exactly one transition in the
data (use bounds= to pick one out of a longer record).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import pulse_transitions as pt
from _plot import (
    ACCENT,
    FALLING,
    REFERENCE,
    RISING,
    SIGNAL,
    post_region,
    second_order_step,
    save,
)

WN = 30.0  # natural frequency (rad/s); the step has settled well before t = 1
ZETA = 0.2  # damping ratio
NOISE_SIGMA = 0.005  # standard deviation of the added noise (V)
LEVELS = (10.0, 50.0, 90.0)  # percent reference levels used for the rise time
TOLERANCE = 2.0  # settling band, percent of the amplitude
# Post-transition region length in edge durations. The default, 3 (MATLAB's too), ends before the first
# trough of this underdamped response, so the undershoot would be read at the region's end; 5 covers it.
ABERRATION_FACTOR = 5.0


def lowpass_if_available(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Zero-phase Butterworth low-pass (order 3, cutoff 0.2 of Nyquist), or `y` if scipy is absent."""
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        print("scipy not installed: measuring the unfiltered signal")
        return y
    b, a = butter(3, 0.2)
    print(
        "scipy found: low-pass filtered (butter order 3, cutoff 0.2) before measuring"
    )
    return filtfilt(b, a, y)


def main() -> None:
    t = np.linspace(-0.25, 1.0, 1000)
    noisy = second_order_step(t, WN, ZETA) + np.random.default_rng(0).normal(
        0.0, NOISE_SIGMA, t.size
    )
    y = lowpass_if_available(noisy)

    step = pt.step_metrics(
        y,
        x=t,
        reference=LEVELS,
        state_tolerance=TOLERANCE,
        aberration_factor=ABERRATION_FACTOR,
    )
    edge = step.transition
    low, high = step.levels
    print(f"State levels: low {low:.4f} V, high {high:.4f} V")
    print(f"Rise time ({LEVELS[0]:.0f}-{LEVELS[2]:.0f} %): {edge.duration:.4f} s")
    print(f"Mid instant: {edge.mid_instant:.4f} s")
    print(f"Post-overshoot: {step.post_overshoot.percent:.2f} %")
    print(f"Post-undershoot: {step.post_undershoot.percent:.2f} %")
    print(f"Settling time: {step.settling_time:.4f} s")

    fig, ax = plt.subplots()
    ax.plot(t, noisy, color="#cbd5e0", linewidth=0.8, label="Measured")
    ax.plot(t, y, color=SIGNAL, label="Filtered")
    band = (high - low) * TOLERANCE / 100
    ax.axhspan(
        high - band,
        high + band,
        color=ACCENT,
        alpha=0.12,
        linewidth=0,
        label=f"±{TOLERANCE:g} % band",
    )
    for percent, value in zip(
        LEVELS, (edge.lower_level, edge.mid_level, edge.upper_level), strict=True
    ):
        ax.axhline(value, color=REFERENCE, linestyle="--", linewidth=0.8)
        ax.annotate(
            f"{percent:.0f} %",
            (t[0], value),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=8,
            color="#718096",
        )

    ax.axvspan(
        edge.start,
        edge.end,
        color=RISING,
        alpha=0.15,
        linewidth=0,
        label=f"Rise time {edge.duration * 1e3:.1f} ms",
    )
    ax.axvspan(
        *post_region(t, y, step, ABERRATION_FACTOR, TOLERANCE),
        color=FALLING,
        alpha=0.08,
        linewidth=0,
        label="Post-transition aberration region",
    )
    for aberration, name, dy, color in (
        (step.post_overshoot, "overshoot", 6, FALLING),
        (step.post_undershoot, "undershoot", -14, FALLING),
    ):
        ax.plot(
            aberration.instant,
            aberration.value,
            "v" if dy > 0 else "^",
            color=color,
            markersize=7,
        )
        ax.annotate(
            f"{name} {aberration.percent:.1f} %",
            (aberration.instant, aberration.value),
            xytext=(6, dy),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    settle = edge.mid_instant + step.settling_time
    ax.annotate(
        "",
        xy=(settle, 0.25),
        xytext=(edge.mid_instant, 0.25),
        arrowprops={"arrowstyle": "<->", "color": ACCENT, "linewidth": 1.0},
    )
    ax.text(
        (edge.mid_instant + settle) / 2,
        0.27,
        f"settling time {step.settling_time * 1e3:.0f} ms",
        ha="center",
        va="bottom",
        fontsize=8,
        color=ACCENT,
    )
    ax.axvline(settle, color=ACCENT, linestyle=":", linewidth=1.0)

    ax.set_title("step_metrics: second-order step response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
