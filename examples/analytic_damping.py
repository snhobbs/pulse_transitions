"""Analytic against measured: step metrics of a noisy second-order system across damping ratios.

    PYTHONPATH=src python examples/analytic_damping.py

For each damping ratio the script synthesises the closed-form step response, adds 0.5 % RMS
noise, low-pass filters it and measures it with step_metrics, over many noise seeds. The
lines are the exact values from the closed form (second_order_exact in _plot.py), the points
the mean of the measurements with their 5-95 % spread. Settling time is a staircase in the
damping ratio: each step is where another ringing peak drops inside the ±2 % band.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import RISING, SIGNAL, lowpass, save, second_order_exact, second_order_step

WN = 1.0  # natural frequency (rad/s), so time is in units of 1/wn
ZETAS = np.linspace(0.1, 0.95, 18)
NOISE = 0.005  # RMS noise, as a fraction of the step
CUTOFF = 15.0  # low-pass -3 dB point, in wn
SEEDS = 30
ABERRATION_FACTOR = 5.0  # long enough to reach the first trough at every damping ratio
DT = 0.02
T = np.arange(-20.0, 80.0, DT)

PANELS = (
    ("overshoot", "Overshoot (%)"),
    ("undershoot", "Undershoot (%)"),
    ("rise_time", "10-90 % rise time (1/ωn)"),
    ("settling_time", "Settling time, ±2 % (1/ωn)"),
)


def measure(y: np.ndarray) -> dict[str, float]:
    m = pt.step_metrics(y, x=T, aberration_factor=ABERRATION_FACTOR)
    return {
        "overshoot": m.post_overshoot.percent,
        "undershoot": m.post_undershoot.percent,
        "rise_time": m.transition.duration,
        "settling_time": m.settling_time,
    }


def main() -> None:
    dense = np.linspace(ZETAS[0], ZETAS[-1], 200)
    exact = [second_order_exact(WN, z) for z in dense]
    measured = {name: np.empty((ZETAS.size, SEEDS)) for name, _ in PANELS}
    for i, zeta in enumerate(ZETAS):
        clean = second_order_step(T, WN, zeta)
        for seed in range(SEEDS):
            noisy = clean + np.random.default_rng(seed).normal(0.0, NOISE, T.size)
            for name, value in measure(
                lowpass(noisy, 1 / DT, CUTOFF * WN / (2 * np.pi))
            ).items():
                measured[name][i, seed] = value

    print(
        " zeta  overshoot exact/measured  rise time exact/measured  settling exact/measured"
    )
    for i, zeta in enumerate(ZETAS):
        e = second_order_exact(WN, zeta)
        mean = {name: np.nanmean(measured[name][i]) for name, _ in PANELS}
        print(
            f" {zeta:4.2f}  {e.overshoot:7.2f} / {mean['overshoot']:6.2f} %      {e.rise_time:5.3f} / {mean['rise_time']:5.3f}"
            f"            {e.settling_time:6.2f} / {mean['settling_time']:6.2f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (name, label) in zip(axes.flat, PANELS, strict=True):
        ax.plot(
            dense,
            [getattr(e, name) for e in exact],
            color=SIGNAL,
            label="Exact (closed form)",
        )
        mean = np.nanmean(measured[name], axis=1)
        low, high = np.nanpercentile(measured[name], [5, 95], axis=1)
        ax.errorbar(
            ZETAS,
            mean,
            yerr=(mean - low, high - mean),
            fmt="o",
            color=RISING,
            markersize=4,
            capsize=2,
            linewidth=1,
            label=f"Measured, mean and 5-95 % of {SEEDS} seeds",
        )
        ax.set_ylabel(label)
    axes[0, 0].legend(loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("Damping ratio ζ")
    fig.suptitle(
        f"Second-order step: exact against measured, {NOISE * 100:g} % RMS noise, "
        f"low-pass at {CUTOFF:g} ωn",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
