"""Analytic against measured: how coarse sampling biases the step metrics.

    PYTHONPATH=src python examples/analytic_sampling.py

A noiseless second-order step (damping ratio 0.5) sampled with fewer and fewer samples per
rise time, each record starting at a random sub-sample offset. The library interpolates the
reference crossings linearly between samples, so the rise time stays close at a few samples
per edge; the overshoot is the largest sample, so it reads low once the peak falls between
samples; the settling time is interpolated at the band edge and follows the rise time.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import ACCENT, RISING, save, second_order_exact, second_order_step

WN = 1.0
ZETA = 0.5
SAMPLES_PER_RISE = np.geomspace(1.5, 50.0, 16)
OFFSETS = 40  # random sub-sample start offsets per sampling rate
ABERRATION_FACTOR = 5.0
PANELS = (
    ("rise_time", "Rise time error (%)"),
    ("overshoot", "Overshoot error (points of %)"),
    ("settling_time", "Settling time error (1/ωn)"),
)


def main() -> None:
    exact = second_order_exact(WN, ZETA)
    rng = np.random.default_rng(0)
    result = {name: np.empty((SAMPLES_PER_RISE.size, OFFSETS)) for name, _ in PANELS}
    for i, n in enumerate(SAMPLES_PER_RISE):
        dt = exact.rise_time / n
        for j in range(OFFSETS):
            t = np.arange(-20.0, 80.0, dt) + rng.uniform(0.0, dt)
            m = pt.step_metrics(
                second_order_step(t, WN, ZETA), x=t, aberration_factor=ABERRATION_FACTOR
            )
            result["rise_time"][i, j] = 100 * (
                m.transition.duration / exact.rise_time - 1
            )
            result["overshoot"][i, j] = m.post_overshoot.percent - exact.overshoot
            result["settling_time"][i, j] = m.settling_time - exact.settling_time

    print(
        f"Exact: rise time {exact.rise_time:.3f}, overshoot {exact.overshoot:.2f} %, settling {exact.settling_time:.2f} (1/wn)"
    )
    for i, n in enumerate(SAMPLES_PER_RISE):
        cells = "  ".join(
            f"{name} {np.mean(result[name][i]):+7.3f}" for name, _ in PANELS
        )
        print(f"  {n:5.1f} samples per rise time: {cells}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True)
    for ax, (name, label) in zip(axes, PANELS, strict=True):
        low, high = np.percentile(result[name], [5, 95], axis=1)
        ax.fill_between(
            SAMPLES_PER_RISE,
            low,
            high,
            color=RISING,
            alpha=0.15,
            linewidth=0,
            label=f"5-95 % of {OFFSETS} offsets",
        )
        ax.plot(
            SAMPLES_PER_RISE,
            result[name].mean(axis=1),
            "o-",
            color=RISING,
            markersize=3.5,
            label="Mean",
        )
        ax.axhline(0.0, color="#718096", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50], labels=["2", "5", "10", "20", "50"])
        ax.minorticks_off()
        ax.set_xlabel("Samples per 10-90 % rise time")
        ax.set_ylabel(label)
    # The nearest sample is at most half a sample period from the peak, where the ringing has fallen
    # by about 1 - cos(wd dt / 2) of its height.
    wd = WN * np.sqrt(1 - ZETA**2)
    worst = -exact.overshoot * (1 - np.cos(wd * exact.rise_time / SAMPLES_PER_RISE / 2))
    axes[1].plot(
        SAMPLES_PER_RISE, worst, ":", color=ACCENT, label="Peak half a sample away"
    )
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="lower right")
    fig.suptitle(
        f"Second-order step (ζ = {ZETA:g}, noiseless): measured minus exact as sampling gets coarser",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
