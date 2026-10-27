"""Analytic against measured: how ADC resolution biases the step metrics.

    PYTHONPATH=src python examples/analytic_quantization.py

A noiseless second-order step (damping ratio 0.5) spanning 80 % of the full scale, quantised
to 4 to 16 bits, each record at a random sub-LSB offset and a random sample phase. Every
quantised value is within half an LSB of the true one, so the overshoot (the largest sample
against the state level) is within about one LSB of exact, and the reference crossings move by
up to half an LSB over the slope there. The settling time needs the ±2 % band to span several
LSBs: once one LSB is a sizeable part of the band, the last sample to leave it is decided by
rounding, and the settling time jumps between ringing peaks. Above that every error halves
with each added bit, following half an LSB over the slope, down to a floor that no longer
depends on the bits: the histogram's estimate of the state levels, as in the noiseless case.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import ACCENT, REFERENCE, RISING, save, second_order_exact, second_order_step

WN = 1.0
ZETA = 0.5
BITS = np.arange(4, 17)
SPAN = 0.8  # step height as a fraction of the full scale
OFFSETS = 40  # random sub-LSB offsets and sample phases per resolution
ABERRATION_FACTOR = 5.0
TOLERANCE = 2.0  # settling band, percent of the step (the library's default)
DT = 0.02
PANELS = (
    ("rise_time", "|Rise time error| (%)"),
    ("overshoot", "|Overshoot error| (points of %)"),
    ("settling_time", "|Settling time error| (1/ωn)"),
)


def lsb(bits: int) -> float:
    """One LSB as a fraction of the step."""
    return 1.0 / (SPAN * 2.0**bits)


def _crossing(level: float) -> float:
    """The first instant the response reaches `level`, from a fine grid (it rises monotonically to the first peak)."""
    t = np.linspace(0.0, np.pi / (WN * np.sqrt(1 - ZETA**2)), 200_001)
    return float(np.interp(level, second_order_step(t, WN, ZETA), t))


def main() -> None:
    exact = second_order_exact(WN, ZETA, TOLERANCE)
    rng = np.random.default_rng(0)
    result = {name: np.empty((BITS.size, OFFSETS)) for name, _ in PANELS}
    for i, bits in enumerate(BITS):
        q = lsb(bits)
        for j in range(OFFSETS):
            t = np.arange(-20.0, 80.0, DT) + rng.uniform(0.0, DT)
            offset = rng.uniform(0.0, q)
            y = np.round((second_order_step(t, WN, ZETA) + offset) / q) * q - offset
            m = pt.step_metrics(
                y, x=t, aberration_factor=ABERRATION_FACTOR, state_tolerance=TOLERANCE
            )
            result["rise_time"][i, j] = 100 * (
                m.transition.duration / exact.rise_time - 1
            )
            result["overshoot"][i, j] = m.post_overshoot.percent - exact.overshoot
            result["settling_time"][i, j] = m.settling_time - exact.settling_time

    print(
        f"Exact: rise time {exact.rise_time:.3f}, overshoot {exact.overshoot:.2f} %, settling {exact.settling_time:.2f} (1/wn)"
    )
    for i, bits in enumerate(BITS):
        cells = "  ".join(
            f"{name} {np.mean(result[name][i]):+7.3f} ({np.std(result[name][i]):.3f})"
            for name, _ in PANELS
        )
        print(f"  {bits:2d} bits (LSB {100 * lsb(bits):6.3f} % of step): {cells}")

    # Guides: half an LSB at each crossing, over the slope there. The rise time moves by both of its
    # crossings, the settling time by the band exit on the last ringing peak; the overshoot by one LSB
    # (the peak and the state level are each within half an LSB).
    def slope(at: float) -> float:
        return float(
            np.diff(second_order_step(np.array([at - 1e-6, at + 1e-6]), WN, ZETA))[0]
            / 2e-6
        )

    lower = _crossing(0.1)
    upper = _crossing(0.9)
    settle = exact.mid_instant + exact.settling_time
    q = np.array([lsb(b) for b in BITS])
    guides = {
        "rise_time": (
            100 * q / 2 * (1 / slope(lower) + 1 / slope(upper)) / exact.rise_time,
            "½ LSB / slope, both crossings",
        ),
        "overshoot": (100 * q, "1 LSB"),
        "settling_time": (q / 2 / abs(slope(settle)), "½ LSB / slope at the band exit"),
    }
    # The bit depth at which one LSB is a quarter of the ±2 % band's width.
    coarse = np.log2(4 / (SPAN * 2 * TOLERANCE / 100))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True)
    for ax, (name, label) in zip(axes, PANELS, strict=True):
        error = np.abs(result[name])
        ax.semilogy(
            BITS,
            np.percentile(error, 95, axis=1),
            "o-",
            color=RISING,
            markersize=3.5,
            label=f"95th percentile of {OFFSETS} offsets",
        )
        ax.semilogy(
            BITS,
            error.mean(axis=1),
            "s--",
            color=RISING,
            markersize=3,
            linewidth=0.8,
            alpha=0.7,
            label="Mean",
        )
        ax.semilogy(BITS, guides[name][0], ":", color=ACCENT, label=guides[name][1])
        ax.axvspan(
            BITS[0] - 0.5,
            coarse,
            color=REFERENCE,
            alpha=0.2,
            linewidth=0,
            label="LSB > ¼ of the ±2 % band" if name == "settling_time" else None,
        )
        ax.set_xlabel("ADC bits (step = 80 % of full scale)")
        ax.set_ylabel(label)
        ax.set_xlim(BITS[0] - 0.5, BITS[-1] + 0.5)
        ax.set_xticks(BITS[::2])
        ax.legend(
            loc="lower left" if name == "settling_time" else "upper right", fontsize=8
        )
    fig.suptitle(
        f"Second-order step (ζ = {ZETA:g}, noiseless): measured minus exact as the ADC gets coarser",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
