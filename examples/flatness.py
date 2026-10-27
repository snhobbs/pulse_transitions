"""flatness: how flat a window of the signal is, normalised to the final level.

    PYTHONPATH=src python examples/flatness.py

flatness is the standard deviation of the samples inside `bounds`, divided by the
record's final level (pass normalize=False for volts). Here it compares the ringing just
after a step with the settled top, and tracks a sliding window across the record.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import ACCENT, FALLING, RISING, SIGNAL, second_order_step, save

WINDOW = 0.1  # s


def main() -> None:
    t = np.linspace(-0.2, 1.5, 6000)
    y = 2.0 * second_order_step(t, wn=30.0, zeta=0.15) + np.random.default_rng(6).normal(0.0, 0.004, t.size)

    windows = {"ringing": ((0.1, 0.4), FALLING), "settled": ((1.1, 1.4), RISING)}
    fig, (ax, flat_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5), gridspec_kw={"height_ratios": [2.2, 1]})
    ax.plot(t, y, color=SIGNAL, linewidth=1.0)
    for name, (bounds, color) in windows.items():
        value = pt.flatness(y, x=t, bounds=bounds)
        volts = pt.flatness(y, x=t, bounds=bounds, normalize=False)
        print(f"{name:>8} {bounds}: flatness {value * 100:.3f} % ({volts * 1e3:.2f} mV)")
        ax.axvspan(*bounds, color=color, alpha=0.12, linewidth=0)
        ax.text(sum(bounds) / 2, 3.55, f"{name}\nflatness {value * 100:.2f} %", ha="center", va="top",
                fontsize=8, color=color)
    ax.set_ylim(-0.1, 3.7)
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("flatness: standard deviation of a window, as a fraction of the final level")

    centres = np.linspace(t[0] + WINDOW / 2, t[-1] - WINDOW / 2, 300)
    sliding = [pt.flatness(y, x=t, bounds=(c - WINDOW / 2, c + WINDOW / 2)) * 100 for c in centres]
    flat_ax.semilogy(centres, sliding, color=ACCENT)
    flat_ax.set_ylabel(f"Flatness, {WINDOW * 1e3:.0f} ms\nwindow (%)")
    flat_ax.set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
