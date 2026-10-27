"""transition_metrics: overshoot, undershoot and settling time of every edge of a ringing pulse.

    PYTHONPATH=src python examples/transition_metrics.py

The post-transition aberration region runs from the end of each edge for
aberration_factor (default 3) edge durations. Its overshoot and undershoot are given as a
percentage of the amplitude, and the settling time runs from the mid instant to the last
entry into the state_tolerance band (default 2 %) around the final level.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import ACCENT, FALLING, RISING, SIGNAL, second_order_step, save

TOLERANCE = 2.0  # percent of the amplitude
ABERRATION_FACTOR = 3.0  # post-transition region length, in edge durations


def main() -> None:
    t = np.linspace(0.0, 2.0, 8000)
    y = sum(sign * second_order_step(t - start, wn=40.0, zeta=0.3) for start, sign in ((0.2, 1), (1.1, -1)))
    y += np.random.default_rng(3).normal(0.0, 0.003, t.size)

    metrics = pt.transition_metrics(y, x=t, state_tolerance=TOLERANCE, aberration_factor=ABERRATION_FACTOR)
    low, high = metrics[0].levels
    band = (high - low) * TOLERANCE / 100

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, y, color=SIGNAL, linewidth=1.0)
    for level in (low, high):
        ax.axhspan(level - band, level + band, color=ACCENT, alpha=0.12, linewidth=0)

    for item in metrics:
        edge = item.transition
        color = RISING if edge.polarity is pt.Polarity.RISING else FALLING
        ax.axvspan(edge.end, edge.end + ABERRATION_FACTOR * edge.duration, color=color, alpha=0.1, linewidth=0)
        print(f"{edge.polarity.name:>7}: overshoot {item.post_overshoot.percent:5.2f} %, "
              f"undershoot {item.post_undershoot.percent:5.2f} %, settling time {item.settling_time * 1e3:6.1f} ms")
        for aberration, name, dy in ((item.post_overshoot, "overshoot", 12), (item.post_undershoot, "undershoot", -18)):
            if edge.polarity is pt.Polarity.FALLING:
                dy = -dy
            ax.plot(aberration.instant, aberration.value, "v" if dy > 0 else "^", color=color, markersize=7)
            ax.annotate(f"{name} {aberration.percent:.1f} %", (aberration.instant, aberration.value),
                        xytext=(6, dy), textcoords="offset points", fontsize=8, color=color)
        if not math.isfinite(item.settling_time):
            ax.text(edge.end, 0.52, "  settling time NaN (not settled)", va="bottom", fontsize=8, color=color)
            continue
        settle = edge.mid_instant + item.settling_time
        ax.annotate("", xy=(settle, 0.5), xytext=(edge.mid_instant, 0.5),
                    arrowprops={"arrowstyle": "<->", "color": color, "linewidth": 1.0})
        ax.text((edge.mid_instant + settle) / 2, 0.52, f"settling time {item.settling_time * 1e3:.0f} ms",
                ha="center", va="bottom", fontsize=8, color=color)
        ax.axvline(settle, color=color, linestyle=":", linewidth=1.0)

    ax.plot([], [], color=ACCENT, alpha=0.3, linewidth=6, label=f"±{TOLERANCE:g} % settling band")
    ax.plot([], [], color=SIGNAL, alpha=0.2, linewidth=6, label="Post-transition aberration region")
    ax.plot([], [], ":", color=SIGNAL, label="Settling instant")
    ax.set_title("transition_metrics: aberrations and settling time of each edge")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_ylim(-0.55, 1.55)
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
