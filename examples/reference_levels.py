"""reference_levels: absolute reference levels from the state levels and percentages.

    PYTHONPATH=src python examples/reference_levels.py

Each reference level is low + (high - low) * percent / 100 (IEEE 181 §5.3.2). The same
edge is measured with the default 10/50/90 % levels and with 20/50/80 %: the narrower
pair gives a shorter rise time on the same edge.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import REFERENCE, SIGNAL, save

REFERENCES = {(10, 50, 90): "#2b6cb0", (20, 50, 80): "#c05621"}


def main() -> None:
    t = np.linspace(0.0, 10e-9, 2000)
    y = 0.4 + 1.4 * (
        1.0 - np.exp(-np.clip(t - 3e-9, 0.0, None) / 0.8e-9)
    )  # RC edge, 0.4 V to 1.8 V

    levels = pt.state_levels(y, x=t)
    fig, ax = plt.subplots()
    ax.plot(t * 1e9, y, color=SIGNAL, linewidth=1.8, label="Signal")
    for value, name in zip(levels.levels, ("low", "high"), strict=True):
        ax.axhline(value, color=REFERENCE, linewidth=1.0)
        ax.annotate(
            f"{name} state {value:.2f} V",
            (0.0, value),
            xytext=(2, 3),
            textcoords="offset points",
            color="#718096",
            fontsize=8,
        )

    for i, (reference, color) in enumerate(REFERENCES.items()):
        lower, mid, upper = pt.reference_levels(levels.levels, reference)
        edge = pt.transitions(y, x=t, levels=levels.levels, reference=reference)[0]
        print(
            f"reference {reference}: levels {lower:.3f}, {mid:.3f}, {upper:.3f} V; "
            f"rise time {edge.duration * 1e9:.3f} ns"
        )
        x_label = 9.95 - i * 2.4
        for percent, value in zip(reference, (lower, mid, upper), strict=True):
            ax.axhline(value, color=color, linestyle="--", linewidth=0.9)
            ax.annotate(
                f"{percent} %: {value:.2f} V",
                (x_label, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=8,
            )
        ax.plot(
            [edge.lower_instant * 1e9, edge.upper_instant * 1e9],
            [lower, upper],
            "o",
            color=color,
            label=f"{reference[0]}/{reference[2]} %: rise time {edge.duration * 1e9:.2f} ns",
        )

    ax.set_title("reference_levels: percentages of the state levels")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_xlim(0, 10)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45))
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
