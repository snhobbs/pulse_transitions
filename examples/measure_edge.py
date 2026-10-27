"""measure_edge: one edge of a multi-level (TDR) trace, between the flat levels either side.

    PYTHONPATH=src python examples/measure_edge.py

A TDR trace steps to a new level at every impedance discontinuity, so it has no single
pair of state levels. measure_edge finds the edge nearest `around` (within
`max_distance`), in either direction, and measures it between its own levels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import FALLING, RISING, SIGNAL, ramp_edges, save

REFERENCE = (10, 50, 90)


def main() -> None:
    x = np.linspace(0.0, 12.0, 6000)  # ns
    # Incident step, then a high-impedance section (up), a low-impedance one (down) and an open end.
    y = ramp_edges(
        x, [(1.0, 0.35, 0.5), (4.0, 0.5, 0.12), (6.5, 0.6, -0.2), (9.0, 0.7, 0.58)]
    )
    y += np.random.default_rng(5).normal(0.0, 0.004, x.size)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, y, color=SIGNAL, linewidth=1.0)
    for around, max_distance in ((4.5, 1.0), (7.0, 1.0), (9.5, 1.0)):
        edge = pt.measure_edge(
            y, x=x, around=around, max_distance=max_distance, reference=REFERENCE
        )
        tr = edge.transition
        color = RISING if tr.polarity is pt.Polarity.RISING else FALLING
        bottom, top = edge.levels
        print(
            f"around {around} ns: {tr.polarity.name.lower()} edge between {bottom:.3f} and {top:.3f}, "
            f"{REFERENCE[0]}-{REFERENCE[2]} % duration {edge.duration:.3f} ns"
        )
        ax.axvspan(
            around - max_distance,
            around + max_distance,
            color=color,
            alpha=0.07,
            linewidth=0,
        )
        ax.axvline(around, color=color, linestyle=":", linewidth=0.9)
        for level in edge.levels:
            ax.hlines(
                level,
                tr.start - 0.8,
                tr.end + 0.8,
                color=color,
                linestyle="--",
                linewidth=0.9,
            )
        ax.plot(
            [tr.lower_instant, tr.mid_instant, tr.upper_instant],
            [tr.lower_level, tr.mid_level, tr.upper_level],
            "o",
            color=color,
            markersize=4,
        )
        ax.annotate(
            f"{edge.duration:.2f} ns\n{bottom:.2f} → {top:.2f}",
            (tr.end, tr.mid_level),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )

    try:
        pt.measure_edge(y, x=x, around=2.5, max_distance=0.5)
    except pt.EdgeNotFoundError as error:
        print(f"around 2.5 ns: EdgeNotFoundError: {error}")

    ax.plot([], [], ":", color=SIGNAL, label="around (shaded: ± max_distance)")
    ax.plot([], [], "--", color=SIGNAL, label="Levels either side of the edge")
    ax.plot([], [], "o", color=SIGNAL, label="10 / 50 / 90 % instants")
    ax.set_title("measure_edge: each edge between its own levels")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Reflected voltage (V)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
