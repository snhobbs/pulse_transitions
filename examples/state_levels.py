"""state_levels: the low and high states of a noisy, ringing pulse train, by each method.

    PYTHONPATH=src python examples/state_levels.py

`mode` (the default) and `mean` split the histogram into a lower and upper half and take
the most common level, or the mean, of each; `peak` takes the extremes and `endpoints`
averages the first and last samples (the record starts low and ends high). The overshoot
pulls `peak` and `mean` away from the settled levels, which is why the histogram mode is
the default.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import SIGNAL, second_order_step, save

METHODS = {"mode": "#2b6cb0", "mean": "#2f855a", "peak": "#c05621", "endpoints": "#805ad5"}


def main() -> None:
    t = np.linspace(0.0, 3.5, 7000)
    y = 3.3 * sum(
        sign * second_order_step(t - start, wn=60.0, zeta=0.35)
        for start, sign in ((0.25, 1), (1.25, -1), (2.25, 1))
    )
    y += np.random.default_rng(1).normal(0.0, 0.04, t.size)

    results = {method: pt.state_levels(y, x=t, method=method) for method in METHODS}
    for method, levels in results.items():
        print(f"{method:>9}: low {levels.low:+.3f} V, high {levels.high:+.3f} V")

    fig, (ax, hist) = plt.subplots(
        1, 2, sharey=True, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05}
    )
    ax.plot(t, y, color=SIGNAL, linewidth=0.8, label="Signal")

    mode = results["mode"]
    edges = mode.bin_edges
    hist.barh((edges[:-1] + edges[1:]) / 2, mode.histogram, height=np.diff(edges), color="#cbd5e0")
    for method, color in METHODS.items():
        levels = results[method]
        for value in levels.levels:
            for axis in (ax, hist):
                axis.axhline(value, color=color, linestyle="--", linewidth=1.1)
        ax.plot([], [], color=color, linestyle="--", label=f"{method}: {levels.low:.2f} / {levels.high:.2f} V")

    ax.set_title("state_levels: low and high state of each method")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5))
    hist.set_title("Histogram (100 bins)")
    hist.set_xlabel("Samples")
    hist.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.9)
    save(fig, __file__)


if __name__ == "__main__":
    main()
