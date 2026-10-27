"""mid_method: three ways to place the mid instant of a noisy edge.

    PYTHONPATH=src python examples/mid_method.py

Noise makes a slow edge cross its mid level several times. "median" (the default) takes
the median crossing, "first" the first one, and "fit" the point where a least-squares line
through the edge crosses the mid level. The fit is the most precise on a straight edge but
is biased on a curved (RC) one, which is why it isn't the default.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import ACCENT, FALLING, RISING, SIGNAL, save

METHODS = {"median": RISING, "first": FALLING, "fit": ACCENT}
NOISE = 0.03
SEEDS = 200


def linear_edge(t: np.ndarray) -> tuple[np.ndarray, float]:
    """A 0 to 1 ramp from 4 to 6 µs, and its true mid instant."""
    return np.clip((t - 4.0) / 2.0, 0.0, 1.0), 5.0


def rc_edge(t: np.ndarray) -> tuple[np.ndarray, float]:
    """An RC step at 4 µs (tau 0.6 µs), and its true mid instant."""
    tau = 0.6
    return 1.0 - np.exp(-np.clip(t - 4.0, 0.0, None) / tau), 4.0 + tau * np.log(2.0)


def rms_errors(shape, t: np.ndarray) -> dict[str, float]:
    """RMS mid-instant error of each method over SEEDS noise realisations, in ns."""
    clean, truth = shape(t)
    errors = {method: [] for method in METHODS}
    for seed in range(SEEDS):
        y = clean + np.random.default_rng(seed).normal(0.0, NOISE, t.size)
        for method in METHODS:
            edge = pt.transitions(y, x=t, levels=(0.0, 1.0), mid_method=method)[0]
            errors[method].append(edge.mid_instant - truth)
    return {
        method: float(np.sqrt(np.mean(np.square(e)))) * 1e3
        for method, e in errors.items()
    }


def main() -> None:
    t = np.linspace(0.0, 10.0, 2000)  # µs
    clean, truth = linear_edge(t)
    y = clean + np.random.default_rng(7).normal(0.0, NOISE, t.size)

    fig, (ax, bars) = plt.subplots(
        1, 2, figsize=(10.5, 4.5), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    window = (t > 4.4) & (t < 5.6)
    ax.plot(t[window], y[window], ".", color=SIGNAL, markersize=2.5, label="Samples")
    ax.axhline(0.5, color="#a0aec0", linestyle="--", linewidth=0.9, label="Mid level")
    ax.axvline(truth, color="black", linewidth=1.0, label="True mid instant")
    for method, color in METHODS.items():
        edge = pt.transitions(y, x=t, levels=(0.0, 1.0), mid_method=method)[0]
        print(
            f"{method:>6}: mid instant {edge.mid_instant:.4f} us (error {(edge.mid_instant - truth) * 1e3:+.1f} ns)"
        )
        ax.axvline(
            edge.mid_instant,
            color=color,
            linestyle="--",
            linewidth=1.3,
            label=f'"{method}": {(edge.mid_instant - truth) * 1e3:+.1f} ns',
        )
    ax.set_xlim(4.4, 5.6)
    ax.set_title("One noisy linear edge, near its mid level")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude (V)")
    ax.legend(loc="upper left", fontsize=8)

    results = {
        "Linear edge": rms_errors(linear_edge, t),
        "RC edge": rms_errors(rc_edge, t),
    }
    width = 0.25
    for k, (method, color) in enumerate(METHODS.items()):
        values = [results[shape][method] for shape in results]
        positions = np.arange(len(results)) + (k - 1) * width
        bars.bar(positions, values, width, color=color, label=f'"{method}"')
        for position, value in zip(positions, values, strict=True):
            bars.text(
                position, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8
            )
    for shape, errors in results.items():
        print(
            f"{shape}: RMS error "
            + ", ".join(f"{m} {v:.1f} ns" for m, v in errors.items())
        )
    bars.set_xticks(range(len(results)), list(results))
    bars.set_ylabel(f"RMS mid-instant error (ns), {SEEDS} seeds")
    bars.set_title("Accuracy by edge shape")
    bars.set_ylim(0, 1.35 * max(max(e.values()) for e in results.values()))
    bars.legend(loc="upper left", ncols=3)
    bars.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
