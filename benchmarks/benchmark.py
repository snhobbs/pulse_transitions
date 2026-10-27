"""Time the v0.3 public API on pulse trains and print the v0.2.0 figures beside it.

Run from the project root:

    PYTHONPATH=src python benchmarks/benchmark.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

import pulse_transitions as pt

SIZES = (10_000, 100_000, 1_000_000)
REPEATS = 3

# v0.2.0 figures, measured on the same pulse train (docs/spec/pulse_transitions.md, Performance).
V02_LEVELS = {10_000: "0.3", 100_000: "3.0", 1_000_000: "14"}
V02_DETECT_EDGES = {10_000: "11", 100_000: "91", 1_000_000: "1009"}
V02_RISETIME = {10_000: "3.6", 100_000: "37", 1_000_000: "373"}
V02_MEASURE_EDGE = {10_000: "3.6", 100_000: "fails", 1_000_000: "fails"}


def pulse_train(
    n: int, periods: int = 10, noise: float = 0.01, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Pulse train with 20 edges over `periods` periods, Gaussian noise, and a fixed seed."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, periods, n)
    ph = x % 1.0
    y = (
        np.clip((ph - 0.2) / 0.02, 0, 1)
        - np.clip((ph - 0.6) / 0.02, 0, 1)
        + noise * rng.standard_normal(n)
    )
    return x, y


def best_of_ms(func: Callable[[], object]) -> str:
    """Best wall time of REPEATS calls in milliseconds, or "fails" if a call raises."""
    best = float("inf")
    try:
        for _ in range(REPEATS):
            start = time.perf_counter()
            func()
            best = min(best, time.perf_counter() - start)
    except Exception:
        return "fails"
    return f"{best * 1e3:.1f}"


def main() -> None:
    header = f"{'method':<34}{'v0.3 ms':>12}{'v0.2 ms':>12}"
    for n in SIZES:
        x, y = pulse_train(n)
        lv = pt.state_levels(y).levels
        rows = (
            (
                "state_levels (v0.2 levels)",
                best_of_ms(lambda: pt.state_levels(y)),
                V02_LEVELS[n],
            ),
            (
                "transitions (v0.2 detect_edges)",
                best_of_ms(lambda: pt.transitions(y, x=x, levels=lv)),
                V02_DETECT_EDGES[n],
            ),
            (
                "step_metrics (v0.2 risetime)",
                best_of_ms(lambda: pt.step_metrics(y, x=x, bounds=(0, 0.5))),
                V02_RISETIME[n],
            ),
            (
                "measure_edge (around=0.2)",
                best_of_ms(lambda: pt.measure_edge(y, x=x, around=0.2)),
                V02_MEASURE_EDGE[n],
            ),
        )
        print(f"\nn = {n:,} samples")
        print(header)
        print("-" * len(header))
        for name, v03, v02 in rows:
            print(f"{name:<34}{v03:>12}{v02:>12}")


if __name__ == "__main__":
    main()
