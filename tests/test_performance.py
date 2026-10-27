"""Timing tests — derived from: pulse_transitions Expanded Spec (v0.3), §C22–§C24, §C30. Run with PT_PERF=1 (D10)."""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import pulse_transitions as pt

from tests.signals import pulse_train

PERF = unittest.skipUnless(os.environ.get("PT_PERF") == "1", "timing tests: set PT_PERF=1 (machine-dependent limits, D10)")
ROOT = Path(__file__).resolve().parents[1]


def best_of(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@PERF
class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x6, cls.y6 = pulse_train(1_000_000)
        cls.x5, cls.y5 = pulse_train(100_000)

    def test_transitions_1e6_under_100ms(self):
        pt.transitions(self.y6, x=self.x6, levels=(0, 1))  # warm-up
        elapsed = best_of(lambda: pt.transitions(self.y6, x=self.x6, levels=(0, 1)))
        self.assertLessEqual(elapsed, 0.100, msg=f"spec §C22: transitions on 1e6 samples took {elapsed * 1e3:.1f} ms, limit 100 ms (v0.2: 1009 ms)")

    def test_transitions_fit_1e6_under_100ms(self):
        pt.transitions(self.y6, x=self.x6, levels=(0, 1), mid_method="fit")  # warm-up
        elapsed = best_of(lambda: pt.transitions(self.y6, x=self.x6, levels=(0, 1), mid_method="fit"))
        self.assertLessEqual(elapsed, 0.100, msg=f"spec §C31 / §C22: transitions(mid_method='fit') on 1e6 samples took {elapsed * 1e3:.1f} ms, limit 100 ms")

    def test_transitions_scale_linearly(self):
        t6 = best_of(lambda: pt.transitions(self.y6, x=self.x6, levels=(0, 1)))
        t5 = best_of(lambda: pt.transitions(self.y5, x=self.x5, levels=(0, 1)))
        self.assertLessEqual(t6, 15 * t5, msg=f"spec §C23: 1e6 took {t6 * 1e3:.1f} ms, 1e5 took {t5 * 1e3:.1f} ms; ratio must be ≤ 15")

    def test_state_levels_1e6_under_20ms(self):
        elapsed = best_of(lambda: pt.state_levels(self.y6))
        self.assertLessEqual(elapsed, 0.020, msg=f"spec §C24: state_levels on 1e6 samples took {elapsed * 1e3:.1f} ms, limit 20 ms (v0.2: 14 ms)")

    def test_benchmark_script_runs(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        done = subprocess.run([sys.executable, str(ROOT / "benchmarks" / "benchmark.py")], capture_output=True, text=True,
                              env=env, timeout=600, cwd=ROOT)
        self.assertEqual(done.returncode, 0, msg=f"spec §C30: benchmark.py must run cleanly; stderr: {done.stderr[-2000:]}")
        self.assertIn("1009", done.stdout, msg="spec §C30: the v0.2 figures are printed beside the v0.3 ones")


if __name__ == "__main__":
    unittest.main()
