"""Smoke test — derived from: pulse_transitions Expanded Spec (v0.3), §C33 (rev 8).

Runs every example script in a subprocess, in parallel, with its figures redirected to a
temporary directory (PT_FIGURES) so examples/figures stays untouched. Skipped when
matplotlib or scipy is missing: the examples need them, the library doesn't.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted(p for p in (ROOT / "examples").glob("*.py") if not p.name.startswith("_"))
MISSING = [name for name in ("matplotlib", "scipy") if importlib.util.find_spec(name) is None]


@unittest.skipIf(MISSING, f"examples need {', '.join(MISSING)}")
class TestExamples(unittest.TestCase):
    def test_every_example_runs_and_writes_its_figure(self):
        self.assertTrue(EXAMPLES, msg="spec §C33: no example scripts found")
        with tempfile.TemporaryDirectory() as figures:
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg", "PT_FIGURES": figures}

            def run(script: Path) -> subprocess.CompletedProcess:
                return subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                                      capture_output=True, text=True, timeout=120)

            with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
                results = dict(zip(EXAMPLES, pool.map(run, EXAMPLES)))
            for script, result in results.items():
                with self.subTest(example=script.name):
                    self.assertEqual(result.returncode, 0,
                                     msg=f"spec §C33: {script.name} failed:\n{result.stderr[-2000:]}")
                    figure = Path(figures) / f"{script.stem}.svg"
                    self.assertTrue(figure.is_file() and figure.stat().st_size > 0,
                                    msg=f"spec §C33: {script.name} wrote no figure to {figure}")


if __name__ == "__main__":
    unittest.main()
