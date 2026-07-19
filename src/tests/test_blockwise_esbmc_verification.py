from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from verification.esbmc_install import resolve_esbmc_executable


def _has_gurobi() -> bool:
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return False
    return True


def _has_tensorflow() -> bool:
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(resolve_esbmc_executable(), "esbmc binary is not installed")
@unittest.skipUnless(_has_gurobi(), "gurobipy is not installed/configured")
@unittest.skipUnless(_has_tensorflow(), "tensorflow is not installed")
class BlockwiseESBMCSanityTest(unittest.TestCase):
    def _run_iris(self, block_size: int, output_root: Path) -> dict:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "src" / "scripts" / "run_robustness_pipeline.py"
        output_dir = output_root / f"iris_block_{block_size}"
        command = [
            sys.executable,
            str(script),
            "--dataset",
            "iris_3x2",
            "--arch",
            "2blk_3_3",
            "--sample-id",
            "0",
            "--eps",
            "0.01",
            "--bit-lb",
            "1",
            "--bit-ub",
            "12",
            "--preimage-mode",
            "abstr",
            "--verify-mode",
            "esbmc",
            "--esbmc-layer-block-size",
            str(block_size),
            "--max-quality-refinement-steps",
            "0",
            "--no-formal-saturation-check",
            "--no-empirical-saturation-check",
            "--accuracy-drop-threshold",
            "-1",
            "--saturation-threshold",
            "-1",
            "--mismatch-threshold",
            "-1",
            "--compare-limit",
            "1",
            "--skip-c-backend",
            "--no-export-paper-tables",
            "--output-dir",
            str(output_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                f"Pipeline failed for block size {block_size}\n"
                f"stdout tail:\n{completed.stdout[-4000:]}\n"
                f"stderr tail:\n{completed.stderr[-4000:]}"
            )
        return json.loads((output_dir / "reports" / "pipeline_summary.json").read_text(encoding="utf-8"))

    def test_iris_block_size_zero_and_one_select_consistent_qif(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            full_layer = self._run_iris(block_size=0, output_root=output_root)
            blockwise = self._run_iris(block_size=1, output_root=output_root)

        self.assertTrue(full_layer["synthesis"]["success"])
        self.assertTrue(blockwise["synthesis"]["success"])
        self.assertEqual(full_layer["synthesis"]["total_bits"], blockwise["synthesis"]["total_bits"])
        self.assertEqual(full_layer["synthesis"]["integer_bits"], blockwise["synthesis"]["integer_bits"])
        self.assertEqual(full_layer["synthesis"]["fractional_bits"], blockwise["synthesis"]["fractional_bits"])
        self.assertFalse(full_layer["blockwise_verification"]["enabled"])
        self.assertTrue(blockwise["blockwise_verification"]["enabled"])
        self.assertEqual(blockwise["blockwise_verification"]["block_size"], 1)
        self.assertGreater(blockwise["blockwise_verification"]["total_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
