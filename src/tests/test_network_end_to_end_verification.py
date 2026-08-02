from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from verification.esbmc_install import resolve_esbmc_executable


def _module_available(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


@unittest.skipUnless(resolve_esbmc_executable(), "esbmc binary is not installed")
@unittest.skipUnless(_module_available("mip"), "python-mip is not installed")
@unittest.skipUnless(_module_available("tensorflow"), "tensorflow is not installed")
class NetworkEndToEndVerificationTest(unittest.TestCase):
    def _run(
        self,
        *,
        dataset: str,
        arch: str,
        sample_id: int,
        invariants: bool,
        timeout_seconds: int,
        output_dir: Path,
        bit_width: int = 8,
        cex_feedback: str = "off",
    ) -> tuple[dict, dict]:
        repo_root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            str(repo_root / "src" / "scripts" / "run_robustness_pipeline.py"),
            "--dataset",
            dataset,
            "--arch",
            arch,
            "--sample-id",
            str(sample_id),
            "--eps",
            "0.01",
            "--bit-lb",
            str(bit_width),
            "--bit-ub",
            str(bit_width),
            "--preimage-mode",
            "abstr",
            "--verify-mode",
            "esbmc",
            "--harness-scope",
            "network",
            "--cex-feedback",
            cex_feedback,
            "--esbmc-timeout",
            str(timeout_seconds),
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
        if not invariants:
            command.append("--no-e2e-invariants")
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                f"Network-scope pipeline failed\n"
                f"stdout tail:\n{completed.stdout[-3000:]}\n"
                f"stderr tail:\n{completed.stderr[-3000:]}"
            )
        reports = output_dir / "reports"
        return (
            json.loads((reports / "pipeline_summary.json").read_text(encoding="utf-8")),
            json.loads((reports / "experiment_summary.json").read_text(encoding="utf-8")),
        )

    def test_iris_network_scopes_verify_in_one_query(self) -> None:
        cases = [
            ("iris", "1blk_10", 1),
            ("iris_15x2", "2blk_15_15", 0),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (dataset, arch, sample_id) in enumerate(cases):
                with self.subTest(arch=arch):
                    pipeline, experiment = self._run(
                        dataset=dataset,
                        arch=arch,
                        sample_id=sample_id,
                        invariants=True,
                        timeout_seconds=60,
                        output_dir=Path(temp_dir) / f"case_{index}",
                    )
                    self.assertEqual(
                        pipeline["end_to_end_verification"]["status"],
                        "VERIFIED",
                    )
                    self.assertEqual(
                        pipeline["timing_metrics"]["number_of_esbmc_calls"],
                        1,
                    )
                    self.assertEqual(experiment["final_status"], "VERIFIED")
                    self.assertEqual(
                        experiment["guarantee_level"],
                        "deployed-transfer",
                    )

    def test_no_invariant_ablation_keeps_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline, experiment = self._run(
                dataset="iris",
                arch="1blk_10",
                sample_id=1,
                invariants=False,
                timeout_seconds=2,
                output_dir=Path(temp_dir) / "no_invariants",
            )

        status = pipeline["end_to_end_verification"]["status"]
        self.assertIn(status, {"VERIFIED", "TIMEOUT"})
        self.assertEqual(
            pipeline["timing_metrics"]["number_of_esbmc_calls"],
            1,
        )
        self.assertFalse(
            pipeline["end_to_end_verification"]["invariants_injected"]
        )
        self.assertEqual(experiment["final_status"], status)

    def test_exact_input_quantizer_removes_floor_ceil_only_counterexample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline, experiment = self._run(
                dataset="iris",
                arch="1blk_10",
                sample_id=1,
                invariants=True,
                timeout_seconds=60,
                output_dir=Path(temp_dir) / "counterexample",
                bit_width=1,
                cex_feedback="filter+jump",
            )

        end_to_end = pipeline["end_to_end_verification"]
        self.assertEqual(end_to_end["status"], "VERIFIED")
        self.assertEqual(end_to_end["assumption_box_cardinality"], "1")
        self.assertIsNone(end_to_end["counterexample_replay"])
        self.assertEqual(
            pipeline["counterexamples"]["counterexamples_total"],
            0,
        )
        self.assertEqual(experiment["final_status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
