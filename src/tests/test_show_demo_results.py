from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.show_demo_results import render_summary


class ShowDemoResultsTests(unittest.TestCase):
    def test_renders_accepted_refined_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            reports = run_dir / "reports"
            reports.mkdir()
            (reports / "experiment_summary.json").write_text(
                json.dumps(
                    {
                        "benchmark": {
                            "dataset": "iris",
                            "arch": "2blk_4_4",
                            "sample_id": 27,
                            "eps": 0.05,
                            "samples_evaluated": 10,
                        },
                        "reference": {"predicted_label": 1, "sample_label": 1, "clean_margin": 4.1},
                        "quality_refined": {
                            "accepted": True,
                            "Q": [7],
                            "I": [2],
                            "F": [4],
                            "final_status": "VERIFIED",
                            "guarantee_level": "harness-verified",
                            "contract_status": "VERIFIED",
                            "no_saturation_status": "VERIFIED",
                            "python_c_exact_match": True,
                            "deployment_metrics": {
                                "c_fixed_accuracy": 1.0,
                                "mismatch_rate_vs_keras": 0.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (reports / "pipeline_summary.json").write_text(
                json.dumps(
                    {
                        "soundness": "degraded",
                        "chaining_ok": {"all_ok": False},
                        "timing_metrics": {"total_runtime_seconds": 5.0},
                    }
                ),
                encoding="utf-8",
            )

            output = render_summary(run_dir)

        self.assertIn("status final: VERIFIED", output)
        self.assertIn("nivel de garantia: harness-verified", output)
        self.assertIn("camada 0: <Q=7, I=2, F=4>", output)
        self.assertIn("nao estabelece a garantia deployed-transfer", output)


if __name__ == "__main__":
    unittest.main()
