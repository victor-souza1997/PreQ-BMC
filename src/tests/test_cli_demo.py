from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cli import _demo_result_summary, build_parser


class DemoResultSummaryTests(unittest.TestCase):
    def test_demo_defaults_to_paper_slack_profile(self) -> None:
        args = build_parser().parse_args(["demo", "--no-gurobi"])

        self.assertEqual(args.contract_profile, "paper-slack")

    def test_demo_accepts_strict_profile(self) -> None:
        args = build_parser().parse_args(["demo", "--contract-profile", "strict"])

        self.assertEqual(args.contract_profile, "strict")

    def test_prefers_accepted_quality_refined_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            reports = output_dir / "reports"
            reports.mkdir()
            (reports / "experiment_summary.json").write_text(
                json.dumps(
                    {
                        "formal_only": {
                            "final_status": "UNKNOWN",
                            "guarantee_level": "unknown",
                        },
                        "quality_refined": {
                            "accepted": True,
                            "final_status": "VERIFIED",
                            "guarantee_level": "harness-verified",
                            "contract_status": "VERIFIED",
                            "no_saturation_status": "VERIFIED",
                            "Q": [7, 6, 6],
                            "I": [2, 3, 4],
                            "F": [4, 2, 1],
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = _demo_result_summary(output_dir)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["method"], "quality_refined")
        self.assertEqual(summary["final_status"], "VERIFIED")
        self.assertEqual(summary["guarantee_level"], "harness-verified")
        self.assertEqual(summary["Q"], [7, 6, 6])

    def test_returns_none_when_report_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(_demo_result_summary(Path(temp_dir)))


if __name__ == "__main__":
    unittest.main()
