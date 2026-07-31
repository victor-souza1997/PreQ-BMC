from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from cli import _demo_result_summary, build_parser, cmd_reproduce


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

    def test_reproduce_resolves_config_before_changing_subprocess_cwd(self) -> None:
        args = SimpleNamespace(
            config=Path("experiments/sound_v2_experiments.json"),
            solver="cbc",
            only=None,
            max_runs=None,
            output_root=None,
            dry_run=False,
            aggregate=False,
            plots=False,
            continue_on_error=False,
            resume=False,
            force=False,
            error_budget_mode=None,
            vacuity_check=None,
            cex_feedback=None,
            harness_scope=None,
            e2e_invariants=None,
        )

        with patch("cli.subprocess.run") as run:
            run.return_value.returncode = 0
            return_code = cmd_reproduce(args, [])

        command = run.call_args.args[0]
        config_argument = Path(command[command.index("--config") + 1])
        self.assertEqual(return_code, 0)
        self.assertTrue(config_argument.is_absolute())
        self.assertEqual(
            config_argument,
            Path("experiments/sound_v2_experiments.json").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
