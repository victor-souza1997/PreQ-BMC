import json
from pathlib import Path
import unittest
from unittest.mock import patch

from cli import main
from scripts.search_ssv_source_model import (
    candidate_plan,
    select_candidate,
    validate_search_config,
)


class SourceModelSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "experiments/sign_source_model_search.json"
        cls.config = json.loads(path.read_text(encoding="utf-8"))

    def test_config_pins_two_feasible_restricted_candidates(self):
        plans = validate_search_config(self.config)
        self.assertEqual([plan.hidden_relu_count for plan in plans], [64, 128])
        self.assertEqual([plan.shared_parameter_count for plan in plans], [3099, 6155])
        self.assertEqual([plan.affine_lowering_entries for plan in plans], [49152, 98304])
        self.assertTrue(all(plan.affine_lowering_entries <= 1_000_000 for plan in plans))

    def test_selection_uses_smallest_passing_model(self):
        results = [
            {"candidate_id": "small", "status": "COMPLETED", "validation_accuracy": .91,
             "shared_parameter_count": 100},
            {"candidate_id": "large", "status": "COMPLETED", "validation_accuracy": .96,
             "shared_parameter_count": 200},
        ]
        self.assertEqual(select_candidate(results, .9)["candidate_id"], "small")
        self.assertIsNone(select_candidate(results, .97))

    def test_selection_tie_prefers_accuracy_then_stable_id(self):
        results = [
            {"candidate_id": "b", "status": "COMPLETED", "validation_accuracy": .92,
             "shared_parameter_count": 100},
            {"candidate_id": "a", "status": "COMPLETED", "validation_accuracy": .93,
             "shared_parameter_count": 100},
        ]
        self.assertEqual(select_candidate(results, .9)["candidate_id"], "a")

    def test_invalid_or_over_budget_candidate_is_rejected(self):
        config = json.loads(json.dumps(self.config))
        config["max_affine_entries"] = 10
        with self.assertRaisesRegex(ValueError, "affine entries"):
            validate_search_config(config)
        with self.assertRaises(ValueError):
            candidate_plan(config["preprocessing"], {"id": "BAD ID", "filters": 4,
                           "kernel_size": [5, 5], "strides": [4, 4], "padding": "SAME"},
                           classes=43, max_affine_entries=1_000_000)

    def test_threshold_and_test_policy_are_explicit(self):
        self.assertEqual(self.config["selection"]["minimum_validation_accuracy"], .9)
        self.assertEqual(self.config["selection"]["test_set_policy"], "evaluate_once_after_selection")
        config = json.loads(json.dumps(self.config))
        config["selection"]["minimum_validation_accuracy"] = 0
        with self.assertRaises(ValueError):
            validate_search_config(config)

    def test_cli_routes_validation_only_model_search(self):
        with patch("cli._run_gtsrb_module", return_value=0) as run:
            result = main(["gtsrb", "search-model", "--config", "search.json",
                           "--output", "new-output"])
        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.args[0], "scripts.search_ssv_source_model")
        self.assertEqual(run.call_args.args[1],
                         ["--config", "search.json", "--output", "new-output"])


if __name__ == "__main__":
    unittest.main()
