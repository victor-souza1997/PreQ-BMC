import unittest
from unittest.mock import patch

import numpy as np

from cli import main
from reports.ssv_quality import QualityComparison


class QualityTest(unittest.TestCase):
    def test_equal_accuracy_does_not_imply_prediction_parity(self):
        metric = QualityComparison()
        metric.add([2., 1.], [0., 3.], 0)
        metric.add([2., 1.], [0., 3.], 1)
        result = metric.summary()
        self.assertEqual(result["source_accuracy"], .5)
        self.assertEqual(result["candidate_accuracy"], .5)
        self.assertEqual(result["accuracy_drop_percentage_points"], 0)
        self.assertEqual(result["prediction_mismatch_rate"], 1)
        self.assertEqual(result["source_correct_candidate_wrong_count"], 1)
        self.assertEqual(result["source_wrong_candidate_correct_count"], 1)
        self.assertEqual(result["mean_abs_logit_error"], 2)

    def test_ties_use_lowest_index_and_logit_error_is_not_prediction_error(self):
        metric = QualityComparison()
        metric.add([1., 1.], [3., 3.], 0)
        self.assertEqual(metric.summary()["candidate_accuracy"], 1)
        self.assertEqual(metric.summary()["prediction_mismatch_rate"], 0)
        self.assertEqual(metric.summary()["max_abs_logit_error"], 2)

    def test_loss_is_percentage_points_not_relative_percent(self):
        metric = QualityComparison()
        metric.add([3, 1], [1, 3], 0)
        metric.add([3, 1], [3, 1], 0)
        metric.add([3, 1], [3, 1], 1)
        metric.add([3, 1], [3, 1], 1)
        self.assertEqual(metric.summary()["accuracy_drop_percentage_points"], 25)

    def test_reject_empty_nonfinite_or_mismatched_measurements(self):
        with self.assertRaises(ValueError):
            QualityComparison().summary()
        for source, candidate, label in [([1, 2], [1], 0), ([1, 2], [np.nan, 1], 0),
                                         ([1, 2], [1, 2], 3)]:
            with self.assertRaises(ValueError):
                QualityComparison().add(source, candidate, label)

    def test_cli_supports_uncertified_frozen_artifact(self):
        with patch("cli._run_gtsrb_module", return_value=0) as run:
            self.assertEqual(main(["gtsrb", "evaluate-host", "--study", "study.json",
                                   "--artifact", "artifact.json", "--output", "quality",
                                   "--max-accuracy-drop-pp", "1.5"]), 0)
        self.assertEqual(run.call_args.args[0], "scripts.evaluate_ssv_host")
        self.assertIn("--artifact", run.call_args.args[1])
        self.assertIn("1.5", run.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
