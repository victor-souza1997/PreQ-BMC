import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN
from scripts.evaluate_ssv_deep_source_model import (
    balanced_accuracy, evaluate, load_selected, reference_parity, restricted_from_npz,
)

GEOMETRY = {"input_shape": [4, 4, 2], "kernel_shape": [3, 3, 2, 3],
            "strides": [2, 2], "padding": "SAME"}


def _write_model(directory: Path, rng):
    arrays = {
        "conv_0_kernel": rng.normal(size=(3, 3, 2, 3)).astype(np.float32),
        "conv_0_bias": rng.normal(size=(3,)).astype(np.float32),
        "dense_0_kernel": rng.normal(size=(12, 5)).astype(np.float32),
        "dense_0_bias": rng.normal(size=(5,)).astype(np.float32),
        "dense_1_kernel": rng.normal(size=(5, 4)).astype(np.float32),
        "dense_1_bias": rng.normal(size=(4,)).astype(np.float32),
    }
    path = directory / "folded_model.npz"
    np.savez(path, **arrays)
    return path


def _selected(path: Path):
    import hashlib

    return {
        "candidate_id": "tiny",
        "validation_accuracy": 0.9,
        "geometries": [GEOMETRY],
        "dense_hidden": [5],
        "relus_per_layer": [12, 5],
        "affine_entries_per_layer": [384, 60, 20],
        "folded_model_path": str(path),
        "folded_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class DeepSourceModelEvaluationTest(unittest.TestCase):
    def test_restricted_model_rebuilt_from_npz_matches_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_model(root, np.random.default_rng(7))
            restricted = restricted_from_npz(_selected(path), path)
        self.assertIsInstance(restricted, RestrictedSequentialCNN)
        self.assertEqual(restricted.layer_sizes, [12, 5, 4])
        self.assertEqual(len(restricted.dense_biases[-1]), 4)

    def test_reference_parity_is_zero_against_itself_and_grows_with_perturbation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_model(root, np.random.default_rng(11))
            restricted = restricted_from_npz(_selected(path), path)
        rng = np.random.default_rng(3)
        images = rng.random((4, 4, 4, 2)).astype(np.float32)
        expected = np.stack([restricted.float_reference(image) for image in images])
        self.assertLess(reference_parity(restricted, images, expected, [0, 3]), 1e-5)
        self.assertGreater(
            reference_parity(restricted, images, expected + np.float32(0.25), [0, 3]), 0.2,
        )

    def test_balanced_accuracy_ignores_absent_classes_and_weights_classes_equally(self):
        labels = np.asarray([0, 0, 0, 0, 2, 2])
        predictions = np.asarray([0, 0, 0, 0, 0, 2])
        balanced, per_class = balanced_accuracy(labels, predictions, 3)
        self.assertEqual(per_class, [1.0, 0.5])
        self.assertAlmostEqual(balanced, 0.75)
        self.assertNotAlmostEqual(balanced, float(np.mean(predictions == labels)))

    def test_load_selected_refuses_unselected_and_already_consumed_searches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_model(root, np.random.default_rng(5))
            summary = {"status": "NO_CANDIDATE_MET_THRESHOLD", "test_status": "NOT_EVALUATED",
                       "selected_candidate_summary": _selected(path)}
            target = root / "search_summary.json"
            target.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not select"):
                load_selected(root)

            summary["status"] = "SELECTED"
            summary["test_status"] = "COMPLETED"
            target.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already consumed"):
                load_selected(root)

            summary["test_status"] = "NOT_EVALUATED"
            target.write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(load_selected(root)[1]["candidate_id"], "tiny")

    def test_evaluate_refuses_a_second_run_before_touching_the_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_evaluation.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already consumed"):
                evaluate(root / "absent-search", root)

    def test_load_selected_rejects_a_tampered_folded_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_model(root, np.random.default_rng(13))
            summary = {"status": "SELECTED", "test_status": "NOT_EVALUATED",
                       "selected_candidate_summary": _selected(path)}
            (root / "search_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            _write_model(root, np.random.default_rng(14))
            with self.assertRaisesRegex(ValueError, "digest"):
                load_selected(root)


class RecordedTestEvaluationTest(unittest.TestCase):
    """Guard the recorded held-out numbers against silent regression."""

    @classmethod
    def setUpClass(cls):
        path = (Path(__file__).resolve().parents[2]
                / "output/sign_deep_source_model_test_20260922/test_evaluation.json")
        if not path.exists():
            raise unittest.SkipTest("Held-out evaluation output is not present")
        cls.report = json.loads(path.read_text(encoding="utf-8"))

    def test_folding_does_not_change_a_single_prediction(self):
        self.assertTrue(self.report["folding_predictions_equal"])
        self.assertEqual(self.report["folding_prediction_mismatches"], 0)
        self.assertEqual(self.report["test_accuracy_source_float32"],
                         self.report["test_accuracy_folded_affine"])

    def test_batched_path_agrees_with_the_independent_reference(self):
        self.assertLess(self.report["batched_vs_reference_max_abs_error"], 1e-4)
        self.assertGreaterEqual(self.report["batched_vs_reference_sample_size"], 32)

    def test_quantization_and_proof_remain_unclaimed(self):
        self.assertEqual(self.report["quantization_status"], "NOT_RUN")
        self.assertEqual(self.report["proof_status"], "NOT_RUN")
        self.assertEqual(self.report["android_status"], "NOT_MEASURED")
        self.assertEqual(self.report["float_lowering_claim"],
                         "tested_numerical_equivalence_not_IEEE_proof")

    def test_test_accuracy_is_above_the_selection_gate_without_a_generalization_gap(self):
        self.assertEqual(self.report["n_test_images"], 12630)
        self.assertGreater(self.report["test_accuracy_folded_affine"], 0.90)
        self.assertLess(self.report["generalization_gap_percentage_points"], 1.0)

    def test_balanced_accuracy_is_reported_and_lower_than_raw_accuracy(self):
        balanced = self.report["test_balanced_accuracy_folded_affine"]
        self.assertLess(balanced, self.report["test_accuracy_folded_affine"])
        self.assertEqual(len(self.report["per_class_accuracy_folded_affine"]), 43)
        self.assertEqual(self.report["worst_class_accuracy"],
                         min(self.report["per_class_accuracy_folded_affine"]))


if __name__ == "__main__":
    unittest.main()
