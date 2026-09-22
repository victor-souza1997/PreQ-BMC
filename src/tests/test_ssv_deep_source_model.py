import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from cli import main
from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN
from scripts.search_ssv_deep_source_model import fold_batch_norm, validate_config


class DeepSourceModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "experiments/sign_deep_source_model_search.json"
        cls.config = json.loads(path.read_text(encoding="utf-8"))

    def test_candidate_ladder_is_bounded_and_increasing(self):
        plans = validate_config(self.config)
        self.assertEqual([plan.candidate_id for plan in plans], [
            "conv8_16_dense128", "conv12_24_dense192", "conv16_32_dense256",
        ])
        self.assertEqual(plans[0].relus_per_layer, (512, 256, 128))
        self.assertTrue(all(max(plan.affine_entries_per_layer) <= 2_000_000 for plan in plans))
        self.assertEqual([plan.shared_parameter_count for plan in plans],
                         sorted(plan.shared_parameter_count for plan in plans))

    def test_batch_norm_folding_matches_inference_equation(self):
        kernel = np.asarray([[1., -2.], [3., 4.]], dtype=np.float32)
        gamma = np.asarray([2., .5], dtype=np.float32)
        beta = np.asarray([1., -1.], dtype=np.float32)
        mean = np.asarray([.25, -.5], dtype=np.float32)
        variance = np.asarray([4., .25], dtype=np.float32)
        folded_kernel, folded_bias = fold_batch_norm(
            kernel, [gamma, beta, mean, variance], 0.001,
        )
        x = np.asarray([.2, -.3], dtype=np.float32)
        original = gamma * ((x @ kernel) - mean) / np.sqrt(variance + .001) + beta
        np.testing.assert_allclose(x @ folded_kernel + folded_bias, original, rtol=1e-6, atol=1e-6)

    def test_sequential_conv_lowers_to_same_float_function(self):
        first = ConvGeometry((4, 4, 1), (2, 2, 1, 2), (2, 2), "SAME")
        second = ConvGeometry(first.output_shape, (1, 1, 2, 2), (1, 1), "SAME")
        rng = np.random.default_rng(4)
        model = RestrictedSequentialCNN(
            (first, second),
            (rng.normal(size=first.kernel_shape), rng.normal(size=second.kernel_shape)),
            (rng.normal(size=2), rng.normal(size=2)),
            (rng.normal(size=(8, 3)), rng.normal(size=(3, 2))),
            (rng.normal(size=3), rng.normal(size=2)),
            max_affine_entries=1000,
        )
        image = rng.normal(size=first.input_shape).astype(np.float32)
        expected = model.float_reference(image)
        actual = np.asarray(model.as_deep_model()(image.reshape(1, -1), training=False))[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        quantized = model.quantized([LayerQuantizationSpec(16, 7, 8)] * 4)
        self.assertEqual(len(quantized.layers), 4)
        self.assertTrue(quantized.layers[-1].is_output_layer)
        self.assertFalse(any(layer.is_output_layer for layer in quantized.layers[:-1]))

    def test_invalid_budget_and_order_are_rejected(self):
        config = json.loads(json.dumps(self.config))
        config["max_affine_entries"] = 10
        with self.assertRaisesRegex(ValueError, "affine entries"):
            validate_config(config)
        config = json.loads(json.dumps(self.config))
        config["candidates"] = list(reversed(config["candidates"]))
        with self.assertRaisesRegex(ValueError, "increasing parameter"):
            validate_config(config)

    def test_cli_routes_deep_search(self):
        with patch("cli._run_gtsrb_module", return_value=0) as run:
            result = main(["gtsrb", "search-deep-model", "--config", "deep.json",
                           "--output", "new-output"])
        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.args[0], "scripts.search_ssv_deep_source_model")

    def test_center_nearest_resize_has_matching_c_mapping(self):
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        encoder = ByteImageEncoder((3, 4, 3), resize_mode="nearest_center")
        expected_rows = (2 * np.arange(3) + 1) * 5 // 6
        expected_cols = (2 * np.arange(4) + 1) * 7 // 8
        np.testing.assert_array_equal(
            encoder.resize(image), image[expected_rows[:, None], expected_cols[None, :], :],
        )
        source = encoder.render_c()
        self.assertIn("(2 * y + 1) * height / 6", source)
        self.assertIn("(2 * x + 1) * width / 8", source)

    def test_training_seed_is_independent_of_frozen_split_seed(self):
        path = Path(__file__).resolve().parents[2] / "experiments/sign_deep_source_model_search_32_regularized.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        plans = validate_config(config)
        self.assertEqual(config["training"]["seed"], 2027)
        self.assertEqual(config["training"]["split_seed"], 2026)
        self.assertEqual(len(plans), 1)
        config["training"]["split_seed"] = 2027
        with self.assertRaisesRegex(ValueError, "physical-track split"):
            validate_config(config)

    def test_four_stage_candidate_reduces_dense_input(self):
        path = Path(__file__).resolve().parents[2] / "experiments/sign_deep_source_model_search_4stage.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        plan = validate_config(config)[0]
        self.assertEqual([geometry.output_shape for geometry in plan.geometries], [
            (16, 16, 24), (8, 8, 48), (4, 4, 96), (2, 2, 128),
        ])
        self.assertEqual(plan.affine_entries_per_layer[-2], 512 * 384)
        self.assertEqual(plan.relus_per_layer, (6144, 3072, 1536, 512, 384))


if __name__ == "__main__":
    unittest.main()
