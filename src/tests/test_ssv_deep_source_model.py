import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from cli import main
from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN
from scripts.search_ssv_deep_source_model import (
    _build_clean_model,
    _fold_model,
    fold_batch_norm,
    search,
    validate_config,
)


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
        config["minimum_available_memory_mib"] = -1
        with self.assertRaisesRegex(ValueError, "minimum_available_memory_mib"):
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

    def test_search_pauses_before_loading_training_data_when_memory_is_low(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "study.json"
            base.write_text(json.dumps({"schema": "ssv_restricted_cnn_v1"}), encoding="utf-8")
            config = json.loads(json.dumps(self.config))
            config["base_study"] = str(base)
            config["minimum_available_memory_mib"] = 10**9
            config["candidates"] = [config["candidates"][0]]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "output"

            result = search(config_path, output)

            self.assertEqual(result["status"], "RESOURCE_PAUSED")
            self.assertEqual(result["candidates"], [])
            self.assertIsNone(result["tensorflow_version"])
            self.assertTrue((output / "search_progress.json").exists())
            self.assertFalse((output / "search_summary.json").exists())


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

    def test_direct_head_candidate_has_no_dense_relu(self):
        config = json.loads(json.dumps(self.config))
        config["candidates"] = [{
            "id": "conv8_16_direct43",
            "conv_blocks": config["candidates"][0]["conv_blocks"],
            "dense_hidden": [],
            "dropout_rates": [0.0, 0.0],
        }]
        plan = validate_config(config)[0]
        self.assertEqual(plan.dense_hidden, ())
        self.assertEqual(plan.relus_per_layer, (512, 256))
        self.assertEqual(plan.affine_entries_per_layer[-1], 256 * 43)

    def test_global_average_pooling_plan_is_sparse_and_has_no_trainable_dense_hidden(self):
        config = json.loads(json.dumps(self.config))
        config["candidates"] = [{
            "id": "conv8_16_gap43",
            "conv_blocks": config["candidates"][0]["conv_blocks"],
            "dense_hidden": [],
            "global_average_pooling": True,
            "dropout_rates": [0.0, 0.0],
        }]
        plan = validate_config(config)[0]
        flattened = int(np.prod(plan.geometries[-1].output_shape))
        channels = plan.geometries[-1].output_shape[-1]
        self.assertTrue(plan.global_average_pooling)
        self.assertEqual(plan.dense_hidden, ())
        self.assertEqual(plan.affine_entries_per_layer[-2], flattened * channels)
        self.assertEqual(plan.affine_entries_per_layer[-1], channels * 43)
        self.assertEqual(plan.relus_per_layer[-1], channels)

    def test_spatial_average_pooling_retains_quadrants(self):
        config = json.loads(json.dumps(self.config))
        config["candidates"] = [{
            "id": "conv8_16_avg2_43",
            "conv_blocks": config["candidates"][0]["conv_blocks"],
            "dense_hidden": [],
            "average_pool_size": [2, 2],
            "dropout_rates": [0.0, 0.0],
        }]
        plan = validate_config(config)[0]
        height, width, channels = plan.geometries[-1].output_shape
        expected_units = (height // 2) * (width // 2) * channels
        self.assertEqual(plan.average_pool_size, (2, 2))
        self.assertFalse(plan.global_average_pooling)
        self.assertEqual(plan.relus_per_layer[-1], expected_units)
        self.assertEqual(plan.affine_entries_per_layer[-2],
                         height * width * channels * expected_units)
        self.assertEqual(plan.affine_entries_per_layer[-1], expected_units * 43)

    def test_global_average_pooling_lowers_to_same_float_function(self):
        import tensorflow as tf

        config = json.loads(json.dumps(self.config))
        config["candidates"] = [{
            "id": "conv8_16_gap43",
            "conv_blocks": config["candidates"][0]["conv_blocks"],
            "dense_hidden": [],
            "global_average_pooling": True,
            "dropout_rates": [0.0, 0.0],
        }]
        plan = validate_config(config)[0]
        tf.keras.utils.set_random_seed(17)
        image = np.random.default_rng(17).uniform(
            0, 1, size=(1, *plan.geometries[0].input_shape),
        ).astype(np.float32)
        with tf.device("/CPU:0"):
            clean = _build_clean_model(tf, plan)
            expected = np.asarray(clean(image, training=False))[0]
        actual = _fold_model(clean, plan).float_reference(image[0])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_depthwise_pooling_lowers_to_diagonal_standard_convolution(self):
        import tensorflow as tf

        config = json.loads(json.dumps(self.config))
        blocks = list(config["candidates"][0]["conv_blocks"])
        channels = blocks[-1]["filters"]
        blocks.append({
            "filters": channels,
            "kernel_size": [2, 2],
            "strides": [2, 2],
            "padding": "VALID",
            "groups": "depthwise",
        })
        config["candidates"] = [{
            "id": "conv8_16_dwpool43",
            "conv_blocks": blocks,
            "dense_hidden": [],
            "dropout_rates": [0.0, 0.0, 0.0],
        }]
        plan = validate_config(config)[0]
        self.assertEqual(plan.conv_groups, (1, 1, channels))
        tf.keras.utils.set_random_seed(23)
        image = np.random.default_rng(23).uniform(
            0, 1, size=(1, *plan.geometries[0].input_shape),
        ).astype(np.float32)
        with tf.device("/CPU:0"):
            clean = _build_clean_model(tf, plan)
            expected = np.asarray(clean(image, training=False))[0]
        restricted = _fold_model(clean, plan)
        depthwise = restricted.conv_kernels[-1]
        for output_channel in range(channels):
            nonzero_inputs = np.flatnonzero(np.any(
                depthwise[:, :, :, output_channel] != 0, axis=(0, 1)
            ))
            np.testing.assert_array_equal(nonzero_inputs, [output_channel])
        np.testing.assert_allclose(
            restricted.float_reference(image[0]), expected, rtol=2e-5, atol=2e-5,
        )

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
