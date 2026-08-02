from __future__ import annotations

import unittest

import numpy as np

from backends.fixed_point import (
    LayerQuantizationSpec,
    compute_accumulator_range_analysis,
    build_fixed_point_network,
    fixed_point_semantics_for_network,
    forward_fixed_point_batch_with_diagnostics,
    forward_fixed_point_single,
)
from reports.experiment_summary import summarize_saturation
from reports.resource_metrics import compute_fixed_point_resource_metrics
from utils.fixed_point import quantize_int, quantize_interval_int_bounds


class _ToyDenseLayer:
    def __init__(self, kernel: np.ndarray, bias: np.ndarray) -> None:
        self._kernel = np.asarray(kernel, dtype=np.float32)
        self._bias = np.asarray(bias, dtype=np.float32)
        self.units = int(self._bias.shape[0])

    def get_weights(self) -> list[np.ndarray]:
        return [self._kernel, self._bias]


class _ToyModel:
    def __init__(self) -> None:
        self.input_scale = 1.0
        self.dense_layers = [
            _ToyDenseLayer(
                np.asarray([[0.5, 0.125], [-0.25, 0.5]], dtype=np.float32),
                np.asarray([0.25, -0.125], dtype=np.float32),
            ),
            _ToyDenseLayer(
                np.asarray([[0.5, -0.25], [0.5, 0.25]], dtype=np.float32),
                np.asarray([0.0, 0.125], dtype=np.float32),
            ),
        ]


def _build_toy_model() -> _ToyModel:
    return _ToyModel()


class FixedPointForwardTest(unittest.TestCase):
    def test_quantized_interval_bounds_are_exact_deployed_endpoint_image(self) -> None:
        lower = np.asarray([-20.0, -0.20, 0.0, 0.49, 7.9], dtype=np.float64)
        upper = np.asarray([-7.9, 0.20, 0.25 / 255.0, 0.76, 20.0], dtype=np.float64)

        lower_int, upper_int = quantize_interval_int_bounds(lower, upper, 8, 8 - 4)

        np.testing.assert_array_equal(lower_int, quantize_int(lower, 8, 4))
        np.testing.assert_array_equal(upper_int, quantize_int(upper, 8, 4))
        for index in range(lower.size):
            samples = np.linspace(lower[index], upper[index], num=2001)
            observed = np.asarray(quantize_int(samples, 8, 4), dtype=np.int64)
            self.assertEqual(int(observed.min()), int(lower_int[index]))
            self.assertEqual(int(observed.max()), int(upper_int[index]))

    def test_exact_input_bounds_remove_unreachable_mnist_lattice_point(self) -> None:
        lower_int, upper_int = quantize_interval_int_bounds(
            np.asarray([0.0]),
            np.asarray([0.25 / 255.0]),
            13,
            8,
        )

        np.testing.assert_array_equal(lower_int, np.asarray([0], dtype=np.int64))
        np.testing.assert_array_equal(upper_int, np.asarray([0], dtype=np.int64))

    def test_layer_quantization_spec_validates_sign_bit_invariant(self) -> None:
        spec = LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4)

        self.assertEqual(spec.signed_range, (-128, 127))
        self.assertEqual(spec.real_range, (-8.0, 7.9375))
        self.assertEqual(spec.semantic_metadata["scale_factor"], 16)
        self.assertEqual(spec.semantic_metadata["q_min_int"], -128)
        self.assertEqual(spec.semantic_metadata["q_max_int"], 127)
        self.assertEqual(spec.semantic_metadata["overflow_mode"], "saturation")
        self.assertEqual(spec.semantic_metadata["rounding_mode"], "round_half_away_from_zero")

    def test_layer_quantization_spec_rejects_inconsistent_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "Q=8, I=2, F=4"):
            LayerQuantizationSpec(total_bits=8, integer_bits=2, fractional_bits=4)

    def test_python_fixed_point_forward_matches_expected_integer_outputs(self) -> None:
        model = _build_toy_model()
        specs = [
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
        ]
        network = build_fixed_point_network(model, specs)
        sample = np.asarray([0.5, 0.25], dtype=np.float64)

        output_int = forward_fixed_point_single(network, sample)

        np.testing.assert_array_equal(output_int, np.asarray([4, 0], dtype=np.int64))

    def test_diagnostic_forward_reports_saturation_fields(self) -> None:
        model = _build_toy_model()
        specs = [
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
        ]
        network = build_fixed_point_network(model, specs)
        samples = np.asarray([[0.5, 0.25], [1.0, -0.5]], dtype=np.float64)

        logits, diagnostics = forward_fixed_point_batch_with_diagnostics(network, samples)

        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(diagnostics["samples"], 2)
        self.assertEqual(len(diagnostics["layers"]), 2)
        for layer in diagnostics["layers"]:
            self.assertIn("saturation_count", layer)
            self.assertIn("saturation_rate", layer)
            self.assertIn("q_min", layer)
            self.assertIn("q_max", layer)

        saturation = summarize_saturation(diagnostics)
        self.assertIn("max_saturation_rate", saturation)
        self.assertIn("mean_saturation_rate", saturation)

    def test_resource_metrics_report_parameter_memory(self) -> None:
        model = _build_toy_model()
        specs = [
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
        ]
        network = build_fixed_point_network(model, specs)

        metrics = compute_fixed_point_resource_metrics(network)

        self.assertEqual(metrics["num_layers"], 2)
        self.assertEqual(metrics["num_parameters"], 12)
        self.assertEqual(metrics["float_parameter_memory_bytes"], 48)
        self.assertEqual(metrics["fixed_parameter_memory_bytes"], 12)
        self.assertEqual(metrics["weighted_avg_bits_per_parameter"], 8.0)

    def test_semantics_and_accumulator_analysis_are_reportable(self) -> None:
        model = _build_toy_model()
        specs = [
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
        ]
        network = build_fixed_point_network(model, specs)

        semantics = fixed_point_semantics_for_network(network)
        accumulator = compute_accumulator_range_analysis(network)

        self.assertEqual(semantics["claim_type"], "declared_backend_semantics")
        self.assertEqual(len(semantics["layers"]), 2)
        self.assertEqual(len(accumulator), 2)
        for layer in accumulator:
            self.assertIn("max_abs_accumulator", layer)
            self.assertIn("fits_int64", layer)
            self.assertIn("per_neuron", layer)
            self.assertEqual(layer["claim_type"], "static_interval_analysis")


if __name__ == "__main__":
    unittest.main()
