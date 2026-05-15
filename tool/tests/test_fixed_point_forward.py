from __future__ import annotations

import unittest

import numpy as np

from backends.fixed_point import (
    LayerQuantizationSpec,
    build_fixed_point_network,
    forward_fixed_point_batch_with_diagnostics,
    forward_fixed_point_single,
)
from reports.experiment_summary import summarize_saturation
from reports.resource_metrics import compute_fixed_point_resource_metrics


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
    def test_layer_quantization_spec_validates_sign_bit_invariant(self) -> None:
        spec = LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4)

        self.assertEqual(spec.signed_range, (-128, 127))
        self.assertEqual(spec.real_range, (-8.0, 7.9375))

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


if __name__ == "__main__":
    unittest.main()
