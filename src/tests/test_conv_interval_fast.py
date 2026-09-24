import unittest

import numpy as np

from backends.conv_fixed_point import QuantizedDense, _finish_value, quantize_restricted_sequential
from backends.fixed_point import LayerQuantizationSpec
from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN
from verification.conv_contracts import IntegerInvariant, propagate_network
from verification.conv_interval_fast import (
    box_margin,
    difference_row_margin,
    propagate_box,
)

Q16 = LayerQuantizationSpec(total_bits=16, integer_bits=7, fractional_bits=8)


def _network(seed, scale=0.4):
    rng = np.random.default_rng(seed)
    geometries = (
        ConvGeometry((8, 8, 3), (3, 3, 3, 4), (2, 2), "SAME"),
        ConvGeometry((4, 4, 4), (3, 3, 4, 6), (1, 1), "SAME"),
        ConvGeometry((4, 4, 6), (2, 2, 6, 6), (2, 2), "VALID"),
    )
    kernels = [rng.normal(0, scale, g.kernel_shape).astype(np.float32) for g in geometries]
    # Depthwise last convolution, stored densely with zeros off the diagonal.
    kernels[-1] *= np.eye(6, dtype=np.float32)[None, None]
    biases = [rng.normal(0, 0.1, g.kernel_shape[-1]).astype(np.float32) for g in geometries]
    dense = (rng.normal(0, scale, (24, 5)).astype(np.float32),)
    model = RestrictedSequentialCNN(
        geometries, tuple(kernels), tuple(biases), dense,
        (rng.normal(0, 0.1, 5).astype(np.float32),), 10**6,
    )
    return quantize_restricted_sequential(model, [Q16] * 4)


def _box(seed, width):
    rng = np.random.default_rng(seed + 100)
    center = rng.integers(0, 256, 8 * 8 * 3)
    return np.maximum(center - width, 0), np.minimum(center + width, 255)


class FastIntervalTests(unittest.TestCase):
    def test_matches_reference_propagation_exactly(self):
        for seed, width, scale in ((0, 1, 0.4), (1, 4, 1.5), (2, 30, 3.0)):
            network = _network(seed, scale)
            low, high = _box(seed, width)
            reference = propagate_network(
                network.layers, IntegerInvariant.from_arrays(low, high, (8, 8, 3)))
            fast = propagate_box(network, low, high)
            for certificate, lo, hi, pre_lo, pre_hi in zip(
                    reference, fast.low, fast.high, fast.pre_low, fast.pre_high, strict=True):
                self.assertEqual(certificate.output_invariant.low, tuple(int(v) for v in lo))
                self.assertEqual(certificate.output_invariant.high, tuple(int(v) for v in hi))
                self.assertEqual(certificate.pre_relu_low, tuple(int(v) for v in pre_lo))
                self.assertEqual(certificate.pre_relu_high, tuple(int(v) for v in pre_hi))

    def test_box_margin_is_target_low_minus_largest_competitor_high(self):
        network = _network(3)
        trace = propagate_box(network, *_box(3, 1))
        high = [int(v) for v in trace.high[-1]]
        expected = int(trace.low[-1][2]) - max(v for i, v in enumerate(high) if i != 2)
        self.assertEqual(box_margin(trace, 2), expected)

    def test_difference_row_bound_holds_on_sampled_final_inputs(self):
        for seed in range(4):
            network = _network(seed, 1.0)
            trace = propagate_box(network, *_box(seed, 6))
            layer = network.layers[-1]
            self.assertIsInstance(layer, QuantizedDense)
            low, high = trace.final_input_low, trace.final_input_high
            rng = np.random.default_rng(seed)
            for target in range(layer.output_size):
                bound, _ = difference_row_margin(network, trace, target)
                for _ in range(200):
                    h = rng.integers(low, high + 1)
                    pre = [
                        int(np.dot(layer.weights_int[i], h)) for i in range(layer.output_size)
                    ]
                    denominator = 1 << layer.input_fractional_bits
                    rounded = [
                        (abs(a) + denominator // 2) // denominator * (1 if a >= 0 else -1)
                        + int(layer.bias_int[i]) for i, a in enumerate(pre)
                    ]
                    worst = min(rounded[target] - v for i, v in enumerate(rounded) if i != target)
                    self.assertGreaterEqual(worst, bound)
                    # Deployed values agree with the unclamped model inside range.
                    self.assertEqual(
                        _finish_value(pre[target], int(layer.bias_int[target]), layer),
                        max(-32768, min(32767, rounded[target])),
                    )


if __name__ == "__main__":
    unittest.main()
