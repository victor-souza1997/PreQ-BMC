from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from backends.fixed_point import LayerQuantizationSpec, build_fixed_point_network, forward_fixed_point_single
from models.deep_model import DeepModel


def _build_toy_model() -> DeepModel:
    model = DeepModel([2, 2], input_scale=1.0)
    model.build((None, 2))
    _ = model(tf.zeros((1, 2), dtype=tf.float32))
    model.dense_layers[0].set_weights(
        [
            np.asarray([[0.5, 0.125], [-0.25, 0.5]], dtype=np.float32),
            np.asarray([0.25, -0.125], dtype=np.float32),
        ]
    )
    model.dense_layers[1].set_weights(
        [
            np.asarray([[0.5, -0.25], [0.5, 0.25]], dtype=np.float32),
            np.asarray([0.0, 0.125], dtype=np.float32),
        ]
    )
    return model


class FixedPointForwardTest(unittest.TestCase):
    def test_python_fixed_point_forward_matches_expected_integer_outputs(self) -> None:
        model = _build_toy_model()
        specs = [
            LayerQuantizationSpec(total_bits=8, integer_bits=4, fractional_bits=4),
            LayerQuantizationSpec(total_bits=8, integer_bits=4, fractional_bits=4),
        ]
        network = build_fixed_point_network(model, specs)
        sample = np.asarray([0.5, 0.25], dtype=np.float64)

        output_int = forward_fixed_point_single(network, sample)

        np.testing.assert_array_equal(output_int, np.asarray([4, 0], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
