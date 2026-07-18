from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

from models.deep_model import DeepModel
from models.loading import build_and_load_deep_model, infer_dense_architecture_from_h5


class ModelLoadingTest(unittest.TestCase):
    def test_subclassed_model_weight_loading_after_materialization(self) -> None:
        model = DeepModel([3, 2], input_scale=1.0)
        model.build((None, 4))
        _ = model(tf.zeros((1, 4), dtype=tf.float32))

        model.dense_layers[0].set_weights(
            [
                np.asarray(
                    [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [1.0, 1.1, 1.2],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.05, -0.1, 0.2], dtype=np.float32),
            ]
        )
        model.dense_layers[1].set_weights(
            [
                np.asarray(
                    [
                        [0.2, -0.1],
                        [0.3, 0.4],
                        [0.5, 0.6],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.01, -0.02], dtype=np.float32),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = Path(tmp_dir) / "toy.weights.h5"
            model.save_weights(str(weights_path))

            inferred_arch = infer_dense_architecture_from_h5(weights_path)
            self.assertEqual(inferred_arch, [4, 3, 2])

            loaded = build_and_load_deep_model(
                input_dim=4,
                layer_units=[3, 2],
                weights_path=weights_path,
                input_scale=1.0,
            )

            sample = np.asarray([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)
            np.testing.assert_allclose(
                model(sample, training=False).numpy(),
                loaded(sample, training=False).numpy(),
                atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()
