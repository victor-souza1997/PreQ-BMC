from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import unittest

import numpy as np

from backends.c_qnn_generator import CompiledCQNN, compile_c_qnn_shared_library, write_c_qnn_source
from backends.fixed_point import LayerQuantizationSpec, build_fixed_point_network, forward_fixed_point_single
from tests.test_fixed_point_forward import _build_toy_model


@unittest.skipUnless(shutil.which("gcc"), "gcc is required for the generated C backend test")
class CQNNGenerationTest(unittest.TestCase):
    def test_generated_c_network_matches_python_backend(self) -> None:
        model = _build_toy_model()
        specs = [
            LayerQuantizationSpec(total_bits=8, integer_bits=4, fractional_bits=4),
            LayerQuantizationSpec(total_bits=8, integer_bits=4, fractional_bits=4),
        ]
        network = build_fixed_point_network(model, specs)
        sample = np.asarray([0.5, 0.25], dtype=np.float64)
        expected = forward_fixed_point_single(network, sample)

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = write_c_qnn_source(network, Path(tmp_dir) / "toy_qnn.c")
            shared_path = compile_c_qnn_shared_library(source_path, Path(tmp_dir) / "toy_qnn.so")
            compiled = CompiledCQNN(network, shared_path)

            actual = compiled.forward(sample)

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
