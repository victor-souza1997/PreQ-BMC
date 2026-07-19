from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import unittest
import ctypes

import numpy as np

from backends.c_qnn_generator import (
    CompiledCQNN,
    compare_python_c_fixed_point_outputs,
    compile_c_qnn_shared_library,
    write_fixed_point_semantics_test_source,
    write_c_qnn_source,
)
from backends.fixed_point import (
    LayerQuantizationSpec,
    apply_fixed_point_affine_value,
    build_fixed_point_network,
    forward_fixed_point_single,
)
from utils.fixed_point import clamp_to_signed_range, round_divide_half_away_from_zero
from tests.test_fixed_point_forward import _build_toy_model


@unittest.skipUnless(shutil.which("gcc"), "gcc is required for the generated C backend test")
class CQNNGenerationTest(unittest.TestCase):
    def test_generated_c_network_matches_python_backend(self) -> None:
        model = _build_toy_model()
        specs = [
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
            LayerQuantizationSpec(total_bits=8, integer_bits=3, fractional_bits=4),
        ]
        network = build_fixed_point_network(model, specs)
        sample = np.asarray([0.5, 0.25], dtype=np.float64)
        expected = forward_fixed_point_single(network, sample)

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = write_c_qnn_source(network, Path(tmp_dir) / "toy_qnn.c")
            shared_path = compile_c_qnn_shared_library(source_path, Path(tmp_dir) / "toy_qnn.so")
            compiled = CompiledCQNN(network, shared_path)

            actual = compiled.forward(sample)
            comparison = compare_python_c_fixed_point_outputs(
                network,
                np.asarray([[0.5, 0.25], [-0.75, 0.5], [1.0, -0.5]], dtype=np.float64),
                compiled,
            )

        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(comparison["exact_match"])
        self.assertEqual(comparison["max_integer_difference"], 0)

    def test_generated_c_semantic_primitives_match_python_random_cases(self) -> None:
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = write_fixed_point_semantics_test_source(Path(tmp_dir) / "semantics.c")
            shared_path = compile_c_qnn_shared_library(source_path, Path(tmp_dir) / "semantics.so")
            library = ctypes.CDLL(str(shared_path))

            library.qnn_semantic_round_div_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
            library.qnn_semantic_round_div_i64.restype = ctypes.c_int64
            library.qnn_semantic_clamp_i64.argtypes = [ctypes.c_int64, ctypes.c_int]
            library.qnn_semantic_clamp_i64.restype = ctypes.c_int64
            library.qnn_semantic_relu_i64.argtypes = [ctypes.c_int64]
            library.qnn_semantic_relu_i64.restype = ctypes.c_int64
            library.qnn_semantic_step_i64.argtypes = [
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int,
            ]
            library.qnn_semantic_step_i64.restype = ctypes.c_int64
            library.qnn_semantic_affine2_i64.argtypes = [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            library.qnn_semantic_affine2_i64.restype = ctypes.c_int64

            for _ in range(200):
                total_bits = int(rng.integers(3, 16))
                input_fractional_bits = int(rng.integers(0, 7))
                denominator = 1 << input_fractional_bits
                accumulator = int(rng.integers(-20000, 20001))
                bias = int(rng.integers(-200, 201))
                apply_relu = int(rng.integers(0, 2))

                self.assertEqual(
                    library.qnn_semantic_round_div_i64(accumulator, denominator),
                    round_divide_half_away_from_zero(accumulator, denominator),
                )

                raw_value = int(rng.integers(-20000, 20001))
                self.assertEqual(
                    library.qnn_semantic_clamp_i64(raw_value, total_bits),
                    clamp_to_signed_range(raw_value, total_bits),
                )
                self.assertEqual(library.qnn_semantic_relu_i64(raw_value), max(raw_value, 0))

                self.assertEqual(
                    library.qnn_semantic_step_i64(
                        accumulator,
                        input_fractional_bits,
                        bias,
                        total_bits,
                        apply_relu,
                    ),
                    apply_fixed_point_affine_value(
                        accumulator,
                        input_fractional_bits,
                        bias,
                        total_bits,
                        apply_relu=bool(apply_relu),
                    ),
                )

    def test_shared_kernel_rounds_half_away_from_zero_at_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = write_fixed_point_semantics_test_source(Path(tmp_dir) / "semantics.c")
            shared_path = compile_c_qnn_shared_library(source_path, Path(tmp_dir) / "semantics.so")
            library = ctypes.CDLL(str(shared_path))
            library.qnn_semantic_round_div_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
            library.qnn_semantic_round_div_i64.restype = ctypes.c_int64

            denominator = 8
            for numerator in (-13, -12, -11, -5, -4, -3, 3, 4, 5, 11, 12, 13):
                self.assertEqual(
                    library.qnn_semantic_round_div_i64(numerator, denominator),
                    round_divide_half_away_from_zero(numerator, denominator),
                )

    def test_shared_kernel_clamps_at_signed_range_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = write_fixed_point_semantics_test_source(Path(tmp_dir) / "semantics.c")
            shared_path = compile_c_qnn_shared_library(source_path, Path(tmp_dir) / "semantics.so")
            library = ctypes.CDLL(str(shared_path))
            library.qnn_semantic_clamp_i64.argtypes = [ctypes.c_int64, ctypes.c_int]
            library.qnn_semantic_clamp_i64.restype = ctypes.c_int64

            total_bits = 4
            for value in (-20, -9, -8, -7, 0, 6, 7, 8, 20):
                self.assertEqual(
                    library.qnn_semantic_clamp_i64(value, total_bits),
                    clamp_to_signed_range(value, total_bits),
                )

    def test_shared_kernel_layer_step_matches_python_for_tiny_affine(self) -> None:
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = write_fixed_point_semantics_test_source(Path(tmp_dir) / "semantics.c")
            shared_path = compile_c_qnn_shared_library(source_path, Path(tmp_dir) / "semantics.so")
            library = ctypes.CDLL(str(shared_path))
            library.qnn_semantic_affine2_i64.argtypes = [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            library.qnn_semantic_affine2_i64.restype = ctypes.c_int64

            x0, x1 = 7, -3
            w0, w1 = 5, -6
            bias = -2
            input_fractional_bits = 3
            total_bits = 5
            accumulator = x0 * w0 + x1 * w1

            for apply_relu in (0, 1):
                self.assertEqual(
                    library.qnn_semantic_affine2_i64(
                        x0,
                        x1,
                        w0,
                        w1,
                        bias,
                        input_fractional_bits,
                        total_bits,
                        apply_relu,
                    ),
                    apply_fixed_point_affine_value(
                        accumulator,
                        input_fractional_bits,
                        bias,
                        total_bits,
                        apply_relu=bool(apply_relu),
                    ),
                )

            for _ in range(10):
                x0 = int(rng.integers(-64, 65))
                x1 = int(rng.integers(-64, 65))
                w0 = int(rng.integers(-32, 33))
                w1 = int(rng.integers(-32, 33))
                affine_accumulator = x0 * w0 + x1 * w1
                for apply_relu in (0, 1):
                    self.assertEqual(
                        library.qnn_semantic_affine2_i64(
                            x0,
                            x1,
                            w0,
                            w1,
                            bias,
                            input_fractional_bits,
                            total_bits,
                            apply_relu,
                        ),
                        apply_fixed_point_affine_value(
                            affine_accumulator,
                            input_fractional_bits,
                            bias,
                            total_bits,
                            apply_relu=bool(apply_relu),
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
