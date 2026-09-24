import ctypes
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backends.c_qnn_generator import compile_c_qnn_shared_library
from backends.conv_fixed_point import (
    ConvFixedPointNetwork,
    QuantizedConv2D,
    QuantizedDense,
    forward_conv_fixed_point_single,
    generate_conv_qnn_source,
    quantize_restricted_sequential,
)
from backends.fixed_point import LayerQuantizationSpec, forward_fixed_point_single
from backends.image_encoder import ByteImageEncoder
from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN
from verification.conv_contracts import (
    IntegerInvariant,
    dense_output_terms,
    propagate_interval,
    propagate_network,
    render_conv_block_contract,
    render_invariant_bridge,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.conv_proof import (
    dense_margin_endpoint_witness,
    render_byte_encoding_contract,
    render_dense_block_contract,
    render_dense_margin_witness_replay,
)
from verification.interval_lemmas import render_dense_interval_certificate
from verification.esbmc_install import resolve_esbmc_executable


def tiny_sequential_model():
    first = ConvGeometry((3, 3, 1), (2, 2, 1, 2), (1, 1), "VALID")
    second = ConvGeometry(first.output_shape, (2, 2, 2, 1), (1, 1), "VALID")
    return RestrictedSequentialCNN(
        geometries=(first, second),
        conv_kernels=(
            np.array([1, -2, 2, 1, -1, 1, 2, -1], dtype=np.float32).reshape(first.kernel_shape) / 4,
            np.array([1, -1, 2, 1, -2, 1, 1, 2], dtype=np.float32).reshape(second.kernel_shape) / 4,
        ),
        conv_biases=(np.array([0.25, -0.25]), np.array([0.125])),
        dense_kernels=(np.array([[0.5, -0.25]], dtype=np.float32),),
        dense_biases=(np.array([0.125, -0.125]),),
    )


class ConvNativeFixedPointTest(unittest.TestCase):
    def setUp(self):
        self.model = tiny_sequential_model()
        self.specs = [LayerQuantizationSpec(8, 3, 4)] * 3
        self.network = quantize_restricted_sequential(
            self.model, self.specs, input_fractional_bits=4, input_total_bits=8
        )

    def test_network_rejects_possible_signed_i128_accumulator_overflow(self):
        spec = LayerQuantizationSpec(63, 62, 0)
        layer = QuantizedDense(
            np.full((2, 9), (1 << 62) - 1, dtype=np.int64),
            np.zeros(2, dtype=np.int64), spec, input_fractional_bits=0,
            apply_relu=False,
        )
        with self.assertRaisesRegex(ValueError, "does not fit signed __int128"):
            ConvFixedPointNetwork((3, 3, 1), 0, 63, (layer,))

    def test_compact_python_matches_existing_dense_semantics(self):
        dense = self.model.quantized(self.specs, input_fractional_bits=4, input_total_bits=8)
        for values in itertools.product((0, 1, 4, 7), repeat=3):
            sample_int = np.resize(np.asarray(values, dtype=np.int64), 9)
            compact = forward_conv_fixed_point_single(self.network, sample_int)
            existing = forward_fixed_point_single(dense, sample_int.astype(np.float64) / 16.0)
            np.testing.assert_array_equal(compact, existing)

    def test_compact_c_matches_python(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "compact.c"
            encoder = ByteImageEncoder((3, 3, 1), fractional_bits=4, total_bits=8)
            source.write_text(
                generate_conv_qnn_source(self.network, encoder_source=encoder.render_c()),
                encoding="utf-8",
            )
            library = ctypes.CDLL(str(compile_c_qnn_shared_library(
                source, source.with_suffix(".so")
            ).resolve()))
            self.assertEqual(library.qnn_encoder_size(), 9)
            pointer = ctypes.POINTER(ctypes.c_int64)
            library.qnn_forward_fixed.argtypes = [pointer, pointer]
            for seed in range(20):
                sample = np.random.default_rng(seed).integers(-8, 16, 9, dtype=np.int64)
                expected = forward_conv_fixed_point_single(self.network, sample)
                actual = np.zeros(2, dtype=np.int64)
                library.qnn_forward_fixed(
                    sample.ctypes.data_as(pointer), actual.ctypes.data_as(pointer)
                )
                np.testing.assert_array_equal(actual, expected)

    def test_interval_certificates_contain_exhaustive_integer_execution(self):
        initial = IntegerInvariant((0,) * 9, (1,) * 9, (3, 3, 1))
        certificates = propagate_network(self.network.layers, initial)
        for values in itertools.product((0, 1), repeat=9):
            final, trace = forward_conv_fixed_point_single(
                self.network, np.asarray(values), return_trace=True
            )
            del final
            for certificate, layer_values in zip(certificates, trace, strict=True):
                self.assertTrue(certificate.output_invariant.contains(layer_values))

    def test_sparse_dense_deployment_and_certificates_skip_zero_macs_exactly(self):
        spec = LayerQuantizationSpec(8, 3, 4)
        layer = QuantizedDense(
            np.asarray([[8, 0, 0, 0], [0, 0, -4, 0]], dtype=np.int64),
            np.asarray([1, -1], dtype=np.int64),
            spec,
            input_fractional_bits=4,
            apply_relu=False,
        )
        network = ConvFixedPointNetwork((1, 4, 1), 4, 8, (layer,))
        source_text = generate_conv_qnn_source(network)
        self.assertIn("LAYER_0_OFFSET", source_text)
        self.assertIn("LAYER_0_INPUT", source_text)
        self.assertEqual(dense_output_terms(layer, 0), [(0, 8)])
        self.assertEqual(dense_output_terms(layer, 1), [(2, -4)])

        invariant = IntegerInvariant((-4, -3, -2, -1), (1, 2, 3, 4), (4,))
        certificate = propagate_interval(layer, invariant, layer_index=0)
        explicit = render_dense_block_contract(layer, certificate, [0, 1])
        interval = render_dense_interval_certificate(layer, certificate, [0, 1])
        for harness in (explicit, interval):
            self.assertIn("#define LOCAL_INPUT_SIZE 2", harness)
            self.assertIn("#define MAX_TERMS 1", harness)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sparse.c"
            source.write_text(source_text, encoding="utf-8")
            library = ctypes.CDLL(str(compile_c_qnn_shared_library(
                source, source.with_suffix(".so")
            ).resolve()))
            pointer = ctypes.POINTER(ctypes.c_int64)
            library.qnn_forward_fixed.argtypes = [pointer, pointer]
            for values in itertools.product((-4, 0, 4), repeat=4):
                sample = np.asarray(values, dtype=np.int64)
                expected = forward_conv_fixed_point_single(network, sample)
                actual = np.zeros(2, dtype=np.int64)
                library.qnn_forward_fixed(
                    sample.ctypes.data_as(pointer), actual.ctypes.data_as(pointer)
                )
                np.testing.assert_array_equal(actual, expected)

    def test_conv_harness_contains_only_receptive_field(self):
        layer = self.network.layers[0]
        self.assertIsInstance(layer, QuantizedConv2D)
        initial = IntegerInvariant((0,) * 9, (3,) * 9, (3, 3, 1))
        certificate = propagate_interval(layer, initial, layer_index=0)
        source = render_conv_block_contract(layer, certificate, [0])
        self.assertIn("#define LOCAL_INPUT_SIZE 4", source)
        self.assertIn("#define MAX_TERMS 4", source)
        self.assertNotIn("[9][8]", source)
        self.assertEqual(ESBMCRunner().infer_unwind(source), 5)

    def test_bridge_direction_is_explicit(self):
        guarantee = IntegerInvariant((1, 2), (3, 4), (2,))
        wider = IntegerInvariant((0, 1), (4, 5), (2,))
        self.assertTrue(guarantee.subset_of(wider))
        source = render_invariant_bridge(guarantee, wider)
        self.assertIn("ASSUMPTION_LOW[i] <= GUARANTEE_LOW[i]", source)

    def test_dense_margin_endpoint_witness_uses_adverse_box_endpoints(self):
        spec = LayerQuantizationSpec(8, 7, 0)
        layer = QuantizedDense(
            np.asarray([[1, 0], [-1, 0]], dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            spec,
            input_fractional_bits=0,
            apply_relu=False,
        )
        invariant = IntegerInvariant((-3, 0), (3, 0), (2,))
        witness = dense_margin_endpoint_witness(layer, invariant, 0, 1)
        self.assertEqual(witness, (-3, 0))
        source = render_dense_margin_witness_replay(
            layer, invariant, 0, 1, witness
        )
        self.assertIn("static const int64_t INPUT[LOCAL_INPUT_SIZE] = {-3, 0}", source)
        self.assertNotIn("nondet_long_long", source)
        self.assertIn("abstract hidden-box witness violates strict margin", source)

    @unittest.skipUnless(resolve_esbmc_executable("esbmc"), "ESBMC is optional")
    def test_esbmc_replays_abstract_margin_witness_as_failed_diagnostic(self):
        spec = LayerQuantizationSpec(8, 7, 0)
        layer = QuantizedDense(
            np.asarray([[1, 0], [-1, 0]], dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            spec,
            input_fractional_bits=0,
            apply_relu=False,
        )
        invariant = IntegerInvariant((-3, 0), (3, 0), (2,))
        witness = dense_margin_endpoint_witness(layer, invariant, 0, 1)
        source = render_dense_margin_witness_replay(
            layer, invariant, 0, 1, witness
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = Path(directory) / "abstract_margin_witness.c"
            harness.write_text(source, encoding="utf-8")
            result = ESBMCRunner(ESBMCConfig(
                timeout_seconds=30, memlimit="1g", default_profile="paper-z3"
            )).run_file(harness, profile="paper-z3")
        self.assertEqual(result.status, "FAILED", result.stderr or result.stdout)

    @unittest.skipUnless(resolve_esbmc_executable("esbmc"), "ESBMC is optional")
    def test_esbmc_proves_raw_byte_encoding_box(self):
        encoder = ByteImageEncoder((1, 2, 1), fractional_bits=4, total_bits=8)
        byte_low = np.array([[[0], [16]]], dtype=np.uint8)
        byte_high = np.array([[[15], [31]]], dtype=np.uint8)
        low, high = encoder.encode(byte_low), encoder.encode(byte_high)
        invariant = IntegerInvariant.from_arrays(low, high, (1, 2, 1), "exact_byte_box")
        source = render_byte_encoding_contract(
            byte_low.reshape(-1), byte_high.reshape(-1), invariant,
            fractional_bits=4, total_bits=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = Path(directory) / "byte_encoding.c"
            harness.write_text(source, encoding="utf-8")
            result = ESBMCRunner(ESBMCConfig(
                timeout_seconds=30, memlimit="1g", default_profile="paper-z3"
            )).run_file(harness, profile="paper-z3")
        self.assertEqual(result.status, "VERIFIED", result.stderr or result.stdout)

    @unittest.skipUnless(resolve_esbmc_executable("esbmc"), "ESBMC is optional")
    def test_esbmc_proves_small_convolution_tile(self):
        layer = self.network.layers[0]
        initial = IntegerInvariant((0,) * 9, (3,) * 9, (3, 3, 1))
        certificate = propagate_interval(layer, initial, layer_index=0)
        source = render_conv_block_contract(layer, certificate, [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            harness = Path(directory) / "conv_tile.c"
            harness.write_text(source, encoding="utf-8")
            result = ESBMCRunner(ESBMCConfig(
                timeout_seconds=30, memlimit="1g", default_profile="paper-z3"
            )).run_file(harness, profile="paper-z3")
        self.assertEqual(result.status, "VERIFIED", result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
