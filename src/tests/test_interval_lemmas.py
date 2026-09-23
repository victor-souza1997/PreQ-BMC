import tempfile
from pathlib import Path
import unittest

import numpy as np

from backends.conv_fixed_point import QuantizedConv2D, QuantizedDense, quantize_restricted_sequential
from backends.fixed_point import LayerQuantizationSpec
from tests.test_conv_native_verification import tiny_sequential_model
from verification.conv_contracts import IntegerInvariant, propagate_interval
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.conv_cegar import SparseRelationalCut
from verification.conv_proof import render_dense_relational_cut_validation
from verification.esbmc_install import resolve_esbmc_executable
from verification.interval_lemma_parts import (
    render_clamp_relu_monotonicity_lemma,
    render_rounding_monotonicity_lemma,
)
from verification.interval_lemmas import (
    render_conv_interval_certificate,
    render_dense_interval_certificate,
)


@unittest.skipUnless(resolve_esbmc_executable("esbmc"), "ESBMC is optional")
class IntervalLemmaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specs = [LayerQuantizationSpec(8, 3, 4)] * 3
        cls.network = quantize_restricted_sequential(
            tiny_sequential_model(), specs, input_fractional_bits=4, input_total_bits=8
        )
        cls.runner = ESBMCRunner(ESBMCConfig(
            timeout_seconds=30, memlimit="1g", default_profile="paper-z3"
        ))

    def _run(self, source, name):
        with tempfile.TemporaryDirectory() as directory:
            harness = Path(directory) / name
            harness.write_text(source, encoding="utf-8")
            return self.runner.run_file(harness, profile="paper-z3")

    def test_esbmc_proves_monotone_fixed_point_postprocessing(self):
        layer = self.network.layers[0]
        sources = {
            "rounding.c": render_rounding_monotonicity_lemma(layer, 8),
            "clamp_relu.c": render_clamp_relu_monotonicity_lemma(layer),
        }
        for name, source in sources.items():
            with self.subTest(name=name):
                result = self._run(source, name)
                self.assertEqual(result.status, "VERIFIED", result.stderr or result.stdout)

    def test_esbmc_checks_concrete_convolution_certificate(self):
        layer = self.network.layers[0]
        self.assertIsInstance(layer, QuantizedConv2D)
        initial = IntegerInvariant((0,) * 9, (3,) * 9, (3, 3, 1))
        certificate = propagate_interval(layer, initial, layer_index=0)
        result = self._run(
            render_conv_interval_certificate(layer, certificate, np.arange(layer.output_size)),
            "interval_certificate.c",
        )
        self.assertEqual(result.status, "VERIFIED", result.stderr or result.stdout)

    def test_esbmc_checks_concrete_dense_certificate(self):
        layer = self.network.layers[2]
        self.assertIsInstance(layer, QuantizedDense)
        initial = IntegerInvariant((0,) * layer.input_size, (3,) * layer.input_size,
                                   (layer.input_size,))
        certificate = propagate_interval(layer, initial, layer_index=1)
        result = self._run(
            render_dense_interval_certificate(layer, certificate, range(layer.output_size)),
            "dense_interval_certificate.c",
        )
        self.assertEqual(result.status, "VERIFIED", result.stderr or result.stdout)

    def test_esbmc_validates_sparse_dense_relational_cut(self):
        layer = self.network.layers[2]
        initial = IntegerInvariant((0,) * layer.input_size, (3,) * layer.input_size,
                                   (layer.input_size,))
        certificate = propagate_interval(layer, initial, layer_index=2)
        cut = SparseRelationalCut(
            layer_index=2, indices=(0,), coefficients=(1,), scale=1,
            lower=certificate.output_invariant.low[0],
            upper=certificate.output_invariant.high[0],
        )
        result = self._run(
            render_dense_relational_cut_validation(layer, certificate, cut),
            "dense_relational_cut.c",
        )
        self.assertEqual(result.status, "VERIFIED", result.stderr or result.stdout)

    def test_corrupted_certificate_is_rejected(self):
        layer = self.network.layers[0]
        initial = IntegerInvariant((0,) * 9, (3,) * 9, (3, 3, 1))
        certificate = propagate_interval(layer, initial, layer_index=0)
        source = render_conv_interval_certificate(layer, certificate, [0])
        source = source.replace(
            f"static const int64_t EXPECTED_LOW[BLOCK_SIZE] = {{{certificate.output_invariant.low[0]}}};",
            f"static const int64_t EXPECTED_LOW[BLOCK_SIZE] = {{{certificate.output_invariant.low[0] + 1}}};",
        )
        result = self._run(source, "corrupted_certificate.c")
        self.assertEqual(result.status, "FAILED")


if __name__ == "__main__":
    unittest.main()
