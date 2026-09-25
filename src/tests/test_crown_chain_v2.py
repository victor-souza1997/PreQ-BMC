"""Focused tests for the strict CROWN-chain v2 checker."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from verification.crown_chain_v2 import (
    bind_layers_to_deployment,
    render_relu_class_lemmas,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.esbmc_install import resolve_esbmc_executable


_DEPLOYMENT = """#include <stdint.h>
static __int128 div_round_half_away_from_zero_i128(__int128 n, __int128 d) { return n / d; }
static __int128 clamp_to_signed_range_i128(__int128 v, int bits) { return v; }
static const int64_t LAYER_0_WEIGHTS[4] = {1, 0, 2, 3};
static const int64_t LAYER_0_BIAS[2] = {4, 5};
void qnn_forward_fixed(const int64_t *input, int64_t *output) {
  for (int out_index = 0; out_index < 2; ++out_index) {
    __int128 acc = 0;
    for (int in_index = 0; in_index < 2; ++in_index)
      acc += LAYER_0_WEIGHTS[out_index * 2 + in_index] * input[in_index];
    output[out_index] = (int64_t)(acc + LAYER_0_BIAS[out_index]);
  }
}
"""


def _layers():
    return {0: {
        "rows": 2, "cols": 2,
        "indptr": [0, 1, 3],
        "indices": [0, 0, 1],
        "data": [1, 2, 3],
        "bias": [4, 5],
    }}


class CrownChainV2BindingTest(unittest.TestCase):
    def test_dense_binding_is_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "qnn.c"
            source.write_text(_DEPLOYMENT, encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            result = bind_layers_to_deployment(_layers(), source, digest)
        self.assertEqual(result["status"], "EXACT_ORDERED_MATCH")
        self.assertEqual(result["nonzero_terms_compared"], 3)

    def test_dense_binding_rejects_weight_permutation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "qnn.c"
            source.write_text(_DEPLOYMENT.replace("{1, 0, 2, 3}", "{2, 0, 1, 3}"), encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "weight mismatch"):
                bind_layers_to_deployment(_layers(), source, digest)

    def test_all_corrected_relu_classes_have_shared_lemmas(self):
        rendered = render_relu_class_lemmas(400_000, 3_300_000)
        names = {name for name, _, _ in rendered}
        self.assertEqual(names, {
            "lemma_relu_bound_free.c",
            "lemma_relu_nosat_upper.c",
            "lemma_relu_lower.c",
            "lemma_relu_upper.c",
            "lemma_relu_lower_upper.c",
            "lemma_relu_positive_product.c",
        })
        nosat = next(source for name, source, _ in rendered if name == "lemma_relu_nosat_upper.c")
        self.assertIn("raw <= 32767", nosat)
        lower = next(source for name, source, _ in rendered if name == "lemma_relu_lower.c")
        self.assertIn("raw >= lower", lower)


@unittest.skipUnless(resolve_esbmc_executable("esbmc"), "ESBMC is optional")
class CrownChainV2LemmaESBMCTest(unittest.TestCase):
    def test_shared_relu_lemmas(self):
        runner = ESBMCRunner(ESBMCConfig(
            timeout_seconds=20, memlimit="1g", default_profile="paper-z3"
        ))
        with tempfile.TemporaryDirectory() as directory:
            for name, source, _ in render_relu_class_lemmas(400_000, 3_300_000):
                with self.subTest(name=name):
                    harness = Path(directory) / name
                    harness.write_text(source, encoding="utf-8")
                    result = runner.run_file(harness, profile="paper-z3")
                    self.assertEqual(result.status, "VERIFIED")


if __name__ == "__main__":
    unittest.main()
