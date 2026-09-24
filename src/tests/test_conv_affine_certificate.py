"""Regression tests for untrusted convolution affine certificates."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from verification.conv_affine_certificate import (
    SCHEMA,
    load_affine_certificate,
    render_scalarized_symbolic_affine_block,
    render_unrolled_recomputed_affine_block,
    slice_affine_certificate,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.esbmc_install import resolve_esbmc_executable


SCALE = 1 << 16


def _form(coefficient: int, constant: int = 0) -> dict:
    return {
        "input_indices": [0],
        "coefficients": [coefficient],
        "constant": constant,
        "scale_bits": 16,
    }


def _certificate() -> dict:
    return {
        "schema": SCHEMA,
        "input_box": {"low": [0], "high": [1]},
        "qif": {"total_bits": 8, "integer_bits": 3, "fractional_bits": 4},
        "previous_layer_envelopes": [{
            "neuron_index": 0,
            "box": [0, 1],
            "h_lower": _form(SCALE),
            "h_upper": _form(SCALE),
        }],
        "block": [{
            "neuron_index": 0,
            "bias": 0,
            "weights": {"input_neurons": [0], "values": [16]},
            "z_lower": _form(SCALE, -(SCALE // 2)),
            "z_upper": _form(SCALE, SCALE // 2),
            "pre_clamp_bounds": [0, 1],
            "esbmc_box_pre_activation": [0, 1],
            "relu_regime": "active",
            "relu_lower_rule": "h>=z",
            "relu_upper_rule": "h<=z",
            "h_lower": _form(SCALE, -(SCALE // 2)),
            "h_upper": _form(SCALE, SCALE // 2),
        }],
    }


def _load(payload: dict) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "certificate.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_affine_certificate(path)


class ConvAffineCertificateTest(unittest.TestCase):
    def test_load_and_slice_preserve_required_predecessors(self):
        certificate = _load(_certificate())
        selected = slice_affine_certificate(certificate, [0])
        self.assertEqual(len(selected["block"]), 1)
        self.assertEqual(
            [row["neuron_index"] for row in selected["previous_layer_envelopes"]],
            [0],
        )

    def test_duplicate_form_input_is_rejected(self):
        payload = _certificate()
        payload["block"][0]["z_lower"] = {
            "input_indices": [0, 0],
            "coefficients": [1, 1],
            "constant": 0,
            "scale_bits": 16,
        }
        with self.assertRaisesRegex(ValueError, "invalid terms"):
            _load(payload)

    def test_straight_line_checker_asserts_exported_relu_regime(self):
        source = render_unrolled_recomputed_affine_block(_load(_certificate()))
        self.assertIn('"active ReLU regime 0"', source)
        self.assertIn('"clamp inactivity 0"', source)
        self.assertNotIn("nondet_long_long", source)
        self.assertNotIn("coefficient[", source)

    def test_symbolic_checker_uses_scalar_state(self):
        source = render_scalarized_symbolic_affine_block(_load(_certificate()))
        self.assertIn("nondet_long_long", source)
        self.assertNotIn("x[", source)
        self.assertNotIn("previous[", source)


@unittest.skipUnless(resolve_esbmc_executable("esbmc"), "ESBMC is optional")
class ConvAffineCertificateESBMCTest(unittest.TestCase):
    def _run(self, certificate: dict) -> str:
        source = render_unrolled_recomputed_affine_block(certificate)
        with tempfile.TemporaryDirectory() as directory:
            harness = Path(directory) / "affine_certificate.c"
            harness.write_text(source, encoding="utf-8")
            result = ESBMCRunner(ESBMCConfig(
                timeout_seconds=20,
                memlimit="1g",
                default_profile="paper-z3",
            )).run_file(harness, profile="paper-z3")
        return result.status

    def test_valid_certificate_verifies_and_corruption_fails(self):
        certificate = _load(_certificate())
        self.assertEqual(self._run(certificate), "VERIFIED")

        corrupted = copy.deepcopy(certificate)
        corrupted["block"][0]["z_lower"]["coefficients"][0] += 1
        self.assertEqual(self._run(corrupted), "FAILED")


if __name__ == "__main__":
    unittest.main()
