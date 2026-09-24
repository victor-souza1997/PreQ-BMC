"""Focused tests for exact-integer CROWN-chain ESBMC obligations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from verification.crown_chain_certificate import (
    render_affine_step,
    render_chain_closure,
    render_margin_closure,
    render_weighted_rounding_lemma,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.esbmc_install import resolve_esbmc_executable


def _affine_fixture() -> tuple[dict, dict, dict]:
    certificate = {
        "qif": {"total_bits": 8, "fractional_bits": 1},
        "input_box": {"low": [0], "high": [1]},
        "layers_by_index": {
            0: {
                "rows": 1,
                "cols": 1,
                "indptr": [0, 1],
                "indices": [0],
                "data": [2],
                "bias": [3],
            },
        },
        "bounds_by_id": {},
    }
    chain = {"certificate_id": "chain/test"}
    step = {
        "step_id": "chain/test/s0",
        "kind": "affine",
        "layer_index": 0,
        "input_coefficients": {"indices": [0], "values": [1]},
        "output_coefficients": {"indices": [0], "values": [1]},
        "residual": {"indices": [], "values": []},
        "residual_lower_values": {"indices": [], "values": []},
        "input_constant": 0,
        "rounding_term": 1,
        "output_constant": 2,
        "max_abs_g": 2,
    }
    return certificate, chain, step


def _margin_fixture(*, target_low: int = -127, competitor_high: int = 126) -> dict:
    return {
        "qif": {"total_bits": 8},
        "chains": [
            {"certificate_id": "chain/margin/12-14", "bound": 1},
            {"certificate_id": "chain/logit/12/lower", "bound": target_low},
        ],
        "margin_closure": {
            "target": 12,
            "competitor": 14,
            "raw_margin_lower_bound": 1,
            "raw_target_lower": target_low,
            "raw_competitor_upper": competitor_high,
            "logit_post_clamp_box": {"14": [-128, competitor_high]},
        },
    }


def _chain_closure_fixture() -> tuple[dict, dict]:
    certificate = {
        "scale_bits": 1,
        "layers_by_index": {0: {"rows": 1}},
    }
    chain = {
        "certificate_id": "chain/L0/n0/lower",
        "claim": "lower",
        "lambda_var": "z0",
        "lambda": {"indices": [0], "values": [2]},
        "scaled_bound": 0,
        "bound": 0,
        "steps": [{
            "step_id": "chain/L0/n0/lower/s0",
            "kind": "concretize",
            "input_coefficients": {"indices": [0], "values": [2]},
            "input_constant": 0,
            "scaled_bound": 0,
        }],
    }
    return certificate, chain


class CrownChainCertificateTest(unittest.TestCase):
    def test_affine_chunks_contain_no_tautological_partial_assertion(self):
        certificate, chain, step = _affine_fixture()
        rendered = render_affine_step(certificate, chain, step)
        chunk = next(source for _, source, meta in rendered if meta["kind"] == "affine_chunk")
        close = next(source for _, source, meta in rendered if meta["kind"] == "affine_close")
        self.assertNotIn("partial 0", chunk)
        self.assertIn("floor_div_i128(residual_dot - rounding_term, 2)", close)
        self.assertIn("computed == (__int128)(2)", close)

    def test_two_sided_margin_closure_emits_both_conditions(self):
        source = render_margin_closure(_margin_fixture())
        self.assertIn('1, "two-sided top clamp condition"', source)
        self.assertIn('1, "two-sided bottom clamp condition"', source)
        bottom_failure = render_margin_closure(_margin_fixture(target_low=-128))
        self.assertIn('0, "two-sided bottom clamp condition"', bottom_failure)
        top_failure = render_margin_closure(_margin_fixture(competitor_high=127))
        self.assertIn('0, "two-sided top clamp condition"', top_failure)


@unittest.skipUnless(resolve_esbmc_executable("esbmc"), "ESBMC is optional")
class CrownChainCertificateESBMCTest(unittest.TestCase):
    def _run(self, source: str, name: str) -> str:
        with tempfile.TemporaryDirectory() as directory:
            harness = Path(directory) / name
            harness.write_text(source, encoding="utf-8")
            result = ESBMCRunner(ESBMCConfig(
                timeout_seconds=20,
                memlimit="1g",
                default_profile="paper-z3",
            )).run_file(harness, profile="paper-z3")
        return result.status

    def test_affine_output_constant_corruption_is_rejected(self):
        certificate, chain, step = _affine_fixture()
        clean = render_affine_step(certificate, chain, step)[-1][1]
        corrupted_step = deepcopy(step)
        corrupted_step["output_constant"] += 1
        corrupted = render_affine_step(certificate, chain, corrupted_step)[-1][1]
        self.assertEqual(self._run(clean, "clean.c"), "VERIFIED")
        self.assertEqual(self._run(corrupted, "corrupted.c"), "FAILED")

    def test_margin_closure_rejects_missing_end_condition(self):
        good = render_margin_closure(_margin_fixture())
        missing_bottom = render_margin_closure(_margin_fixture(target_low=-128))
        self.assertEqual(self._run(good, "good_margin.c"), "VERIFIED")
        self.assertEqual(self._run(missing_bottom, "bad_margin.c"), "FAILED")

    def test_chain_identity_corruption_is_rejected(self):
        certificate, chain = _chain_closure_fixture()
        clean = render_chain_closure(certificate, chain)[1]
        corrupted_chain = deepcopy(chain)
        corrupted_chain["lambda"]["values"][0] = 1
        corrupted_chain["steps"][0]["input_coefficients"]["values"][0] = 1
        corrupted = render_chain_closure(certificate, corrupted_chain)[1]
        self.assertEqual(self._run(clean, "clean_identity.c"), "VERIFIED")
        self.assertEqual(
            self._run(corrupted, "corrupted_identity.c"),
            "FAILED",
        )

    def test_round_half_away_error_bound(self):
        source = render_weighted_rounding_lemma(8)
        self.assertEqual(self._run(source, "rounding_error.c"), "VERIFIED")


if __name__ == "__main__":
    unittest.main()
