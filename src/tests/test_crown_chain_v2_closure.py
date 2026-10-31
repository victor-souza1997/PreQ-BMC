"""Negative mutations and positive controls for the v2 checker closure (P1).

Each mutation is a malformed certificate the checker must reject before any
VERIFIED; the positive controls must still be accepted. No ESBMC is needed:
these test what the Python checker admits, not what ESBMC proves.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from verification.crown_chain_v2 import (
    RELU_CLASSES, CertificateV2, _csr_times_transpose, _sparse_payload,
    _validate_facts, render_v2_harnesses, validate_chain, validate_margin_closure,
)
from verification.crown_chain_v2_coverage import verify_shard_coverage

SCALE_BITS = 8
LOW, HIGH = [100, 120, 90], [101, 121, 91]


def _layers():
    # Layer 0: 3 hidden rows over 3 input bytes. Layer 1: 2 logits over 3 hidden.
    # Logit rows agree on hidden column 1, so the margin's g cancels to 0 there.
    return {
        0: {"rows": 3, "cols": 3, "indptr": [0, 2, 4, 6], "indices": [0, 1, 0, 1, 0, 1],
            "data": [500, 300, 1, 1, 260, -40], "bias": [10, 20, 30]},
        1: {"rows": 2, "cols": 3, "indptr": [0, 3, 6], "indices": [0, 1, 2, 0, 1, 2],
            "data": [3, 2, 1, 1, 2, -1], "bias": [100, -50]},
    }


def _facts():
    return {
        f"fact/L0/n{index}/upper": {"value": 900} for index in range(3)
    }


def _traversal(lambda_layer):
    steps = []
    for layer in range(lambda_layer, -1, -1):
        steps.append(("affine", layer))
        if layer > 0:
            steps.append(("relu", layer - 1))
    return steps + [("concretize", None)]


def _build_chain(certificate_id, claim, lambda_layer, initial, traversal, layers, facts):
    """Serialize a chain with the arithmetic the checker recomputes."""

    current, constant, steps = dict(initial), 0, []
    for ordinal, (kind, layer) in enumerate(traversal):
        step = {"step_id": f"{certificate_id}/s{ordinal}", "kind": kind, "input_constant": constant}
        if ordinal == 0:
            step["input_coefficients"] = _sparse_payload(current)
        if kind == "affine":
            csr = layers[layer]
            g = _csr_times_transpose(csr, current)
            output = {index: value // 256 for index, value in g.items() if value // 256}
            residual = {index: value - 256 * output.get(index, 0) for index, value in g.items()}
            lower = {index: LOW[index] for index in residual} if layer == 0 else {}
            rounding = 128 * sum(abs(value) for value in current.values())
            constant += sum(c * csr["bias"][row] for row, c in current.items()) + (
                sum(value * lower.get(index, 0) for index, value in residual.items()) - rounding
            ) // 256
            step.update(
                layer_index=layer, output_coefficients=_sparse_payload(output),
                residual_lower_sources="input_box.low" if layer == 0 else {"indices": [], "fact_ids": []},
                rounding_term=rounding, output_constant=constant,
            )
            current = output
        elif kind == "relu":
            classes = {name: [] for name in RELU_CLASSES}
            output, constants = {}, {}
            for index, a in sorted(current.items()):
                upper = facts[f"fact/L{layer}/n{index}/upper"]["value"]
                if a > 0:
                    classes["nosat_upper"].append(index)
                    output[index] = a
                else:
                    classes["upper"].append(index)
                    constants[index] = a * max(upper, 0)
                    constant += constants[index]
            step.update(
                layer_index=layer, output_coefficients=_sparse_payload(output),
                relu_constants=_sparse_payload(constants), coordinate_classes=classes,
                output_constant=constant,
            )
            current = output
        else:
            scaled = constant + sum(c * (LOW[i] if c >= 0 else HIGH[i]) for i, c in current.items())
            step["scaled_bound"] = scaled
        steps.append(step)
    scale = 1 << SCALE_BITS
    bound = (-scaled) // scale if claim == "upper" else -((-scaled) // scale)
    return {"certificate_id": certificate_id, "claim": claim, "lambda_layer": lambda_layer,
            "scaled_bound": scaled, "bound": bound, "steps": steps}


def _margin(traversal=None):
    return _build_chain(
        "chain/margin/0-1", "margin", 1, {0: 1 << SCALE_BITS, 1: -(1 << SCALE_BITS)},
        traversal or _traversal(1), _layers(), _facts(),
    )


def _validate(chain):
    return validate_chain(chain, _layers(), _facts(), LOW, HIGH, SCALE_BITS)


class ChainTraversalTest(unittest.TestCase):
    def test_positive_control_valid_chain_is_accepted(self):
        chain = _margin()
        _validate(chain)
        self.assertGreaterEqual(chain["bound"], 1)

    def test_m1_skipped_layer_is_rejected(self):
        # affine(1) then concretize: the hidden coefficients are read as input
        # bytes. Every local check passes; only the traversal check stops it.
        traversal = [("affine", 1), ("concretize", None)]
        with self.assertRaisesRegex(ValueError, "does not traverse"):
            _validate(_margin(traversal))
        # Without the traversal check the malformed chain is accepted, so this
        # was a real false-acceptance path, not a redundant guard.
        with mock.patch("verification.crown_chain_v2.expected_traversal", return_value=traversal):
            _validate(_margin(traversal))

    def test_m2_duplicated_layer_is_rejected(self):
        chain = _margin([("affine", 1), ("relu", 0), ("relu", 0), ("affine", 0), ("concretize", None)])
        with self.assertRaisesRegex(ValueError, "does not traverse"):
            _validate(chain)

    def test_step_identity_must_follow_position(self):
        chain = _margin()
        chain["steps"][1]["step_id"] = "chain/margin/0-1/s9"
        with self.assertRaisesRegex(ValueError, "has identity"):
            _validate(chain)


class FactIdentityTest(unittest.TestCase):
    def _check(self, fact_id, chain_id, chains):
        layer, neuron, side = fact_id.split("/")[1:]
        facts = {fact_id: {
            "fact_id": fact_id, "layer_index": int(layer[1:]), "neuron_index": int(neuron[1:]),
            "side": side, "value": 7, "source": {"kind": "chain", "chain_id": chain_id},
        }}
        _validate_facts({}, facts, chains, {}, 2)

    def test_positive_control_own_chain_is_accepted(self):
        self._check("fact/L0/n1/lower", "chain/L0/n1/lower", {"chain/L0/n1/lower": 7})

    def test_m3_other_neurons_chain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cites chain/L0/n2/lower"):
            self._check("fact/L0/n1/lower", "chain/L0/n2/lower", {"chain/L0/n2/lower": 7})

    def test_m4_other_sides_chain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cites chain/L0/n1/upper"):
            self._check("fact/L0/n1/lower", "chain/L0/n1/upper", {"chain/L0/n1/upper": 7})


def _closure_case():
    layers = {0: {"rows": 3}, 1: {"rows": 3}}
    facts = {"fact/L1/n1/upper": {"value": 30000}, "fact/L1/n2/upper": {"value": 30000}}
    bounds = {"chain/margin/0-1": 5, "chain/margin/0-2": 9, "chain/logit/0/lower": 200}

    def record(competitor, margin):
        return {
            "raw_margin_lower_bound": margin, "closed": True,
            "top": {"fact": f"fact/L1/n{competitor}/upper", "value": 30000, "rule": "raw_c <= 32766"},
            "bottom": {"fact": "chain/logit/0/lower", "value": 200, "rule": "raw_t >= -32767"},
        }

    root = {
        "provenance": {"target": 0},
        "manifest": {"required_competitors": [1, 2], "root_chains": ["chain/margin/0-1", "chain/margin/0-2"]},
        "margin_closure": {"target": 0, "competitors": {"1": record(1, 5), "2": record(2, 9)}},
    }
    return root, layers, facts, bounds


class ClampClosureTest(unittest.TestCase):
    def test_positive_control_is_closed(self):
        result = validate_margin_closure(*_closure_case())
        self.assertEqual(result["minimum_raw_margin_lower_bound"], 5)

    def _rejects(self, mutate, pattern):
        root, layers, facts, bounds = _closure_case()
        mutate(root, facts, bounds)
        with self.assertRaisesRegex(ValueError, pattern):
            validate_margin_closure(root, layers, facts, bounds)

    def test_clamp_edge_citing_another_logit_is_rejected(self):
        def mutate(root, facts, bounds):
            root["margin_closure"]["competitors"]["1"]["top"]["fact"] = "fact/L1/n2/upper"
        self._rejects(mutate, "cites fact/L1/n2/upper")

    def test_clamp_edge_citing_wrong_side_is_rejected(self):
        def mutate(root, facts, bounds):
            facts["fact/L1/n1/lower"] = {"value": 30000}
            root["margin_closure"]["competitors"]["1"]["top"]["fact"] = "fact/L1/n1/lower"
        self._rejects(mutate, "cites fact/L1/n1/lower")

    def test_bottom_rule_in_top_slot_is_rejected(self):
        def mutate(root, facts, bounds):
            root["margin_closure"]["competitors"]["1"]["top"]["rule"] = "raw_t >= -32767"
        self._rejects(mutate, "top edge has rule")

    def test_m7_missing_competitor_is_rejected(self):
        def mutate(root, facts, bounds):
            del root["margin_closure"]["competitors"]["2"]
        self._rejects(mutate, "exactly the competitors")

    def test_m8_target_differs_from_nominal_class_is_rejected(self):
        def mutate(root, facts, bounds):
            root["provenance"]["target"] = 1
        self._rejects(mutate, "nominal class")

    def test_nonpositive_margin_is_rejected(self):
        def mutate(root, facts, bounds):
            bounds["chain/margin/0-2"] = 0
            root["margin_closure"]["competitors"]["2"]["raw_margin_lower_bound"] = 0
        self._rejects(mutate, "positive closed margin")


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        chain = _margin()
        _validate(chain)
        self.certificate = CertificateV2(
            root_dir=Path(self.directory.name), root={"input_box": {"low": LOW, "high": HIGH}, "scale_bits": SCALE_BITS},
            layers=_layers(), facts=_facts(), chain_bounds={}, chains=[chain], selected_layers=(1,),
            preflight={"complete_certificate_scope": False},
        )
        manifest = render_v2_harnesses(self.certificate, Path(self.directory.name) / "harnesses")
        self.records = [{**record, "status": "VERIFIED"} for record in manifest["records"]]

    def tearDown(self):
        self.directory.cleanup()

    def _rewrite(self, record, transform):
        path = Path(record["harness"])
        path.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")
        record["harness_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_positive_control_complete_check_is_accepted(self):
        self.assertEqual(verify_shard_coverage(self.certificate, self.records)["status"], "COMPLETE")

    def test_m10_reordered_records_are_accepted(self):
        self.assertEqual(verify_shard_coverage(self.certificate, self.records[::-1])["status"], "COMPLETE")

    def test_zero_product_column_is_checked_by_esbmc(self):
        # The renderer used to skip columns whose g cancels to 0; that claim
        # then rested on Python alone. Simulate the old renderer.
        record = next(r for r in self.records if r["kind"] == "affine_chunk" and r["step_id"].endswith("/s0"))
        self.assertIn('"chain/margin/0-1/s0 g 1"', Path(record["harness"]).read_text(encoding="utf-8"))
        self._rewrite(record, lambda text: "\n".join(
            line for line in text.splitlines()
            if not any(f'"chain/margin/0-1/s0 {check} 1"' in line for check in ("g", "floor", "residual"))
            and "g_1 " not in line and "q_1 " not in line
        ))
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_shard_coverage(self.certificate, self.records)

    def test_m5_missing_affine_chunk_is_rejected(self):
        records = [r for r in self.records if r["kind"] != "affine_chunk" or not r["step_id"].endswith("/s2")]
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_shard_coverage(self.certificate, records)

    def test_m6_missing_relu_chunk_is_rejected(self):
        records = [r for r in self.records if r["kind"] != "relu_chunk"]
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_shard_coverage(self.certificate, records)

    def test_missing_shared_lemma_is_rejected(self):
        records = [r for r in self.records if r["kind"] != "relu_lemma_lower_upper"]
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_shard_coverage(self.certificate, records)

    def test_missing_concretize_and_close_are_rejected(self):
        for kind in ("concretize", "affine_close", "relu_close", "chain_close"):
            with self.subTest(kind=kind):
                records = [r for r in self.records if r["kind"] != kind]
                with self.assertRaisesRegex(ValueError, "missing"):
                    verify_shard_coverage(self.certificate, records)

    def test_m11_duplicated_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicated"):
            verify_shard_coverage(self.certificate, self.records + [self.records[-1]])

    def test_m12_harness_edited_after_verification_is_rejected(self):
        path = Path(self.records[-1]["harness"])
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after"):
            verify_shard_coverage(self.certificate, self.records)

    def test_unverified_record_is_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]["status"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "not VERIFIED"):
            verify_shard_coverage(self.certificate, records)

    def test_instance_outside_lemma_envelope_is_rejected(self):
        record = next(r for r in self.records if r["kind"] == "relu_lemma_positive_product")
        self._rewrite(record, lambda text: re.sub(r"coefficient <= \d+", "coefficient <= 1", text))
        with self.assertRaisesRegex(ValueError, "outside the shared lemma envelope"):
            verify_shard_coverage(self.certificate, self.records)


if __name__ == "__main__":
    unittest.main()
