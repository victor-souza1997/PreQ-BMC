"""Focused tests for the phase-1 CROWN diagnostics (spec section 13, items 1, 3, 4, 5, 10).

Run with the worktree source first:
  PYTHONPATH=/home/joao/code/preqbmc-qformat/src python -m unittest tests.test_crown_diagnostics
The import-isolation tests start subprocesses from the main checkout on purpose.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

SRC = Path(__file__).resolve().parents[1]
WORKTREE = SRC.parent
MAIN = Path("/home/joao/code/preqbmc")
sys.path.insert(0, str(SRC / "scripts" / "crown_finder"))

from backends.conv_fixed_point import ConvFixedPointNetwork, QuantizedDense, generate_conv_qnn_source  # noqa: E402
from backends.c_qnn_generator import compile_c_qnn_shared_library  # noqa: E402
from backends.fixed_point import LayerQuantizationSpec  # noqa: E402
from backends.image_encoder import ByteImageEncoder  # noqa: E402
import crown_chain_export as cce  # noqa: E402
import crown_chain_export_v2 as v2  # noqa: E402
from utils.fixed_point import round_divide_half_away_from_zero  # noqa: E402
from verification import crown_diagnostics as cd  # noqa: E402

SPEC = LayerQuantizationSpec(16, 7, 8)


def tiny_network(seed=0, big_bias=False):
    """Two hidden dense layers and a 3-class output over a 3x3x1 byte image."""

    rng = np.random.default_rng(seed)
    shapes = [(6, 9), (5, 6), (3, 5)]
    layers = []
    for i, (o, n) in enumerate(shapes):
        w = rng.integers(-300, 301, size=(o, n))
        w[rng.random((o, n)) < 0.3] = 0
        b = rng.integers(-2000, 2001, size=o)
        if big_bias and i == 0:
            b[0], b[1] = 40000, -40000          # raw values outside Q16: saturation both ways
        layers.append(QuantizedDense(w, b, SPEC, 8, apply_relu=i < len(shapes) - 1))
    return ConvFixedPointNetwork((3, 3, 1), 8, 16, tuple(layers))


def lowered(net):
    return [(sp.csr_matrix(np.asarray(l.weights_int, dtype=np.int64)), np.asarray(l.bias_int, dtype=np.int64))
            for l in net.layers]


def sound_bounds(mats, xl, xu):
    """Exact interval bounds on raw r for the hidden layers, in the v2 bounds tuple form."""

    lo_h, hi_h = xl.astype(np.int64), xu.astype(np.int64)
    bounds = []
    for w, b in mats[:-1]:
        wd = w.toarray()
        pos, neg = np.maximum(wd, 0), np.minimum(wd, 0)
        s_lo, s_hi = pos @ lo_h + neg @ hi_h, pos @ hi_h + neg @ lo_h
        r_lo, r_hi = cd.rha_vec(s_lo) + b, cd.rha_vec(s_hi) + b
        saturated = r_lo >= cd.Q_HIGH
        clipped = ~saturated & (r_hi > cd.Q_HIGH)
        bounds.append((r_lo, r_hi, saturated, clipped))
        lo_h = np.clip(np.maximum(np.clip(r_lo, cd.Q_LOW, cd.Q_HIGH), 0), 0, cd.Q_HIGH)
        hi_h = np.clip(np.maximum(np.clip(r_hi, cd.Q_LOW, cd.Q_HIGH), 0), 0, cd.Q_HIGH)
    return bounds


class ExactArithmetic(unittest.TestCase):
    def test_half_ties_match_deployed_rounding(self):
        for s in range(-1024, 1025):
            self.assertEqual(cd.rha(s, 256), round_divide_half_away_from_zero(s, 256))
        self.assertEqual((cd.rha(128, 256), cd.rha(-128, 256)), (1, -1))
        self.assertEqual((cd.rha(384, 256), cd.rha(-384, 256)), (2, -2))
        s = np.array([128, -128, 384, -384, 127, -127, 0])
        self.assertEqual(cd.rha_vec(s).tolist(), [1, -1, 2, -2, 0, 0, 0])
        e = 256 * cd.rha_vec(np.arange(-1024, 1025)) - np.arange(-1024, 1025)
        self.assertEqual((int(e.min()), int(e.max())), (-128, 128))

    def test_saturation_raw_versus_clamped(self):
        net = tiny_network(big_bias=True)
        mats = lowered(net)
        x = np.full(9, 100, dtype=np.int64)
        trace = cd.exact_forward(mats, x, [True, True, False])
        self.assertGreater(int(trace[0]["r"][0]), cd.Q_HIGH)
        self.assertEqual(int(trace[0]["z"][0]), cd.Q_HIGH)
        self.assertEqual(int(trace[0]["h"][0]), cd.Q_HIGH)
        self.assertLess(int(trace[0]["r"][1]), cd.Q_LOW)
        self.assertEqual((int(trace[0]["z"][1]), int(trace[0]["h"][1])), (cd.Q_LOW, 0))

    def test_tie_is_a_violation_and_raw_margin_does_not_decide(self):
        self.assertTrue(cd.deployed_violation(0))
        self.assertTrue(cd.deployed_violation(-3))
        self.assertFalse(cd.deployed_violation(1))
        # both raw logits saturate: raw margin 5000, deployed margin 0 -> violation
        margins = cd.margins_from_logits(np.array([32767, 32767]), np.array([40000, 35000]), 0)
        self.assertEqual(margins[1], {"deployed": 0, "raw": 5000})
        self.assertTrue(cd.deployed_violation(margins[1]["deployed"]))
        # the classification reads only the deployed margin
        self.assertFalse(cd.deployed_violation(1))

    def test_material_gain_and_singleton_thresholds(self):
        self.assertEqual(cd.material_gain_threshold(-997, False), 50)      # ceil(0.05 * 998)
        self.assertEqual(cd.material_gain_threshold(5, False), 1)
        self.assertEqual(cd.material_gain_threshold(-997, True), 1)
        self.assertTrue(cd.significant_singleton_gap(50, 1000))
        self.assertFalse(cd.significant_singleton_gap(49, 1000))
        self.assertTrue(cd.significant_singleton_gap(1, 0))
        self.assertEqual(cd.gap_fraction(3, -6), {"num": "1", "den": "2"})


class InputDomain(unittest.TestCase):
    def setUp(self):
        self.enc = ByteImageEncoder((4, 4, 1), resize_mode="nearest_center")

    def test_alias_map_matches_encoder_resize_and_byte_clamping(self):
        crop = np.array([[0, 255], [17, 200]], dtype=np.uint8).reshape(2, 2, 1)    # upsampled: aliases
        idx = cd.resize_source_index(crop.shape, (4, 4, 1), "nearest_center")
        self.assertTrue(np.array_equal(crop.reshape(-1)[idx], self.enc.resize(crop).reshape(-1)))
        groups = cd.alias_groups(idx)
        self.assertEqual(groups.sizes, {"4": 4})
        allowed = cd.group_allowed_values(crop, groups, 1)
        self.assertEqual(allowed[0], [0, 1])            # byte 0 clamps at 0
        self.assertEqual(allowed[1], [254, 255])        # byte 255 clamps at 255

    def test_alias_inconsistent_box_point_has_no_crop_lift(self):
        crop = np.array([[10, 20], [30, 40]], dtype=np.uint8).reshape(2, 2, 1)
        groups = cd.alias_groups(cd.resize_source_index(crop.shape, (4, 4, 1), "nearest_center"))
        resized = self.enc.resize(crop).copy()
        consistent = cd.lift_resized(resized, groups, crop)
        self.assertTrue(np.array_equal(consistent, crop))
        resized[0, 0, 0] = 11                       # only one copy of source pixel (0, 0) moves
        self.assertIsNone(cd.lift_resized(resized, groups, crop))
        resized[0, 1, 0] = resized[1, 0, 0] = resized[1, 1, 0] = 11
        lifted = cd.lift_resized(resized, groups, crop)
        self.assertEqual(int(lifted[0, 0, 0]), 11)
        self.assertTrue(all(cd.in_crop_domain(lifted, crop, groups, 1).values()))
        self.assertFalse(cd.in_crop_domain(lifted, crop, groups, 0)["source_bounds_satisfied"])


class Accounting(unittest.TestCase):
    """Exact fixed-chain, fixed-witness identities on a synthetic network (spec section 7)."""

    @classmethod
    def setUpClass(cls):
        v2.configure_format(16, 8)

    def run_case(self, seed, big_bias=False):
        net = tiny_network(seed, big_bias)
        mats = lowered(net)
        rng = np.random.default_rng(seed)
        nominal = rng.integers(1, 255, size=9)
        xl, xu = np.maximum(nominal - 3, 0), np.minimum(nominal + 3, 255)
        bounds = sound_bounds(mats, xl, xu)
        lam = np.zeros((2, 3), dtype=np.int64)
        lam[0, 0], lam[0, 1] = 1 << 16, -(1 << 16)
        lam[1, 2], lam[1, 0] = 1 << 16, -(1 << 16)
        with self.pin(xu):
            scaled, _, steps = v2.run_chain(lam, 2, mats, bounds, xl, record=True)
        checked = 0
        for _ in range(40):
            x = rng.integers(xl, xu + 1)
            trace = cd.exact_forward(mats, x, [True, True, False])
            for row in range(2):
                acc = cd.account_chain(steps, row, lam[row], trace, x, xl, xu, [m[1] for m in mats],
                                       int(scaled[row]))
                self.assertTrue(acc["accounting_identity_passed"])
                self.assertGreaterEqual(int(acc["total_slack"]), 0)
                credit = acc["final_rounding_credit"]
                self.assertTrue(0 <= int(credit["num"]) < int(credit["den"]))
                checked += 1
        return net, mats, steps, scaled, xl, xu, bounds, lam, checked

    def pin(self, xu):
        from scripts.diagnose_crown_finder import Run
        return Run.XUState(cce, xu)

    def test_identities_hold_with_negative_coefficients(self):
        for seed in range(5):
            *_, checked = self.run_case(seed)
            self.assertEqual(checked, 80)

    def test_identities_hold_with_saturation(self):
        net, mats, steps, *_ = self.run_case(7, big_bias=True)
        classes = {int(c) for s in steps if s["kind"] == "relu" for c in
                   np.unique(v2.classify(s["a"], s["a_out"])).tolist()}
        self.assertTrue(classes)

    def test_negative_half_tie_witness(self):
        w = np.array([[128, 0], [-128, 0]], dtype=np.int64)        # s = +-128 at x = (1, 0)
        mats = [(sp.csr_matrix(w), np.array([0, 0])), (sp.csr_matrix(np.array([[1, -1]])), np.array([0]))]
        x, xl, xu = np.array([1, 0]), np.array([1, 0]), np.array([1, 0])
        trace = cd.exact_forward(mats, x, [True, False])
        self.assertEqual(trace[0]["e"].tolist(), [128, -128])
        bounds = sound_bounds(mats, xl, xu)
        lam = np.array([[1 << 16]], dtype=np.int64)
        with self.pin(xu):
            scaled, _, steps = v2.run_chain(lam, 1, mats, bounds, xl, record=True)
        acc = cd.account_chain(steps, 0, lam[0], trace, x, xl, xu, [m[1] for m in mats], int(scaled[0]))
        self.assertTrue(acc["accounting_identity_passed"])

    def test_zero_cancelled_column(self):
        w = np.array([[2, 3], [2, -3]], dtype=np.int64)
        lam = np.array([[1 << 16, 1 << 16]], dtype=np.int64)        # g column 1 = 3 - 3 = 0
        g = np.asarray(sp.csr_matrix(w).T @ lam.T).T
        self.assertEqual(int(g[0, 1]), 0)
        mats = [(sp.csr_matrix(w), np.array([5, -7]))]
        x, xl, xu = np.array([9, 4]), np.array([8, 3]), np.array([10, 5])
        with self.pin(xu):
            scaled, _, steps = v2.run_chain(lam, 0, mats, [], xl, record=True)
        trace = cd.exact_forward(mats, x, [False])
        acc = cd.account_chain(steps, 0, lam[0], trace, x, xl, xu, [mats[0][1]], int(scaled[0]))
        self.assertTrue(acc["accounting_identity_passed"])

    def test_negative_slack_is_an_implementation_error(self):
        net, mats, steps, scaled, xl, xu, bounds, lam, _ = self.run_case(3)
        x = np.minimum(xu + 5, 255)                                 # outside the box
        trace = cd.exact_forward(mats, x, [True, True, False])
        with self.assertRaises(cd.ImplementationError):
            cd.account_chain(steps, 0, lam[0], trace, x, xl, xu, [m[1] for m in mats], int(scaled[0]))

    def test_recorded_constant_mismatch_is_an_implementation_error(self):
        net, mats, steps, scaled, xl, xu, bounds, lam, _ = self.run_case(4)
        steps[0]["c_out"] = steps[0]["c_out"] + 1
        x = xl.copy()
        trace = cd.exact_forward(mats, x, [True, True, False])
        with self.assertRaises(cd.ImplementationError):
            cd.account_chain(steps, 0, lam[0], trace, x, xl, xu, [m[1] for m in mats], int(scaled[0]))


class BoundIntervention(unittest.TestCase):
    """I-B / I-S support: endpoint arithmetic, candidate family, one-coordinate override, cone size."""

    @classmethod
    def setUpClass(cls):
        v2.configure_format(16, 8)

    def pin(self, xu):
        from scripts.diagnose_crown_finder import Run
        return Run.XUState(cce, xu)

    def test_midpoint_rounds_toward_the_nominal_value(self):
        self.assertEqual(cd.midpoint_endpoint("lower", -423, 10), -206)
        self.assertEqual(cd.midpoint_endpoint("lower", -5, 0), -2)
        self.assertEqual(cd.midpoint_endpoint("upper", 5, 0), 2)
        self.assertEqual(cd.midpoint_endpoint("upper", -1, -4), -3)
        self.assertEqual(cd.midpoint_endpoint("lower", 7, 7), 7)          # unchanged: caller skips
        self.assertEqual(cd.midpoint_endpoint("lower", 6, 7), 7)
        with self.assertRaises(cd.ImplementationError):
            cd.midpoint_endpoint("lower", 3, 2)
        with self.assertRaises(cd.ImplementationError):
            cd.midpoint_endpoint("upper", 3, 4)
        self.assertTrue(cd.endpoint_refuted("lower", 5, 4))
        self.assertFalse(cd.endpoint_refuted("lower", 5, 5))
        self.assertTrue(cd.endpoint_refuted("upper", 5, 6))
        self.assertTrue(cd.tighter("lower", 3, 2) and cd.tighter("upper", 2, 3))
        self.assertFalse(cd.tighter("lower", 2, 2) or cd.tighter("upper", 2, 2))

    def test_candidate_family_and_eligibility(self):
        self.assertEqual(cd.slope_candidates(5, 5), [0, 2])
        self.assertEqual(cd.slope_candidates(-3, 0), [-3, -2])
        self.assertEqual(cd.slope_candidates(4, 2), [0, 4])
        self.assertEqual(cd.slope_candidates(1, 1), [0])
        self.assertTrue(cd.slope_eligible(-1, 1, False, False, -2))
        self.assertFalse(cd.slope_eligible(0, 1, False, False, 2))
        self.assertFalse(cd.slope_eligible(-1, 1, False, True, 2))
        self.assertFalse(cd.slope_eligible(-1, 1, False, False, 0))
        bounds = [(np.array([-5, -5, -5]), np.array([5, 5, 5]), np.zeros(3, bool), np.array([False, False, True]))]
        ranked = cd.rank_coordinates({(0, 0): 7, (0, 1): 7, (0, 2): 99}, {0: np.array([1, 3, 1])}, bounds)
        self.assertEqual([(e[2], e[3]) for e in ranked], [(0, 1), (0, 0)])     # tie broken by |a| (u - l)

    def test_override_is_sound_shared_and_restored(self):
        original = v2.relu_choice
        tested = 0
        for seed in range(12):
            net = tiny_network(seed)
            mats = lowered(net)
            rng = np.random.default_rng(seed)
            nominal = rng.integers(1, 255, size=9)
            xl, xu = np.maximum(nominal - 20, 0), np.minimum(nominal + 20, 255)
            bounds = sound_bounds(mats, xl, xu)
            A = np.zeros((1, 5), dtype=np.int64)
            A[0, int(rng.integers(5))] = 1 << 16
            with self.pin(xu):
                base, _, steps = v2.run_chain(A, 1, mats, bounds, xl, record=True)
            relu = next(s for s in steps if s["kind"] == "relu")
            lo, hi, sat, clip = bounds[0]
            for i in range(6):
                a, a_base = int(relu["a"][0, i]), int(relu["a_out"][0, i])
                if not cd.slope_eligible(int(lo[i]), int(hi[i]), bool(sat[i]), bool(clip[i]), a):
                    continue
                for value in cd.slope_candidates(a, a_base):
                    with self.pin(xu), cd.SlopeOverride(v2, bounds[0], 0, i, a, value) as ov:
                        scaled, _, st = v2.run_chain(A, 1, mats, bounds, xl, record=True)
                        v2.needs(A, 1, mats, bounds)
                    self.assertEqual(ov.applied, 2)
                    r = next(s for s in st if s["kind"] == "relu")
                    self.assertEqual(int(r["a_out"][0, i]), value)
                    self.assertEqual(int(r["d"][0, i]), int(v2.relu_const(np.array([[a]]), np.array([[value]]),
                                                                          lo[i:i + 1], hi[i:i + 1])[0, 0]))
                    others = np.arange(6) != i
                    np.testing.assert_array_equal(r["a_out"][0, others], relu["a_out"][0, others])
                    for _ in range(30):     # the overridden chain stays a sound lower bound
                        x = rng.integers(xl, xu + 1)
                        trace = cd.exact_forward(mats, x, [True, True, False])
                        acc = cd.account_chain(st, 0, A[0], trace[:2], x, xl, xu, [m[1] for m in mats], int(scaled[0]))
                        self.assertGreaterEqual(int(acc["total_slack"]), 0)
                    tested += 1
                with self.assertRaises(cd.ImplementationError):
                    with self.pin(xu), cd.SlopeOverride(v2, bounds[0], 0, i, a + 1, 0):
                        v2.run_chain(A, 1, mats, bounds, xl)
            self.assertIs(v2.relu_choice, original)
        self.assertGreater(tested, 5)

    def test_cone_size_counts_ancestor_terms(self):
        w0 = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 0]]))
        w1 = sp.csr_matrix(np.array([[5, 0, 6], [0, 7, 0]]))
        cone = cd.cone_size([(w0, None), (w1, None)], 1, 0)
        self.assertEqual(cone["multiply_terms"], 2 + 2 + 1)       # row 0 of w1; rows 0 and 2 of w0
        self.assertEqual(cone["neurons_by_layer"], {"1": 1, "0": 2})
        self.assertEqual(cone["source_inputs"], 2)


class StateIsolation(unittest.TestCase):
    def test_xu_is_pinned_and_restored(self):
        from scripts.diagnose_crown_finder import Run
        v2.configure_format(16, 8)
        w = np.array([[1, -1]], dtype=np.int64)
        mats = [(sp.csr_matrix(w), np.array([0]))]
        lam = np.array([[-(1 << 16)]], dtype=np.int64)              # depends on the upper corner of x_0
        xl = np.array([0, 0])
        before = cce.XU[0]
        with Run.XUState(cce, np.array([10, 0])):
            eps1, _, _ = v2.run_chain(lam, 0, mats, [], xl)
            with Run.XUState(cce, np.array([0, 0])):
                eps0, _, _ = v2.run_chain(lam, 0, mats, [], xl)
            again, _, _ = v2.run_chain(lam, 0, mats, [], xl)
        self.assertIs(cce.XU[0], before)
        self.assertNotEqual(int(eps1[0]), int(eps0[0]))
        self.assertEqual(int(eps1[0]), int(again[0]))
        with self.assertRaises(RuntimeError):
            with Run.XUState(cce, np.array([1, 1])):
                cce.XU[0] = np.array([2, 2])


class SearchSemantics(unittest.TestCase):
    def setUp(self):
        crop = np.full((4, 4, 1), 100, dtype=np.uint8)
        self.crop = crop
        self.groups = cd.alias_groups(cd.resize_source_index(crop.shape, (4, 4, 1), "nearest_center"))
        self.allowed = cd.group_allowed_values(crop, self.groups, 1)
        self.nominal = [100] * len(self.allowed)

    def evaluator(self, offset):
        def evaluate(values):
            margin = sum(values) - 1600 + offset
            return cd.Evaluation(deployed=margin, raw=margin, margins={1: margin}, key=str(values), new=True)
        return evaluate

    def test_finds_violation_and_stops(self):
        rec = cd.search_focus(self.groups, self.allowed, self.nominal, [1] * 16, self.evaluator(8),
                              lambda r: 20261031 + r)
        self.assertIsNotNone(rec["violation"])
        self.assertLessEqual(rec["violation"]["deployed"], 0)

    def test_no_witness_is_not_robust(self):
        rec = cd.search_focus(self.groups, self.allowed, self.nominal, [1] * 16, self.evaluator(10_000),
                              lambda r: 20261031 + r)
        self.assertIsNone(rec["violation"])
        self.assertEqual(rec["status"], "COMPLETED")
        self.assertEqual(rec["proposals"], 4 * 256)
        self.assertNotIn("ROBUST", json.dumps(rec))

    def test_deadline_marks_budget_and_unexecuted(self):
        rec = cd.search_focus(self.groups, self.allowed, self.nominal, [1] * 16, self.evaluator(10_000),
                              lambda r: r, deadline=0.0)
        self.assertEqual(rec["status"], "NOT_RUN_BUDGET")
        self.assertEqual(rec["unexecuted_proposals"], 4 * 256)

    def test_search_is_deterministic(self):
        a = cd.search_focus(self.groups, self.allowed, self.nominal, [1] * 16, self.evaluator(10_000), lambda r: r)
        b = cd.search_focus(self.groups, self.allowed, self.nominal, [1] * 16, self.evaluator(10_000), lambda r: r)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


class Envelope(unittest.TestCase):
    def base(self, **over):
        rec = {"schema": cd.SCHEMA, "run_id": "x", "parent_run_id": "p", "diagnostic": "D-P", "ordinal": "013",
               "image_id": "i", "target": 1, "competitor": 2, "epsilon_raw_bytes": 1, "domain": "D_box",
               "region_sha256": "r", "assumption_context_sha256": "a", "deployment_sha256": "d",
               "source_manifest_sha256": "s", "scale_bits": 16, "status": "COMPLETED",
               "evidence_kind": "SCREEN_BOUND_UNCHECKED", "solver_status": "NOT_RUN", "dependency_status": "SCREEN_ONLY",
               "parameters": {}, "metrics": {k: 0 for k in cd.METRIC_FIELDS}, "metric_scope": "this_probe",
               "missing_reason": {}, "artifacts": {},
               "interpretation": {"hypothesis_id": "H-P", "assessment": "INCONCLUSIVE", "reason_code": "x"}}
        rec.update(over)
        return rec

    def test_valid_record(self):
        cd.validate_envelope(self.base())

    def test_evidence_cannot_be_promoted(self):
        with self.assertRaises(cd.ImplementationError):
            cd.validate_envelope(self.base(solver_status="VERIFIED"))          # screen evidence with a solver status
        with self.assertRaises(cd.ImplementationError):
            cd.validate_envelope(self.base(evidence_kind="ESBMC_CHECKED_OBLIGATION"))   # no solver run
        with self.assertRaises(cd.ImplementationError):
            cd.validate_envelope(self.base(evidence_kind="VERIFIED"))
        cd.validate_envelope(self.base(evidence_kind="ESBMC_CHECKED_OBLIGATION", solver_status="TIMEOUT"))

    def test_null_needs_missing_reason(self):
        metrics = {k: 0 for k in cd.METRIC_FIELDS} | {"cpu_seconds": None}
        with self.assertRaises(cd.ImplementationError):
            cd.validate_envelope(self.base(metrics=metrics))
        cd.validate_envelope(self.base(metrics=metrics, missing_reason={"metrics.cpu_seconds": "not measured"}))

    def test_runlog_refuses_rerun_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = cd.RunLog(Path(tmp) / "runs.jsonl", {})
            log.write(**self.base())
            with self.assertRaises(cd.ImplementationError):
                log.write(**self.base())


class SolverParsing(unittest.TestCase):
    def test_violation_kinds(self):
        log = "Violated property:\n  file h.c line 9 column 3 function main\n  diag/x/claim\n  out[0] > out[1]\n\n"
        self.assertEqual(cd.parse_violation(log, "diag/x/claim")["failed_claim_kind"], "TARGET")
        unwind = "Violated property:\n  file h.c line 4 function main\n  unwinding assertion loop 0\n\n"
        self.assertEqual(cd.parse_violation(unwind, "diag/x/claim")["failed_claim_kind"], "UNWINDING")
        safety = "Violated property:\n  file h.c line 4 function main\n  array bounds violated: x upper bound\n\n"
        self.assertEqual(cd.parse_violation(safety, "diag/x/claim")["failed_claim_kind"], "SAFETY")
        self.assertEqual(cd.parse_violation("VERIFICATION FAILED (truncated)", "diag/x/claim")["failed_claim_kind"],
                         "UNPARSED")

    def test_loop_inventory_covers_generated_c(self):
        net = tiny_network()
        enc = ByteImageEncoder((3, 3, 1), resize_mode="nearest_center")
        source = generate_conv_qnn_source(net, encoder_source=enc.render_c())
        inventory = cd.loop_inventory(source)
        self.assertEqual(inventory["unresolved"], [])
        self.assertGreaterEqual(inventory["max_iterations"], 9)

    @unittest.skipUnless(shutil.which("gcc"), "gcc required")
    def test_solver_controls_with_esbmc(self):
        from verification.esbmc import ESBMCConfig, ESBMCRunner
        from verification.esbmc_install import resolve_esbmc_executable
        if not resolve_esbmc_executable():
            self.skipTest("ESBMC not installed")

        class DiagnosticRunner(ESBMCRunner):
            def build_command(self, c_file, unwind, profile):
                return tuple(a for a in super().build_command(c_file, unwind, profile) if a != "--result-only")

        runner = DiagnosticRunner(ESBMCConfig(timeout_seconds=10, memlimit="4g", default_profile="paper-z3"))
        with tempfile.TemporaryDirectory() as tmp:
            for control in cd.solver_control_harnesses():
                path = Path(tmp) / f"{control['name']}.c"
                path.write_text(control["source"])
                result = runner.run_file(path, "paper-z3", extract_counterexample=False)
                self.assertEqual(result.status, control["expected"], control["name"])
                if control["expected"] == "FAILED":
                    log = Path(result.stdout_log_path).read_text() + Path(result.stderr_log_path).read_text()
                    self.assertEqual(cd.parse_violation(log, control["label"])["failed_claim_kind"], "TARGET",
                                     control["name"])


@unittest.skipUnless(shutil.which("gcc"), "gcc required")
class ObserverParity(unittest.TestCase):
    """Observer C equals unmodified C byte-for-byte after stripping, and agrees on every coordinate."""

    def test_observer_and_python_replay(self):
        net = tiny_network(11, big_bias=True)
        enc = ByteImageEncoder((3, 3, 1), resize_mode="nearest_center")
        source = generate_conv_qnn_source(net, encoder_source=enc.render_c())
        sizes = [l.output_size for l in net.layers]
        observer = cd.observer_source(source, sizes)
        self.assertEqual(cd.strip_observer(observer), source)
        mats = lowered(net)
        with tempfile.TemporaryDirectory() as tmp:
            plain, obs = Path(tmp) / "plain.c", Path(tmp) / "obs.c"
            plain.write_text(source)
            obs.write_text(observer)
            compile_c_qnn_shared_library(plain, Path(tmp) / "plain.so", optimization="-O0")
            compile_c_qnn_shared_library(obs, Path(tmp) / "obs.so", optimization="-O2")
            c_plain = cd.COracle(Path(tmp) / "plain.so", 9, 3)
            c_obs = cd.COracle(Path(tmp) / "obs.so", 9, 3, sizes)
            rng = np.random.default_rng(1)
            for _ in range(25):
                image = rng.integers(0, 256, size=(3, 3, 1)).astype(np.uint8)
                x = c_plain.encode(image)
                self.assertTrue(np.array_equal(x, enc.encode(image)))
                y = c_plain.forward(x)
                self.assertTrue(np.array_equal(c_obs.forward(c_obs.encode(image)), y))
                observed = c_obs.observed()
                exact = cd.exact_forward(mats, x, [True, True, False])
                self.assertTrue(np.array_equal(exact[-1]["z"], y))
                for k in range(3):
                    for key in "srzh":
                        self.assertTrue(np.array_equal(observed[k][key], exact[k][key]), (k, key))


class ImportIsolation(unittest.TestCase):
    """The diagnostic bootstrap resolves every project module to the worktree, from the main checkout."""

    config = WORKTREE / "experiments" / "crown_diagnostics_013_016_017.json"

    @unittest.skipUnless(MAIN.joinpath("src").is_dir(), "main checkout not present")
    def test_main_checkout_first_on_path_still_resolves_to_worktree(self):
        env = dict(os.environ, PYTHONPATH=f"{MAIN / 'src'}:{SRC}")
        proc = subprocess.run([sys.executable, str(SRC / "scripts" / "diagnose_crown_finder.py"), "--config",
                               str(self.config), "--import-audit-only"], cwd=MAIN, env=env,
                              capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        audit = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(audit["passed"], audit["violations"])
        files = [m["file"] for m in audit["modules"] if m["file"]]
        self.assertTrue(files)
        self.assertTrue(all(f.startswith(str(SRC)) for f in files))
        names = {m["module"] for m in audit["modules"]}
        for required in ("affine_export", "crown_chain_export_v2", "backends.conv_fixed_point",
                         "verification.crown_diagnostics", "verification.esbmc", "utils.fixed_point", "datasets.gtsrb_study"):
            self.assertIn(required, names)
        # affine_export inserted the relative "src" (= main checkout src from this cwd), yet nothing escaped
        self.assertIn("src", audit["sys_path"])
        self.assertFalse(any(Path(p or ".").resolve() == MAIN / "src" for p in audit["sys_path"] if p != "src"))

    @unittest.skipUnless(MAIN.joinpath("src").is_dir(), "main checkout not present")
    def test_import_before_guard_fails_explicitly(self):
        code = ("import sys, importlib.util\n"
                f"sys.path.insert(0, {str(MAIN / 'src')!r})\n"
                "import backends\n"
                f"spec = importlib.util.spec_from_file_location('g', {str(SRC / 'scripts' / 'diagnostic_import_guard.py')!r})\n"
                "g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)\n"
                "from pathlib import Path\n"
                f"g.install(Path({str(SRC)!r}), Path({str(MAIN / 'src')!r}))\n")
        proc = subprocess.run([sys.executable, "-c", code], cwd=MAIN, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("project modules imported before the guard", proc.stderr)

    @unittest.skipUnless(MAIN.joinpath("src").is_dir(), "main checkout not present")
    def test_audit_flags_a_module_outside_the_worktree(self):
        code = ("import sys, importlib.util, json\n"
                f"spec = importlib.util.spec_from_file_location('g', {str(SRC / 'scripts' / 'diagnostic_import_guard.py')!r})\n"
                "g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)\n"
                "from pathlib import Path\n"
                f"f = g.install(Path({str(SRC)!r}), Path({str(MAIN / 'src')!r}))\n"
                f"spec = importlib.util.spec_from_file_location('verification.leak', {str(MAIN / 'src/verification/properties.py')!r})\n"
                "m = importlib.util.module_from_spec(spec); sys.modules['verification.leak'] = m\n"
                f"print(json.dumps(g.audit(f, Path({str(MAIN / 'src')!r}))['violations']))\n")
        proc = subprocess.run([sys.executable, "-c", code], cwd=MAIN, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        violations = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual([v["module"] for v in violations], ["verification.leak"])


if __name__ == "__main__":
    unittest.main()
