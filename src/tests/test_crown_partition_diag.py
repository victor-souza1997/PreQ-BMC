"""Tests for the I-P partition diagnostic support (frozen-choice replay, split, accounting, CSR check).

  PYTHONPATH=/home/joao/code/preqbmc-qformat/src python -m unittest tests.test_crown_partition_diag
"""
from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from tests.test_crown_diagnostics import cce, cd, lowered, sound_bounds, tiny_network, v2
from verification import crown_partition_diag as pd


def pinned(xu):
    from scripts.diagnose_crown_finder import Run
    return Run.XUState(cce, xu)


class SplitAndAccounting(unittest.TestCase):
    def test_split_covers_parent_exactly(self):
        blo, bhi = np.array([4, 0, 254]), np.array([6, 1, 255])
        for i in range(3):
            kids = pd.split_bytes(blo, bhi, i)
            check = pd.check_partition((blo, bhi), kids, i)
            self.assertTrue(check["passed"], check)
        kids = pd.split_bytes(blo, bhi, 0)
        self.assertEqual(pd.check_partition((blo, bhi), kids, 0)["child_values"], [[4, 5], [6]])
        with self.assertRaises(ValueError):
            pd.split_bytes(np.array([3]), np.array([3]), 0)
        bad = [(blo.copy(), bhi.copy()), (blo.copy(), bhi.copy())]
        bad[0][1][0], bad[1][0][0] = 5, 5          # value 5 duplicated
        self.assertFalse(pd.check_partition((blo, bhi), bad, 0)["passed"])
        gap = [(blo.copy(), bhi.copy()), (blo.copy(), bhi.copy())]
        gap[0][1][0], gap[1][0][0] = 4, 6          # value 5 dropped
        self.assertFalse(pd.check_partition((blo, bhi), gap, 0)["passed"])

    def test_select_split_tie_and_zero(self):
        a = np.array([0, 3, -3, 1])
        lo, hi = np.array([0, 0, 0, 0]), np.array([2, 2, 2, 0])
        self.assertEqual(pd.select_split(a, lo, hi)["index"], 1)       # tie 1 vs 2 -> lowest index
        self.assertIsNone(pd.select_split(np.array([0, 0]), lo[:2], hi[:2]))

    def test_worst_child_reproduces_parent_form(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            a = rng.integers(-500, 500, size=6)
            blo = rng.integers(0, 250, size=6)
            bhi = blo + rng.integers(1, 3, size=6)
            i = int(rng.integers(6))
            kids = pd.split_bytes(blo, bhi, i)
            parent = pd.worst_child_of_form(17, a, [(blo, bhi)])[0]
            self.assertEqual(min(pd.worst_child_of_form(17, a, kids)), parent)

    def test_csr_overlap(self):
        w = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]]))
        rep = pd.overlap_report(pd.csr_predecessors(w, [0, 1]))
        self.assertTrue(rep["all_disjoint"])
        rep = pd.overlap_report(pd.csr_predecessors(w, [0, 1, 2]))
        self.assertFalse(rep["all_disjoint"])
        self.assertEqual(len(rep["overlapping_pairs"]), 2)


class FrozenReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v2.configure_format(16, 8)

    def unit_chains(self, mats, bounds, xl, xu, layer):
        n = bounds[layer][0].size
        E = np.zeros((2 * n, n), dtype=np.int64)
        E[np.arange(n), np.arange(n)] = 1 << 16
        E[n + np.arange(n), np.arange(n)] = -(1 << 16)
        with pinned(xu):
            sc, _, steps = v2.run_chain(E, layer, mats, bounds, xl, record=True)
        return sc, steps

    def test_identity_replay_reproduces_and_children_are_sound(self):
        checked = 0
        for seed in range(10):
            net = tiny_network(seed, big_bias=seed % 3 == 0)
            mats = lowered(net)
            rng = np.random.default_rng(seed)
            nominal = rng.integers(2, 253, size=9)
            xl, xu = np.maximum(nominal - 2, 0), np.minimum(nominal + 2, 255)
            parent = sound_bounds(mats, xl, xu)
            lam = np.zeros((1, 3), dtype=np.int64)
            lam[0, 0], lam[0, 1] = 1 << 16, -(1 << 16)
            with pinned(xu):
                p_scaled, _, p_steps = v2.run_chain(lam, 2, mats, parent, xl, record=True)
                p_unit, _ = self.unit_chains(mats, parent, xl, xu, 1)
            # identity replay: identical constants and bounds
            rep = pd.Replay(v2, cd.Q_HIGH)
            rep.map(parent, parent)
            with rep, pinned(xu):
                i_scaled, _, _ = v2.run_chain(lam, 2, mats, parent, xl)
                i_unit, _ = self.unit_chains(mats, parent, xl, xu, 1)
            self.assertEqual(int(i_scaled[0]), int(p_scaled[0]))
            np.testing.assert_array_equal(i_unit, p_unit)
            # children: split the coordinate with the largest |A| width
            final = p_steps[-1]["a"][0]
            sel = pd.select_split(final, xl, xu)
            if sel is None:
                continue
            kids = pd.split_bytes(xl, xu, sel["index"])
            self.assertTrue(pd.check_partition((xl, xu), kids, sel["index"])["passed"])
            const = int(p_steps[-1]["c_in"][0])
            self.assertEqual(min(pd.worst_child_of_form(const, final, kids)), int(p_scaled[0]))
            for cxl, cxu in kids:
                child = [(lo.copy(), hi.copy(), s, c) for lo, hi, s, c in parent]
                # tighten layer 0 from exact child chains, layer 1 from replayed chains
                for k in range(2):
                    with pinned(xu):
                        p_sc, _ = self.unit_chains(mats, parent, xl, xu, k)
                    rep = pd.Replay(v2, cd.Q_HIGH)
                    rep.map(child, parent)
                    with rep:     # unit_chains pins XU to the child box
                        sc, _ = self.unit_chains(mats, child, cxl, cxu, k)
                    self.assertTrue((sc >= p_sc).all(), "child chain looser than the same parent chain")
                    n = child[k][0].size
                    lo_c = np.array([cd.ceil_div(int(v), 1 << 16) for v in sc[:n]])
                    hi_c = np.array([(-int(v)) // (1 << 16) for v in sc[n:]])
                    new = (np.maximum(child[k][0], lo_c), np.minimum(child[k][1], hi_c), parent[k][2], parent[k][3])
                    child[k] = new
                rep = pd.Replay(v2, cd.Q_HIGH)
                rep.map(child, parent)
                with rep, pinned(cxu):
                    c_scaled, _, c_steps = v2.run_chain(lam, 2, mats, child, cxl, record=True)
                self.assertGreaterEqual(int(c_scaled[0]), int(p_scaled[0]))
                for s_p, s_c in zip(p_steps, c_steps):     # same coefficient path
                    np.testing.assert_array_equal(s_p["a"], s_c["a"])
                for _ in range(25):
                    x = rng.integers(cxl, cxu + 1)
                    trace = cd.exact_forward(mats, x, [True, True, False])
                    acc = cd.account_chain(c_steps, 0, lam[0], trace, x, cxl, cxu, [m[1] for m in mats],
                                           int(c_scaled[0]))
                    self.assertGreaterEqual(int(acc["total_slack"]), 0)
                    checked += 1
        self.assertGreater(checked, 100)

    def test_unmapped_bounds_fail_closed(self):
        net = tiny_network(1)
        mats = lowered(net)
        xl, xu = np.full(9, 10), np.full(9, 12)
        parent = sound_bounds(mats, xl, xu)
        rep = pd.Replay(v2, cd.Q_HIGH)
        with self.assertRaises(pd.ReplayError):
            with rep, pinned(xu):
                v2.run_chain(np.eye(3, dtype=np.int64)[:1] << 16, 2, mats, parent, xl)
        self.assertIsNot(v2.relu_choice, None)

    def test_keyed_override_applies_once_per_call(self):
        found = None
        for seed in range(40):
            net = tiny_network(seed)
            mats = lowered(net)
            xl, xu = np.full(9, 20), np.full(9, 120)
            parent = sound_bounds(mats, xl, xu)
            lo, hi, sat, clip = parent[0]
            for row in range(5):
                A = np.zeros((1, 5), dtype=np.int64)
                A[0, row] = 1 << 16
                with pinned(xu):
                    _, _, steps = v2.run_chain(A, 1, mats, parent, xl, record=True)
                relu = next(s for s in steps if s["kind"] == "relu")
                idx = [i for i in range(6) if lo[i] < 0 < hi[i] and not sat[i] and not clip[i]
                       and relu["a"][0, i] != 0]
                if idx:
                    found = (mats, xl, xu, parent, A, relu, idx[0])
                    break
            if found:
                break
        self.assertIsNotNone(found, "no eligible coordinate in 40 synthetic networks")
        mats, xl, xu, parent, A, relu, i = found
        rep = pd.Replay(v2, cd.Q_HIGH)
        rep.map(parent, parent)
        with rep, rep.keyed_override(parent[0], 0, i, int(relu["a"][0, i]), 0), pinned(xu):
            _, _, st = v2.run_chain(A, 1, mats, parent, xl, record=True)
            v2.needs(A, 1, mats, parent)
            applied = rep.override_applied
        self.assertEqual(applied, 2)
        self.assertEqual(int(next(s for s in st if s["kind"] == "relu")["a_out"][0, i]), 0)


if __name__ == "__main__":
    unittest.main()
