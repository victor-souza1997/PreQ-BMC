"""I-P input-partition diagnostic support (one bounded two-child split; docs/ssv2026/IP_partition_016.md).

NOT A PROOF COMPONENT. Untrusted finder arithmetic for a diagnostic screen. Nothing here labels
anything verified, and nothing here is imported by the production finder, exporter or checker.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class ReplayError(RuntimeError):
    """A frozen parent choice cannot be replayed soundly: the experiment is INCONCLUSIVE."""


class PartitionAccountingError(RuntimeError):
    """The partition does not reproduce the parent: an audit failure, never a result."""


# ---------------------------------------------------------------- architecture check

def csr_predecessors(w, rows) -> dict[int, set[int]]:
    """Immediate predecessor coordinates (nonzero CSR columns) of the given rows of w."""

    w = w.tocsr()
    return {int(r): set(int(c) for c in w.indices[w.indptr[r]:w.indptr[r + 1]]) for r in rows}


def overlap_report(pred: dict[int, set[int]]) -> dict[str, Any]:
    """Pairwise intersections of predecessor sets; disjoint iff every intersection is empty."""

    keys = sorted(pred)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = pred[a] & pred[b]
            pairs.append({"a": a, "b": b, "intersection": sorted(inter)})
    return {"neurons": keys, "support_sizes": {str(k): len(pred[k]) for k in keys},
            "supports": {str(k): sorted(pred[k]) for k in keys}, "pairs": len(pairs),
            "overlapping_pairs": [p for p in pairs if p["intersection"]],
            "all_disjoint": all(not p["intersection"] for p in pairs)}


# ---------------------------------------------------------------- partition

def select_split(final_a: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, Any] | None:
    """Varying coordinate maximizing |A_i| (high_i - low_i), lowest index on ties."""

    score = np.abs(final_a.astype(object)) * (high.astype(object) - low.astype(object))
    varying = high > low
    best, index = None, None
    for i in np.nonzero(varying)[0].tolist():
        s = int(score[i])
        if best is None or s > best:
            best, index = s, i
    if index is None or best == 0:
        return None
    return {"index": int(index), "score": int(best), "a": int(final_a[index]), "low": int(low[index]),
            "high": int(high[index]), "mid": (int(low[index]) + int(high[index])) // 2}


def split_bytes(blo: np.ndarray, bhi: np.ndarray, index: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two children: b_i <= m and b_i >= m + 1; every other byte interval unchanged."""

    lo, hi = int(blo[index]), int(bhi[index])
    if hi <= lo:
        raise ValueError("split coordinate does not vary")
    m = (lo + hi) // 2
    c0 = (blo.copy(), bhi.copy())
    c1 = (blo.copy(), bhi.copy())
    c0[1][index] = m
    c1[0][index] = m + 1
    return [c0, c1]


def check_partition(parent: tuple[np.ndarray, np.ndarray], children, index: int) -> dict[str, Any]:
    """Exact checks: disjoint, union equals the parent, boundaries covered, no value dropped or duplicated."""

    plo, phi = (np.asarray(a, dtype=np.int64) for a in parent)
    (l0, h0), (l1, h1) = ((np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)) for a, b in children)
    others = np.arange(plo.size) != index
    same_elsewhere = bool(np.array_equal(l0[others], plo[others]) and np.array_equal(h0[others], phi[others])
                          and np.array_equal(l1[others], plo[others]) and np.array_equal(h1[others], phi[others]))
    parent_values = list(range(int(plo[index]), int(phi[index]) + 1))
    v0 = list(range(int(l0[index]), int(h0[index]) + 1))
    v1 = list(range(int(l1[index]), int(h1[index]) + 1))
    union = sorted(v0 + v1)
    result = {"same_elsewhere": same_elsewhere, "child_values": [v0, v1], "parent_values": parent_values,
              "nonempty": bool(v0) and bool(v1), "disjoint": not set(v0) & set(v1),
              "union_equals_parent": union == parent_values, "no_duplicates": len(union) == len(set(union)),
              "boundaries_covered": bool(parent_values) and parent_values[0] in union and parent_values[-1] in union}
    result["passed"] = all(result[k] for k in ("same_elsewhere", "nonempty", "disjoint", "union_equals_parent",
                                               "no_duplicates", "boundaries_covered"))
    return result


def worst_child_of_form(constant: int, a: np.ndarray, boxes) -> list[int]:
    """Exact minimum of constant + a.x over each child box (corner rule), in Python integers."""

    out = []
    for xl, xu in boxes:
        total = int(constant)
        for ai, lo, hi in zip(a.tolist(), xl.tolist(), xu.tolist()):
            total += ai * (lo if ai >= 0 else hi)
        out.append(total)
    return out


# ---------------------------------------------------------------- frozen-choice replay

class Replay:
    """Replace finder.relu_choice in-process so every chain reuses the PARENT's coefficient choice.

    For a ReLU step on a layer whose child bounds tuple maps to a parent tuple P, the output
    coefficients a' are exactly what relu_choice(A, P, policy) (plus at most one keyed override)
    gives under the parent bounds; only the constants d are recomputed from the child interval
    with the parent's class flags (saturated: d = a Q_HIGH; clipped: clipped_const; otherwise
    relu_const). Fixed (a, a') with the exact breakpoint minimum over the child interval is a valid
    relaxation for any child interval inside the parent one. With the identity mapping (child is P)
    the recomputed d must equal the parent's d exactly; a mismatch raises.

    The incoming coefficients a do not depend on bounds once every a' is fixed, so a child chain
    traverses the same coefficients as its parent chain."""

    def __init__(self, finder, q_high: int):
        self.finder, self.q_high = finder, q_high
        self.parent_of: dict[int, tuple] = {}
        self._keep: list = []
        self.override = None
        self.override_applied = 0
        self.calls = 0

    def map(self, child_bounds: list, parent_bounds: list):
        if len(child_bounds) != len(parent_bounds):
            raise ReplayError("layer count mismatch")
        for c, p in zip(child_bounds, parent_bounds):
            if c[0].shape != p[0].shape:
                raise ReplayError("layer size mismatch")
            if not (np.array_equal(c[2], p[2]) and np.array_equal(c[3], p[3])):
                raise ReplayError("child bounds must carry the parent's saturated/clipped flags")
            self.parent_of[id(c)] = p
            self._keep.append(c)

    def __enter__(self):
        self._original = self.finder.relu_choice
        finder, q_high = self.finder, self.q_high
        original = self._original

        def relu_choice(A, bounds, policy=None):
            parent = self.parent_of.get(id(bounds))
            if parent is None:
                raise ReplayError("bounds tuple without a frozen parent")
            self.calls += 1
            Ap, D_parent = original(A, parent, policy)
            Ap = Ap.copy()
            if self.override is not None and self.override["parent"] is parent:
                row, n = self.override["row"], self.override["neuron"]
                if int(A[row, n]) != self.override["a"]:
                    raise ReplayError(f"override expects a={self.override['a']}, got {int(A[row, n])}")
                Ap[row, n] = self.override["a_out"]
                self.override_applied += 1
            l, u, _, _ = bounds
            sat, clip = parent[2], parent[3]
            D = np.where(sat[None, :], A * q_high, finder.relu_const(A, Ap, l, u))
            cols = np.nonzero(clip)[0]
            if cols.size:
                D[:, cols] = finder.clipped_const(A[:, cols], Ap[:, cols], l[cols], u[cols])
            overridden = self.override is not None and self.override["parent"] is parent
            if bounds is parent and not overridden and not np.array_equal(D, D_parent):
                raise ReplayError("identity replay changed a ReLU constant")
            return Ap, D

        self._patched = relu_choice
        finder.relu_choice = relu_choice
        return self

    def __exit__(self, *exc):
        if self.finder.relu_choice is not self._patched:
            raise RuntimeError("relu_choice was replaced inside a replay block")
        self.finder.relu_choice = self._original

    def keyed_override(self, parent_layer_bounds, row: int, neuron: int, a: int, a_out: int):
        replay = self

        class _Ctx:
            def __enter__(self_inner):
                if replay.override is not None:
                    raise ReplayError("nested overrides")
                replay.override = {"parent": parent_layer_bounds, "row": row, "neuron": neuron, "a": a,
                                   "a_out": a_out}
                replay.override_applied = 0
                return replay

            def __exit__(self_inner, *exc):
                replay.override = None

        return _Ctx()
