"""Brute-force soundness gate for an exact-integer difference-row margin bound.

The output margin is currently bounded by taking each logit's range over the
assumption box independently and subtracting, which evaluates
``min(t) - max(c)`` where the sound quantity is ``min(t - c)``. Bounding the
difference row ``w_t - w_c`` directly is tighter and still sound -- but only
with a correction, because round-half-away-from-zero does not distribute over
subtraction:

    rdiv(a, S) - rdiv(b, S)  !=  rdiv(a - b, S)   in general

The residue is at most one output ULP, so the difference-row bound must be
lowered by 1. These tests enumerate every point of a small input box through
the deployed integer kernel and check both directions: the corrected bound is
never above ground truth, and the uncorrected bound demonstrably is -- so the
correction is load-bearing, not defensive padding.
"""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from verification.invariants import propagate_exact_intervals
from verification.replay import LayerReplayFormat, replay_on_python


def _round_half_away(numerator: int, denominator: int) -> int:
    numerator = int(numerator)
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -(((-numerator) + denominator // 2) // denominator)


def _saturates(value: int, total_bits: int) -> bool:
    low, high = -(1 << (total_bits - 1)), (1 << (total_bits - 1)) - 1
    return not low <= int(value) <= high


class _Case:
    """One tiny network plus the three margin quantities it induces."""

    def __init__(self, seed: int, *, inputs: int, hidden: int,
                 frac: int, total: int, span: int, wmax: int) -> None:
        rng = np.random.default_rng(seed)
        self.first = SimpleNamespace(
            weights_int=rng.integers(-wmax, wmax + 1, (hidden, inputs)).astype(np.int64),
            biases_int=rng.integers(-4, 5, hidden).astype(np.int64),
        )
        self.second = SimpleNamespace(
            weights_int=rng.integers(-wmax, wmax + 1, (2, hidden)).astype(np.int64),
            biases_int=rng.integers(-4, 5, 2).astype(np.int64),
        )
        self.fmt = LayerQuantizationSpec(
            total_bits=total, integer_bits=total - frac - 1, fractional_bits=frac
        )
        self.low = np.zeros(inputs, dtype=np.int64)
        self.high = np.full(inputs, span, dtype=np.int64)
        self.frac, self.total = frac, total

    def ground_truth(self) -> int:
        """Exhaustive minimum of (logit0 - logit1) over the whole input box."""
        best = None
        ranges = [range(int(a), int(b) + 1) for a, b in zip(self.low, self.high)]
        for point in itertools.product(*ranges):
            hidden = replay_on_python(
                list(point), self.first,
                LayerReplayFormat(self.frac, self.total, apply_relu=True),
            )
            out = replay_on_python(
                list(hidden), self.second,
                LayerReplayFormat(self.frac, self.total, apply_relu=False),
            )
            margin = int(out[0]) - int(out[1])
            best = margin if best is None else min(best, margin)
        return int(best)

    def bounds(self) -> tuple[int, int, bool]:
        """Return (independent bound, corrected difference bound, saturated?)."""
        (h_lo, h_hi), (o_lo, o_hi) = propagate_exact_intervals(
            [self.first, self.second], [self.fmt, self.fmt], self.low, self.high
        )
        independent = int(o_lo[0]) - int(o_hi[1])
        row = self.second.weights_int[0] - self.second.weights_int[1]
        accumulator = sum(
            min(int(w) * int(a), int(w) * int(b))
            for w, a, b in zip(row, h_lo, h_hi)
        )
        bias = int(self.second.biases_int[0]) - int(self.second.biases_int[1])
        difference = _round_half_away(accumulator, 1 << self.frac) + bias
        saturated = any(
            _saturates(v, self.total) for v in list(o_lo) + list(o_hi)
        )
        return independent, difference - 1, saturated


CONFIGS = [
    dict(inputs=3, hidden=3, frac=2, total=8, span=3, wmax=6),
    dict(inputs=3, hidden=4, frac=3, total=10, span=4, wmax=8),
    dict(inputs=4, hidden=3, frac=1, total=8, span=2, wmax=5),
]


class DifferenceRowMarginBoundTest(unittest.TestCase):
    def test_corrected_bound_never_exceeds_enumerated_truth(self) -> None:
        checked = 0
        for config in CONFIGS:
            for seed in range(30):
                case = _Case(seed, **config)
                independent, corrected, saturated = case.bounds()
                if saturated:
                    continue
                truth = case.ground_truth()
                self.assertLessEqual(
                    corrected, truth,
                    f"difference-row bound {corrected} exceeds enumerated "
                    f"minimum {truth} for seed {seed} config {config}",
                )
                self.assertLessEqual(independent, truth)
                checked += 1
        self.assertGreater(checked, 60, "too few networks actually enumerated")

    def test_rounding_correction_is_load_bearing(self) -> None:
        """Without the 1-ULP residue the bound is unsound on real cases.

        This is the reason the correction exists. If this test ever stops
        finding violations the enumeration has become too weak to defend the
        bound, which is itself a failure.
        """
        violations = 0
        for config in CONFIGS:
            for seed in range(30):
                case = _Case(seed, **config)
                _, corrected, saturated = case.bounds()
                if saturated:
                    continue
                if corrected + 1 > case.ground_truth():
                    violations += 1
        self.assertGreater(
            violations, 0,
            "uncorrected difference bound never violated ground truth; "
            "the enumeration is not exercising the rounding residue",
        )

    def test_neither_bound_dominates_so_the_maximum_is_the_one_to_use(self) -> None:
        """The difference row is usually tighter, but not always.

        Recovering the correlation gains ``min(t-c) - (min(t)-max(c))``, while
        the rounding residue costs a flat 1 ULP. When a box has little
        correlation to recover the residue dominates and the difference bound
        comes out *below* the independent one. Both are sound, so the bound to
        report is their maximum -- taking the difference row unconditionally
        would lose ground on those cases.
        """
        tighter = looser = 0
        for config in CONFIGS:
            for seed in range(30):
                case = _Case(seed, **config)
                independent, corrected, saturated = case.bounds()
                if saturated:
                    continue
                truth = case.ground_truth()
                # the combined bound is what an implementation should emit
                self.assertLessEqual(max(independent, corrected), truth)
                if corrected > independent:
                    tighter += 1
                elif corrected < independent:
                    looser += 1
        self.assertGreater(tighter, 0, "difference row never helped")
        self.assertGreater(
            looser, 0,
            "difference row never lost; if this holds the max() is unnecessary "
            "and the implementation can be simplified",
        )


if __name__ == "__main__":
    unittest.main()
