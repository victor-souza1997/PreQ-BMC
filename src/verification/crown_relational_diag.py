"""Bounded upstream relational diagnostic (two layer-3 neurons, shared layer-1 context).

NOT A PROOF COMPONENT. Builds one ESBMC harness that asserts e_A + e_B >= Delta for two ReLU
relaxation residuals e_j = a_j H(r_j) - a'_j r_j - d_j, where r_j is computed exactly from symbolic
layer-1 activations through the deployed layer-2 and layer-3 arithmetic, and a decoupled control in
which the layer-1 coordinates shared by both cones get independent copies. Spec:
docs/ssv2026/IR_upstream_016.md. A VERIFIED result here is at most a CONDITIONAL_ESBMC_LEMMA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np

Q_LOW, Q_HIGH = -(1 << 15), (1 << 15) - 1
F = 8

# verbatim from the deployed C (dd535f54...): rounding and clamp helpers
DEPLOYED_HELPERS = """static inline __int128 div_round_half_away_from_zero_i128(__int128 num, __int128 den) {
    if (num >= 0) {
        return (num + den / 2) / den;
    }
    return -(((-num) + den / 2) / den);
}

static inline __int128 clamp_to_signed_range_i128(__int128 v, int total_bits) {
    const __int128 lower = -((__int128)1 << (total_bits - 1));
    const __int128 upper = (((__int128)1 << (total_bits - 1)) - 1);
    if (v < lower) return lower;
    if (v > upper) return upper;
    return v;
}

static inline __int128 mac_i128(__int128 acc, int64_t w, int64_t x) {
    return acc + ((__int128)w * (__int128)x);
}
"""


def rha(num: int, den: int = 1 << F) -> int:
    num = int(num)
    return (num + den // 2) // den if num >= 0 else -(((-num) + den // 2) // den)


def H(r: int) -> int:
    """Deployed hidden activation: clamp to Q16, ReLU, clamp (= min(Q_HIGH, max(0, r)))."""
    return min(Q_HIGH, max(0, int(r)))


def residual(a: int, a_out: int, d: int, r: int) -> int:
    return int(a) * H(r) - int(a_out) * int(r) - int(d)


@dataclass
class Cone:
    """One layer-3 neuron and its four layer-2 predecessors, with their unary assumptions."""
    name: str
    neuron3: int
    w3: list[tuple[int, int]]                 # (layer-2 row, weight)
    b3: int
    rows2: dict[int, list[tuple[int, int]]]   # layer-2 row -> [(layer-1 coord, weight)]
    b2: dict[int, int]
    bounds2: dict[int, tuple[int | None, int | None]]   # raw r2 assumptions (None = not assumed)
    bounds3: tuple[int | None, int | None]
    a: int
    a_out: int
    d: int

    def coords(self) -> set[int]:
        return {c for row in self.rows2.values() for c, _ in row}


@dataclass
class Context:
    cones: tuple[Cone, Cone]
    h1: dict[int, tuple[int, int]]            # layer-1 activation interval per coordinate
    delta: int
    label: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def shared(self) -> list[int]:
        return sorted(self.cones[0].coords() & self.cones[1].coords())

    def variable(self, coord: int, cone: int, decoupled: bool) -> str:
        if decoupled and coord in self._shared_set():
            return f"x_{coord}_{'ab'[cone]}"
        return f"x_{coord}"

    def _shared_set(self) -> set[int]:
        if not hasattr(self, "_shared_cache"):
            self._shared_cache = set(self.shared())
        return self._shared_cache

    def variables(self, decoupled: bool) -> dict[str, int]:
        """Variable name -> layer-1 coordinate."""
        out = {}
        for ci, cone in enumerate(self.cones):
            for c in cone.coords():
                out[self.variable(c, ci, decoupled)] = c
        return dict(sorted(out.items(), key=lambda kv: (kv[1], kv[0])))


def structure(mats, neurons: tuple[int, int]) -> dict[str, Any]:
    w3, w2 = mats[3][0].tocsr(), mats[2][0].tocsr()
    rows, coords, terms = {}, {}, 0
    for n in neurons:
        p = w3.indices[w3.indptr[n]:w3.indptr[n + 1]].tolist()
        terms += len(p)
        cs = set()
        for q in p:
            cols = w2.indices[w2.indptr[q]:w2.indptr[q + 1]]
            terms += len(cols)
            cs |= set(cols.tolist())
        rows[n], coords[n] = p, cs
    a, b = neurons
    return {"layer2_rows": {str(n): rows[n] for n in neurons}, "layer2_row_count": len(rows[a]) + len(rows[b]),
            "layer1_coords_per_cone": {str(n): len(coords[n]) for n in neurons},
            "layer1_union": len(coords[a] | coords[b]), "layer1_shared": len(coords[a] & coords[b]),
            "multiply_terms": terms, "layer2_rows_disjoint": not set(rows[a]) & set(rows[b])}


def harness(ctx: Context, decoupled: bool, concrete: dict[str, int] | None = None) -> str:
    """C harness. concrete=None: nondet inputs with assumptions and the target assertion.
    concrete=values: the same arithmetic on fixed inputs, printing e_A, e_B (gcc parity test)."""

    lines = ["#include <stdint.h>"]
    if concrete is not None:
        lines.append("#include <stdio.h>")
    else:
        lines += ["void __ESBMC_assume(_Bool);", "void __ESBMC_assert(_Bool, const char *);",
                  "int64_t nondet_int64(void);"]
    lines += ["", DEPLOYED_HELPERS,
              "static inline __int128 act_i128(__int128 v) {",
              "    v = clamp_to_signed_range_i128(v, 16);",
              "    if (v < 0) v = 0;",
              "    v = clamp_to_signed_range_i128(v, 16);",
              "    return v;",
              "}", "", "int main(void) {"]
    if concrete is not None:
        lines.append("    int ok = 1;")
    for name, coord in ctx.variables(decoupled).items():
        lo, hi = ctx.h1[coord]
        if concrete is None:
            lines.append(f"    int64_t {name} = nondet_int64();")
            lines.append(f"    __ESBMC_assume({name} >= {lo} && {name} <= {hi});")
        else:
            lines.append(f"    int64_t {name} = {int(concrete[name])};")
            lines.append(f"    ok &= ({name} >= {lo} && {name} <= {hi});")

    def assume(expr):
        return f"    __ESBMC_assume({expr});" if concrete is None else f"    ok &= ({expr});"

    for ci, cone in enumerate(ctx.cones):
        tag = "AB"[ci]
        for q, row in cone.rows2.items():
            acc = f"acc2_{tag}_{q}"
            lines.append(f"    __int128 {acc} = 0;")
            for c, w in row:
                lines.append(f"    {acc} = mac_i128({acc}, {int(w)}, {ctx.variable(c, ci, decoupled)});")
            lines.append(f"    __int128 r2_{tag}_{q} = div_round_half_away_from_zero_i128({acc}, ((__int128)1 << 8))"
                         f" + (__int128)({int(cone.b2[q])});")
            lo, hi = cone.bounds2[q]
            if lo is not None:
                lines.append(assume(f"r2_{tag}_{q} >= (__int128)({lo})"))
            if hi is not None:
                lines.append(assume(f"r2_{tag}_{q} <= (__int128)({hi})"))
            lines.append(f"    int64_t h2_{tag}_{q} = (int64_t)act_i128(r2_{tag}_{q});")
        lines.append(f"    __int128 acc3_{tag} = 0;")
        for q, w in cone.w3:
            lines.append(f"    acc3_{tag} = mac_i128(acc3_{tag}, {int(w)}, h2_{tag}_{q});")
        lines.append(f"    __int128 r3_{tag} = div_round_half_away_from_zero_i128(acc3_{tag}, ((__int128)1 << 8))"
                     f" + (__int128)({int(cone.b3)});")
        lo, hi = cone.bounds3
        if lo is not None:
            lines.append(assume(f"r3_{tag} >= (__int128)({lo})"))
        if hi is not None:
            lines.append(assume(f"r3_{tag} <= (__int128)({hi})"))
        lines.append(f"    __int128 h3_{tag} = act_i128(r3_{tag});")
        lines.append(f"    __int128 e_{tag} = (__int128)({int(cone.a)}) * h3_{tag} - (__int128)({int(cone.a_out)}) * r3_{tag}"
                     f" - (__int128)({int(cone.d)});")
    if concrete is None:
        lines.append(f'    __ESBMC_assert(e_A + e_B >= (__int128)({int(ctx.delta)}), "{ctx.label}");')
    else:
        lines.append('    printf("%d %lld %lld\\n", ok, (long long)e_A, (long long)e_B);')
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)


def replay(ctx: Context, decoupled: bool, values: dict[str, int]) -> dict[str, Any]:
    """Exact Python replay of the harness arithmetic on a full assignment."""

    names = ctx.variables(decoupled)
    missing = sorted(set(names) - set(values))
    if missing:
        return {"complete": False, "missing": missing[:20], "missing_count": len(missing)}
    failed = []
    for name, coord in names.items():
        lo, hi = ctx.h1[coord]
        if not lo <= int(values[name]) <= hi:
            failed.append(f"{name} outside [{lo}, {hi}]")
    es, r3s = [], []
    for ci, cone in enumerate(ctx.cones):
        h2 = {}
        for q, row in cone.rows2.items():
            s = sum(int(w) * int(values[ctx.variable(c, ci, decoupled)]) for c, w in row)
            r2 = rha(s) + int(cone.b2[q])
            lo, hi = cone.bounds2[q]
            if (lo is not None and r2 < lo) or (hi is not None and r2 > hi):
                failed.append(f"r2 {'AB'[ci]}:{q}={r2} outside assumption")
            h2[q] = H(r2)
        r3 = rha(sum(int(w) * h2[q] for q, w in cone.w3)) + int(cone.b3)
        lo, hi = cone.bounds3
        if (lo is not None and r3 < lo) or (hi is not None and r3 > hi):
            failed.append(f"r3 {'AB'[ci]}={r3} outside assumption")
        r3s.append(r3)
        es.append(residual(cone.a, cone.a_out, cone.d, r3))
    total = es[0] + es[1]
    return {"complete": True, "assumptions_hold": not failed, "failed_assumptions": failed[:20],
            "r3": r3s, "e": es, "e_sum": total, "violates": total < int(ctx.delta)}


def assignment_from_hidden(ctx: Context, decoupled: bool, h1: np.ndarray) -> dict[str, int]:
    """Assignment taking every copy of coordinate k from a concrete layer-1 activation vector."""
    return {name: int(h1[coord]) for name, coord in ctx.variables(decoupled).items()}


_TRACE_LINE = re.compile(r"^\s*(x_\d+(?:_[ab])?)\s*=\s*(-?\d+)\b")


def parse_assignment(log: str) -> dict[str, int]:
    """Values of the harness input variables from an ESBMC counterexample trace (last write wins)."""
    out = {}
    for line in log.splitlines():
        m = _TRACE_LINE.match(line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


# ---------------------------------------------------------------- exact reduced encoding (I-R encoding pilot)

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


class WidthProofFailed(RuntimeError):
    """Some intermediate of the reduced program may leave int32: the transformation is refused."""


def fixed_inputs(ctx: Context) -> dict[int, int]:
    """Layer-1 coordinates whose assumed interval is a single value."""
    return {c: lo for c, (lo, hi) in ctx.h1.items() if lo == hi}


def _rha_interval(lo: int, hi: int, steps: list, tag: str) -> tuple[int, int]:
    """Interval of div_round_half_away_from_zero(num, 256) over num in [lo, hi], recording every
    intermediate of the helper: num + 128 (num >= 0 branch), -num and (-num) + 128 (num < 0 branch)."""
    if lo >= 0 or hi >= 0:
        steps.append((f"{tag}:num+128", max(lo, 0) + 128, hi + 128))
    if lo < 0:
        steps.append((f"{tag}:-num", -min(hi, -1), -lo))
        steps.append((f"{tag}:-num+128", -min(hi, -1) + 128, -lo + 128))
    return rha(lo), rha(hi)


def _act_interval(lo: int, hi: int) -> tuple[int, int]:
    return H(lo), H(hi)


def _mul_interval(w: int, lo: int, hi: int) -> tuple[int, int]:
    p, q = w * lo, w * hi
    return min(p, q), max(p, q)


def width_proof(ctx: Context, decoupled: bool) -> dict[str, Any]:
    """Interval bound of EVERY intermediate value of the reduced program, in its exact evaluation
    order, over the domain fixed by the assumptions that precede it. Raises WidthProofFailed if any
    value (or the negation of one) can leave int32. Values before an assumption use the
    unassumed range; values after it use the intersected range."""

    fixed = fixed_inputs(ctx)
    steps: list[tuple[str, int, int]] = []
    sums = []
    for ci, cone in enumerate(ctx.cones):
        tag = "AB"[ci]
        h2 = {}
        for q, row in cone.rows2.items():
            const = sum(int(w) * fixed[c] for c, w in row if c in fixed)
            steps.append((f"{tag}{q}:fixed_const", const, const))
            lo = hi = const
            for c, w in row:
                if c in fixed:
                    continue
                plo, phi = _mul_interval(int(w), *ctx.h1[c])
                steps.append((f"{tag}{q}:w*x{c}", plo, phi))
                lo, hi = lo + plo, hi + phi
                steps.append((f"{tag}{q}:acc", lo, hi))
            rlo, rhi = _rha_interval(lo, hi, steps, f"{tag}{q}:rha")
            r2lo, r2hi = rlo + int(cone.b2[q]), rhi + int(cone.b2[q])
            steps.append((f"{tag}{q}:r2", r2lo, r2hi))
            alo, ahi = cone.bounds2[q]
            r2lo, r2hi = max(r2lo, alo if alo is not None else r2lo), min(r2hi, ahi if ahi is not None else r2hi)
            if r2lo > r2hi:
                raise WidthProofFailed(f"{tag}{q}: assumed r2 interval empty")
            h2[q] = _act_interval(r2lo, r2hi)
            steps.append((f"{tag}{q}:h2", *h2[q]))
        lo = hi = 0
        for q, w in cone.w3:
            plo, phi = _mul_interval(int(w), *h2[q])
            steps.append((f"{tag}:w3*h2_{q}", plo, phi))
            lo, hi = lo + plo, hi + phi
            steps.append((f"{tag}:acc3", lo, hi))
        rlo, rhi = _rha_interval(lo, hi, steps, f"{tag}:rha3")
        r3lo, r3hi = rlo + int(cone.b3), rhi + int(cone.b3)
        steps.append((f"{tag}:r3", r3lo, r3hi))
        alo, ahi = cone.bounds3
        r3lo, r3hi = max(r3lo, alo if alo is not None else r3lo), min(r3hi, ahi if ahi is not None else r3hi)
        if r3lo > r3hi:
            raise WidthProofFailed(f"{tag}: assumed r3 interval empty")
        h3 = _act_interval(r3lo, r3hi)
        t1 = _mul_interval(int(cone.a), *h3)
        t2 = _mul_interval(int(cone.a_out), r3lo, r3hi)
        steps += [(f"{tag}:a*h3", *t1), (f"{tag}:a_out*r3", *t2)]
        diff = (t1[0] - t2[1], t1[1] - t2[0])
        steps.append((f"{tag}:a*h3-a_out*r3", *diff))
        e = (diff[0] - int(cone.d), diff[1] - int(cone.d))
        steps.append((f"{tag}:e", *e))
        sums.append(e)
    steps.append(("e_A+e_B", sums[0][0] + sums[1][0], sums[0][1] + sums[1][1]))
    for name, value in (("delta", ctx.delta), *((f"{'AB'[i]}:{k}", int(getattr(c, k)))
                                                  for i, c in enumerate(ctx.cones) for k in ("a", "a_out", "d", "b3"))):
        steps.append((f"const:{name}", value, value))
    worst = max(steps, key=lambda s: max(abs(s[1]), abs(s[2])))
    bad = [s for s in steps if s[1] <= INT32_MIN or s[2] > INT32_MAX]    # <= MIN: keeps -v representable
    if bad:
        raise WidthProofFailed(f"{len(bad)} intermediates leave int32, first {bad[0]}")
    return {"intermediates": len(steps), "max_abs": max(abs(worst[1]), abs(worst[2])), "max_at": worst[0],
            "max_abs_layer2_acc": max(max(abs(s[1]), abs(s[2])) for s in steps if s[0].endswith(":acc")),
            "int32_ok": True}


def harness_reduced(ctx: Context, decoupled: bool, concrete: dict[str, int] | None = None) -> str:
    """The same obligation with (1) fixed inputs substituted as constants and folded into each layer-2
    accumulator's starting value, and (2) int32 arithmetic, valid only after width_proof() succeeds.
    Every layer-2 row, every r2/r3 assumption and the target assertion are kept."""

    width_proof(ctx, decoupled)
    fixed = fixed_inputs(ctx)
    if concrete is None:
        lines = ["#include <stdint.h>", "void __ESBMC_assume(_Bool);", "void __ESBMC_assert(_Bool, const char *);",
                 "int32_t nondet_int32(void);", ""]
    else:
        lines = ["#include <stdint.h>", "#include <stdio.h>", "static int ok = 1;",
                 "#define __ESBMC_assume(c) (ok &= (c))", ""]
    lines += [
             "/* int32 forms of the deployed helpers; equivalence by width_proof() (no intermediate leaves int32) */",
             "static inline int32_t rha32(int32_t num) {",
             "    if (num >= 0) {", "        return (num + 128) / 256;", "    }",
             "    return -(((-num) + 128) / 256);", "}",
             "static inline int32_t clamp16(int32_t v) {",
             "    if (v < -32768) return -32768;", "    if (v > 32767) return 32767;", "    return v;", "}",
             "static inline int32_t act32(int32_t v) {",
             "    v = clamp16(v);", "    if (v < 0) v = 0;", "    v = clamp16(v);", "    return v;", "}",
             "", "int main(void) {"]
    for name, coord in ctx.variables(decoupled).items():
        if coord in fixed:
            continue
        lo, hi = ctx.h1[coord]
        init = "nondet_int32()" if concrete is None else str(int(concrete[name]))
        lines.append(f"    int32_t {name} = {init};")
        lines.append(f"    __ESBMC_assume({name} >= {lo} && {name} <= {hi});")
    for ci, cone in enumerate(ctx.cones):
        tag = "AB"[ci]
        for q, row in cone.rows2.items():
            const = sum(int(w) * fixed[c] for c, w in row if c in fixed)
            acc = f"acc2_{tag}_{q}"
            lines.append(f"    int32_t {acc} = {const};")
            for c, w in row:
                if c in fixed:
                    continue
                lines.append(f"    {acc} = {acc} + ({int(w)}) * {ctx.variable(c, ci, decoupled)};")
            lines.append(f"    int32_t r2_{tag}_{q} = rha32({acc}) + ({int(cone.b2[q])});")
            lo, hi = cone.bounds2[q]
            if lo is not None:
                lines.append(f"    __ESBMC_assume(r2_{tag}_{q} >= {lo});")
            if hi is not None:
                lines.append(f"    __ESBMC_assume(r2_{tag}_{q} <= {hi});")
            lines.append(f"    int32_t h2_{tag}_{q} = act32(r2_{tag}_{q});")
        lines.append(f"    int32_t acc3_{tag} = 0;")
        for q, w in cone.w3:
            lines.append(f"    acc3_{tag} = acc3_{tag} + ({int(w)}) * h2_{tag}_{q};")
        lines.append(f"    int32_t r3_{tag} = rha32(acc3_{tag}) + ({int(cone.b3)});")
        lo, hi = cone.bounds3
        if lo is not None:
            lines.append(f"    __ESBMC_assume(r3_{tag} >= {lo});")
        if hi is not None:
            lines.append(f"    __ESBMC_assume(r3_{tag} <= {hi});")
        lines.append(f"    int32_t h3_{tag} = act32(r3_{tag});")
        lines.append(f"    int32_t e_{tag} = ({int(cone.a)}) * h3_{tag} - ({int(cone.a_out)}) * r3_{tag}"
                     f" - ({int(cone.d)});")
    if concrete is None:
        lines.append(f'    __ESBMC_assert(e_A + e_B >= {int(ctx.delta)}, "{ctx.label}");')
    else:
        lines.append('    printf("%d %d %d\\n", ok, (int)e_A, (int)e_B);')
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)


def complete_assignment(ctx: Context, decoupled: bool, values: dict[str, int]) -> dict[str, int]:
    """A reduced-harness trace assigns only the free inputs; fixed inputs have exactly one admissible value."""
    fixed = fixed_inputs(ctx)
    out = dict(values)
    for name, coord in ctx.variables(decoupled).items():
        if coord in fixed:
            out.setdefault(name, fixed[coord])
    return out
