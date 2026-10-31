"""Diagnostic-only support for docs/ssv2026/finder_diagnostic_spec_013_016.md.

NOT A PROOF COMPONENT. Nothing here is imported by the production finder,
exporter or checker, and nothing here may label a result as verified: ESBMC
statuses are recorded only from actual solver runs.

Names follow spec section 2: s = W h (accumulator without bias),
r = RHA(s / 256) + bias (raw), z = clamp(r), h = relu(z) at hidden layers,
final deployed logits z_4. e = 256 * RHA(s / 256) - s, an integer in [-128, 128].
All identity checks use exact Python integers.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterable

import numpy as np

SCHEMA = "preqbmc_crown_diagnostic_v1"
Q_LOW, Q_HIGH = -(1 << 15), (1 << 15) - 1
F = 8
SCALE_BITS = 16

EVIDENCE_KINDS = (
    "MEASURED_EXACT_EXECUTION",
    "SCREEN_BOUND_UNCHECKED",
    "CONDITIONAL_ESBMC_LEMMA",
    "ESBMC_CHECKED_OBLIGATION",
    "COUNTERFACTUAL_ONLY",
    "UNSUPPORTED_CERTIFICATE_FORM",
)
SOLVER_STATUSES = ("NOT_RUN", "VERIFIED", "FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN")
RUN_STATUSES = ("COMPLETED", "SKIPPED_NO_CANDIDATE", "UNSUPPORTED", "NOT_RUN_BUDGET", "DEPENDENCY_FAILED",
                "OVERFLOW_GUARD", "CANCELLED", "INVALID_DOMAIN", "PARITY_FAILURE")
ASSESSMENTS = ("SUPPORTS", "NO_SUPPORT_WITHIN_PROBE", "INCONCLUSIVE", "IMPLEMENTATION_ERROR")
ESBMC_EVIDENCE = ("CONDITIONAL_ESBMC_LEMMA", "ESBMC_CHECKED_OBLIGATION")
METRIC_FIELDS = ("wall_seconds", "cpu_seconds", "peak_process_tree_rss_bytes", "peak_cgroup_memory_bytes",
                 "solver_elapsed_sum_seconds", "serialized_bytes", "compressed_bytes", "planned_chains",
                 "planned_multiply_terms", "planned_assertions", "planned_harnesses", "additional_assertions",
                 "additional_harnesses", "executed_solver_calls")


class ImplementationError(RuntimeError):
    """An exact identity, nonnegativity or parity check failed: stop dependent interpretation."""


class OverflowGuard(RuntimeError):
    """Native-width arithmetic could wrap; abort instead."""


# ---------------------------------------------------------------- exact numbers

def dec(value) -> str:
    """Exact decimal string of an integer (numpy or Python)."""

    return str(int(value))


def rational(num: int, den: int) -> dict[str, str]:
    value = Fraction(int(num), int(den))
    return {"num": str(value.numerator), "den": str(value.denominator)}


def idot(u, v) -> int:
    """Exact integer dot product (Python integers, no wrap)."""

    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    support = np.nonzero(u)[0]
    return sum(int(a) * int(b) for a, b in zip(u[support].tolist(), v[support].tolist()))


def rha(num: int, den: int) -> int:
    """div_round_half_away_from_zero_i128 for positive den, exact."""

    num, den = int(num), int(den)
    if den <= 0:
        raise ValueError("denominator must be positive")
    if num >= 0:
        return (num + den // 2) // den
    return -(((-num) + den // 2) // den)


def rha_vec(s: np.ndarray, den: int = 1 << F) -> np.ndarray:
    s = np.asarray(s, dtype=np.int64)
    return np.where(s >= 0, (s + den // 2) // den, -((-s + den // 2) // den))


def ceil_div(a: int, d: int) -> int:
    return -((-int(a)) // int(d))


def material_gain_threshold(baseline_raw_lb: int, control: bool) -> int:
    """Spec section 3: max(1, ceil(0.05 * max(1, 1 - baseline_raw_lb))); one unit for a control margin."""

    if control:
        return 1
    return max(1, math.ceil(Fraction(5, 100) * max(1, 1 - int(baseline_raw_lb))))


def significant_singleton_gap(raw_gap: int, nominal_raw_margin: int) -> bool:
    """Spec section 6: raw_gap >= max(1, ceil(0.05 * abs(nominal_raw_margin)))."""

    return int(raw_gap) >= max(1, math.ceil(Fraction(5, 100) * abs(int(nominal_raw_margin))))


def gap_fraction(raw_gap: int, nominal_raw_margin: int) -> dict[str, str]:
    return rational(raw_gap, max(1, abs(int(nominal_raw_margin))))


# ---------------------------------------------------------------- deployed semantics

def exact_forward(mats, x, relu_flags) -> list[dict[str, np.ndarray]]:
    """Per layer s, r, z, h from the lowered CSR, exact.

    int64 is used only after proving |s| < 2^62 from Python-integer row sums,
    so no product or sum can wrap; otherwise OverflowGuard."""

    h = np.asarray(x, dtype=np.int64)
    trace = []
    for (w, b), relu in zip(mats, relu_flags):
        row_abs = np.abs(w).sum(axis=1)
        worst = int(np.asarray(row_abs).max(initial=0)) * int(np.abs(h).max(initial=0)) + int(np.abs(b).max())
        if worst >= 1 << 62:
            raise OverflowGuard(f"accumulator envelope {worst} reaches 2^62")
        s = np.asarray(w @ h, dtype=np.int64)
        r = rha_vec(s) + np.asarray(b, dtype=np.int64)
        z = np.clip(r, Q_LOW, Q_HIGH)
        hh = np.clip(np.maximum(z, 0), Q_LOW, Q_HIGH) if relu else z
        trace.append({"s": s, "r": r, "z": z, "h": hh, "e": (r - b) * (1 << F) - s})
        h = hh
    return trace


def deployed_violation(deployed_margin: int) -> bool:
    """Strict dominance z_t - z_c >= 1; a tie (0) is a violation regardless of argmax tie-breaking."""

    return int(deployed_margin) <= 0


def margins_from_logits(z_out, r_out, target: int) -> dict[int, dict[str, int]]:
    return {c: {"deployed": int(z_out[target]) - int(z_out[c]), "raw": int(r_out[target]) - int(r_out[c])}
            for c in range(len(z_out)) if c != target}


# ---------------------------------------------------------------- input domains

def resize_source_index(crop_shape, output_shape, resize_mode: str) -> np.ndarray:
    """Flat crop index of each resized coordinate, exactly as ByteImageEncoder.resize."""

    H, W, C = crop_shape
    h, w, c = output_shape
    if C != c:
        raise ValueError("channel mismatch")
    if resize_mode == "nearest_center":
        rows = (2 * np.arange(h) + 1) * H // (2 * h)
        cols = (2 * np.arange(w) + 1) * W // (2 * w)
    else:
        rows = np.arange(h) * H // h
        cols = np.arange(w) * W // w
    index = (rows[:, None, None] * W + cols[None, :, None]) * C + np.arange(C)[None, None, :]
    return index.reshape(-1).astype(np.int64)


@dataclass(frozen=True)
class AliasGroups:
    source: np.ndarray        # flat crop index per group, ascending
    members: tuple            # resized coordinates per group
    of_coordinate: np.ndarray  # group id per resized coordinate

    @property
    def sizes(self) -> dict[str, int]:
        counts: dict[int, int] = {}
        for m in self.members:
            counts[len(m)] = counts.get(len(m), 0) + 1
        return {str(k): v for k, v in sorted(counts.items())}


def alias_groups(source_index: np.ndarray) -> AliasGroups:
    source, inverse = np.unique(source_index, return_inverse=True)
    members = tuple(tuple(np.nonzero(inverse == g)[0].tolist()) for g in range(source.size))
    return AliasGroups(source=source, members=members, of_coordinate=inverse.astype(np.int64))


def group_allowed_values(crop: np.ndarray, groups: AliasGroups, eps: int) -> list[list[int]]:
    flat = crop.reshape(-1)
    return [list(range(max(0, int(flat[s]) - eps), min(255, int(flat[s]) + eps) + 1)) for s in groups.source]


def crop_from_assignment(crop: np.ndarray, groups: AliasGroups, values) -> np.ndarray:
    out = crop.copy().reshape(-1)
    out[groups.source] = np.asarray(values, dtype=np.uint8)
    return out.reshape(crop.shape)


def lift_resized(resized_bytes: np.ndarray, groups: AliasGroups, crop: np.ndarray):
    """The crop with changed source bytes reproducing resized_bytes, or None if aliases disagree."""

    flat = np.asarray(resized_bytes).reshape(-1)
    values = []
    for members in groups.members:
        group_values = {int(flat[m]) for m in members}
        if len(group_values) != 1:
            return None
        values.append(group_values.pop())
    return crop_from_assignment(crop, groups, values)


def in_crop_domain(candidate: np.ndarray, crop: np.ndarray, groups: AliasGroups, eps: int) -> dict[str, bool]:
    """Source-byte bounds, unused-pixel invariance; alias consistency holds by construction of a crop."""

    cand = candidate.reshape(-1).astype(np.int64)
    orig = crop.reshape(-1).astype(np.int64)
    used = np.zeros(orig.size, bool)
    used[groups.source] = True
    return {"source_bounds_satisfied": bool(np.all(np.abs(cand - orig) <= eps)),
            "unused_pixels_nominal": bool(np.array_equal(cand[~used], orig[~used])),
            "byte_range": bool(cand.min() >= 0 and cand.max() <= 255)}


def sha256_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


# ---------------------------------------------------------------- D-V fixed search

@dataclass
class Evaluation:
    deployed: int
    raw: int
    margins: dict
    key: str
    new: bool


def search_focus(groups: AliasGroups, allowed: list[list[int]], nominal_values: list[int],
                 group_coefficient: list[int], evaluate: Callable[[list[int]], Evaluation],
                 seed_of_restart: Callable[[int], int], deadline: float | None = None,
                 restarts: int = 4, per_restart: int = 256, cd_proposals: int = 127, cd_groups: int = 32,
                 log: Callable[[dict], None] | None = None) -> dict:
    """Spec section 5 steps 2-6 for one focus competitor. evaluate() returns the deployed and raw
    focus margins of a source-group assignment. Returns the search record."""

    G = len(allowed)
    movable = [g for g in range(G) if len(allowed[g]) > 1]
    ranked = sorted(movable, key=lambda g: (-abs(int(group_coefficient[g])) * (allowed[g][-1] - allowed[g][0]),
                                            int(groups.source[g])))[:cd_groups]
    record = {"movable_groups": len(movable), "ranked_groups": [int(g) for g in ranked], "starts": [],
              "proposals": 0, "evaluations": 0, "unexecuted_proposals": 0, "incomplete_starts": 0,
              "violation": None, "best": None, "seeds": [], "status": "COMPLETED"}
    if not movable:
        record["status"] = "SKIPPED_NO_CANDIDATE"
        return record
    best_overall = None
    for restart in range(restarts):
        seed = seed_of_restart(restart)
        record["seeds"].append(seed)
        rng = np.random.Generator(np.random.PCG64(seed))
        if restart == 0:
            current = list(nominal_values)
        elif restart == 1:
            current = [allowed[g][0] if group_coefficient[g] > 0 else allowed[g][-1] if group_coefficient[g] < 0
                       else nominal_values[g] for g in range(G)]
        else:
            current = [allowed[g][int(rng.integers(len(allowed[g])))] for g in range(G)]
        start = {"restart": restart, "seed": seed, "proposals": 0, "accepted": 0, "complete": False}
        record["starts"].append(start)

        def propose(values, kind):
            nonlocal current, incumbent
            if deadline is not None and time.monotonic() > deadline:
                return "deadline"
            result = evaluate(values)
            start["proposals"] += 1
            record["proposals"] += 1
            record["evaluations"] += int(result.new)
            accepted = incumbent is None or (result.deployed, result.raw) < (incumbent.deployed, incumbent.raw)
            if log is not None:
                log({"restart": restart, "proposal_index": start["proposals"] - 1, "kind": kind,
                     "key": result.key, "deployed": result.deployed, "raw": result.raw, "accepted": accepted})
            if accepted:
                incumbent, current = result, list(values)
                start["accepted"] += 1
            if deployed_violation(result.deployed):
                record["violation"] = {"restart": restart, "proposal_index": start["proposals"] - 1,
                                       "values": [int(v) for v in values], "key": result.key,
                                       "deployed": result.deployed, "raw": result.raw}
                return "violation"
            return None

        incumbent = None
        stop = propose(list(current), "start")
        cursor = 0
        while stop is None and start["proposals"] < 1 + cd_proposals:
            g = ranked[cursor % len(ranked)]
            cursor += 1
            for value in [v for v in allowed[g] if v != current[g]]:
                if start["proposals"] >= 1 + cd_proposals:
                    break
                candidate = list(current)
                candidate[g] = value
                stop = propose(candidate, "coordinate")
                if stop:
                    break
        block = 0
        while stop is None and start["proposals"] < per_restart:
            size = min((8, 32)[block % 2], len(movable))
            block += 1
            chosen = rng.choice(len(movable), size=size, replace=False)
            candidate = list(current)
            for index in chosen:
                g = movable[int(index)]
                others = [v for v in allowed[g] if v != current[g]]
                candidate[g] = others[int(rng.integers(len(others)))]
            stop = propose(candidate, "block")
        start["complete"] = stop is None and start["proposals"] == per_restart
        start["best"] = None if incumbent is None else {"deployed": incumbent.deployed, "raw": incumbent.raw,
                                                        "key": incumbent.key, "values": [int(v) for v in current]}
        if incumbent is not None and (best_overall is None or
                                      (incumbent.deployed, incumbent.raw) < (best_overall["deployed"],
                                                                             best_overall["raw"])):
            best_overall = dict(start["best"], restart=restart)
        if not start["complete"]:
            record["incomplete_starts"] += 1
        if stop == "violation":
            break
        if stop == "deadline":
            record["status"] = "NOT_RUN_BUDGET"
            record["unexecuted_proposals"] = restarts * per_restart - record["proposals"]
            break
    record["best"] = best_overall
    return record


# ---------------------------------------------------------------- clamp closure (mirror of the exporter)

def closure_edges(margins: dict[int, int], logit_hh, logit_hl, logit_lower_ok, raw_t_low: int, raw_t_high: int,
                  target: int) -> dict[int, dict]:
    """The exporter's per-competitor clamp closure (crown_chain_export_v2.main), replicated for logging.
    Its result is cross-checked against the production screen's closed count in preflight."""

    out = {}
    for c, margin in margins.items():
        if int(logit_hh[c]) <= Q_HIGH - 1:
            top = {"rule": f"raw_c <= {Q_HIGH - 1}", "reference": f"fact/L4/n{c}/upper", "value": int(logit_hh[c])}
        elif raw_t_high <= Q_HIGH:
            top = {"rule": f"raw_t <= {Q_HIGH}", "reference": f"chain/logit/{target}/upper", "value": raw_t_high}
        else:
            top = None
        if bool(logit_lower_ok[c]) and int(logit_hl[c]) >= Q_LOW:
            bottom = {"rule": f"raw_c >= {Q_LOW}", "reference": f"fact/L4/n{c}/lower", "value": int(logit_hl[c])}
        elif raw_t_low >= Q_LOW + 1:
            bottom = {"rule": f"raw_t >= {Q_LOW + 1}", "reference": f"chain/logit/{target}/lower", "value": raw_t_low}
        else:
            bottom = None
        out[c] = {"raw_lb": int(margin), "top": top, "bottom": bottom,
                  "closed": top is not None and bottom is not None and int(margin) >= 1}
    return out


# ---------------------------------------------------------------- I-B / I-S support

def midpoint_endpoint(side: str, endpoint: int, nominal: int) -> int:
    """Spec section 8 I-B hypothetical endpoint: l' = ceil((l + r0) / 2), u' = floor((u + r0) / 2)."""

    if side == "lower":
        if int(nominal) < int(endpoint):
            raise ImplementationError("nominal raw value below the baseline lower bound")
        return ceil_div(int(endpoint) + int(nominal), 2)
    if side == "upper":
        if int(nominal) > int(endpoint):
            raise ImplementationError("nominal raw value above the baseline upper bound")
        return (int(endpoint) + int(nominal)) // 2
    raise ValueError(side)


def endpoint_refuted(side: str, endpoint: int, raw_value: int) -> bool:
    """True when a concrete raw value lies outside the proposed one-sided bound."""

    return int(raw_value) < int(endpoint) if side == "lower" else int(raw_value) > int(endpoint)


def tighter(side: str, new: int, old: int) -> bool:
    return int(new) > int(old) if side == "lower" else int(new) < int(old)


def slope_eligible(l: int, u: int, saturated: bool, clipped: bool, a: int) -> bool:
    """Spec section 8 I-S eligibility: l < 0 < u, not saturated or clipped, inside Q, nonzero incoming a."""

    return (int(l) < 0 < int(u) and not saturated and not clipped and Q_LOW <= int(l) <= int(u) <= Q_HIGH
            and int(a) != 0)


def slope_candidates(a: int, baseline_a_out: int) -> list[int]:
    """Distinct values of {0, a, floor(a/2), baseline_a_out} without the unchanged baseline value, in that order."""

    out = []
    for v in (0, int(a), int(a) // 2, int(baseline_a_out)):
        if v != int(baseline_a_out) and v not in out:
            out.append(v)
    return out


def rank_coordinates(per_neuron: dict[tuple[int, int], int], relu_a: dict[int, np.ndarray], bounds) -> list[tuple]:
    """Spec section 7 ranking over (layer, neuron): max observed ReLU slack desc, |a| (u - l) desc,
    layer desc, neuron asc. Only I-S eligible coordinates are returned, as (slack, width_term, layer, neuron)."""

    entries = []
    for (layer, neuron), slack in per_neuron.items():
        lo, hi, sat, clip = bounds[layer]
        a = int(relu_a[layer][neuron])
        if not slope_eligible(int(lo[neuron]), int(hi[neuron]), bool(sat[neuron]), bool(clip[neuron]), a):
            continue
        entries.append((int(slack), abs(a) * (int(hi[neuron]) - int(lo[neuron])), layer, neuron))
    entries.sort(key=lambda e: (-e[0], -e[1], -e[2], e[3]))
    return entries


class SlopeOverride:
    """One immutable keyed a_out override (layer, row, neuron, expected_a, a_out), spec section 8 I-S.

    Wraps finder.relu_choice in-process for the duration of a with-block, so run_chain() and needs(),
    which both look the function up in the finder module, see the same choice. The finder file is not
    changed. The layer is identified by the identity of its bounds tuple, so pass the very list whose
    element is `layer_bounds`. The incoming coefficient must equal expected_a, else ImplementationError."""

    def __init__(self, finder, layer_bounds, row: int, neuron: int, expected_a: int, a_out: int):
        self.finder, self.layer_bounds = finder, layer_bounds
        self.key = (int(row), int(neuron), int(expected_a), int(a_out))
        self.applied = 0

    def __enter__(self):
        finder, target = self.finder, self.layer_bounds
        row, neuron, expected_a, a_out = self.key
        original = finder.relu_choice
        self._original = original

        def relu_choice(A, bounds, policy=None):
            Ap, D = original(A, bounds, policy)
            if bounds is not target:
                return Ap, D
            if int(A[row, neuron]) != expected_a:
                raise ImplementationError(f"incoming a {int(A[row, neuron])} != expected {expected_a}")
            l, u, saturated, clipped = bounds
            if bool(saturated[neuron]) or bool(clipped[neuron]):
                raise ImplementationError("override on a saturated or clipped coordinate")
            Ap, D = Ap.copy(), D.copy()
            Ap[row, neuron] = a_out
            D[row, neuron] = int(finder.relu_const(np.array([[expected_a]], dtype=np.int64),
                                                   np.array([[a_out]], dtype=np.int64),
                                                   l[neuron:neuron + 1], u[neuron:neuron + 1])[0, 0])
            self.applied += 1
            return Ap, D

        self._patched = relu_choice
        finder.relu_choice = relu_choice
        return self

    def __exit__(self, *exc):
        if self.finder.relu_choice is not self._patched:
            raise RuntimeError("relu_choice was replaced inside an override block")
        self.finder.relu_choice = self._original


def cone_size(mats, layer: int, neuron: int) -> dict[str, Any]:
    """Exact ancestor cone of one pre-activation: nonzero multiply terms, neurons per layer, input bytes."""

    need = np.array([int(neuron)])
    terms, sizes = 0, {}
    for j in range(int(layer), -1, -1):
        sizes[j] = int(need.size)
        sub = mats[j][0].tocsr()[need]
        terms += int(sub.nnz)
        need = np.unique(sub.indices)
    return {"multiply_terms": terms, "neurons_by_layer": {str(k): v for k, v in sizes.items()},
            "source_inputs": int(need.size)}


# ---------------------------------------------------------------- D-T exact accounting

def account_chain(steps: list[dict], row: int, lam: np.ndarray, trace: list[dict], x: np.ndarray,
                  xl: np.ndarray, xu: np.ndarray, biases: list[np.ndarray], scaled_bound: int,
                  coordinate_sink: Callable[[dict], None] | None = None) -> dict:
    """Spec section 7: exact fixed-chain, fixed-witness slack accounting for one recorded chain row.

    steps come from crown_chain_export_v2.run_chain(record=True). Raises ImplementationError on any
    negative slack, recorded-constant mismatch or identity failure."""

    records, total = [], 0
    top = len(trace) - 1
    for n, s in enumerate(steps):
        kind = s["kind"]
        a = np.asarray(s["a"][row], dtype=np.int64)
        rec = {"step_index": n, "kind": kind, "coefficient_l1": dec(sum(abs(int(v)) for v in a.tolist())),
               "nonzero_coefficients": int(np.count_nonzero(a)), "a_hash": sha256_array(a)}
        if kind == "affine":
            j = int(s["j"])
            a_out = np.asarray(s["a_out"][row], dtype=np.int64)
            R = np.asarray(s["R"][row], dtype=np.int64)
            hlow = np.asarray(s["hlow"], dtype=np.int64)
            h_prev = np.asarray(x if j == 0 else trace[j - 1]["h"], dtype=np.int64)
            r = trace[j]["r"]
            e = trace[j]["e"]
            c_in, c_out = int(s["c_in"][row]), int(s["c_out"][row])
            if R.min(initial=0) < 0 or R.max(initial=0) >= 1 << F:
                raise ImplementationError(f"step {n}: residual outside [0, 256)")
            P = (1 << (F - 1)) * sum(abs(int(v)) for v in a.tolist())
            B = idot(R, hlow) - P
            q = B // (1 << F)
            rho = B - (q << F)
            bias_dot = idot(a, biases[j])
            if c_out != c_in + bias_dot + q:
                raise ImplementationError(f"step {n}: recorded output constant != c_in + a.b + floor(B/256)")
            slack = idot(a, r) - idot(a_out, h_prev) - bias_dot - q
            res = idot(R, h_prev - hlow)
            rnd = idot(a, e) + P
            if res < 0 or rnd < 0 or not 0 <= rho < (1 << F):
                raise ImplementationError(f"step {n}: negative affine slack component ({res}, {rnd}, {rho})")
            if (1 << F) * slack != res + rnd + rho:
                raise ImplementationError(f"step {n}: 256 * affine_slack != residual + rounding + floor terms")
            rec.update(layer=j, a_out_hash=sha256_array(a_out), c_in=dec(c_in), c_out=dec(c_out),
                       bias_dot=dec(bias_dot), residual_dot=dec(idot(R, hlow)), rounding_term=dec(P), B=dec(B),
                       q=dec(q), rho=dec(rho), slack=dec(slack), residual_endpoint_slack_num=dec(res),
                       rounding_envelope_slack_num=dec(rnd), constant_floor_slack_num=dec(rho), input_slack=None)
            if coordinate_sink is not None:
                for i in np.nonzero(a)[0].tolist():
                    coordinate_sink({"step_index": n, "layer": j, "neuron": i, "a": dec(a[i]), "a_out": None,
                                     "raw_r": dec(r[i]), "clamped_z": dec(trace[j]["z"][i]),
                                     "activation_h": dec(trace[j]["h"][i]), "accumulator_s": dec(trace[j]["s"][i]),
                                     "rounding_e": dec(e[i]), "residual_coefficient": None,
                                     "residual_endpoint_slack_num": None,
                                     "rounding_envelope_slack_num": dec(int(a[i]) * int(e[i]) + 128 * abs(int(a[i])))})
                for i in np.nonzero(R)[0].tolist():
                    coordinate_sink({"step_index": n, "layer": j - 1, "neuron": i, "a": None, "a_out": dec(a_out[i]),
                                     "activation_h": dec(h_prev[i]), "lower": dec(hlow[i]),
                                     "residual_coefficient": dec(R[i]),
                                     "residual_endpoint_slack_num": dec(int(R[i]) * (int(h_prev[i]) - int(hlow[i])))})
        elif kind == "relu":
            j = int(s["j"])
            a_out = np.asarray(s["a_out"][row], dtype=np.int64)
            d = np.asarray(s["d"][row], dtype=np.int64)
            h, r = trace[j]["h"], trace[j]["r"]
            c_in, c_out = int(s["c_in"][row]), int(s["c_out"][row])
            if c_out != c_in + sum(int(v) for v in d.tolist()):
                raise ImplementationError(f"step {n}: recorded output constant != c_in + sum d")
            support = np.nonzero((a != 0) | (a_out != 0) | (d != 0))[0]
            per = {int(i): int(a[i]) * int(h[i]) - int(a_out[i]) * int(r[i]) - int(d[i]) for i in support.tolist()}
            negative = [i for i, v in per.items() if v < 0]
            if negative:
                raise ImplementationError(f"step {n}: negative ReLU slack at layer {j} neurons {negative[:5]}")
            slack = sum(per.values())
            rec.update(layer=j, a_out_hash=sha256_array(a_out), c_in=dec(c_in), c_out=dec(c_out), slack=dec(slack),
                       input_slack=None, max_coordinate_slack=dec(max(per.values(), default=0)))
            rec["_per_coordinate"] = per
            if coordinate_sink is not None:
                for i in support.tolist():
                    coordinate_sink({"step_index": n, "layer": j, "neuron": i, "a": dec(a[i]), "a_out": dec(a_out[i]),
                                     "raw_r": dec(r[i]), "clamped_z": dec(trace[j]["z"][i]),
                                     "activation_h": dec(h[i]), "accumulator_s": dec(trace[j]["s"][i]),
                                     "rounding_e": dec(trace[j]["e"][i]), "d": dec(d[i]), "relu_slack": dec(per[i])})
        else:
            c_in = int(s["c_in"][row])
            low = idot(np.maximum(a, 0), xl) + idot(np.minimum(a, 0), xu)
            if c_in + low != int(scaled_bound):
                raise ImplementationError("concretize: c_in + corner sum != recorded scaled bound")
            slack = idot(a, x) - low
            if slack < 0:
                raise ImplementationError("concretize: negative input slack (witness outside the box?)")
            rec.update(layer=None, c_in=dec(c_in), slack=dec(slack), input_slack=dec(slack))
        total += int(rec["slack"])
        records.append(rec)
    lam_value = idot(lam, trace[top]["r"])
    identity = total == lam_value - int(scaled_bound)
    if not identity:
        raise ImplementationError("sum of step slacks != 65536 * actual raw margin - scaled bound")
    bound = ceil_div(scaled_bound, 1 << SCALE_BITS)
    return {"steps": records, "total_slack": dec(total), "lambda_dot_raw": dec(lam_value),
            "scaled_bound": dec(scaled_bound), "raw_lb": bound,
            "final_rounding_credit": rational(bound * (1 << SCALE_BITS) - int(scaled_bound), 1 << SCALE_BITS),
            "accounting_identity_passed": True}


# ---------------------------------------------------------------- observer C

_BIAS_LINE = re.compile(r"\) \+ \(__int128\)LAYER_(\d+)_BIAS\[")
_CLAMP_LINE = re.compile(r"^\s*value = clamp_to_signed_range_i128\(value, \d+\);\s*$")
_STORE_LINE = re.compile(r"^(\s*)buffer_\d+\[(.+)\] = \(int64_t\)value;\s*$")
_OBS_MARK = "/*QNN_OBS*/"


def observer_source(source: str, layer_sizes: list[int]) -> str:
    """Add observer stores of s, r, z, h after the deployed arithmetic, changing nothing else.
    Every added line carries _OBS_MARK; strip_observer() must give back the input exactly."""

    offsets = np.concatenate([[0], np.cumsum(layer_sizes)]).astype(int)
    prelude = [
        f"{_OBS_MARK}static int64_t qnn_obs_s[{offsets[-1]}], qnn_obs_r[{offsets[-1]}], qnn_obs_z[{offsets[-1]}], "
        f"qnn_obs_h[{offsets[-1]}];",
        f"{_OBS_MARK}static int qnn_obs_overflow = 0;",
        f"{_OBS_MARK}static const int qnn_obs_offset[{len(offsets)}] = {{{', '.join(map(str, offsets))}}};",
        f"{_OBS_MARK}static int64_t qnn_obs_i64(__int128 v) {{ if (v > (__int128)INT64_MAX || v < (__int128)INT64_MIN) "
        f"qnn_obs_overflow = 1; return (int64_t)v; }}",
        f"{_OBS_MARK}static void qnn_observe(int layer, int index, __int128 s, __int128 r, __int128 z, __int128 h) {{ "
        f"int k = qnn_obs_offset[layer] + index; qnn_obs_s[k] = qnn_obs_i64(s); qnn_obs_r[k] = qnn_obs_i64(r); "
        f"qnn_obs_z[k] = qnn_obs_i64(z); qnn_obs_h[k] = qnn_obs_i64(h); }}",
        f"{_OBS_MARK}int qnn_observed(int64_t *s, int64_t *r, int64_t *z, int64_t *h) {{ for (int k = 0; k < "
        f"{offsets[-1]}; ++k) {{ s[k] = qnn_obs_s[k]; r[k] = qnn_obs_r[k]; z[k] = qnn_obs_z[k]; h[k] = qnn_obs_h[k]; }} "
        f"int o = qnn_obs_overflow; qnn_obs_overflow = 0; return o; }}",
    ]
    out, state, layer, seen = [], 0, None, []
    lines = source.split("\n")
    anchor = next(i for i, line in enumerate(lines) if line.startswith("int qnn_input_dim"))
    for i, line in enumerate(lines):
        if i == anchor:
            out.extend(prelude)
        out.append(line)
        match = _BIAS_LINE.search(line)
        if match and state == 0:
            layer, state = int(match.group(1)), 1
            out.append(f"{_OBS_MARK}        __int128 qnn_obs_rv = value;")
        elif state == 1 and _CLAMP_LINE.match(line):
            out.append(f"{_OBS_MARK}        __int128 qnn_obs_zv = value;")
            state = 2
        elif state == 2 and _STORE_LINE.match(line):
            store = _STORE_LINE.match(line)
            out.pop()
            out.append(f"{_OBS_MARK}{store.group(1)}qnn_observe({layer}, {store.group(2)}, acc, qnn_obs_rv, "
                       f"qnn_obs_zv, value);")
            out.append(line)
            seen.append(layer)
            state = 0
    if seen != list(range(len(layer_sizes))):
        raise ImplementationError(f"observer anchors found for layers {seen}, expected {len(layer_sizes)}")
    if "#include <stdint.h>" not in source:
        raise ImplementationError("deployment source lacks stdint.h")
    return "\n".join(out)


def strip_observer(source: str) -> str:
    return "\n".join(line for line in source.split("\n") if not line.startswith(_OBS_MARK))


class COracle:
    """ctypes binding of the pinned C: qnn_encode_bytes, qnn_forward_fixed and (observer builds) qnn_observed."""

    def __init__(self, library: Path, n_input: int, n_output: int, layer_sizes: list[int] | None = None):
        import ctypes
        self.ct = ctypes
        self.lib = ctypes.CDLL(str(library))
        self.n_input, self.n_output = n_input, n_output
        self.layer_sizes = layer_sizes
        i64p = ctypes.POINTER(ctypes.c_int64)
        self.lib.qnn_encode_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, i64p]
        self.lib.qnn_encode_bytes.restype = ctypes.c_int
        self.lib.qnn_forward_fixed.argtypes = [i64p, i64p]
        self.lib.qnn_forward_fixed.restype = None
        if layer_sizes is not None:
            self.lib.qnn_observed.argtypes = [i64p, i64p, i64p, i64p]
            self.lib.qnn_observed.restype = ctypes.c_int

    def _ptr(self, array, kind):
        return array.ctypes.data_as(self.ct.POINTER(kind))

    def encode(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image, dtype=np.uint8)
        out = np.zeros(self.n_input, dtype=np.int64)
        status = self.lib.qnn_encode_bytes(self._ptr(image, self.ct.c_uint8), image.shape[0], image.shape[1],
                                           self._ptr(out, self.ct.c_int64))
        if status != 0:
            raise ImplementationError(f"qnn_encode_bytes returned {status}")
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.int64)
        out = np.zeros(self.n_output, dtype=np.int64)
        self.lib.qnn_forward_fixed(self._ptr(x, self.ct.c_int64), self._ptr(out, self.ct.c_int64))
        return out

    def observed(self) -> list[dict[str, np.ndarray]]:
        total = int(sum(self.layer_sizes))
        arrays = [np.zeros(total, dtype=np.int64) for _ in range(4)]
        overflow = self.lib.qnn_observed(*(self._ptr(a, self.ct.c_int64) for a in arrays))
        if overflow:
            raise OverflowGuard("observer value outside int64")
        offsets = np.concatenate([[0], np.cumsum(self.layer_sizes)])
        return [{k: arrays[n][offsets[i]:offsets[i + 1]].copy() for n, k in enumerate("srzh")}
                for i in range(len(self.layer_sizes))]


# ---------------------------------------------------------------- ESBMC harnesses and parsing

def parse_violation(log: str, label: str) -> dict:
    """Classify a FAILED run by the property ESBMC reports as violated.

    TARGET: the named diagnostic assertion; UNWINDING: an unwinding assertion; SAFETY: any other
    property (bounds, pointer, division, ...). Only TARGET is evidence about the asserted claim."""

    lines = log.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().startswith("Violated property"))
    except StopIteration:
        return {"failed_claim_kind": "UNPARSED", "violated_property_text": None}
    block = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        block.append(line.strip())
    text = "\n".join(block)
    if any(line == label for line in block):
        kind = "TARGET"
    elif "unwinding assertion" in text:
        kind = "UNWINDING"
    else:
        kind = "SAFETY"
    return {"failed_claim_kind": kind, "violated_property_text": text}


def _control(name: str, body: str, expected: str) -> dict:
    from verification.arith_kernel import render_arith_kernel
    label = f"diag/controls/{name}/claim"
    source = ("#include <stdint.h>\n" + render_arith_kernel() + "\nint nondet_int(void);\n"
              "int main(void) {\n" + body.replace("LABEL", label) + "  return 0;\n}\n")
    return {"name": name, "source": source, "label": label, "expected": expected}


def solver_control_harnesses() -> list[dict]:
    """Spec section 9 compact controls: status and assertion-identity parsing only."""

    rounding = ("  int s = nondet_int();\n  __ESBMC_assume(s >= -1024 && s <= 1024);\n"
                "  __int128 e = 256 * div_round_half_away_from_zero_i128(s, 256) - s;\n")
    tie = "  __int128 r = div_round_half_away_from_zero_i128(S, 256);\n"
    pair = "  int x = nondet_int();\n  __ESBMC_assume(x >= 0 && x <= 3);\n  int y1 = x, y2 = 2 * x;\n"
    logits = "  int64_t out[2] = {T, C};\n"
    return [
        _control("rounding_true", rounding + '  __ESBMC_assert(e >= -128 && e <= 128, "LABEL");\n', "VERIFIED"),
        _control("rounding_false", rounding + '  __ESBMC_assert(e > -128 && e < 128, "LABEL");\n', "FAILED"),
        _control("tie_pos_128", tie.replace("S", "128") + '  __ESBMC_assert(r == 1, "LABEL");\n', "VERIFIED"),
        _control("tie_neg_128", tie.replace("S", "-128") + '  __ESBMC_assert(r == -1, "LABEL");\n', "VERIFIED"),
        _control("tie_neg_128_false", tie.replace("S", "-128") + '  __ESBMC_assert(r == 0, "LABEL");\n', "FAILED"),
        _control("relation_valid", pair + '  __ESBMC_assert(y2 - y1 >= 0, "LABEL");\n', "VERIFIED"),
        _control("relation_invalid", pair + '  __ESBMC_assert(y2 - y1 >= 1, "LABEL");\n', "FAILED"),
        _control("target_dominance", logits.replace("T", "6").replace("C", "5") +
                 '  __ESBMC_assert(out[0] > out[1], "LABEL");\n', "VERIFIED"),
        _control("tie_is_violation", logits.replace("T", "5").replace("C", "5") +
                 '  __ESBMC_assert(out[0] > out[1], "LABEL");\n', "FAILED"),
    ]


_FOR_LITERAL = re.compile(r"for\s*\(\s*int\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(\d+)\s*;")
_FOR_ANY = re.compile(r"for\s*\(")
_OFFSET = re.compile(r"static const int (LAYER_\d+_OFFSET)\[\d+\] = \{([^}]*)\};")


def loop_inventory(source: str) -> dict:
    """Conservative maximum iteration count over every loop of the generated C (spec section 9)."""

    loops, unresolved = [], []
    offsets = {m.group(1): [int(v) for v in m.group(2).split(",")] for m in _OFFSET.finditer(source)}
    for number, line in enumerate(source.splitlines(), 1):
        if not _FOR_ANY.search(line):
            continue
        literal = _FOR_LITERAL.search(line)
        if literal:
            loops.append({"line": number, "bound": int(literal.group(1)), "kind": "literal"})
            continue
        name = next((n for n in offsets if n in line), None)
        if name is not None:
            widths = [b - a for a, b in zip(offsets[name], offsets[name][1:])]
            loops.append({"line": number, "bound": max(widths, default=0), "kind": f"csr {name}"})
            continue
        unresolved.append({"line": number, "text": line.strip()})
    # CSR term loops span two lines ("for (int term = ...;" then the bound); match them by offsets
    for name, values in offsets.items():
        widths = [b - a for a, b in zip(values, values[1:])]
        if not any(l["kind"] == f"csr {name}" for l in loops):
            loops.append({"line": None, "bound": max(widths, default=0), "kind": f"csr {name}"})
    unresolved = [u for u in unresolved if "term" not in u["text"]]
    return {"loops": loops, "unresolved": unresolved,
            "max_iterations": max((l["bound"] for l in loops), default=0)}


def witness_replay_harness(deployed_source: str, image_bytes: np.ndarray, target: int, competitor: int,
                           label: str) -> tuple[str, dict]:
    """Pinned deployed C plus a fixed-byte main asserting strict dominance of the deployed logits."""

    image_bytes = np.asarray(image_bytes, dtype=np.uint8)
    if image_bytes.ndim != 3:
        raise ValueError("witness must be HWC bytes")
    inventory = loop_inventory(deployed_source)
    if inventory["unresolved"]:
        raise ImplementationError(f"loops without a derived bound: {inventory['unresolved'][:3]}")
    h, w, _ = image_bytes.shape
    n_input = int(re.search(r"int qnn_input_dim\(void\) \{ return (\d+); \}", deployed_source).group(1))
    n_output = int(re.search(r"int qnn_output_dim\(void\) \{ return (\d+); \}", deployed_source).group(1))
    data = ", ".join(str(int(v)) for v in image_bytes.reshape(-1))
    main = f"""
#define INPUT_SIZE {inventory['max_iterations']}
static const uint8_t DIAG_WITNESS[{image_bytes.size}] = {{{data}}};
int main(void) {{
    int64_t x[{n_input}];
    int64_t out[{n_output}];
    int status = qnn_encode_bytes(DIAG_WITNESS, {h}, {w}, x);
    __ESBMC_assert(status == 0, "{label.rsplit('/', 1)[0]}/encoder_status");
    qnn_forward_fixed(x, out);
    __ESBMC_assert(out[{int(target)}] > out[{int(competitor)}], "{label}");
    return 0;
}}
"""
    return deployed_source + main, inventory


# ---------------------------------------------------------------- records and resources

class RunLog:
    """Append-only runs.jsonl writer enforcing the common envelope and the evidence rules."""

    def __init__(self, path: Path, context: dict):
        self.path, self.context, self.ids = path, context, set()
        if path.exists():
            self.ids = {json.loads(line)["run_id"] for line in path.read_text().splitlines() if line}

    def write(self, **fields) -> dict:
        record = {"schema": SCHEMA, "parent_run_id": None, "competitor": None, "epsilon_raw_bytes": None,
                  "domain": None, "region_sha256": None, "assumption_context_sha256": None,
                  "status": "COMPLETED", "solver_status": "NOT_RUN", "dependency_status": None,
                  "parameters": {}, "metric_scope": "this_probe", "missing_reason": {}, "artifacts": {},
                  **self.context, **fields}
        metrics = {k: None for k in METRIC_FIELDS}
        metrics.update(record.get("metrics") or {})
        record["metrics"] = metrics
        validate_envelope(record)
        if record["run_id"] in self.ids:
            raise ImplementationError(f"run_id {record['run_id']} already written; reruns need a fresh id")
        self.ids.add(record["run_id"])
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record


REQUIRED = ("schema", "run_id", "parent_run_id", "diagnostic", "ordinal", "image_id", "target", "competitor",
            "epsilon_raw_bytes", "domain", "region_sha256", "assumption_context_sha256", "deployment_sha256",
            "source_manifest_sha256", "scale_bits", "status", "evidence_kind", "solver_status", "dependency_status",
            "parameters", "metrics", "metric_scope", "missing_reason", "artifacts", "interpretation")


def validate_envelope(record: dict) -> None:
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ImplementationError(f"envelope fields missing: {missing}")
    if record["evidence_kind"] not in EVIDENCE_KINDS:
        raise ImplementationError(f"unknown evidence_kind {record['evidence_kind']}")
    if record["solver_status"] not in SOLVER_STATUSES:
        raise ImplementationError(f"unknown solver_status {record['solver_status']}")
    if record["status"] not in RUN_STATUSES:
        raise ImplementationError(f"unknown status {record['status']}")
    if record["solver_status"] != "NOT_RUN" and record["evidence_kind"] not in ESBMC_EVIDENCE:
        raise ImplementationError("a solver status requires an ESBMC evidence kind")
    if record["evidence_kind"] in ESBMC_EVIDENCE and record["solver_status"] == "NOT_RUN":
        raise ImplementationError("ESBMC evidence without a solver run")
    interpretation = record["interpretation"]
    if interpretation.get("assessment") not in ASSESSMENTS:
        raise ImplementationError(f"unknown assessment {interpretation.get('assessment')}")
    nulls = [k for k in REQUIRED if record[k] is None] + \
            [f"metrics.{k}" for k, v in record["metrics"].items() if v is None]
    unexplained = [k for k in nulls if k not in record["missing_reason"]]
    if unexplained:
        raise ImplementationError(f"null fields without missing_reason: {unexplained}")


def null_reasons(fields: Iterable[str], reason: str) -> dict[str, str]:
    return {f: reason for f in fields}


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def _tree_rss(root: int) -> int:
    total, stack, seen = 0, [root], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            stack.extend(int(c) for c in children)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def cgroup_memory_file(name: str) -> Path | None:
    try:
        line = next(l for l in Path("/proc/self/cgroup").read_text().splitlines() if l.startswith("0::"))
    except (StopIteration, FileNotFoundError):
        return None
    path = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/") / name
    return path if path.exists() else None


class ResourceMonitor:
    """Poll process-tree RSS every 50 ms; report cgroup memory.peak when the kernel exposes it."""

    def __init__(self, interval: float = 0.05):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _tree_rss(os.getpid()))
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t0 = time.monotonic()
        self.cpu0 = _cpu()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()
        self.wall = time.monotonic() - self.t0
        self.cpu = _cpu() - self.cpu0

    def metrics(self) -> tuple[dict, dict]:
        peak_file = cgroup_memory_file("memory.peak")
        metrics = {"wall_seconds": round(self.wall, 3), "cpu_seconds": round(self.cpu, 3),
                   "peak_process_tree_rss_bytes": self.peak or None,
                   "peak_cgroup_memory_bytes": int(peak_file.read_text()) if peak_file else None}
        reasons = {}
        if not self.peak:
            reasons["metrics.peak_process_tree_rss_bytes"] = "no RSS sample collected"
        if peak_file is None:
            reasons["metrics.peak_cgroup_memory_bytes"] = "cgroup memory.peak not exposed by this kernel/cgroup"
        return metrics, reasons


def _cpu() -> float:
    import resource
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


class GzipJsonl:
    """Streaming gzip JSON-lines writer that counts uncompressed bytes."""

    def __init__(self, path: Path, cap_bytes: int | None = None):
        self.path, self.cap, self.bytes, self.rows = path, cap_bytes, 0, 0
        self.fh = gzip.open(path, "at", compresslevel=6)

    def write(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        self.bytes += len(line)
        if self.cap is not None and self.bytes > self.cap:
            raise OverflowGuard(f"{self.path.name} exceeds its artifact cap")
        self.fh.write(line)
        self.rows += 1

    def close(self):
        self.fh.close()
