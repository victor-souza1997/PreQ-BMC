"""Dependency-complete CROWN chain certificate export (schema crown_chain_certificate_v2).

UNTRUSTED FINDER. Exact integer arithmetic; ESBMC re-checks every step.

One signed format Q = [Q_LOW, Q_HIGH] with F fractional bits for the input and every
layer (--bits/--frac, default Q16/I7/F8, the pinned deployment).
Variables: r_k is the RAW pre-activation (exact affine value), z_k = clamp_q(r_k),
h_k = relu(z_k). Chains bound r. A ReLU step uses h_j >= / <= relations to r_j; per
nonzero coordinate (a = input coefficient, a' = output coefficient) it needs:
  a > 0, a' = 0          bound_free     a h >= 0 always
  a > 0, a' = a          nosat_upper    h >= r needs r <= Q_HIGH (h = Q_HIGH < r otherwise)
  a < 0, a' = a          lower          h - r <= max(-l, 0) for r >= l (saturation only lowers h)
  a < 0, a' = 0          upper          h <= max(u, 0) for r <= u
  otherwise              lower_upper    l <= r <= u inside Q, so z = r
  any a, a' = 0          saturated_top  r >= l >= Q_HIGH, so h = Q_HIGH and a h = a Q_HIGH
  any a, a' (u > Q_HIGH)  clipped       l <= r <= u, h = clamp(r, 0, Q_HIGH) for every integer r
Affine residual folding uses h >= min(relu(l), Q_HIGH); it needs the lower fact only when l > 0.
Facts are one-sided, one source each: an ESBMC box block (box_ledger.py) or a chain.

Usage:
  crown_chain_export_v2.py screen <eps> <image> [--report path]
  crown_chain_export_v2.py export <eps> <image> <out_dir> --ledger proof_summary.json [--protocol p.json]
<image> is a validation index or a record id (Final_Test/Images/NNNNN.ppm). screen needs no
ledger: it takes the facts a VERIFIED box ledger would yield from the propagated box and
only predicts whether export will close. It never writes a certificate.
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import affine_export  # noqa: E402
import box_ledger  # noqa: E402
from box_ledger import facts_from_box, facts_from_ledger, load_ledger  # noqa: E402
from crown_chain_export import ENC, S, XU, _check_headroom, ceil_div, choose_relu, relu_const, setup  # noqa: E402
from backends.conv_fixed_point import generate_conv_qnn_source  # noqa: E402

DEPLOYMENT_Q16 = "dd535f5480bec45cdae7774f849afca49f7afb5d645f02037e86a44d16f45097"
CLASSES = ("bound_free", "nosat_upper", "lower", "upper", "lower_upper", "saturated_top", "clipped")
BITS, F, HALF, Q_LOW, Q_HIGH = 16, 8, 1 << 7, -(1 << 15), (1 << 15) - 1


def configure_format(bits, frac):
    """One signed format for the input and every layer, set in every finder module before setup()."""
    global BITS, F, HALF, Q_LOW, Q_HIGH
    if frac < 1:
        raise ValueError("the rounding term needs at least one fractional bit")
    affine_export.configure(bits, frac)
    BITS, F, HALF = bits, frac, 1 << (frac - 1)
    Q_LOW, Q_HIGH = affine_export.Q_LOW, affine_export.Q_HIGH
    box_ledger.Q_LOW, box_ledger.Q_HIGH = Q_LOW, Q_HIGH


def classify(A, Ap, saturated=None, clipped=None):
    """Per coordinate class index into CLASSES (-1 where A == 0)."""
    cls = np.full(A.shape, 4, dtype=np.int8)
    cls[(A > 0) & (Ap == 0)] = 0
    cls[(A > 0) & (Ap == A)] = 1
    cls[(A < 0) & (Ap == A)] = 2
    cls[(A < 0) & (Ap == 0)] = 3
    if saturated is not None:
        cls[np.broadcast_to(saturated, A.shape)] = 5
    if clipped is not None:
        cls[np.broadcast_to(clipped, A.shape)] = 6
    cls[A == 0] = -1
    return cls


def clipped_const(A, Ap, l, u):
    """min of a clamp(z, 0, Q_HIGH) - a' z over [l, u]: at l, u, or a breakpoint 0, Q_HIGH inside."""
    def f(z):
        return A * np.clip(z, 0, Q_HIGH) - Ap * z
    d = np.minimum(f(l), f(u))
    d = np.where((l < 0) & (u > 0), np.minimum(d, 0), d)
    return np.where((l < Q_HIGH) & (u > Q_HIGH), np.minimum(d, f(np.full_like(l, Q_HIGH))), d)


ADAPTIVE = -1
POLICY_ORDER = (ADAPTIVE, 0, 1, 2, 3, 4)


def relu_choice(A, bounds, policy=None):
    """CROWN z-coefficients and constants. A saturated neuron is the constant h = Q_HIGH; a
    clipped neuron (u > Q_HIGH) takes the best of a' in {0, a, chord} by the area score.

    policy: optional per-row int array. ADAPTIVE (-1) keeps the coordinatewise heuristic;
    p in 0..4 sets a' = (p a + 2) // 4 on every unstable (l < 0 < u), unsaturated, unclipped
    coordinate with a > 0. Everything else keeps the heuristic. The checker re-derives each
    coordinate's class from (a, a') alone, so no schema change is needed."""
    Ap, D = _relu_choice_adaptive(A, bounds)
    if policy is None:
        return Ap, D
    policy = np.asarray(policy, dtype=np.int64)
    rows = policy >= 0
    if not rows.any():
        return Ap, D
    l, u, saturated, clipped = bounds
    free = (l < 0) & (u > 0) & ~saturated & ~clipped
    mask = rows[:, None] & free[None, :] & (A > 0)
    scalar = (policy[:, None] * A + 2) // 4
    Ap = np.where(mask, scalar, Ap)
    D = np.where(mask, relu_const(A, Ap, l, u), D)
    return Ap, D


def _relu_choice_adaptive(A, bounds):
    l, u, saturated, clipped = bounds
    # u <= 0 gives h = 0 even when l = u = 0, so a' = 0 needs no lower side
    Ap = np.where(saturated | (u <= 0), 0, choose_relu(A, l, u))
    D = np.where(saturated, A * Q_HIGH, relu_const(A, Ap, l, u))
    cols = np.nonzero(clipped)[0]
    if cols.size:
        a, lc, uc = A[:, cols], l[cols], u[cols]
        rise, span = np.clip(uc, 0, Q_HIGH) - np.clip(lc, 0, Q_HIGH), uc - lc
        chord = np.floor_divide(2 * a * rise + span, 2 * span)
        best_ap, best_d, best = None, None, None
        for candidate in (np.zeros_like(a), a, chord):
            d = clipped_const(a, candidate, lc, uc)
            score = candidate * (lc + uc) + 2 * d          # twice the mean of a' z + d over [l, u]
            if best is None:
                best_ap, best_d, best = candidate, d, score
            else:
                take = score > best
                best_ap, best_d, best = (np.where(take, candidate, best_ap), np.where(take, d, best_d),
                                         np.maximum(score, best))
        Ap[:, cols], D[:, cols] = best_ap, best_d
    return Ap, D


def run_chain(A, k, mats, bounds, xl, record=False, policies=None):
    """Batched exact chain over rows of A @ r_k. Returns scaled bounds, terms, per-step dense records.

    policies: optional (rows, hidden layers) int array; column j is the ReLU policy at layer j."""
    A = A.astype(np.int64)
    C = np.zeros(A.shape[0], dtype=np.int64)
    terms = np.zeros(A.shape[0], dtype=np.int64)
    steps = []
    for j in range(k, -1, -1):
        w, b = mats[j]
        _check_headroom(w, A)
        G = np.asarray((w.T @ A.T).T, dtype=np.int64)
        An = np.floor_divide(G, 1 << F)
        R = G - (An << F)
        hlow = xl if j == 0 else np.clip(bounds[j - 1][0], 0, Q_HIGH)
        terms += (A != 0).astype(np.int64) @ np.diff(w.indptr)
        C_in = C
        C = C + A @ b + np.floor_divide(R @ hlow - HALF * np.abs(A).sum(1), 1 << F)
        if record:
            absw = abs(w).astype(np.int64)
            steps.append({"kind": "affine", "j": j, "a": A, "c_in": C_in, "a_out": An, "R": R, "hlow": hlow,
                          "c_out": C, "g_env": np.asarray((absw.T @ np.abs(A).T).T).max(1),
                          "max_prod": np.array([int(abs(w).max()) * int(np.abs(r).max(initial=0)) for r in A]),
                          "bias_env": np.abs(A) @ np.abs(b), "res_env": R @ np.abs(hlow)})
        A = An
        if j == 0:
            break
        Ap, D = relu_choice(A, bounds[j - 1], None if policies is None else policies[:, j - 1])
        C_in = C
        C = C + D.sum(1)
        if record:
            steps.append({"kind": "relu", "j": j - 1, "a": A, "c_in": C_in, "a_out": Ap, "d": D, "c_out": C})
        A = Ap
    scaled = C + np.maximum(A, 0) @ xl + np.minimum(A, 0) @ XU[0]
    if record:
        steps.append({"kind": "concretize", "a": A, "c_in": C, "scaled": scaled})
    return scaled, terms, steps


def bounds_v2(mats, xl, facts, chunk=256):
    """Raw bounds per hidden layer from proved sources only. Lower: box iff STABLE_RELU = +1,
    else chain. Upper: the box when proved (hh < Q_HIGH), replaced by the chain where the chain
    is tighter on unstable neurons or the box upper is not proved. A stable-active neuron whose
    proved lower side reaches Q_HIGH is saturated (h = Q_HIGH) and needs no upper side."""
    bounds, info = [], []
    for k in range(len(mats) - 1):
        f = facts[k]
        n = mats[k][0].shape[0]
        lo_c, hi_c = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
        for s in range(0, n, chunk):
            rows = np.arange(s, min(s + chunk, n))
            eye = np.zeros((rows.size, n), dtype=np.int64)
            eye[np.arange(rows.size), rows] = 1 << S
            lo, _, _ = run_chain(eye, k, mats, bounds, xl)
            nhi, _, _ = run_chain(-eye, k, mats, bounds, xl)
            lo_c[rows], hi_c[rows] = ceil_div(lo, 1 << S), np.floor_divide(-nhi, 1 << S)
        lower_box = f["stable"] > 0
        if not f["lower_ok"][lower_box].all():
            raise ValueError(f"layer {k}: stable-active box lower not proved")
        lo = np.where(lower_box, f["hl"], lo_c)
        saturated = lower_box & (f["hl"] >= Q_HIGH)
        # an unproved box upper side (hh = Q_HIGH) is always replaced by the chain
        upper_chain = ~saturated & ((f["stable"] == 0) | ~f["upper_ok"]) & ((hi_c < f["hh"]) | ~f["upper_ok"])
        hi = np.where(upper_chain, hi_c, f["hh"])
        # the chain cannot exclude r > Q_HIGH: the clipped class covers clamp(r, 0, Q_HIGH) on [lo, hi]
        clipped = ~saturated & (hi > Q_HIGH)
        # a stable-inactive lower side is never used (h = 0), so only unstable lower sides must fit
        unstable = (f["stable"] == 0) & ~clipped
        plain = ~saturated & ~clipped
        if (lo[unstable] < Q_LOW).any() or (lo > hi)[plain | clipped].any():
            raise ValueError(f"layer {k}: bounds outside Q{BITS} or empty")
        bounds.append((lo, hi, saturated, clipped))
        info.append({"lo_chain": lo_c, "hi_chain": hi_c, "lower_box": lower_box, "upper_chain": upper_chain,
                     "saturated": saturated, "clipped": clipped})
    return bounds, info


def needs(A, k, mats, bounds, policies=None):
    """Per lower layer: [need_lower, need_upper_value, need_upper_nosat] over rows of A @ r_k.
    policies must be the same table run_chain() uses for these rows."""
    out = {}
    A = A.astype(np.int64)
    for j in range(k, 0, -1):
        w, _ = mats[j]
        G = np.asarray((w.T @ A.T).T, dtype=np.int64)
        An = np.floor_divide(G, 1 << F)
        R = G - (An << F)
        l, _, saturated, clipped = bounds[j - 1]
        Ap, _ = relu_choice(An, bounds[j - 1], None if policies is None else policies[:, j - 1])
        cls = classify(An, Ap, saturated, clipped)
        out[j - 1] = [(((cls == 2) | (cls == 4) | (cls == 5) | (cls == 6)).any(0)) | ((R != 0) & (l > 0)[None, :]).any(0),
                      ((cls == 3) | (cls == 4) | (cls == 6)).any(0), (cls == 1).any(0)]
        A = Ap
    return out


def search_policies(lams, top, mats, bounds, xl, deadline=None):
    """Bounded discrete search over integer ReLU coefficients, per root row (P3 pilot).

    Every row starts ADAPTIVE at every layer. Visit ReLU layers top-1 .. 0 once; at each
    layer try the five scalar policies with the other layers at their current choice. Switch
    only on a strict improvement of the exact scaled bound; among equal improvements take
    fewer multiply terms, then the lower policy. Intermediate bounds stay the baseline ones.
    Returns (policies, scaled, trace); trace is None for rows not finished before deadline."""
    rows = lams.shape[0]
    policies = np.full((rows, top), ADAPTIVE, dtype=np.int64)
    best, best_terms, _ = run_chain(lams, top, mats, bounds, xl, policies=policies)
    best, best_terms = best.copy(), best_terms.copy()
    trace = [[] for _ in range(rows)]
    finished = np.zeros(rows, bool)
    for r in range(rows):
        for layer in range(top - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            trial = np.repeat(policies[r][None], 5, axis=0)
            trial[:, layer] = np.arange(5)
            scaled, terms, _ = run_chain(np.repeat(lams[r][None], 5, axis=0), top, mats, bounds, xl,
                                         policies=trial)
            order = sorted(range(5), key=lambda p: (-int(scaled[p]), int(terms[p]), p))
            p = order[0]
            improved = int(scaled[p]) > int(best[r])
            trace[r].append({"layer": layer, "scaled": [int(v) for v in scaled], "incumbent": int(best[r]),
                             "chosen": int(p) if improved else int(policies[r, layer])})
            if improved:
                policies[r, layer], best[r], best_terms[r] = p, scaled[p], terms[p]
        else:
            finished[r] = True
    return policies, best, [t if f else None for t, f in zip(trace, finished)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("screen", "export"))
    ap.add_argument("eps", type=int)
    ap.add_argument("image")
    ap.add_argument("out_dir", nargs="?", type=Path)
    ap.add_argument("--ledger")
    ap.add_argument("--protocol", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--bits", type=int, default=16, help="signed width of the input and every layer")
    ap.add_argument("--frac", type=int, default=8, help="fractional bits of the input and every layer")
    ap.add_argument("--search", action="store_true",
                    help="P3 pilot: discrete ReLU-coefficient search on margin roots with margin < 1")
    ap.add_argument("--policy", type=Path,
                    help="export: reuse the margin-root policy table from a --search screen report")
    ap.add_argument("--deadline", type=float, help="unix time after which the search stops (NOT_RUN_PILOT_BUDGET)")
    ap.add_argument("--cost", action="store_true", help="screen: count terms and two-sided ReLU coordinates "
                    "over the whole planned dependency graph")
    args = ap.parse_args()
    eps, image, out_dir, screen = args.eps, args.image, args.out_dir, args.mode == "screen"
    if not screen and (out_dir is None or args.ledger is None):
        raise SystemExit("export needs <out_dir> and --ledger")
    t0 = time.time()
    configure_format(args.bits, args.frac)
    sel, row, net, mats, target, correct, xl, xu, box = setup(eps, image)
    deployment = hashlib.sha256(
        generate_conv_qnn_source(net, encoder_source=ENC[0].render_c()).encode()).hexdigest()
    if (BITS, F) == (16, 8) and deployment != DEPLOYMENT_Q16:
        raise ValueError("regenerated Q16 deployment C is not the pinned deployment")
    sizes = [m[0].shape[0] for m in mats]
    report = {"mode": args.mode, "image": row["id"], "split": row["split"], "epsilon_raw_bytes": eps,
              "qif": {"total_bits": BITS, "integer_bits": BITS - F - 1, "fractional_bits": F},
              "deployment_sha256": deployment, "target": target, "deployed_correct": bool(correct)}

    def finish(status, **extra):
        report.update(status=status, seconds=round(time.time() - t0, 1), **extra)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report), flush=True)

    if not correct:
        if screen:
            return finish("MISCLASSIFIED")
        raise ValueError("deployed prediction is not the target; nothing to certify")
    protocol = None
    if not screen:
        if row["split"] == "test" and args.protocol is None:
            raise SystemExit("a test-split export must cite the pre-registered protocol (--protocol)")
        if args.protocol is not None:
            protocol = {"path": str(args.protocol),
                        "sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest()}
    if screen:
        summary = blocks = None
        facts = facts_from_box(box, sizes)
    else:
        summary, blocks = load_ledger(args.ledger, len(mats))
        if summary["deployment_source_sha256"] != deployment:
            raise ValueError("ledger deployment mismatch")
        facts = facts_from_ledger(summary, blocks, sizes)
    # the ledger must equal the box this finder would otherwise have used
    for k in range(len(mats)):
        pl, ph = box.pre_low[k].astype(np.int64), box.pre_high[k].astype(np.int64)
        hidden = k < len(mats) - 1
        if not (np.array_equal(np.maximum(pl, 0) if hidden else pl, facts[k]["hl"])
                and np.array_equal(np.maximum(ph, 0) if hidden else ph, facts[k]["hh"])):
            raise ValueError(f"layer {k}: ledger interval != propagated box")
    try:
        bounds, info = bounds_v2(mats, xl, facts)
    except ValueError as error:
        if screen:
            return finish("FINDER_INCONCLUSIVE", cause=f"bounds: {error}")
        raise
    t_bounds = time.time() - t0

    top = len(mats) - 1
    n_out = sizes[-1]
    comps = [c for c in range(n_out) if c != target]
    roots = []  # (chain_id, claim, layer, lambda row)

    def unit(n, i, v):
        e = np.zeros(n, dtype=np.int64)
        e[i] = v
        return e

    for c in comps:
        lam = unit(n_out, target, 1 << S)
        lam[c] = -(1 << S)
        roots.append((f"chain/margin/{target}-{c}", "margin", top, lam))
    margin_lams = np.stack([r[3] for r in roots])
    margin_policies = np.full((len(roots), top), ADAPTIVE, dtype=np.int64)
    margin_scaled, _, _ = run_chain(margin_lams, top, mats, bounds, xl)
    baseline_margins = dict(zip(comps, ceil_div(margin_scaled, 1 << S).tolist()))
    search_report = None
    if args.search:
        open_rows = np.nonzero(ceil_div(margin_scaled, 1 << S) < 1)[0]
        t_search = time.time()
        if open_rows.size:
            pol, scaled, trace = search_policies(margin_lams[open_rows], top, mats, bounds, xl, args.deadline)
            margin_policies[open_rows] = pol
            margin_scaled = margin_scaled.copy()
            margin_scaled[open_rows] = scaled
        search_report = {
            "searched": [comps[r] for r in open_rows],
            "not_run_pilot_budget": [comps[r] for r, t in zip(open_rows, trace) if t is None] if open_rows.size else [],
            "trace": {str(comps[r]): t for r, t in zip(open_rows, trace)} if open_rows.size else {},
            "seconds": round(time.time() - t_search, 1)}
    if args.policy is not None:
        saved = json.loads(args.policy.read_text())
        if saved.get("image") != row["id"] or saved.get("epsilon_raw_bytes") != eps:
            raise ValueError("policy table belongs to another region")
        for index, root in enumerate(roots):
            margin_policies[index] = saved["root_policies"][root[0]]
        margin_scaled, _, _ = run_chain(margin_lams, top, mats, bounds, xl, policies=margin_policies)
        if [int(v) for v in margin_scaled] != [int(saved["root_scaled"][r[0]]) for r in roots]:
            raise ValueError("policy table does not reproduce the screened margin bounds")
    margins = dict(zip(comps, ceil_div(margin_scaled, 1 << S).tolist()))
    root_policies = {roots[r][0]: margin_policies[r] for r in range(len(roots))}
    # clamp closure per competitor: top (raw_c <= Q_HIGH - 1 or raw_t <= Q_HIGH),
    # bottom (raw_t >= Q_LOW + 1 or raw_c >= Q_LOW)
    fl = facts[top]
    t_lo_scaled, _, _ = run_chain(unit(n_out, target, 1 << S)[None], top, mats, bounds, xl)
    t_hi_scaled, _, _ = run_chain(unit(n_out, target, -(1 << S))[None], top, mats, bounds, xl)
    raw_t_low, raw_t_high = int(ceil_div(t_lo_scaled, 1 << S)[0]), int(np.floor_divide(-t_hi_scaled, 1 << S)[0])
    closure, logit_roots = {}, set()
    for c in comps:
        entry = {"raw_margin_lower_bound": int(margins[c])}
        if fl["hh"][c] <= Q_HIGH - 1:
            entry["top"] = {"fact": f"fact/L{top}/n{c}/upper", "value": int(fl["hh"][c]),
                            "rule": f"raw_c <= {Q_HIGH - 1}"}
        elif raw_t_high <= Q_HIGH:
            entry["top"] = {"fact": f"chain/logit/{target}/upper", "value": raw_t_high, "rule": f"raw_t <= {Q_HIGH}"}
            logit_roots.add(("upper", target))
        else:
            entry["top"] = None
        if fl["lower_ok"][c] and fl["hl"][c] >= Q_LOW:
            entry["bottom"] = {"fact": f"fact/L{top}/n{c}/lower", "value": int(fl["hl"][c]),
                               "rule": f"raw_c >= {Q_LOW}"}
        elif raw_t_low >= Q_LOW + 1:
            entry["bottom"] = {"fact": f"chain/logit/{target}/lower", "value": raw_t_low,
                               "rule": f"raw_t >= {Q_LOW + 1}"}
            logit_roots.add(("lower", target))
        else:
            entry["bottom"] = None
        entry["closed"] = entry["top"] is not None and entry["bottom"] is not None and margins[c] >= 1
        closure[str(c)] = entry
    for side, i in sorted(logit_roots):
        roots.append((f"chain/logit/{i}/{side}", side, top, unit(n_out, i, (1 if side == "lower" else -1) << S)))
    # one immutable policy row per root; logit roots and every supporting chain stay ADAPTIVE
    root_table = np.stack([root_policies.get(r[0], np.full(top, ADAPTIVE, dtype=np.int64)) for r in roots])
    root_table.setflags(write=False)

    # dependency closure, top-down; each side has exactly one source
    pending = {j: [np.zeros(sizes[j], bool) for _ in range(3)] for j in range(top)}

    def add(nd):
        for j, arrs in nd.items():
            for p, a in zip(pending[j], arrs):
                p |= a

    add(needs(np.stack([r[3] for r in roots]), top, mats, bounds, policies=root_table))
    plan = {}
    for k in range(top - 1, -1, -1):
        need_l, need_uv, need_un = pending[k]
        stable = facts[k]["stable"]
        if (need_l & (stable < 0)).any():
            if screen:
                return finish("FINDER_INCONCLUSIVE", cause=f"layer {k}: lower side of a stable-inactive neuron needed")
            raise ValueError(f"layer {k}: lower side of a stable-inactive neuron needed")
        chain_l = need_l & ~info[k]["lower_box"]
        chain_u = need_uv & info[k]["upper_chain"]
        plan[k] = (chain_l, chain_u)
        for sign, mask in ((1, chain_l), (-1, chain_u)):
            idx = np.nonzero(mask)[0]
            for s in range(0, idx.size, 256):
                rows = idx[s:s + 256]
                E = np.zeros((rows.size, sizes[k]), dtype=np.int64)
                E[np.arange(rows.size), rows] = sign << S
                add(needs(E, k, mats, bounds))
    t_closure = time.time() - t0 - t_bounds
    closed = sum(e["closed"] for e in closure.values())
    finder = {"margin_min": int(min(margins.values())), "tightest": int(min(margins, key=margins.get)),
              "positive": sum(m >= 1 for m in margins.values()), "closed": closed, "of": len(comps),
              "raw_t": [raw_t_low, raw_t_high],
              "chains_per_layer": {str(k): int(plan[k][0].sum() + plan[k][1].sum()) for k in plan},
              "chains_planned": int(sum(plan[k][0].sum() + plan[k][1].sum() for k in plan)) + len(roots)}
    if search_report is not None or args.policy is not None:
        finder["baseline_margin_min"] = int(min(baseline_margins.values()))
        finder["baseline_closed"] = sum(baseline_margins[int(c)] >= 1 and e["top"] is not None
                                        and e["bottom"] is not None for c, e in closure.items())
        finder["root_policies"] = {r[0]: [int(p) for p in root_table[i]] for i, r in enumerate(roots)}
        finder["root_scaled"] = {roots[i][0]: int(margin_scaled[i]) for i in range(len(comps))}
        if search_report is not None:
            finder["search"] = search_report
    if screen and args.cost:
        # whole planned graph: roots with their policies, supporting chains ADAPTIVE
        t_cost = time.time()
        cost = {"multiply_terms": 0, "two_sided_relu": 0, "relu_coordinates": 0, "chains": 0}

        def tally(E, layer, policies=None):
            _, terms, steps = run_chain(E, layer, mats, bounds, xl, record=True, policies=policies)
            cost["multiply_terms"] += int(terms.sum())
            cost["chains"] += E.shape[0]
            for s in steps:
                if s["kind"] == "relu":
                    j = s["j"]
                    cls = classify(s["a"], s["a_out"], bounds[j][2], bounds[j][3])
                    cost["two_sided_relu"] += int(((cls == 4) | (cls == 6)).sum())
                    cost["relu_coordinates"] += int((cls >= 0).sum())

        tally(np.stack([r[3] for r in roots]), top, root_table)
        for k in range(top - 1, -1, -1):
            for sign, mask in ((1, plan[k][0]), (-1, plan[k][1])):
                idx = np.nonzero(mask)[0]
                for s in range(0, idx.size, 64):
                    rows = idx[s:s + 64]
                    E = np.zeros((rows.size, sizes[k]), dtype=np.int64)
                    E[np.arange(rows.size), rows] = sign << S
                    tally(E, k)
        cost["seconds"] = round(time.time() - t_cost, 1)
        finder["cost"] = cost
    if screen:
        open_causes = {c: ("margin < 1" if e["raw_margin_lower_bound"] < 1 else
                           "no top clamp separation" if e["top"] is None else "no bottom clamp separation")
                       for c, e in closure.items() if not e["closed"]}
        return finish("FINDER_POSITIVE" if closed == len(comps) else "FINDER_INCONCLUSIVE",
                      **finder, **({"open_competitors": open_causes} if open_causes else {}))

    # fact table
    fact_table = []
    for k in range(top + 1):
        if k < top:
            need_l, need_uv, need_un = pending[k]
            chain_l, chain_u = plan[k]
            lo, hi = bounds[k][:2]
        else:
            need_l = np.zeros(sizes[k], bool)
            need_uv = np.zeros(sizes[k], bool)
            need_un = need_l
            for c, e in closure.items():
                for key in ("top", "bottom"):
                    if e[key] and e[key]["fact"].startswith("fact/"):
                        (need_uv if key == "top" else need_l)[int(c)] = True
            chain_l = chain_u = np.zeros(sizes[k], bool)
            lo, hi = facts[k]["hl"], facts[k]["hh"]
        f = facts[k]
        for side, need, chained, values in (("lower", need_l, chain_l, lo), ("upper", need_uv | need_un, chain_u, hi)):
            for i in np.nonzero(need)[0]:
                endpoint = f["hl" if side == "lower" else "hh"][i]
                rec = {"fact_id": f"fact/L{k}/n{i}/{side}", "layer_index": k, "neuron_index": int(i), "side": side,
                       "value": int(values[i] if chained[i] else endpoint), "statement": f"r{k}[{i}] {'>=' if side == 'lower' else '<='} value"}
                if chained[i]:
                    rec["source"] = {"kind": "chain", "chain_id": f"chain/L{k}/n{i}/{side}"}
                else:
                    ok = f["lower_ok"][i] if side == "lower" else f["upper_ok"][i]
                    if not ok:
                        raise ValueError(f"{rec['fact_id']}: box side not proved by the ledger")
                    field = "EXPECTED_LOW" if side == "lower" else "EXPECTED_HIGH"
                    if side == "lower" and int(values[i]) != int(endpoint):
                        raise ValueError(f"{rec['fact_id']}: lower value used != ledger endpoint")
                    if side == "upper" and need_uv[i] and int(values[i]) != int(endpoint):
                        raise ValueError(f"{rec['fact_id']}: upper value used != ledger endpoint")
                    rec["source"] = {"kind": "box", "digest": f["digest"][i], "slot": int(f["slot"][i]),
                                     "field": field,
                                     "derivation": (f"hl > {Q_LOW} and (STABLE_RELU[slot] = +1 or hl > 0 or logit layer)"
                                                    " => r >= hl") if side == "lower" else
                                     f"hh < {Q_HIGH} => clamp(r) <= hh < {Q_HIGH} => r <= hh"}
                fact_table.append(rec)
    fact_ids = {r["fact_id"] for r in fact_table}

    # export chains, streamed per layer
    out_dir.mkdir(parents=True, exist_ok=True)
    shards, stats = [], {"chains": 0, "multiply_terms": 0, "relu_coordinates": 0,
                         "class_counts": dict.fromkeys(CLASSES, 0), "max_abs_coefficient": 0,
                         "max_abs_g_envelope": 0, "max_abs_constant": 0}
    chain_bounds = {}

    def sp(v):
        i = np.nonzero(v)[0]
        return {"indices": i.tolist(), "values": v[i].tolist()}

    def emit(fh, ids, claims, layer, E, policies=None):
        scaled, terms, steps = run_chain(E, layer, mats, bounds, xl, record=True, policies=policies)
        for r, (cid, claim) in enumerate(zip(ids, claims)):
            out_steps = []
            for n, s in enumerate(steps):
                a = s["a"][r]
                stats["max_abs_coefficient"] = max(stats["max_abs_coefficient"], int(np.abs(a).max(initial=0)))
                rec = {"step_id": f"{cid}/s{n}", "kind": s["kind"], "input_constant": int(s["c_in"][r])}
                if n == 0:
                    rec["input_coefficients"] = sp(a)
                if s["kind"] == "affine":
                    j = s["j"]
                    R = s["R"][r]
                    fold = np.nonzero((R != 0) & (s["hlow"] > 0))[0]
                    if j > 0:
                        missing = [i for i in fold if f"fact/L{j - 1}/n{i}/lower" not in fact_ids]
                        if missing:
                            raise ValueError(f"{cid}: residual fold without fact")
                    rec.update(layer_index=j, output_coefficients=sp(s["a_out"][r]),
                               residual_lower_sources=("input_box.low" if j == 0 else
                                                       {"indices": fold.tolist(),
                                                        "fact_ids": [f"fact/L{j - 1}/n{i}/lower" for i in fold]}),
                               rounding_term=int(HALF * np.abs(a).sum()), output_constant=int(s["c_out"][r]),
                               overflow={"max_abs_product_envelope": int(s["max_prod"][r]),
                                         "g_column_abs_sum_max": int(s["g_env"][r]),
                                         "bias_dot_abs_sum": int(s["bias_env"][r]),
                                         "residual_dot_abs_sum": int(s["res_env"][r]),
                                         "abs_output_constant": abs(int(s["c_out"][r]))})
                    stats["max_abs_g_envelope"] = max(stats["max_abs_g_envelope"], int(s["g_env"][r]))
                elif s["kind"] == "relu":
                    j = s["j"]
                    cls = classify(a, s["a_out"][r], bounds[j][2], bounds[j][3])
                    groups = {}
                    for ci, name in enumerate(CLASSES):
                        idx = np.nonzero(cls == ci)[0]
                        groups[name] = idx.tolist()
                        stats["class_counts"][name] += int(idx.size)
                        sides = {"nosat_upper": ("upper",), "lower": ("lower",), "upper": ("upper",),
                                 "lower_upper": ("lower", "upper"), "saturated_top": ("lower",),
                                 "clipped": ("lower", "upper")}.get(name, ())
                        for side in sides:
                            for i in idx:
                                if f"fact/L{j}/n{i}/{side}" not in fact_ids:
                                    raise ValueError(f"{cid}: relu L{j} n{i} needs missing {side} fact")
                    stats["relu_coordinates"] += int((cls >= 0).sum())
                    rec.update(layer_index=j, output_coefficients=sp(s["a_out"][r]), relu_constants=sp(s["d"][r]),
                               coordinate_classes=groups, output_constant=int(s["c_out"][r]))
                else:
                    rec["scaled_bound"] = int(s["scaled"][r])
                stats["max_abs_constant"] = max(stats["max_abs_constant"], abs(rec["input_constant"]))
                out_steps.append(rec)
            sb = int(scaled[r])
            bound = int(ceil_div(sb, 1 << S)) if claim in ("lower", "margin") else int(np.floor_divide(-sb, 1 << S))
            chain_bounds[cid] = bound
            fh.write(json.dumps({"certificate_id": cid, "claim": claim, "lambda_layer": layer,
                                 "scaled_bound": sb, "bound": bound, "steps": out_steps}) + "\n")
            stats["chains"] += 1
            stats["multiply_terms"] += int(terms[r])

    for k in list(range(top - 1, -1, -1)) + [top]:
        path = out_dir / f"chains_L{k}.jsonl.gz"
        with gzip.open(path, "wt", compresslevel=6) as fh:
            if k == top:
                emit(fh, [r[0] for r in roots], [r[1] for r in roots], top, np.stack([r[3] for r in roots]),
                     policies=root_table)
            else:
                chain_l, chain_u = plan[k]
                for side, sign, mask in (("lower", 1, chain_l), ("upper", -1, chain_u)):
                    idx = np.nonzero(mask)[0]
                    for s in range(0, idx.size, 64):
                        rows = idx[s:s + 64]
                        E = np.zeros((rows.size, sizes[k]), dtype=np.int64)
                        E[np.arange(rows.size), rows] = sign << S
                        emit(fh, [f"chain/L{k}/n{i}/{side}" for i in rows], [side] * rows.size, k, E)
        shards.append({"file": path.name, "layer": k, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "bytes": path.stat().st_size})
    # the exported root chains must reproduce the screened margins exactly
    for c in comps:
        if chain_bounds[f"chain/margin/{target}-{c}"] != margins[c]:
            raise ValueError(f"margin {target}-{c}: exported chain != screened bound")
    if args.policy is not None:
        (out_dir / "finder_policy.json").write_text(json.dumps(
            {"note": "experiment provenance only; the checker never reads this file",
             "policy_source": str(args.policy),
             "policy_source_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
             "root_policies": finder["root_policies"], "root_scaled": finder["root_scaled"]}, indent=1) + "\n")
    # chain facts must equal the exported chain bounds
    for rec in fact_table:
        if rec["source"]["kind"] == "chain" and chain_bounds[rec["source"]["chain_id"]] != rec["value"]:
            raise ValueError(f"{rec['fact_id']}: fact value != chain bound")
    for e in closure.values():
        for key in ("top", "bottom"):
            if e[key] and e[key]["fact"].startswith("chain/") and chain_bounds[e[key]["fact"]] != e[key]["value"]:
                raise ValueError("closure chain value mismatch")

    layers = []
    for index, (w, b) in enumerate(mats):
        csr = {"rows": int(w.shape[0]), "cols": int(w.shape[1]), "indptr": w.indptr.tolist(),
               "indices": w.indices.tolist(), "data": w.data.tolist(), "bias": b.tolist()}
        csr["sha256"] = hashlib.sha256(json.dumps(csr, sort_keys=True).encode()).hexdigest()
        layers.append({"layer_index": index, "lowered_csr": csr})
    (out_dir / "layers.json").write_text(json.dumps(layers))
    used_digests = sorted({r["source"]["digest"] for r in fact_table if r["source"]["kind"] == "box"})
    encoding = next(o for o in summary["obligations"] if o["identity"]["property_type"] == "input_encoding")
    need_bits = int(max(stats["max_abs_coefficient"], 1)).bit_length() + 1
    root = {
        "schema": "crown_chain_certificate_v2",
        "claim": "untrusted_proposal_not_verified",
        "scope": f"{row['split']} image {row['id']}, eps = {eps} raw byte, target {target} vs all {len(comps)} "
                 f"competitors; {stats['chains']} chains, all dependencies exported or bound to ESBMC box blocks",
        "semantics": {
            "deployed": f"r_k = round_half_away((sum_i W_k[j,i] * h_{{k-1}}[i]) / 2^{F}) + b_k[j]; "
                        f"z_k = clamp_q{BITS}(r_k); h_k = relu(z_k); h_{{-1}} = x (encoded input); "
                        f"logits = clamp_q{BITS}(r_4)",
            "chain_invariant": "lambda . r_L >= sum_i coeff[i] * var[i] + constant, lambda at scale 2^scale_bits",
            "input_coefficients": "stored on step 0 only; step n input = step n-1 output_coefficients",
            "affine_step": f"g = W^T a; output_coefficients = floor(g / 2^{F}); residual = g - 2^{F} a_out in "
                           f"[0, 2^{F}) (derived by the checker); output_constant = input_constant + a.b + "
                           f"floor((residual . hlow - rounding_term) / 2^{F}); hlow_i = input_box.low at layer 0, "
                           f"else min(fact value, {Q_HIGH}) for residual_lower_sources indices and 0 elsewhere (h >= 0)",
            "relu_step": "per nonzero input coordinate, coordinate_classes gives the class (re-derived by the "
                         "checker from a, a_out); relu_constants_i: bound_free and nosat_upper 0; lower "
                         "min(a max(l,0) - a l, 0 if l < 0) = a max(-l, 0); upper a max(u, 0); lower_upper "
                         "min over {l, u, 0 if l < 0 < u} of a max(z,0) - a_out z; saturated_top a Q_HIGH; clipped "
                         "min over {l, u, 0 if l < 0 < u, Q_HIGH if l < Q_HIGH < u} of a clamp(z, 0, Q_HIGH) - a_out z. "
                         f"Values l, u are the cited fact values. Sound for raw r: saturation above {Q_HIGH} "
                         "only lowers h, and the lower and lower_upper classes exclude "
                         f"r < {Q_LOW} through l >= {Q_LOW}; r >= l >= Q_HIGH gives h = Q_HIGH; output_constant = "
                         "input_constant + sum relu_constants",
            "relu_classes": {"bound_free": "a > 0, a_out = 0: no fact", "nosat_upper": "a > 0, a_out = a: upper fact "
                             f"value <= {Q_HIGH}", "lower": "a < 0, a_out = a: lower fact", "upper": "a < 0, a_out = 0: "
                             f"upper fact", "lower_upper": f"any other a_out: both facts, inside Q{BITS}",
                             "saturated_top": f"any a, a_out = 0: lower fact >= {Q_HIGH}",
                             "clipped": f"any a, a_out: both facts, upper fact value > {Q_HIGH}; hidden h = "
                                        f"relu(clamp_q{BITS}(r)) = clamp(r, 0, {Q_HIGH}) for every integer r"},
            "concretize_step": "scaled_bound = input_constant + sum_i a_i * (a_i >= 0 ? low_i : high_i)",
            "bound": "lower/margin: ceil(scaled_bound / 2^scale_bits); upper (lambda = -e): "
                     "floor(-scaled_bound / 2^scale_bits)",
            "margin_closure": f"out = clamp_q{BITS}(raw). out_t > out_c if raw_t - raw_c >= 1 AND top "
                              f"(raw_c <= {Q_HIGH - 1} or raw_t <= {Q_HIGH}) AND bottom (raw_t >= {Q_LOW + 1} "
                              f"or raw_c >= {Q_LOW})",
        },
        "scale_bits": S,
        "qif": {"total_bits": BITS, "integer_bits": BITS - F - 1, "fractional_bits": F},
        "coefficient_width": {"max_abs_coefficient": stats["max_abs_coefficient"], "signed_bits_needed": need_bits,
                              "max_abs_g_envelope": stats["max_abs_g_envelope"],
                              "max_abs_constant": stats["max_abs_constant"],
                              "note": "derived from this export; the checker must re-derive and reject a narrower "
                                      "encoding"},
        "deployment": {"qnn_conv_native_c_sha256": deployment,
                       "encoder": {"obligation_digest": encoding["digest"], "identity": encoding["identity"]},
                       "lowered_constants_sha256": hashlib.sha256(
                           "".join(l["lowered_csr"]["sha256"] for l in layers).encode()).hexdigest(),
                       "layers_file": "layers.json",
                       "layers_file_sha256": hashlib.sha256((out_dir / "layers.json").read_bytes()).hexdigest()},
        "provenance": {"script": "scratchpad/arch/crown_chain_export_v2.py",
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "candidate_id": sel["candidate_id"], "folded_model_sha256": sel["folded_model_sha256"],
                       "split": row["split"], "image_id": row["id"], "image_sha256": row.get("sha256"),
                       "target": target, "correct": bool(correct), "epsilon_raw_bytes": eps,
                       "box_ledger": args.ledger, "protocol": protocol},
        "input_box": {"low": xl.tolist(), "high": xu.tolist()},
        "box_blocks": {d: {"identity": blocks[d]["identity"], "harness_sha256": blocks[d]["harness_sha256"],
                           "harness": blocks[d]["harness"], "cache_file": blocks[d]["cache_file"]}
                       for d in used_digests},
        "facts": fact_table,
        "margin_closure": {"target": target, "competitors": closure},
        "shards": shards,
        "manifest": {
            "required_competitors": comps,
            "root_chains": [r[0] for r in roots],
            "diagnostic_counts": {**stats, "facts": len(fact_table),
                                  "box_facts": sum(r["source"]["kind"] == "box" for r in fact_table),
                                  "chain_facts": sum(r["source"]["kind"] == "chain" for r in fact_table),
                                  "box_blocks_used": len(used_digests),
                                  "chains_per_layer": {str(k): int(plan[k][0].sum() + plan[k][1].sum())
                                                       for k in plan}},
            "note": "counts are diagnostic; completeness is every fact/chain reference resolving",
        },
    }
    (out_dir / "certificate.json").write_text(json.dumps(root))
    finish("EXPORTED", **{**finder, **root["manifest"]["diagnostic_counts"]}, out=str(out_dir),
           seconds_split={"bounds": round(t_bounds, 1), "closure": round(t_closure, 1)},
           coefficient_width=root["coefficient_width"],
           bytes={"certificate": (out_dir / "certificate.json").stat().st_size,
                  "layers": (out_dir / "layers.json").stat().st_size, "shards": sum(s["bytes"] for s in shards)})


if __name__ == "__main__":
    main()
