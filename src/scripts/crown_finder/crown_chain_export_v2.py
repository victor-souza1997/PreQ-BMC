"""Dependency-complete CROWN chain certificate export (schema crown_chain_certificate_v2).

UNTRUSTED FINDER. Exact integer arithmetic; ESBMC re-checks every step.

Variables: r_k is the RAW pre-activation (exact affine value), z_k = clamp_q16(r_k),
h_k = relu(z_k). Chains bound r. A ReLU step uses h_j >= / <= relations to r_j; per
nonzero coordinate (a = input coefficient, a' = output coefficient) it needs:
  a > 0, a' = 0          bound_free   a h >= 0 always
  a > 0, a' = a          nosat_upper  h >= r needs r <= 32767 (h = 32767 < r otherwise)
  a < 0, a' = a          lower        h - r <= max(-l, 0) for r >= l (saturation only lowers h)
  a < 0, a' = 0          upper        h <= max(u, 0) for r <= u
  otherwise              lower_upper  l <= r <= u inside Q16, so z = r
Affine residual folding uses h >= relu(l); it needs the lower fact only when l > 0.
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
from box_ledger import Q_HIGH, Q_LOW, facts_from_box, facts_from_ledger, load_ledger  # noqa: E402
from crown_chain_export import F, HALF, S, XU, _check_headroom, ceil_div, choose_relu, relu_const, setup  # noqa: E402

DEPLOYMENT = "dd535f5480bec45cdae7774f849afca49f7afb5d645f02037e86a44d16f45097"
CLASSES = ("bound_free", "nosat_upper", "lower", "upper", "lower_upper")


def classify(A, Ap):
    """Per coordinate class index into CLASSES (-1 where A == 0)."""
    cls = np.full(A.shape, 4, dtype=np.int8)
    cls[(A > 0) & (Ap == 0)] = 0
    cls[(A > 0) & (Ap == A)] = 1
    cls[(A < 0) & (Ap == A)] = 2
    cls[(A < 0) & (Ap == 0)] = 3
    cls[A == 0] = -1
    return cls


def run_chain(A, k, mats, bounds, xl, record=False):
    """Batched exact chain over rows of A @ r_k. Returns scaled bounds, terms, per-step dense records."""
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
        hlow = xl if j == 0 else np.maximum(bounds[j - 1][0], 0)
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
        l, u = bounds[j - 1]
        Ap = choose_relu(A, l, u)
        D = relu_const(A, Ap, l, u)
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
    else chain. Upper: the tighter of box (always proved in hidden layers) and chain."""
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
        if not f["upper_ok"].all():
            raise ValueError(f"layer {k}: a hidden box saturates at the top")
        lower_box = f["stable"] > 0
        if not f["lower_ok"][lower_box].all():
            raise ValueError(f"layer {k}: stable-active box lower not proved")
        lo = np.where(lower_box, f["hl"], lo_c)
        upper_chain = (f["stable"] == 0) & (hi_c < f["hh"])
        hi = np.where(upper_chain, hi_c, f["hh"])
        if (lo[~lower_box] < Q_LOW).any() or (hi > Q_HIGH).any() or (lo > hi).any():
            raise ValueError(f"layer {k}: bounds outside Q16 or empty")
        bounds.append((lo, hi))
        info.append({"lo_chain": lo_c, "hi_chain": hi_c, "lower_box": lower_box, "upper_chain": upper_chain})
    return bounds, info


def needs(A, k, mats, bounds):
    """Per lower layer: [need_lower, need_upper_value, need_upper_nosat] over rows of A @ r_k."""
    out = {}
    A = A.astype(np.int64)
    for j in range(k, 0, -1):
        w, _ = mats[j]
        G = np.asarray((w.T @ A.T).T, dtype=np.int64)
        An = np.floor_divide(G, 1 << F)
        R = G - (An << F)
        l, u = bounds[j - 1]
        Ap = choose_relu(An, l, u)
        cls = classify(An, Ap)
        out[j - 1] = [(((cls == 2) | (cls == 4)).any(0)) | ((R != 0) & (l > 0)[None, :]).any(0),
                      ((cls == 3) | (cls == 4)).any(0), (cls == 1).any(0)]
        A = Ap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("screen", "export"))
    ap.add_argument("eps", type=int)
    ap.add_argument("image")
    ap.add_argument("out_dir", nargs="?", type=Path)
    ap.add_argument("--ledger")
    ap.add_argument("--protocol", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    eps, image, out_dir, screen = args.eps, args.image, args.out_dir, args.mode == "screen"
    if not screen and (out_dir is None or args.ledger is None):
        raise SystemExit("export needs <out_dir> and --ledger")
    t0 = time.time()
    sel, row, net, mats, target, correct, xl, xu, box = setup(eps, image)
    sizes = [m[0].shape[0] for m in mats]
    report = {"mode": args.mode, "image": row["id"], "split": row["split"], "epsilon_raw_bytes": eps,
              "target": target, "deployed_correct": bool(correct)}

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
        if summary["deployment_source_sha256"] != DEPLOYMENT:
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
    margin_scaled, _, _ = run_chain(np.stack([r[3] for r in roots]), top, mats, bounds, xl)
    margins = dict(zip(comps, ceil_div(margin_scaled, 1 << S).tolist()))
    # clamp closure per competitor: top (raw_c <= 32766 or raw_t <= 32767), bottom (raw_t >= -32767 or raw_c >= -32768)
    fl = facts[top]
    t_lo_scaled, _, _ = run_chain(unit(n_out, target, 1 << S)[None], top, mats, bounds, xl)
    t_hi_scaled, _, _ = run_chain(unit(n_out, target, -(1 << S))[None], top, mats, bounds, xl)
    raw_t_low, raw_t_high = int(ceil_div(t_lo_scaled, 1 << S)[0]), int(np.floor_divide(-t_hi_scaled, 1 << S)[0])
    closure, logit_roots = {}, set()
    for c in comps:
        entry = {"raw_margin_lower_bound": int(margins[c])}
        if fl["hh"][c] <= Q_HIGH - 1:
            entry["top"] = {"fact": f"fact/L{top}/n{c}/upper", "value": int(fl["hh"][c]), "rule": "raw_c <= 32766"}
        elif raw_t_high <= Q_HIGH:
            entry["top"] = {"fact": f"chain/logit/{target}/upper", "value": raw_t_high, "rule": "raw_t <= 32767"}
            logit_roots.add(("upper", target))
        else:
            entry["top"] = None
        if fl["lower_ok"][c] and fl["hl"][c] >= Q_LOW:
            entry["bottom"] = {"fact": f"fact/L{top}/n{c}/lower", "value": int(fl["hl"][c]), "rule": "raw_c >= -32768"}
        elif raw_t_low >= Q_LOW + 1:
            entry["bottom"] = {"fact": f"chain/logit/{target}/lower", "value": raw_t_low, "rule": "raw_t >= -32767"}
            logit_roots.add(("lower", target))
        else:
            entry["bottom"] = None
        entry["closed"] = entry["top"] is not None and entry["bottom"] is not None and margins[c] >= 1
        closure[str(c)] = entry
    for side, i in sorted(logit_roots):
        roots.append((f"chain/logit/{i}/{side}", side, top, unit(n_out, i, (1 if side == "lower" else -1) << S)))

    # dependency closure, top-down; each side has exactly one source
    pending = {j: [np.zeros(sizes[j], bool) for _ in range(3)] for j in range(top)}

    def add(nd):
        for j, arrs in nd.items():
            for p, a in zip(pending[j], arrs):
                p |= a

    add(needs(np.stack([r[3] for r in roots]), top, mats, bounds))
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
            lo, hi = bounds[k]
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
                                     "derivation": ("hl > -32768 and (STABLE_RELU[slot] = +1 or hl > 0 or logit layer)"
                                                    " => r >= hl") if side == "lower" else
                                     "hh < 32767 => clamp(r) <= hh < 32767 => r <= hh"}
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

    def emit(fh, ids, claims, layer, E):
        scaled, terms, steps = run_chain(E, layer, mats, bounds, xl, record=True)
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
                    cls = classify(a, s["a_out"][r])
                    groups = {}
                    for ci, name in enumerate(CLASSES):
                        idx = np.nonzero(cls == ci)[0]
                        groups[name] = idx.tolist()
                        stats["class_counts"][name] += int(idx.size)
                        sides = {"nosat_upper": ("upper",), "lower": ("lower",), "upper": ("upper",),
                                 "lower_upper": ("lower", "upper")}.get(name, ())
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
                emit(fh, [r[0] for r in roots], [r[1] for r in roots], top, np.stack([r[3] for r in roots]))
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
            "deployed": "r_k = round_half_away((sum_i W_k[j,i] * h_{k-1}[i]) / 2^8) + b_k[j]; z_k = clamp_q16(r_k); "
                        "h_k = relu(z_k); h_{-1} = x (encoded input); logits = clamp_q16(r_4)",
            "chain_invariant": "lambda . r_L >= sum_i coeff[i] * var[i] + constant, lambda at scale 2^scale_bits",
            "input_coefficients": "stored on step 0 only; step n input = step n-1 output_coefficients",
            "affine_step": "g = W^T a; output_coefficients = floor(g / 2^8); residual = g - 2^8 a_out in [0, 2^8) "
                           "(derived by the checker); output_constant = input_constant + a.b + "
                           "floor((residual . hlow - rounding_term) / 2^8); hlow_i = input_box.low at layer 0, "
                           "else fact value for residual_lower_sources indices and 0 elsewhere (h >= 0)",
            "relu_step": "per nonzero input coordinate, coordinate_classes gives the class (re-derived by the "
                         "checker from a, a_out); relu_constants_i: bound_free and nosat_upper 0; lower "
                         "min(a max(l,0) - a l, 0 if l < 0) = a max(-l, 0); upper a max(u, 0); lower_upper "
                         "min over {l, u, 0 if l < 0 < u} of a max(z,0) - a_out z. Values l, u are the cited "
                         "fact values. Sound for raw r: saturation above 32767 only lowers h, and the lower and "
                         "lower_upper classes exclude r < -32768 through l >= -32768; output_constant = "
                         "input_constant + sum relu_constants",
            "relu_classes": {"bound_free": "a > 0, a_out = 0: no fact", "nosat_upper": "a > 0, a_out = a: upper fact "
                             "value <= 32767", "lower": "a < 0, a_out = a: lower fact", "upper": "a < 0, a_out = 0: "
                             "upper fact", "lower_upper": "any other a_out: both facts, inside Q16"},
            "concretize_step": "scaled_bound = input_constant + sum_i a_i * (a_i >= 0 ? low_i : high_i)",
            "bound": "lower/margin: ceil(scaled_bound / 2^scale_bits); upper (lambda = -e): "
                     "floor(-scaled_bound / 2^scale_bits)",
            "margin_closure": "out = clamp_q16(raw). out_t > out_c if raw_t - raw_c >= 1 AND top (raw_c <= 32766 or "
                              "raw_t <= 32767) AND bottom (raw_t >= -32767 or raw_c >= -32768)",
        },
        "scale_bits": S,
        "qif": {"total_bits": 16, "integer_bits": 7, "fractional_bits": F},
        "coefficient_width": {"max_abs_coefficient": stats["max_abs_coefficient"], "signed_bits_needed": need_bits,
                              "max_abs_g_envelope": stats["max_abs_g_envelope"],
                              "max_abs_constant": stats["max_abs_constant"],
                              "note": "derived from this export; the checker must re-derive and reject a narrower "
                                      "encoding"},
        "deployment": {"qnn_conv_native_c_sha256": DEPLOYMENT,
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
