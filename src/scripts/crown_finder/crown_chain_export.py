"""Exact-integer CROWN back-substitution chains and pilot certificate export.

UNTRUSTED FINDER (not a proof). Every step uses exact integer arithmetic with
outward rounding, so the bounds are sound by construction. Soundness still
comes only from ESBMC re-checking each exported step.

Chain invariant for a query lambda over layer L (lambda at scale 2^S):

    lambda . z_L  >=  sum_i a[i] * v[i] + c      for all x in the input box

where v is the current layer variable (z_j, h_j or the encoded input x).

  affine step  (z_j -> h_{j-1}):  z = round_half_away(W h / 2^F) + b, raw value
      in Q16. With g = W^T a, a' = floor(g / 2^F), r = g - 2^F a' in [0, 2^F):
      a.z >= a'.h + a.b + floor((r . hlow - 2^(F-1) |a|_1) / 2^F)
  relu step    (h_j -> z_j):  h = ReLU(z), l <= z <= u inside Q16. For ANY
      integer a'_i, a_i ReLU(z) - a'_i z >= d_i := min over z in {l, u, 0 if l<0<u}
      (piecewise linear, breakpoint 0). The finder picks a'_i by CROWN.
  final step:  sum a[i] x[i] >= sum a[i] * (xl if a >= 0 else xu).

Usage:
  crown_chain_export.py bounds <eps> <validation_index|record_id>
  crown_chain_export.py export <eps> <validation_index|record_id> <out.json>
  crown_chain_export.py check  <eps> <validation_index|record_id> <samples>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from affine_export import Q_HIGH, Q_LOW, image_region, load, lowered  # noqa: E402
from verification.conv_interval_fast import propagate_box  # noqa: E402

S = 16
F = 8
HALF = 1 << (F - 1)
HEADROOM = float(1 << 62)


def _check_headroom(w, A):
    worst = np.abs(A).astype(np.float64) @ abs(w).astype(np.float64)
    if worst.size and worst.max() >= HEADROOM:
        raise OverflowError("W^T a exceeds int64 headroom")


def choose_relu(A, l, u):
    """CROWN choice of the integer z-coefficients for rows of a (over h)."""
    active, dead = l >= 0, u <= 0
    unstable = ~active & ~dead
    span = np.where(unstable, u - l, 1)
    slope = np.floor_divide(2 * A * u + span, 2 * span)          # round(a u / (u - l))
    keep = u >= -l
    Ap = np.where(A >= 0, np.where(keep, A, 0), slope)
    return np.where(active, A, np.where(dead, 0, Ap))


def relu_const(A, Ap, l, u):
    """d_i = min over the breakpoints of a_i ReLU(z) - a'_i z on [l, u]."""
    fl = A * np.maximum(l, 0) - Ap * l
    fu = A * np.maximum(u, 0) - Ap * u
    d = np.minimum(fl, fu)
    unstable = (l < 0) & (u > 0)
    return np.where(unstable, np.minimum(d, 0), d)               # f(0) = 0


def lower_chain(A, k, mats, bounds, xl, record=False):
    """Exact lower bound of rows of A @ z_k (rows at scale 2^S) over the input box."""
    A = A.astype(np.int64)
    C = np.zeros(A.shape[0], dtype=np.int64)
    terms = np.zeros(A.shape[0], dtype=np.int64)
    steps = []
    for j in range(k, -1, -1):
        w, b = mats[j]
        _check_headroom(w, A)
        G = np.asarray((w.T @ A.T).T, dtype=np.int64)
        Anew = np.floor_divide(G, 1 << F)
        R = G - (Anew << F)
        hlow = xl if j == 0 else np.maximum(bounds[j - 1][0], 0)
        terms += (A != 0).astype(np.int64) @ np.diff(w.indptr)
        C_in = C
        C = C + A @ b + np.floor_divide(R @ hlow - HALF * np.abs(A).sum(1), 1 << F)
        if record:
            steps.append({"kind": "affine", "layer_index": j, "a": A[0], "c_in": int(C_in[0]),
                          "a_out": Anew[0], "residual": R[0], "hlow": hlow, "c_out": int(C[0])})
        A = Anew
        if j == 0:
            break
        l, u = bounds[j - 1]
        Ap = choose_relu(A, l, u)
        D = relu_const(A, Ap, l, u)
        C_in = C
        C = C + D.sum(1)
        if record:
            steps.append({"kind": "relu", "layer_index": j - 1, "a": A[0], "c_in": int(C_in[0]),
                          "a_out": Ap[0], "d": D[0], "c_out": int(C[0])})
        A = Ap
    scaled = C + np.maximum(A, 0) @ xl + np.minimum(A, 0) @ XU[0]
    if record:
        steps.append({"kind": "concretize", "a": A[0], "c_in": int(C[0]), "scaled_bound": int(scaled[0])})
    return scaled, terms, steps


XU = [None]  # the upper input corner, set once per region (keeps signatures short)


def ceil_div(a, d):
    return -np.floor_divide(-a, d)


def all_bounds(mats, box, xl, chunk=256):
    """Raw pre-clamp bounds per hidden layer: chain bounds intersected with the box."""
    bounds, info = [], []
    for k in range(len(mats) - 1):
        n = mats[k][0].shape[0]
        lo_c, hi_c = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
        terms_lo, terms_hi = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
        for s in range(0, n, chunk):
            rows = np.arange(s, min(s + chunk, n))
            eye = np.zeros((rows.size, n), dtype=np.int64)
            eye[np.arange(rows.size), rows] = 1 << S
            lo, t_lo, _ = lower_chain(eye, k, mats, bounds, xl)
            nhi, t_hi, _ = lower_chain(-eye, k, mats, bounds, xl)
            lo_c[rows], hi_c[rows] = ceil_div(lo, 1 << S), np.floor_divide(-nhi, 1 << S)
            terms_lo[rows], terms_hi[rows] = t_lo, t_hi
        box_lo, box_hi = box.pre_low[k].astype(np.int64), box.pre_high[k].astype(np.int64)
        box_unstable = (box_lo < 0) & (box_hi > 0)
        if lo_c[box_unstable].min(initial=0) < Q_LOW or hi_c[box_unstable].max(initial=0) > Q_HIGH:
            raise ValueError(f"layer {k}: chain bounds leave Q16; clamp needs a relaxation")
        # box-stable neurons use the ESBMC box alone (no chain); a post-clamp box strictly
        # inside Q16 implies the raw value equals it
        if box_lo[~box_unstable].min(initial=0) <= Q_LOW or box_hi[~box_unstable].max(initial=0) >= Q_HIGH:
            raise ValueError(f"layer {k}: box-stable neuron may saturate")
        # box-unstable: the chain proves the raw value is in Q16, so the post-clamp box applies too
        lo = np.where(box_unstable, np.maximum(lo_c, box_lo), box_lo)
        hi = np.where(box_unstable, np.minimum(hi_c, box_hi), box_hi)
        bounds.append((lo, hi))
        info.append({"lo_chain": lo_c, "hi_chain": hi_c, "terms_lo": terms_lo, "terms_hi": terms_hi,
                     "box_unstable": box_unstable, "unstable": (lo < 0) & (hi > 0)})
    return bounds, info


def margin_chains(mats, bounds, xl, target, record_competitor=None):
    w, _ = mats[-1]
    n_out = w.shape[0]
    rows = [c for c in range(n_out) if c != target]
    A = np.zeros((len(rows), n_out), dtype=np.int64)
    A[np.arange(len(rows)), target] = 1 << S
    A[np.arange(len(rows)), rows] = -(1 << S)
    scaled, terms, _ = lower_chain(A, len(mats) - 1, mats, bounds, xl)
    margins = ceil_div(scaled, 1 << S)
    steps = None
    if record_competitor is not None:
        one = A[[rows.index(record_competitor)]]
        _, _, steps = lower_chain(one, len(mats) - 1, mats, bounds, xl, record=True)
    return dict(zip(rows, margins.tolist())), dict(zip(rows, terms.tolist())), steps


def setup(eps, image):
    """image: a validation index, or a record id such as Final_Test/Images/06330.ppm."""
    sel, base, net, enc, rows = load()
    mats = [lowered(layer) for layer in net.layers]
    row = rows[int(image)] if image.isdigit() else next(r for r in base["records"] if r["id"] == image)
    target, correct, xl, xu, box = image_region(base, net, enc, row, eps)
    XU[0] = xu
    return sel, row, net, mats, target, correct, xl, xu, box


def bounds_mode(eps, image_index):
    _, row, _, mats, target, correct, xl, xu, box = setup(eps, image_index)
    t = time.time()
    bounds, info = all_bounds(mats, box, xl)
    margins, mterms, _ = margin_chains(mats, bounds, xl, target)
    worst = min(margins, key=margins.get)
    report = {"image": image_index, "id": row["id"], "target": target, "correct": correct,
              "seconds": round(time.time() - t, 1), "margin_lower_bound": margins[worst], "tightest_competitor": worst,
              "competitors_positive": sum(v >= 1 for v in margins.values()), "layers": []}
    for k, (inf, (lo, hi)) in enumerate(zip(info, bounds)):
        need = inf["box_unstable"]        # box-stable neurons keep their ESBMC box and need no chain
        report["layers"].append({
            "layer": k, "neurons": int(lo.size), "box_unstable": int(need.sum()),
            "chain_unstable": int(inf["unstable"].sum()),
            "median_width_box": float(np.median(box.pre_high[k] - box.pre_low[k])),
            "median_width_chain": float(np.median(hi - lo)),
            "chains_needed": int(2 * need.sum()),
            "terms_needed": int((inf["terms_lo"] + inf["terms_hi"])[need].sum()),
            "terms_per_chain_median_max": [int(np.median(inf["terms_lo"][need])) if need.any() else 0,
                                           int(inf["terms_lo"].max())]})
    report["margin_chains"] = len(mterms)
    report["margin_terms_total"] = int(sum(mterms.values()))
    report["total_chains"] = sum(r["chains_needed"] for r in report["layers"]) + len(mterms)
    report["total_terms"] = sum(r["terms_needed"] for r in report["layers"]) + report["margin_terms_total"]
    print(json.dumps(report, indent=1))


def check_mode(eps, image_index, samples):
    """Sampling sanity check of the exact chain bounds against deployed execution."""
    _, _, net, mats, target, _, xl, xu, box = setup(eps, image_index)
    bounds, _ = all_bounds(mats, box, xl)
    margins, _, _ = margin_chains(mats, bounds, xl, target)
    rng = np.random.default_rng(7)
    violations, checks = 0, 0
    for s in range(samples):
        x = xl if s == 0 else xu if s == 1 else np.where(rng.random(xl.shape) < 0.5, xl, xu)
        run = propagate_box(net, x, x)
        for k, (lo, hi) in enumerate(bounds):
            z = run.pre_low[k].astype(np.int64)
            violations += int(np.sum((z < lo) | (z > hi)))
            checks += z.size
        logits = run.low[-1].astype(np.int64)
        for c, m in margins.items():
            violations += int(logits[target] - logits[c] < m)
            checks += 1
    print(json.dumps({"samples": samples, "checks": checks, "violations": violations,
                      "margin_lower_bound": min(margins.values())}))


def sparse(v):
    idx = np.nonzero(v)[0]
    return {"indices": idx.tolist(), "values": v[idx].tolist()}


def export_mode(eps, image_index, out):
    sel, row, net, mats, target, correct, xl, xu, box = setup(eps, image_index)
    bounds, info = all_bounds(mats, box, xl)
    margins, _, _ = margin_chains(mats, bounds, xl, target)
    competitor = min(margins, key=margins.get)
    _, _, margin_steps = margin_chains(mats, bounds, xl, target, record_competitor=competitor)
    # pilot neuron: the unstable layer-3 neuron carrying the largest coefficient in the margin chain
    last_hidden = len(mats) - 2
    relu3 = next(s for s in margin_steps if s["kind"] == "relu" and s["layer_index"] == last_hidden)
    lo3, hi3 = bounds[last_hidden]
    unstable3 = (lo3 < 0) & (hi3 > 0)
    neuron = int(np.argmax(np.where(unstable3, np.abs(relu3["a"]), -1)))
    n3 = mats[last_hidden][0].shape[0]
    e = np.zeros((1, n3), dtype=np.int64)
    e[0, neuron] = 1 << S
    lo_scaled, _, lo_steps = lower_chain(e, last_hidden, mats, bounds, xl, record=True)
    hi_scaled, _, hi_steps = lower_chain(-e, last_hidden, mats, bounds, xl, record=True)

    def bound_id(layer, index):
        return f"bound/L{layer}/n{index}"

    used = set()

    def render(steps, cid):
        rendered = []
        for i, s in enumerate(steps):
            rec = {"certificate_id": cid, "step_id": f"{cid}/s{i}", "kind": s["kind"],
                   "input_coefficients": sparse(s["a"]), "input_constant": s["c_in"]}
            if s["kind"] == "affine":
                j = s["layer_index"]
                rec.update(layer_index=j, from_var=f"z{j}", to_var="x" if j == 0 else f"h{j - 1}",
                           output_coefficients=sparse(s["a_out"]), residual=sparse(s["residual"]),
                           residual_lower_values={"source": "input_box.low" if j == 0 else
                                                  f"relu(lower bound of z{j - 1})",
                                                  **sparse(s["hlow"] * (s["residual"] != 0))},
                           rounding_term=int(HALF * np.abs(s["a"]).sum()),
                           output_constant=s["c_out"],
                           max_abs_g=int(np.abs(s["residual"] + (s["a_out"] << F)).max(initial=0)))
                if j > 0:
                    used.update((j - 1, int(i_)) for i_ in np.nonzero(s["residual"])[0])
            elif s["kind"] == "relu":
                j = s["layer_index"]
                nz = np.nonzero(s["a"])[0]
                used.update((j, int(i_)) for i_ in nz)
                rec.update(layer_index=j, from_var=f"h{j}", to_var=f"z{j}",
                           output_coefficients=sparse(s["a_out"]),
                           relu_constants=sparse(s["d"]),
                           bound_ids=[bound_id(j, int(i_)) for i_ in nz],
                           output_constant=s["c_out"])
            else:
                rec.update(scaled_bound=s["scaled_bound"])
            rendered.append(rec)
        return rendered

    chains = [
        {"certificate_id": f"chain/L{last_hidden}/n{neuron}/lower", "claim": "lower",
         "lambda_var": f"z{last_hidden}", "lambda": {"indices": [neuron], "values": [1 << S]},
         "steps": render(lo_steps, f"chain/L{last_hidden}/n{neuron}/lower"),
         "scaled_bound": int(lo_scaled[0]), "bound": int(ceil_div(lo_scaled, 1 << S)[0]),
         "statement": f"z{last_hidden}[{neuron}] >= bound"},
        {"certificate_id": f"chain/L{last_hidden}/n{neuron}/upper", "claim": "upper",
         "lambda_var": f"z{last_hidden}", "lambda": {"indices": [neuron], "values": [-(1 << S)]},
         "steps": render(hi_steps, f"chain/L{last_hidden}/n{neuron}/upper"),
         "scaled_bound": int(hi_scaled[0]), "bound": int(np.floor_divide(-hi_scaled, 1 << S)[0]),
         "statement": f"z{last_hidden}[{neuron}] <= bound"},
        {"certificate_id": f"chain/margin/{target}-{competitor}", "claim": "margin",
         "lambda_var": f"z{len(mats) - 1}",
         "lambda": {"indices": sorted([target, competitor]),
                    "values": [(1 << S) if i == target else -(1 << S) for i in sorted([target, competitor])]},
         "steps": render(margin_steps, f"chain/margin/{target}-{competitor}"),
         "scaled_bound": int(margin_steps[-1]["scaled_bound"]), "bound": int(margins[competitor]),
         "statement": f"z{len(mats) - 1}[{target}] - z{len(mats) - 1}[{competitor}] >= bound (raw, pre-clamp)"},
    ]
    out_layer = len(mats) - 1
    logit_box = {str(c): [int(box.pre_low[out_layer][c]), int(box.pre_high[out_layer][c])]
                 for c in (target, competitor)}
    # raw target logit lower bound: the logit box saturates, so closure needs its own chain
    n_out = mats[-1][0].shape[0]
    et = np.zeros((1, n_out), dtype=np.int64)
    et[0, target] = 1 << S
    t_scaled, _, t_steps = lower_chain(et, out_layer, mats, bounds, xl, record=True)
    raw_t_low = int(ceil_div(t_scaled, 1 << S)[0])
    ec = np.zeros((1, n_out), dtype=np.int64)
    ec[0, competitor] = -(1 << S)
    c_scaled, _, _ = lower_chain(ec, out_layer, mats, bounds, xl)
    raw_c_high_chain = int(np.floor_divide(-c_scaled, 1 << S)[0])
    box_c_high = int(box.pre_high[out_layer][competitor])
    raw_c_high = box_c_high if box_c_high < Q_HIGH else raw_c_high_chain   # prefer the ESBMC box
    top_ok = raw_c_high <= Q_HIGH - 1
    bottom_ok = raw_t_low >= Q_LOW + 1
    chains.append({
        "certificate_id": f"chain/logit/{target}/lower", "claim": "lower", "lambda_var": f"z{out_layer}",
        "lambda": {"indices": [target], "values": [1 << S]},
        "steps": render(t_steps, f"chain/logit/{target}/lower"),
        "scaled_bound": int(t_scaled[0]), "bound": raw_t_low,
        "statement": f"z{out_layer}[{target}] >= bound (raw, pre-clamp)"})
    bound_records = []
    for layer, index in sorted(used):
        inf = info[layer]
        bound_records.append({
            "bound_id": bound_id(layer, index), "layer_index": layer, "neuron_index": index,
            "lower": int(bounds[layer][0][index]), "upper": int(bounds[layer][1][index]),
            "chain_lower": int(inf["lo_chain"][index]), "chain_upper": int(inf["hi_chain"][index]),
            "post_clamp_pre_relu_box": [int(box.pre_low[layer][index]), int(box.pre_high[layer][index])],
            "source": "box_only" if not inf["box_unstable"][index] else "chain_intersect_box",
            "status": "dependency_not_exported_in_pilot"})
    layers = []
    for index, (w, b) in enumerate(mats):
        csr = {"rows": int(w.shape[0]), "cols": int(w.shape[1]), "indptr": w.indptr.tolist(),
               "indices": w.indices.tolist(), "data": w.data.tolist(), "bias": b.tolist()}
        csr["sha256"] = hashlib.sha256(json.dumps(csr, sort_keys=True).encode()).hexdigest()
        layers.append({"layer_index": index, "lowered_csr": csr})
    cert = {
        "schema": "crown_chain_certificate_pilot_v1",
        "claim": "untrusted_proposal_not_verified",
        "semantics": {
            "deployed": "z_k = clamp_q16(round_half_away((sum_i W_k[j,i] * h_{k-1}[i]) / 2^8) + b_k[j]); "
                        "h_k = clamp_q16(relu(z_k)); h_{-1} = x (encoded input); logits = clamp_q16(z_4)",
            "variables": "z_k are RAW pre-clamp values; each bound record proves them inside Q16",
            "chain_invariant": "lambda . z_L >= sum_i coeff[i] * var[i] + constant (lambda at scale 2^scale_bits)",
            "affine_step": "g = W^T a (weights: lowered_csr of layer_index); a_out = floor(g / 2^8); "
                           "residual = g - 2^8 a_out in [0, 2^8); output_constant = input_constant + a.b + "
                           "floor((residual . residual_lower_values - rounding_term) / 2^8), "
                           "rounding_term = 2^7 * sum |a|",
            "relu_step": "for each i: a_i relu(z_i) - a_out_i z_i >= relu_constants_i on [lower_i, upper_i], "
                         "checked at z in {lower, upper, 0 if lower < 0 < upper}; "
                         "output_constant = input_constant + sum relu_constants",
            "concretize_step": "scaled_bound = input_constant + sum_i a_i * (a_i >= 0 ? low_i : high_i)",
            "bound": "lower claim: ceil(scaled_bound / 2^scale_bits); upper claim (lambda = -e): "
                     "floor(-scaled_bound / 2^scale_bits)",
            "lowered_index": "conv rows = output_position * out_channels + out_channel; "
                             "columns = flattened HWC input index",
        },
        "scale_bits": S,
        "provenance": {"script": "scratchpad/arch/crown_chain_export.py",
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "candidate_id": sel["candidate_id"], "folded_model_sha256": sel["folded_model_sha256"],
                       "split": row["split"], "image_id": row["id"], "image_sha256": row.get("sha256"),
                       "target": target, "correct": bool(correct), "epsilon_raw_bytes": eps,
                       "relu_choice": "CROWN (lower slope 1 if u >= -l else 0; upper slope rounded)"},
        "qif": {"total_bits": 16, "integer_bits": 7, "fractional_bits": F},
        "input_box": {"low": xl.tolist(), "high": xu.tolist()},
        "layers": layers,
        "bound_records": bound_records,
        "margin_closure": {
            "target": target, "competitor": competitor, "raw_margin_lower_bound": int(margins[competitor]),
            "logit_post_clamp_box": logit_box,
            "raw_target_lower": raw_t_low, "raw_target_lower_source": f"chain/logit/{target}/lower",
            "raw_competitor_upper": raw_c_high,
            "raw_competitor_upper_source": "post-clamp box high < 32767 implies raw <= box high"
            if raw_c_high == box_c_high else "chain (not exported in pilot)",
            "top_condition": top_ok, "bottom_condition": bottom_ok,
            "rule": "out = clamp_q16(raw). With raw_t - raw_c >= 1, out_t > out_c iff the clamp does not merge "
                    "them at either end: top (raw_c <= 32766 or raw_t <= 32767) AND bottom "
                    "(raw_t >= -32767 or raw_c >= -32768). Sufficient: raw_c <= 32766 and raw_t >= -32767.",
            "other_competitors": {str(c): int(m) for c, m in margins.items() if c != competitor}},
        "chains": chains,
        "manifest": {"chains": [c["certificate_id"] for c in chains],
                     "bound_records": len(bound_records),
                     "pilot_scope": "3 chains exported; their bound-record dependencies are listed but their "
                                    "own chains are not exported"},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(cert))
    print(json.dumps({"out": out, "bytes": Path(out).stat().st_size, "neuron": neuron, "competitor": competitor,
                      "neuron_bounds": [chains[0]["bound"], chains[1]["bound"]],
                      "box": [int(box.pre_low[last_hidden][neuron]), int(box.pre_high[last_hidden][neuron])],
                      "margin": chains[2]["bound"], "bound_records": len(bound_records),
                      "step_sizes": {c["certificate_id"]: [len(s["input_coefficients"]["indices"]) for s in c["steps"]]
                                     for c in chains}}))


def main():
    mode = sys.argv[1]
    if mode == "bounds":
        bounds_mode(int(sys.argv[2]), sys.argv[3])
    elif mode == "check":
        check_mode(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    else:
        export_mode(int(sys.argv[2]), sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
