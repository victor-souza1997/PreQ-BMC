"""Exact-integer forward input-affine envelopes and block certificate export.

UNTRUSTED FINDER (not a proof). All arithmetic is exact integer arithmetic
with outward rounding, so every envelope is sound by construction. That makes
these useful diagnostics. Soundness still comes only from ESBMC checking the
exported certificates.

Envelope of neuron j at layer k, over the encoded input x (uint8 values, the
same integers the deployed C sees after encoding):

    CL[j] + sum_i AL[j,i] * x[i]  <=  2^s * v[j]  <=  CU[j] + sum_i AU[j,i] * x[i]

where v is the pre-activation integer z (pre-clamp) or the post-ReLU output h.

Deployed semantics used: z = round_half_away(acc / 2^F) + b, with
|round_half_away(a) - a| <= 1/2. The clamp must be inactive: pre-clamp bounds
are checked against Q16, and the run fails loudly otherwise. h = ReLU(z),
followed by a clamp that is inactive for z in range.

Usage:
  affine_export.py sweep  <eps> <count> <K|full> [...]
  affine_export.py export <eps> <image_index> <layer> <first_neuron> <block> <K|full> <out.json>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "src")
from backends.conv_fixed_point import QuantizedConv2D, quantize_restricted_sequential  # noqa: E402
from backends.fixed_point import LayerQuantizationSpec  # noqa: E402
from backends.image_encoder import ByteImageEncoder  # noqa: E402
from datasets.gtsrb_study import load_crop  # noqa: E402
from scripts.evaluate_ssv_deep_source_model import restricted_from_npz  # noqa: E402
from verification.conv_interval_fast import _im2col, propagate_box  # noqa: E402

S = 16          # envelope scale bits
F = 8           # layer input fractional bits (Q16/I7/F8 everywhere)
Q_LOW, Q_HIGH = -32768, 32767
HEADROOM = 1 << 62


def lowered(layer):
    if not isinstance(layer, QuantizedConv2D):
        return sp.csr_matrix(np.asarray(layer.weights_int, dtype=np.int64)), \
            np.asarray(layer.bias_int, dtype=np.int64)
    g = layer.geometry
    n_in = int(np.prod(g.input_shape))
    index = (np.arange(n_in, dtype=np.int64) + 1).reshape(g.input_shape)
    cols = _im2col(index, layer)
    kernel = np.asarray(layer.kernel_int, dtype=np.int64).reshape(-1, g.kernel_shape[-1])
    co = kernel.shape[1]
    p_idx, k_idx, oc = np.nonzero((cols[:, :, None] > 0) & (kernel[None, :, :] != 0))
    w = sp.csr_matrix((kernel[k_idx, oc], (p_idx * co + oc, cols[p_idx, k_idx] - 1)),
                      shape=(cols.shape[0] * co, n_in), dtype=np.int64)
    return w, np.tile(np.asarray(layer.bias_int, dtype=np.int64), cols.shape[0])


def _guard(A, C, xl):
    worst = int(np.abs(A).max(initial=0)) * 256 * A.shape[1] + int(np.abs(C).max(initial=0))
    if worst >= HEADROOM:
        raise OverflowError(f"envelope magnitude {worst} exceeds int64 headroom")


def box_min(A, C, xl, xu):
    return C + np.maximum(A, 0) @ xl + np.minimum(A, 0) @ xu


def box_max(A, C, xl, xu):
    return C + np.maximum(A, 0) @ xu + np.minimum(A, 0) @ xl


def scale_down(A, C, num, den, xl, lower):
    """Outward integer form for (num/den) * (A x + C), num >= 0, den > 0 per row."""
    num, den = num[:, None], den[:, None]
    P = A * num
    if lower:
        Aq = np.floor_divide(P, den)            # residual r = P - den*Aq in [0, den)
        r = P - den * Aq                        # r >= 0 -> r.x >= r.xl
        Cq = np.floor_divide(C * num[:, 0] + r @ xl, den[:, 0])
    else:
        Aq = -np.floor_divide(-P, den)          # r <= 0 -> r.x <= r.xl
        r = P - den * Aq
        Cq = -np.floor_divide(-(C * num[:, 0] + r @ xl), den[:, 0])
    return Aq, Cq


def truncate(A, C, xl, xu, k, lower):
    """Keep the k largest |a|*(xu-xl) terms per row; fold the rest outward."""
    if k is None or A.shape[1] <= k:
        return A, C
    cost = np.abs(A) * (xu - xl)[None, :]
    drop = np.ones(A.shape, dtype=bool)
    np.put_along_axis(drop, np.argpartition(-cost, k - 1, axis=1)[:, :k], False, axis=1)
    D = np.where(drop, A, 0)
    fold = (np.minimum(D * xl, D * xu) if lower else np.maximum(D * xl, D * xu)).sum(1)
    return np.where(drop, 0, A), C + fold


def affine_layer(w, b, hL, hLc, hU, hUc, xl, xu, k):
    """z envelopes from the previous layer's h envelopes (both at scale 2^S)."""
    wp, wn = w.maximum(0), w.minimum(0)
    AL = np.asarray(wp @ hL + wn @ hU)
    CL = wp @ hLc + wn @ hUc
    AU = np.asarray(wp @ hU + wn @ hL)
    CU = wp @ hUc + wn @ hLc
    n = w.shape[0]
    one, den = np.ones(n, dtype=np.int64), np.full(n, 1 << F, dtype=np.int64)
    AL, CL = scale_down(AL, CL, one, den, xl, True)     # acc / 2^F, lower
    AU, CU = scale_down(AU, CU, one, den, xl, False)
    CL = CL + (b << S) - (1 << (S - 1))                  # + b - 1/2 (rounding)
    CU = CU + (b << S) + (1 << (S - 1))
    AL, CL = truncate(AL, CL, xl, xu, k, True)
    AU, CU = truncate(AU, CU, xl, xu, k, False)
    _guard(AL, CL, xl), _guard(AU, CU, xl)
    return AL, CL, AU, CU


def relu_layer(AL, CL, AU, CU, lo, hi, xl):
    active, dead = lo >= 0, hi <= 0
    unstable = ~active & ~dead
    keep_lower = active | (unstable & (hi >= -lo))       # h >= z, else h >= 0
    hL = np.where(keep_lower[:, None], AL, 0)
    hLc = np.where(keep_lower, CL, 0)
    # unstable upper: (hi-lo) h <= hi (z - lo) <= hi (U(x)/2^S - lo)
    num = np.where(unstable, hi, 1)
    den = np.where(unstable, hi - lo, 1)
    Au, Cu = scale_down(AU, CU - np.where(unstable, lo << S, 0), num, den, xl, False)
    hU = np.where(active[:, None], AU, np.where(unstable[:, None], Au, 0))
    hUc = np.where(active, CU, np.where(unstable, Cu, 0))
    regime = np.where(active, 1, np.where(dead, -1, 0))
    return hL, hLc, hU, hUc, regime


def forward(mats, xl, xu, box, k, keep=None):
    n_x = xl.shape[0]
    hL = hU = np.eye(n_x, dtype=np.int64) << S
    hLc = hUc = np.zeros(n_x, dtype=np.int64)
    layers = []
    for index, (w, b) in enumerate(mats[:-1]):
        AL, CL, AU, CU = affine_layer(w, b, hL, hLc, hU, hUc, xl, xu, k)
        lo_raw = -np.floor_divide(-box_min(AL, CL, xl, xu), 1 << S)   # ceil
        hi_raw = np.floor_divide(box_max(AU, CU, xl, xu), 1 << S)
        if lo_raw.min() < Q_LOW or hi_raw.max() > Q_HIGH:
            raise ValueError(f"layer {index}: clamp may be active; needs a clamp relaxation")
        lo = np.maximum(lo_raw, box.pre_low[index])
        hi = np.minimum(hi_raw, box.pre_high[index])
        hL2, hLc2, hU2, hUc2, regime = relu_layer(AL, CL, AU, CU, lo, hi, xl)
        record = {"lo": lo, "hi": hi, "regime": regime,
                  "nnz_L": (AL != 0).sum(1), "nnz_U": (AU != 0).sum(1)}
        if keep is not None and index in keep:
            record.update(z=(AL, CL, AU, CU), h_prev=(hL, hLc, hU, hUc), h=(hL2, hLc2, hU2, hUc2))
        layers.append(record)
        hL, hLc, hU, hUc = hL2, hLc2, hU2, hUc2
    return layers, (hL, hLc, hU, hUc)


def margin_bounds(mats, h, xl, xu, target):
    """Lower envelope of 2^S (v_t - v_c) for each competitor c (pre-clamp)."""
    hL, hLc, hU, hUc = h
    w, b = mats[-1]
    w = w.toarray()
    d = w[[target]] - w
    dp, dn = np.maximum(d, 0), np.minimum(d, 0)
    A, C = dp @ hL + dn @ hU, dp @ hLc + dn @ hUc
    n = d.shape[0]
    A, C = scale_down(A, C, np.ones(n, dtype=np.int64), np.full(n, 1 << F, dtype=np.int64), xl, True)
    C = C + ((b[target] - b) << S) - (1 << S)          # two roundings of 1/2 each
    lower = np.floor_divide(box_min(A, C, xl, xu), 1 << S)
    return np.delete(lower, target)


def load():
    search = json.loads(Path("output/sign_conv_depthwise_pool_search_20260924/search_summary.json").read_text())
    sel = search["selected_candidate_summary"]
    base = json.loads(Path(search["base_study_path"]).read_text())
    model = restricted_from_npz(sel, Path(sel["folded_model_path"]))
    net = quantize_restricted_sequential(model, [LayerQuantizationSpec(16, 7, 8)] * 5)
    enc = ByteImageEncoder(model.input_shape, fractional_bits=8, total_bits=16, resize_mode="nearest_center")
    rows = [r for r in base["records"] if r["split"] == "validation"]
    return sel, base, net, enc, rows


def image_region(base, net, enc, row, eps):
    img = load_crop(base["dataset_root"], row)
    target = int(row["class_id"])
    center = enc.encode(img)
    correct = int(np.argmax(propagate_box(net, center, center).low[-1])) == target
    blo, bhi = enc.byte_box(img, eps)
    xl, xu = enc.encode(blo).astype(np.int64), enc.encode(bhi).astype(np.int64)
    return target, correct, xl, xu, propagate_box(net, xl, xu)


def sweep(eps, count, ks):
    _, base, net, enc, rows = load()
    mats = [lowered(layer) for layer in net.layers]
    chosen = np.random.default_rng(2026).choice(len(rows), size=count, replace=False)
    for i in sorted(chosen):
        target, correct, xl, xu, box = image_region(base, net, enc, rows[i], eps)
        if not correct:
            continue
        for k in ks:
            t = time.time()
            try:
                layers, h = forward(mats, xl, xu, box, k)
                margin = int(margin_bounds(mats, h, xl, xu, target).min())
                status = "ok"
            except (ValueError, OverflowError) as error:
                layers, margin, status = [], None, str(error)
            print(json.dumps({
                "image_index": int(i), "id": rows[i]["id"], "K": k or "full", "status": status,
                "margin_lower_bound": margin,
                "nnz_median_max_per_layer": [[int(np.median(r["nnz_L"])), int(r["nnz_L"].max()),
                                              int(np.median(r["nnz_U"])), int(r["nnz_U"].max())] for r in layers],
                "median_width_per_layer": [float(np.median(r["hi"] - r["lo"])) for r in layers],
                "unstable_per_layer": [int(np.sum(r["regime"] == 0)) for r in layers],
                "seconds": round(time.time() - t, 1)}), flush=True)


def sparse_form(A_row, c):
    idx = np.nonzero(A_row)[0]
    return {"input_indices": idx.tolist(), "coefficients": A_row[idx].tolist(), "constant": int(c), "scale_bits": S}


def export(eps, image_index, layer, first, size, k, out):
    sel, base, net, enc, rows = load()
    mats = [lowered(layer_) for layer_ in net.layers]
    target, correct, xl, xu, box = image_region(base, net, enc, rows[image_index], eps)
    layers, _ = forward(mats, xl, xu, box, k, keep={layer})
    rec = layers[layer]
    AL, CL, AU, CU = rec["z"]
    pL, pLc, pU, pUc = rec["h_prev"]
    hL, hLc, hU, hUc = rec["h"]
    w, b = mats[layer]
    block = list(range(first, first + size))
    field = sorted(set(np.concatenate([w[j].indices for j in block]).tolist()))
    previous = [] if layer == 0 else [{
        "neuron_index": int(i),
        "box": [int(box.low[layer - 1][i]), int(box.high[layer - 1][i])],
        "h_lower": sparse_form(pL[i], pLc[i]), "h_upper": sparse_form(pU[i], pUc[i])} for i in field]
    neurons = []
    for j in block:
        lo, hi, regime = int(rec["lo"][j]), int(rec["hi"][j]), int(rec["regime"][j])
        neurons.append({
            "neuron_index": j, "bias": int(b[j]),
            "weights": {"input_neurons": w[j].indices.tolist(), "values": w[j].data.tolist()},
            "z_lower": sparse_form(AL[j], CL[j]), "z_upper": sparse_form(AU[j], CU[j]),
            "pre_clamp_bounds": [lo, hi],
            "esbmc_box_pre_activation": [int(box.pre_low[layer][j]), int(box.pre_high[layer][j])],
            "relu_regime": {1: "active", -1: "dead", 0: "unstable"}[regime],
            "relu_lower_rule": "h>=z" if (regime == 1 or (regime == 0 and hi >= -lo)) else "h>=0",
            "relu_upper_rule": {1: "h<=z", -1: "h<=0", 0: f"({hi}-({lo}))*h <= {hi}*(z-({lo}))"}[regime],
            "h_lower": sparse_form(hL[j], hLc[j]), "h_upper": sparse_form(hU[j], hUc[j])})
    script = Path(__file__).read_bytes()
    cert = {
        "schema": "input_affine_envelope_block_probe_v1",
        "claim": "untrusted_proposal_not_verified",
        "semantics": "CL + sum A*x <= 2^scale_bits * value <= CU + sum A*x, x = encoded uint8 input",
        "provenance": {"script": "scratchpad/arch/affine_export.py",
                       "script_sha256": hashlib.sha256(script).hexdigest(),
                       "candidate_id": sel["candidate_id"], "folded_model_sha256": sel["folded_model_sha256"],
                       "split": "validation", "image_index": image_index, "image_id": rows[image_index]["id"],
                       "target": target, "correct": bool(correct), "epsilon_raw_bytes": eps,
                       "K": k or "full", "rounding": "outward, exact integer; truncated terms folded over the input box"},
        "qif": {"total_bits": 16, "integer_bits": 7, "fractional_bits": F},
        "input_box": {"low": xl.tolist(), "high": xu.tolist()},
        "layer_index": layer, "layer_input_size": int(w.shape[1]),
        "previous_layer_envelopes": previous, "block": neurons}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(cert))
    nnz = [len(n["z_lower"]["input_indices"]) for n in neurons]
    prev_nnz = [len(p["h_lower"]["input_indices"]) for p in previous] or [0]
    print(json.dumps({"out": out, "bytes": Path(out).stat().st_size, "field": len(field),
                      "z_nnz_max": max(nnz), "prev_nnz_max": max(prev_nnz),
                      "regimes": [n["relu_regime"] for n in neurons]}))


def main():
    mode = sys.argv[1]
    if mode == "sweep":
        sweep(int(sys.argv[2]), int(sys.argv[3]), [None if v == "full" else int(v) for v in sys.argv[4:]])
    else:
        eps, image_index, layer, first, size = map(int, sys.argv[2:7])
        export(eps, image_index, layer, first, size, None if sys.argv[7] == "full" else int(sys.argv[7]), sys.argv[8])


if __name__ == "__main__":
    main()
