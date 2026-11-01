#!/usr/bin/env python
"""I-R (upstream): shared layer-1 context for L3:13 and L3:436 of 016 c5 (docs/ssv2026/IR_upstream_016.md).

DIAGNOSTIC ONLY. Two ESBMC queries at most (shared context, decoupled control). A VERIFIED shared
query is at most a CONDITIONAL_ESBMC_LEMMA; no certificate, no export, no VERIFIED count change.
The frozen-parent reconstruction (load_cache .. reconstruct) is copied verbatim from
ip_partition_016.py, whose run reproduced P3c exactly.

  cd /home/joao/code/preqbmc
  systemd-run --user --scope -q -p MemoryMax=8G /home/joao/miniconda3/envs/quad/bin/python -u \
    /home/joao/code/preqbmc-qformat/src/scripts/ir_upstream_016.py \
    --run output/crown_diagnostics_20261031T161801Z --p3c output/p3c_cascade_pilot_20261031T192435Z \
    --output output/ir_upstream_016_<UTC>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORKTREE = Path(__file__).resolve().parents[2]
SRC = WORKTREE / "src"
MAIN = Path("/home/joao/code/preqbmc")
_spec = importlib.util.spec_from_file_location("diagnostic_import_guard", SRC / "scripts" / "diagnostic_import_guard.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

ORD, COMP, PAIR = "016", 5, (13, 436)
PARENT_C5, TARGET_C5 = -647, -614
DELTA = 33 * (1 << 16)
EXPECTED = {"layer2_row_count": 8, "layer1_union": 2541, "layer1_shared": 316, "multiply_terms": 4182}
QUERY_TIMEOUT, QUERY_MEM, STAGE_CAP, TERM_CAP = 60, "4g", 600.0, 10000
FROZEN = {
    "src/scripts/crown_finder/crown_chain_export_v2.py": "1541896e43a01ada3d688d311f8c3db676b83ba371a88920cc5d3f2db94a9146",
    "src/scripts/crown_finder/crown_chain_export.py": "84e61a573f180e39a6e537caf3232405d77e2f48fac15eae178ffd6df4342c36",
    "src/scripts/crown_finder/affine_export.py": "7657897bac0420885f6825f1e15179ce498a77134e4c13065abe8a4ef4fdbfc0",
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditStop(RuntimeError):
    pass


class Stop(RuntimeError):
    """A pre-registered early outcome (PREFLIGHT_MISMATCH, ALGEBRAIC_PRECONDITION_FAILED, INCONCLUSIVE)."""

    def __init__(self, outcome, reason):
        super().__init__(reason)
        self.outcome = outcome


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--p3c", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if Path.cwd().resolve() != MAIN.resolve():
        raise SystemExit(f"run from {MAIN}")
    finder = guard.install(SRC, MAIN / "src")
    import numpy as np
    import affine_export
    import crown_chain_export as cce
    import crown_chain_export_v2 as v2
    from verification import crown_diagnostics as cd
    from verification import crown_partition_diag as pd
    from verification import crown_relational_diag as rd
    from verification.esbmc import ESBMCConfig, ESBMCRunner
    audit = guard.audit(finder, MAIN / "src")
    if not audit["passed"]:
        raise SystemExit(f"import audit failed: {audit['violations']}")
    for path, sha in FROZEN.items():
        if sha256_file(WORKTREE / path) != sha:
            raise SystemExit(f"frozen source changed: {path}")
    run, p3c, out = args.run.resolve(), args.p3c.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; use a fresh directory")
    p3c_summary = json.loads((p3c / "summary.json").read_text())
    if (p3c / "VOID.txt").exists() or p3c_summary["gate"] != "PASS":
        raise SystemExit("P3c run is void or did not pass")
    out.mkdir(parents=True)
    t_start = time.monotonic()

    config = json.loads((run / "manifest.json").read_text())["config"]
    preflight = json.loads((run / "preflight.json").read_text())
    if not preflight["passed"]:
        raise SystemExit("source run preflight did not pass")
    dv = json.loads((run / "violations.json").read_text())
    deployed_c = run / "oracle" / "qnn_conv_native.c"
    manifest = {
        "schema": "preqbmc_ir_upstream_v1", "spec": "docs/ssv2026/IR_upstream_016.md",
        "spec_sha256": sha256_file(WORKTREE / "docs/ssv2026/IR_upstream_016.md"),
        "script_sha256": sha256_file(__file__), "relational_module_sha256": sha256_file(rd.__file__),
        "partition_module_sha256": sha256_file(pd.__file__), "diagnostics_module_sha256": sha256_file(cd.__file__),
        "guard_sha256": sha256_file(SRC / "scripts" / "diagnostic_import_guard.py"), "frozen": FROZEN,
        "source_run": str(run), "source_run_manifest_sha256": sha256_file(run / "manifest.json"),
        "p3c_run": str(p3c), "p3c_summary_sha256": sha256_file(p3c / "summary.json"),
        "p3c_sides_sha256": sha256_file(p3c / "sides.jsonl.gz"),
        "deployed_c_sha256": sha256_file(deployed_c), "deployment_sha256_pinned": config["deployment_sha256"],
        "import_audit": {k: v for k, v in audit.items() if k != "modules"},
        "imported_modules": [{"module": m["module"], "file": m["file"], "sha256": m["sha256"]} for m in audit["modules"]],
        "budget": {"queries": 2, "query_timeout_s": QUERY_TIMEOUT, "query_memlimit": QUERY_MEM,
                   "stage_cap_s": STAGE_CAP, "local_term_cap": TERM_CAP, "stage_memory": "8G (systemd scope)"},
        "delta": DELTA, "expected_structure": EXPECTED,
    }
    if manifest["deployed_c_sha256"] != config["deployment_sha256"]:
        raise SystemExit("deployed C changed")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    v2.configure_format(int(config["qif"]["total_bits"]), int(config["qif"]["fractional_bits"]))
    _, _, net, enc, _ = affine_export.load()
    mats = [affine_export.lowered(layer) for layer in net.layers]
    relu = [bool(l.apply_relu) for l in net.layers]
    biases = [m[1] for m in mats]
    sizes = [m[0].shape[0] for m in mats]
    top = len(mats) - 1
    ONE = 1 << cd.SCALE_BITS
    Q_LOW, Q_HIGH = cd.Q_LOW, cd.Q_HIGH
    ADAPTIVE = v2.ADAPTIVE
    sides_log = {}
    with gzip.open(p3c / "sides.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            sides_log.setdefault(r["ordinal"], []).append(r)
    images = {im["ordinal"]: im for im in config["images"]}
    summary = {"outcome": None}

    def endpoint(side, scaled):
        return cd.ceil_div(int(scaled), ONE) if side == "lower" else (-int(scaled)) // ONE

    def load_cache(o):
        entry = preflight["caches"][o]
        if sha256_file(entry["npz"]) != entry["npz_sha256"] or sha256_file(entry["json"]) != entry["json_sha256"]:
            raise AuditStop(f"{o}: cache changed after preflight")
        data = dict(np.load(entry["npz"], allow_pickle=False))
        meta = json.loads(Path(entry["json"]).read_text())
        bounds = [(data[f"b{k}_lo"], data[f"b{k}_hi"], data[f"b{k}_sat"], data[f"b{k}_clip"]) for k in range(top)]
        info = [{key: data[f"i{k}_{key}"] for key in ("lo_chain", "hi_chain", "lower_box", "upper_chain")}
                for k in range(top)]
        facts = [{key: data[f"f{k}_{key}"] for key in ("hl", "hh", "stable", "upper_ok", "lower_ok")}
                 for k in range(top + 1)]
        return data, meta, bounds, info, facts

    def witnesses(o, data):
        x_nom = enc.encode(data["crop"]).astype(np.int64)
        tr = {f"{o}/nominal": (x_nom, cd.exact_forward(mats, x_nom, relu))}
        for w in dv["retained"]:
            if w["ordinal"] != o:
                continue
            if sha256_file(w["encoded_path"]) != w["encoded_sha256"]:
                raise AuditStop(f"{w['witness_id']}: witness file changed")
            x = np.load(w["encoded_path"], allow_pickle=False)
            if np.all(x >= data["xl"]) and np.all(x <= data["xu"]):
                tr[w["witness_id"]] = (x, cd.exact_forward(mats, x, relu))
        return tr

    def select(L, lo_c, hi_c, f):
        """bounds_v2 selection rule for one layer (as in P3c)."""
        lower_box = f["stable"] > 0
        if not f["lower_ok"][lower_box].all():
            raise AuditStop(f"layer {L}: stable-active box lower not proved")
        lo = np.where(lower_box, f["hl"], lo_c)
        saturated = lower_box & (f["hl"] >= Q_HIGH)
        upper_chain = ~saturated & ((f["stable"] == 0) | ~f["upper_ok"]) & ((hi_c < f["hh"]) | ~f["upper_ok"])
        hi = np.where(upper_chain, hi_c, f["hh"])
        clipped = ~saturated & (hi > Q_HIGH)
        unstable = (f["stable"] == 0) & ~clipped
        plain = ~saturated & ~clipped
        if (lo[unstable] < Q_LOW).any() or (lo > hi)[plain | clipped].any():
            raise AuditStop(f"layer {L}: selected bounds outside Q16 or empty")
        return (lo, hi, saturated, clipped), {"lower_box": lower_box, "upper_chain": upper_chain}

    def unit_rows(idx, signs, width):
        E = np.zeros((len(idx), width), dtype=np.int64)
        E[np.arange(len(idx)), idx] = np.asarray(signs, dtype=np.int64) * ONE
        return E

    def side_chains(bounds, parents, L, items, xl, xu):
        """Frozen-policy unit chains of layer L under `bounds` (replay mapped to `parents`).
        items: list of (neuron, side, policy) with policy None | ("A", list) | ("B", dict)."""
        rep = pd.Replay(v2, Q_HIGH)
        rep.map(bounds, parents)
        out = {}
        cce.XU[0] = xu
        with rep:
            plain = [(n, s) for n, s, p in items if p is None]
            for s0 in range(0, len(plain), 128):
                chunk = plain[s0:s0 + 128]
                sc, _, _ = v2.run_chain(unit_rows([n for n, _ in chunk], [1 if s == "lower" else -1 for _, s in chunk],
                                                  sizes[L]), L, mats, bounds, xl)
                for (n, s), v in zip(chunk, sc.tolist()):
                    out[(n, s)] = endpoint(s, v)
            for n, s, p in items:
                if p is None:
                    continue
                E = unit_rows([n], [1 if s == "lower" else -1], sizes[L])
                if p[0] == "A":
                    sc, _, _ = v2.run_chain(E, L, mats, bounds, xl, policies=np.array([p[1]], dtype=np.int64))
                else:
                    ov = p[1]
                    with rep.keyed_override(parents[ov["layer"]], 0, ov["neuron"], ov["a"], ov["trial_a_out"]):
                        sc, _, _ = v2.run_chain(E, L, mats, bounds, xl)
                        if rep.override_applied != 1:
                            raise pd.ReplayError(f"override applied {rep.override_applied} times")
                out[(n, s)] = endpoint(s, sc[0])
        return out

    def evaluate(bounds, parents, facts, xl, xu, margin_pols, comps, target, focus_c, record_rows=()):
        rep = pd.Replay(v2, Q_HIGH)
        rep.map(bounds, parents)
        cce.XU[0] = xu
        lams = np.zeros((len(comps), sizes[-1]), dtype=np.int64)
        lams[:, target] = ONE
        lams[np.arange(len(comps)), comps] = -ONE
        with rep:
            sc, terms, _ = v2.run_chain(lams, top, mats, bounds, xl, policies=margin_pols)
            ts, _, _ = v2.run_chain(unit_rows([target, target], [1, -1], sizes[-1]), top, mats, bounds, xl)
            recorded = {}
            for c in record_rows:
                k = comps.index(c)
                s1, _, st = v2.run_chain(lams[k:k + 1], top, mats, bounds, xl, record=True,
                                         policies=margin_pols[k:k + 1])
                if int(s1[0]) != int(sc[k]):
                    raise AuditStop("single-row margin chain differs from the batched one")
                recorded[c] = (lams[k:k + 1], st, int(s1[0]))
        raw = {c: cd.ceil_div(int(v), ONE) for c, v in zip(comps, sc.tolist())}
        rt = [cd.ceil_div(int(ts[0]), ONE), (-int(ts[1])) // ONE]
        clo = cd.closure_edges(raw, facts[top]["hh"], facts[top]["hl"], facts[top]["lower_ok"], rt[0], rt[1], target)
        return {"raw": raw, "scaled": {c: int(v) for c, v in zip(comps, sc.tolist())}, "raw_t": rt, "closure": clo,
                "closed": sorted(c for c, v in clo.items() if v["closed"]), "focus": focus_c and
                {str(c): raw[c] for c in focus_c}, "recorded": recorded}

    def reconstruct(o):
        """Frozen P3c parent: cascade bounds (L2 then L3) and P3 margin search, checked against P3c."""
        image = images[o]
        target = image["target"]
        data, meta, bounds0, info0, facts = load_cache(o)
        xl, xu = data["xl"], data["xu"]
        comps = meta["comps"]
        log = sides_log[o]
        side_pol = {}
        lo2, hi2 = info0[2]["lo_chain"].copy(), info0[2]["hi_chain"].copy()
        for r in (r for r in log if r["config"] == "cascade_L2"):
            arr = lo2 if r["side"] == "lower" else hi2
            if int(arr[r["neuron"]]) != r["baseline"]:
                raise AuditStop(f"{o} L2:{r['neuron']}: cached chain != logged baseline")
            if r["chosen"] != "baseline":
                arr[r["neuron"]] = r["chain"]
                side_pol[(2, r["neuron"], r["side"])] = ("A", r["arm_A_policy"]) if r["chosen"] == "A" \
                    else ("B", r["arm_B_policy"])
            elif r["chain"] != r["baseline"]:
                raise AuditStop("baseline-chosen side changed value")
        b2, f2 = select(2, lo2, hi2, facts[2])
        bounds = [bounds0[0], bounds0[1], b2, bounds0[3]]
        n3 = sizes[3]
        all3 = [(n, s, None) for n in range(n3) for s in ("lower", "upper")]
        vals = side_chains(bounds, bounds, 3, all3, xl, xu)
        lo3 = np.array([vals[(n, "lower")] for n in range(n3)], dtype=np.int64)
        hi3 = np.array([vals[(n, "upper")] for n in range(n3)], dtype=np.int64)
        for r in (r for r in log if r["config"] == "cascade_L3"):
            arr = lo3 if r["side"] == "lower" else hi3
            if int(arr[r["neuron"]]) != r["baseline"]:
                raise AuditStop(f"{o} L3:{r['neuron']}: recomputed ADAPTIVE != logged baseline")
            if r["chosen"] != "baseline":
                arr[r["neuron"]] = r["chain"]
                side_pol[(3, r["neuron"], r["side"])] = ("A", r["arm_A_policy"]) if r["chosen"] == "A" \
                    else ("B", r["arm_B_policy"])
        b3, f3 = select(3, lo3, hi3, facts[3])
        bounds = [bounds0[0], bounds0[1], b2, b3]
        flags = [{"lower_box": info0[k]["lower_box"], "upper_chain": info0[k]["upper_chain"]} for k in (0, 1)] + [f2, f3]
        for L in (2, 3):     # every policy side reproduces its logged chain value under the parent
            items = [(n, s, p) for (k, n, s), p in side_pol.items() if k == L]
            got = side_chains(bounds, bounds, L, items, xl, xu)
            for n, s, p in items:
                want = int((lo2 if s == "lower" else hi2)[n]) if L == 2 else int((lo3 if s == "lower" else hi3)[n])
                if got[(n, s)] != want:
                    raise AuditStop(f"{o} L{L}:{n} {s}: policy replay {got[(n, s)]} != logged {want}")
        pols = np.full((len(comps), top), ADAPTIVE, dtype=np.int64)
        ev = evaluate(bounds, bounds, facts, xl, xu, pols, comps, target, image["focus"])
        ref = p3c_summary["images"][o]
        if ev["raw"][image["primary"]] != ref["cascade"]["primary"] or len(ev["closed"]) != ref["cascade"]["closed"] \
                or min(ev["raw"].values()) != ref["cascade"]["margin_min"]:
            raise AuditStop(f"{o}: reconstructed cascade does not reproduce P3c")
        open_rows = [k for k, c in enumerate(comps) if ev["raw"][c] < 1]
        if open_rows:
            lams = np.zeros((len(open_rows), sizes[-1]), dtype=np.int64)
            lams[:, target] = ONE
            lams[np.arange(len(open_rows)), [comps[k] for k in open_rows]] = -ONE
            rep = pd.Replay(v2, Q_HIGH)
            rep.map(bounds, bounds)
            cce.XU[0] = xu
            with rep:
                pol, _, tr = v2.search_policies(lams, top, mats, bounds, xl, None)
            if any(t is None for t in tr):
                raise AuditStop("margin search unfinished")
            pols[open_rows] = pol
        evs = evaluate(bounds, bounds, facts, xl, xu, pols, comps, target, image["focus"],
                       record_rows=[image["primary"]] + list(image["focus"]))
        rs = ref["cascade_margin_search"]
        if evs["raw"][image["primary"]] != rs["primary"] or len(evs["closed"]) != rs["closed"] \
                or min(evs["raw"].values()) != rs["margin_min"]:
            raise AuditStop(f"{o}: reconstructed cascade + margin search does not reproduce P3c")
        return {"data": data, "meta": meta, "facts": facts, "bounds": bounds, "flags": flags, "side_pol": side_pol,
                "pols": pols, "eval": evs, "comps": comps, "target": target, "image": image}


    def citable(L, n, flags_L, facts_L, bounds_L, side_pol, side):
        """(value, provenance) of a frozen parent fact the current checker can cite, else (None, reason)."""
        lo, hi, sat, _ = bounds_L
        stable = int(facts_L["stable"][n])
        if side == "lower":
            if bool(flags_L["lower_box"][n]):
                return int(lo[n]), {"fact": f"fact/L{L}/n{n}/lower", "source": "box", "value": int(lo[n])}
            if stable == 0:
                pol = side_pol.get((L, n, "lower"))
                return int(lo[n]), {"fact": f"chain/L{L}/n{n}/lower", "source": "chain", "value": int(lo[n]),
                                    "policy": None if pol is None else [pol[0], pol[1]]}
            return None, {"fact": None, "reason": "stable-inactive lower side is not citable"}
        if bool(sat[n]):
            return None, {"fact": None, "reason": "saturated: upper side unused"}
        if bool(flags_L["upper_chain"][n]):
            pol = side_pol.get((L, n, "upper"))
            return int(hi[n]), {"fact": f"chain/L{L}/n{n}/upper", "source": "chain", "value": int(hi[n]),
                                "policy": None if pol is None else [pol[0], pol[1]]}
        if bool(facts_L["upper_ok"][n]):
            return int(hi[n]), {"fact": f"fact/L{L}/n{n}/upper", "source": "box", "value": int(hi[n])}
        return None, {"fact": None, "reason": "box upper not proved"}

    try:
        t0 = time.monotonic()
        par = reconstruct(ORD)
        lam5, steps5, scaled5 = par["eval"]["recorded"][COMP]
        if par["eval"]["raw"][COMP] != PARENT_C5:
            raise Stop("PREFLIGHT_MISMATCH", f"parent c5 {par['eval']['raw'][COMP]} != {PARENT_C5}")
        bounds, facts, flags, side_pol = par["bounds"], par["facts"], par["flags"], par["side_pol"]
        summary["parent"] = {"c5": PARENT_C5, "c5_scaled": str(scaled5), "closed": len(par["eval"]["closed"]),
                             "rebuild_seconds": round(time.monotonic() - t0, 1)}

        # ---------------- 1. preflight: frozen relaxation parameters and CSR structure
        relu3 = [i for i, s in enumerate(steps5) if s["kind"] == "relu" and s["j"] == 3]
        if len(relu3) != 1:
            raise Stop("PREFLIGHT_MISMATCH", "c5 chain does not have exactly one layer-3 ReLU step")
        si = relu3[0]
        st = steps5[si]
        lo3, hi3, sat3, clip3 = bounds[3]
        params = {}
        for n in PAIR:
            a, ap, d = int(st["a"][0, n]), int(st["a_out"][0, n]), int(st["d"][0, n])
            cls = int(v2.classify(st["a"][0:1, n:n + 1], st["a_out"][0:1, n:n + 1], sat3[n:n + 1], clip3[n:n + 1])[0, 0])
            d_re = int(v2.relu_const(np.array([[a]]), np.array([[ap]]), lo3[n:n + 1], hi3[n:n + 1])[0, 0])
            if bool(sat3[n]) or bool(clip3[n]) or d_re != d:
                raise Stop("PREFLIGHT_MISMATCH", f"L3:{n}: d {d} != recomputed {d_re} or saturated/clipped")
            lo_v, lo_p = citable(3, n, flags[3], facts[3], bounds[3], side_pol, "lower")
            hi_v, hi_p = citable(3, n, flags[3], facts[3], bounds[3], side_pol, "upper")
            params[n] = {"a": a, "a_out": ap, "d": d, "raw_interval": [int(lo3[n]), int(hi3[n])],
                         "class": v2.CLASSES[cls] if cls >= 0 else None, "lower_fact": lo_p, "upper_fact": hi_p,
                         "assumed": [lo_v, hi_v]}
        struct = rd.structure(mats, PAIR)
        summary["preflight"] = {"relaxation": {str(k): v for k, v in params.items()}, "structure": struct}
        print(json.dumps({"preflight": {"relaxation": {str(k): {kk: v[kk] for kk in ("a", "a_out", "d", "raw_interval",
                                                                                     "class")}
                                                       for k, v in params.items()}, "structure": struct}}), flush=True)
        mismatch = {k: (struct[k], v) for k, v in EXPECTED.items() if struct[k] != v}
        if mismatch or not struct["layer2_rows_disjoint"]:
            raise Stop("PREFLIGHT_MISMATCH", f"structure differs from the expected values: {mismatch}")
        if struct["multiply_terms"] > TERM_CAP:
            raise Stop("INCONCLUSIVE", "local multiply-term cap exceeded")

        # ---------------- 2. algebraic precondition
        later = steps5[si + 1:]
        recon = int(st["c_out"][0])
        for s in later:
            if s["kind"] == "concretize":
                if int(s["c_in"][0]) != recon:
                    raise Stop("ALGEBRAIC_PRECONDITION_FAILED", "constant chain does not link to concretize")
                corner = int(s["scaled"][0]) - int(s["c_in"][0])
                recon += corner
            else:
                if int(s["c_in"][0]) != recon:
                    raise Stop("ALGEBRAIC_PRECONDITION_FAILED", "constant chain is not linked step to step")
                recon = int(s["c_out"][0])
        if recon != scaled5:
            raise Stop("ALGEBRAIC_PRECONDITION_FAILED", "replayed constants do not give the parent scaled bound")
        # every later step adds a term independent of its incoming constant (run_chain: C = C + term), so
        # raising the layer-3 ReLU constant by DELTA shifts the final scaled bound by exactly DELTA
        new_scaled = scaled5 + DELTA
        new_raw = cd.ceil_div(new_scaled, ONE)
        data = par["data"]
        # exact check in the implemented representation: rerun the frozen c5 chain with the layer-3
        # ReLU constant of L3:PAIR[0] raised by DELTA (all other choices replayed unchanged)
        k5 = par["comps"].index(COMP)
        rep_alg = pd.Replay(v2, Q_HIGH)
        rep_alg.map(bounds, bounds)
        original_choice = None

        def shifted(A, bnds, policy=None):
            Ap, D = original_choice(A, bnds, policy)
            if bnds is bounds[3]:
                D = D.copy()
                D[0, PAIR[0]] += DELTA
            return Ap, D

        cce.XU[0] = data["xu"]
        with rep_alg:
            original_choice = v2.relu_choice
            v2.relu_choice = shifted
            try:
                sc_shift, _, _ = v2.run_chain(lam5, top, mats, bounds, data["xl"], policies=par["pols"][k5:k5 + 1])
            finally:
                v2.relu_choice = original_choice
        additive = int(sc_shift[0]) == new_scaled
        x_nom = enc.encode(data["crop"]).astype(np.int64)
        tr_nom = cd.exact_forward(mats, x_nom, relu)
        acc = cd.account_chain(steps5, 0, lam5[0], tr_nom, x_nom, data["xl"], data["xu"], biases, scaled5)
        per3 = next(r for r in acc["steps"] if r["kind"] == "relu" and r["layer"] == 3)["_per_coordinate"]
        e_nom = {n: rd.residual(params[n]["a"], params[n]["a_out"], params[n]["d"], int(tr_nom[3]["r"][n])) for n in PAIR}
        e_match = all(int(per3.get(n, 0)) == e_nom[n] for n in PAIR)
        summary["algebra"] = {"delta": DELTA, "parent_scaled": str(scaled5), "new_scaled": str(new_scaled),
                              "new_raw": new_raw, "linear_in_constant": additive,
                              "e_nominal": {str(n): e_nom[n] for n in PAIR},
                              "accounting_relu_slack_nominal": {str(n): int(per3.get(n, 0)) for n in PAIR},
                              "e_matches_accounting": e_match,
                              "chain_rerun_scaled": str(int(sc_shift[0])),
                              "passed": new_raw == TARGET_C5 and e_match and additive}
        print(json.dumps({"algebra": summary["algebra"]}), flush=True)
        if not summary["algebra"]["passed"]:
            raise Stop("ALGEBRAIC_PRECONDITION_FAILED", f"new raw {new_raw}, e match {e_match}, linear {additive}")

        # ---------------- 3. context and assumptions
        w2, w3 = mats[2][0].tocsr(), mats[3][0].tocsr()
        assumptions = {"layer1": {}, "layer2": {}, "layer3": {}}
        cones = []
        for ci, n in enumerate(PAIR):
            rows = w3.indices[w3.indptr[n]:w3.indptr[n + 1]].tolist()
            wts = w3.data[w3.indptr[n]:w3.indptr[n + 1]].tolist()
            rows2, b2, bounds2 = {}, {}, {}
            for q in rows:
                cols = w2.indices[w2.indptr[q]:w2.indptr[q + 1]].tolist()
                ws = w2.data[w2.indptr[q]:w2.indptr[q + 1]].tolist()
                rows2[q] = [(int(c), int(w)) for c, w in zip(cols, ws)]
                b2[q] = int(biases[2][q])
                lv, lp = citable(2, q, flags[2], facts[2], bounds[2], side_pol, "lower")
                hv, hp = citable(2, q, flags[2], facts[2], bounds[2], side_pol, "upper")
                bounds2[q] = (lv, hv)
                assumptions["layer2"][str(q)] = {"lower": lp, "upper": hp}
            assumptions["layer3"][str(n)] = {"lower": params[n]["lower_fact"], "upper": params[n]["upper_fact"]}
            cones.append(rd.Cone(name="AB"[ci], neuron3=n, w3=[(int(q), int(w)) for q, w in zip(rows, wts)],
                                 b3=int(biases[3][n]), rows2=rows2, b2=b2, bounds2=bounds2,
                                 bounds3=tuple(params[n]["assumed"]), a=params[n]["a"], a_out=params[n]["a_out"],
                                 d=params[n]["d"]))
        h1 = {}
        for c in sorted(cones[0].coords() | cones[1].coords()):
            lv, lp = citable(1, c, flags[1], facts[1], bounds[1], side_pol, "lower")
            hv, hp = citable(1, c, flags[1], facts[1], bounds[1], side_pol, "upper")
            h1[c] = (max(0, rd.H(lv)) if lv is not None else 0, rd.H(hv) if hv is not None else Q_HIGH)
            if h1[c][0] > h1[c][1]:
                raise Stop("INCONCLUSIVE", f"empty layer-1 activation interval at {c}")
            assumptions["layer1"][str(c)] = {"interval": list(h1[c]), "lower": lp, "upper": hp}
        ctx = rd.Context(cones=tuple(cones), h1=h1, delta=DELTA, label="")
        if len(ctx.shared()) != EXPECTED["layer1_shared"]:
            raise Stop("PREFLIGHT_MISMATCH", "context shared-coordinate count differs")
        (out / "assumptions.json").write_text(json.dumps(assumptions, indent=1, sort_keys=True) + "\n")
        summary["assumption_counts"] = {
            "layer1_lower_cited": sum(v["lower"].get("fact") is not None for v in assumptions["layer1"].values()),
            "layer1_upper_cited": sum(v["upper"].get("fact") is not None for v in assumptions["layer1"].values()),
            "layer2_sides_cited": sum((v["lower"].get("fact") is not None) + (v["upper"].get("fact") is not None)
                                      for v in assumptions["layer2"].values()),
            "layer3_sides_cited": sum((v["lower"].get("fact") is not None) + (v["upper"].get("fact") is not None)
                                      for v in assumptions["layer3"].values()),
            "policy_chain_facts": sorted({v[s]["fact"] for L in ("layer1", "layer2", "layer3")
                                          for v in assumptions[L].values() for s in ("lower", "upper")
                                          if v[s].get("policy")}),
            "dependency_status": "UNCHECKED: box facts are screen facts (facts_from_box), chain facts are finder "
                                 "values; none ESBMC-checked in this run"}

        # ---------------- 4. fidelity, non-vacuity and measured e-sums
        traces = {f"{ORD}/nominal": (x_nom, tr_nom)}
        for w in dv["retained"]:
            if w["ordinal"] != ORD:
                continue
            if sha256_file(w["encoded_path"]) != w["encoded_sha256"]:
                raise AuditStop(f"{w['witness_id']}: witness file changed")
            x = np.load(w["encoded_path"], allow_pickle=False)
            if np.all(x >= data["xl"]) and np.all(x <= data["xu"]):
                traces[w["witness_id"]] = (x, cd.exact_forward(mats, x, relu))
        measured = {}
        for wid, (x, tr) in traces.items():
            for dec in (False, True):
                rep = rd.replay(ctx, dec, rd.assignment_from_hidden(ctx, dec, tr[1]["h"]))
                if not rep["complete"]:
                    raise AuditStop("replay of a concrete state is incomplete")
                if [int(v) for v in rep["r3"]] != [int(tr[3]["r"][n]) for n in PAIR]:
                    raise AuditStop(f"{wid}: harness arithmetic r3 {rep['r3']} != exact forward")
                if not rep["assumptions_hold"]:
                    raise AuditStop(f"{wid}: a real state violates an assumption: {rep['failed_assumptions'][:3]}")
            measured[wid] = rep["e_sum"]
        summary["measured_e_sum"] = {"delta": DELTA, "values": measured, "min": min(measured.values()),
                                     "any_real_state_below_delta": min(measured.values()) < DELTA}
        print(json.dumps({"measured_e_sum": summary["measured_e_sum"]}), flush=True)
        parity = {}
        if shutil.which("gcc"):
            for dec in (False, True):
                vals = rd.assignment_from_hidden(ctx, dec, tr_nom[1]["h"])
                src = out / f"parity_{'decoupled' if dec else 'shared'}.c"
                src.write_text(rd.harness(ctx, dec, concrete=vals))
                exe = out / f"parity_{'decoupled' if dec else 'shared'}"
                subprocess.run(["gcc", "-O0", str(src), "-o", str(exe)], check=True)
                ok, ea, eb = (int(v) for v in subprocess.run([str(exe)], capture_output=True, text=True,
                                                               check=True).stdout.split())
                parity["decoupled" if dec else "shared"] = {"ok": ok, "e": [ea, eb], "python": [e_nom[n] for n in PAIR]}
                if not ok or [ea, eb] != [e_nom[n] for n in PAIR]:
                    raise AuditStop(f"gcc parity failed: {parity}")
        summary["gcc_parity_nominal"] = parity

        # ---------------- 5. two solver queries
        class DiagnosticRunner(ESBMCRunner):
            def build_command(self, c_file, unwind, profile):
                return tuple(a for a in super().build_command(c_file, unwind, profile) if a != "--result-only")

        runner = DiagnosticRunner(ESBMCConfig(timeout_seconds=QUERY_TIMEOUT, memlimit=QUERY_MEM,
                                              default_profile="paper-z3"))
        stage_deadline = time.monotonic() + STAGE_CAP
        queries = {}
        for arm, dec in (("shared", False), ("decoupled", True)):
            ctx.label = f"diag/{ORD}/I-R/L3n{PAIR[0]}+L3n{PAIR[1]}/{arm}"
            src = rd.harness(ctx, dec)
            qdir = out / "obligations" / arm
            qdir.mkdir(parents=True)
            path = qdir / "harness.c"
            path.write_text(src)
            rec = {"label": ctx.label, "harness": str(path), "harness_sha256": sha256_file(path),
                   "harness_bytes": path.stat().st_size, "variables": len(ctx.variables(dec)),
                   "multiply_terms": struct["multiply_terms"]}
            if time.monotonic() + QUERY_TIMEOUT > stage_deadline:
                rec.update(status="NOT_RUN_BUDGET")
                queries[arm] = rec
                continue
            res = runner.run_file(path, "paper-z3", extract_counterexample=False)
            log = "\n".join(Path(p).read_text(errors="replace") for p in (res.stdout_log_path, res.stderr_log_path)
                            if p and Path(p).exists())
            rec.update(status=res.status, command=list(res.command), elapsed_seconds=res.elapsed_seconds,
                       peak_memory_bytes=res.peak_memory_bytes, cpu_time_seconds=res.cpu_time_seconds,
                       stdout_log=res.stdout_log_path, stderr_log=res.stderr_log_path,
                       effective_profile="diagnostic-paper-z3-with-trace")
            if res.status == "FAILED":
                rec.update(cd.parse_violation(log, ctx.label))
                if rec["failed_claim_kind"] == "TARGET":
                    values = rd.parse_assignment(log)
                    rep = rd.replay(ctx, dec, values)
                    rec["replay"] = rep
                    (qdir / "assignment.json").write_text(json.dumps(values, sort_keys=True) + "\n")
            queries[arm] = rec
            print(json.dumps({"query": arm, **{k: rec.get(k) for k in ("status", "failed_claim_kind",
                                                                        "elapsed_seconds", "peak_memory_bytes")},
                              "replay": {k: rec.get("replay", {}).get(k) for k in ("complete", "assumptions_hold",
                                                                                   "e", "e_sum", "violates")}}),
                  flush=True)
        summary["queries"] = queries

        def confirmed_violation(q):
            r = q.get("replay") or {}
            return q.get("status") == "FAILED" and q.get("failed_claim_kind") == "TARGET" and r.get("complete") \
                and r.get("assumptions_hold") and r.get("violates")

        sh, de = queries["shared"], queries["decoupled"]
        if confirmed_violation(sh):
            outcome = "FAIL"
        elif sh.get("status") == "VERIFIED" and confirmed_violation(de):
            outcome = "PASS"
        elif sh.get("status") == "VERIFIED" and de.get("status") == "VERIFIED":
            outcome = "DEPENDENCE_NOT_DEMONSTRATED"
        else:
            outcome = "INCONCLUSIVE"
        summary["outcome"] = outcome
        summary["evidence_kind"] = "CONDITIONAL_ESBMC_LEMMA" if sh.get("status") == "VERIFIED" else \
            ("ESBMC_CHECKED_OBLIGATION" if sh.get("status") in ("FAILED",) else "NOT_ESTABLISHED")
    except Stop as stop:
        summary.update(outcome=stop.outcome, reason=str(stop))
    except (AuditStop, cd.ImplementationError, pd.ReplayError) as exc:
        summary.update(outcome="AUDIT_STOP" if isinstance(exc, AuditStop) else "INCONCLUSIVE",
                       reason=f"{type(exc).__name__}: {exc}")
    summary["wall_seconds"] = round(time.monotonic() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"outcome": summary["outcome"], "reason": summary.get("reason"),
                      "wall_seconds": summary["wall_seconds"]}), flush=True)
    return 0 if summary["outcome"] != "AUDIT_STOP" else 4


if __name__ == "__main__":
    sys.exit(main())
