#!/usr/bin/env python
"""I-P: one bounded two-child input partition of 016 c5 from the frozen P3c parent (017: unchanged control).

DIAGNOSTIC SCREEN ONLY. Untrusted finder arithmetic; no ESBMC, no export, no VERIFIED label.
Frozen parent = the P3c cascade bounds plus P3 margin search (016: c5 -647, 38/42 closed), rebuilt
from the P3c side log and checked against the P3c summary. Children replay every parent coefficient
choice (crown_partition_diag.Replay); only ReLU constants, residual lower endpoints and the input
corners use the child domain. Spec: docs/ssv2026/IP_partition_016.md.

  cd /home/joao/code/preqbmc
  systemd-run --user --scope -q -p MemoryMax=8G /home/joao/miniconda3/envs/quad/bin/python -u \
    /home/joao/code/preqbmc-qformat/src/scripts/ip_partition_016.py \
    --run output/crown_diagnostics_20261031T161801Z --p3c output/p3c_cascade_pilot_20261031T192435Z \
    --output output/ip_partition_016_<UTC>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

WORKTREE = Path(__file__).resolve().parents[2]
SRC = WORKTREE / "src"
MAIN = Path("/home/joao/code/preqbmc")
_spec = importlib.util.spec_from_file_location("diagnostic_import_guard", SRC / "scripts" / "diagnostic_import_guard.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

TARGET_ORDINAL, TARGET_COMPETITOR, CONTROL_ORDINAL = "016", 5, "017"
PARENT_C5 = -647
PASS_C5 = -614             # parent + 33 (5 % of the remaining 648-unit deficit), pre-registered
STRONG_GAIN = 100
IP_WALL_CAP = 1200.0       # spec I-P cap for the child stage of 016
GLOBAL_CAP = 5400.0
BOX_BLOCK = 24             # neurons per ESBMC box block (pipeline description); used for an ASSUMED count
FROZEN = {
    "src/scripts/crown_finder/crown_chain_export_v2.py": "1541896e43a01ada3d688d311f8c3db676b83ba371a88920cc5d3f2db94a9146",
    "src/scripts/crown_finder/crown_chain_export.py": "84e61a573f180e39a6e537caf3232405d77e2f48fac15eae178ffd6df4342c36",
    "src/scripts/crown_finder/affine_export.py": "7657897bac0420885f6825f1e15179ce498a77134e4c13065abe8a4ef4fdbfc0",
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_arr(a) -> str:
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


class AuditStop(RuntimeError):
    pass


class Inconclusive(RuntimeError):
    pass


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
    from box_ledger import facts_from_box
    from verification import crown_diagnostics as cd
    from verification import crown_partition_diag as pd
    from verification.conv_interval_fast import propagate_box
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
    global_deadline = t_start + GLOBAL_CAP

    config = json.loads((run / "manifest.json").read_text())["config"]
    preflight = json.loads((run / "preflight.json").read_text())
    if not preflight["passed"]:
        raise SystemExit("source run preflight did not pass")
    dv = json.loads((run / "violations.json").read_text())
    rankings = json.loads((run / "rankings.json").read_text())
    manifest = {
        "schema": "preqbmc_ip_partition_v1", "spec": "docs/ssv2026/IP_partition_016.md",
        "spec_sha256": sha256_file(WORKTREE / "docs/ssv2026/IP_partition_016.md"),
        "script_sha256": sha256_file(__file__), "partition_module_sha256": sha256_file(pd.__file__),
        "diagnostics_module_sha256": sha256_file(cd.__file__),
        "guard_sha256": sha256_file(SRC / "scripts" / "diagnostic_import_guard.py"), "frozen": FROZEN,
        "source_run": str(run), "source_run_manifest_sha256": sha256_file(run / "manifest.json"),
        "rankings_sha256": sha256_file(run / "rankings.json"),
        "p3c_run": str(p3c), "p3c_summary_sha256": sha256_file(p3c / "summary.json"),
        "p3c_sides_sha256": sha256_file(p3c / "sides.jsonl.gz"),
        "import_audit": {k: v for k, v in audit.items() if k != "modules"},
        "imported_modules": [{"module": m["module"], "file": m["file"], "sha256": m["sha256"]} for m in audit["modules"]],
        "gate": {"parent_c5": PARENT_C5, "pass_c5": PASS_C5, "strong_gain": STRONG_GAIN, "ip_wall_cap": IP_WALL_CAP},
        "evidence_kind": "SCREEN_BOUND_UNCHECKED", "esbmc_runs": 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    v2.configure_format(int(config["qif"]["total_bits"]), int(config["qif"]["fractional_bits"]))
    F = int(config["qif"]["fractional_bits"])
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

    def graph(bounds, parents, facts, flags, side_pol, xl, xu, margin_pols, comps, target, closure):
        """Planned dependency graph (exporter closure rule) and cost tally, with frozen policies."""
        rep = pd.Replay(v2, Q_HIGH)
        rep.map(bounds, parents)
        cce.XU[0] = xu
        lams = np.zeros((len(comps), sizes[-1]), dtype=np.int64)
        lams[:, target] = ONE
        lams[np.arange(len(comps)), comps] = -ONE
        logit = sorted({ref for e in closure.values() for ref in
                        [(e["top"] or {}).get("reference"), (e["bottom"] or {}).get("reference")]
                        if ref and ref.startswith("chain/logit/")})
        roots = [lams] + [unit_rows([target], [1 if r.endswith("lower") else -1], sizes[-1]) for r in logit]
        table = np.concatenate([margin_pols, np.full((len(logit), top), ADAPTIVE, dtype=np.int64)])
        cost = {"multiply_terms": 0, "chains": 0, "two_sided_relu": 0, "relu_coordinates": 0, "policy_chains": 0,
                "box_facts_needed": 0, "logit_roots": logit}
        pending = {j: [np.zeros(sizes[j], bool) for _ in range(3)] for j in range(top)}

        def add(nd):
            for j, arrs in nd.items():
                for p, a in zip(pending[j], arrs):
                    p |= a

        def tally(E, layer, policies=None, ov=None):
            if ov is not None:
                with rep.keyed_override(parents[ov["layer"]], 0, ov["neuron"], ov["a"], ov["trial_a_out"]):
                    _, terms, steps = v2.run_chain(E, layer, mats, bounds, xl, record=True)
                    nd = v2.needs(E, layer, mats, bounds)
                    if rep.override_applied != 2:
                        raise pd.ReplayError(f"override applied {rep.override_applied} times")
            else:
                _, terms, steps = v2.run_chain(E, layer, mats, bounds, xl, record=True, policies=policies)
                nd = v2.needs(E, layer, mats, bounds, policies=policies)
            cost["multiply_terms"] += int(terms.sum())
            cost["chains"] += E.shape[0]
            for st in steps:
                if st["kind"] == "relu":
                    j = st["j"]
                    cls = v2.classify(st["a"], st["a_out"], bounds[j][2], bounds[j][3])
                    cost["two_sided_relu"] += int(((cls == 4) | (cls == 6)).sum())
                    cost["relu_coordinates"] += int((cls >= 0).sum())
            return nd, steps

        per_layer = {}
        with rep:
            nd, _ = tally(np.concatenate(roots), top, table)
            add(nd)
            for k in range(top - 1, -1, -1):
                need_l, need_uv, need_un = pending[k]
                if (need_l & (facts[k]["stable"] < 0)).any():
                    return {"status": "FINDER_INCONCLUSIVE", "cause": f"layer {k}: stable-inactive lower needed"}
                chain_l = need_l & ~flags[k]["lower_box"]
                chain_u = need_uv & flags[k]["upper_chain"]
                cost["box_facts_needed"] += int((need_l & flags[k]["lower_box"]).sum()
                                                + ((need_uv | need_un) & ~flags[k]["upper_chain"]).sum())
                per_layer[str(k)] = int(chain_l.sum() + chain_u.sum())
                for sign, mask, side in ((1, chain_l, "lower"), (-1, chain_u, "upper")):
                    idx = np.nonzero(mask)[0].tolist()
                    plain = [n for n in idx if (k, n, side) not in side_pol]
                    want = bounds[k][0] if side == "lower" else bounds[k][1]
                    for s0 in range(0, len(plain), 64):
                        rows = plain[s0:s0 + 64]
                        nd, steps = tally(unit_rows(rows, [sign] * len(rows), sizes[k]), k)
                        got = [endpoint(side, v) for v in steps[-1]["scaled"].tolist()]
                        if got != [int(want[n]) for n in rows]:
                            raise AuditStop(f"layer {k} {side}: graph chain does not reproduce its bound")
                        add(nd)
                    for n in idx:
                        p = side_pol.get((k, n, side))
                        if p is None:
                            continue
                        cost["policy_chains"] += 1
                        E = unit_rows([n], [sign], sizes[k])
                        nd, steps = tally(E, k, policies=np.array([p[1]], dtype=np.int64)) if p[0] == "A" \
                            else tally(E, k, ov=p[1])
                        if endpoint(side, steps[-1]["scaled"][0]) != int(want[n]):
                            raise AuditStop(f"L{k}:{n} {side}: policy chain does not reproduce its bound")
                        add(nd)
        cost["chains_planned"] = int(sum(per_layer.values()) + sum(r.shape[0] for r in roots))
        cost["chains_per_layer"] = per_layer
        cost["status"] = "OK"
        return cost

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

    def build_child(parent, xl_c, xu_c, deadline):
        """Child bounds with frozen sources and policies; box facts recomputed on the child box."""
        P = parent["bounds"]
        flags, side_pol, pfacts = parent["flags"], parent["side_pol"], parent["facts"]
        box = propagate_box(net, xl_c, xu_c)
        facts_c = facts_from_box(box, sizes)
        for k in range(top + 1):
            if (facts_c[k]["hl"] < pfacts[k]["hl"]).any() or (facts_c[k]["hh"] > pfacts[k]["hh"]).any():
                raise AuditStop(f"child box looser than parent at layer {k}")
        for k in range(top):
            lb, uc, sat = flags[k]["lower_box"], flags[k]["upper_chain"], P[k][2]
            if not ((facts_c[k]["stable"][lb] > 0).all() and facts_c[k]["lower_ok"][lb].all()):
                raise pd.ReplayError(f"layer {k}: a parent box lower side is not available in the child")
            ub = ~uc & ~sat
            if not facts_c[k]["upper_ok"][ub].all():
                raise pd.ReplayError(f"layer {k}: a parent box upper side is not available in the child")
        child = []
        for k in range(top):
            if time.monotonic() > deadline:
                raise Inconclusive("timeout while building child bounds")
            lb, uc = flags[k]["lower_box"], flags[k]["upper_chain"]
            items = [(n, "lower", side_pol.get((k, n, "lower"))) for n in np.nonzero(~lb)[0].tolist()] + \
                    [(n, "upper", side_pol.get((k, n, "upper"))) for n in np.nonzero(uc)[0].tolist()]
            cur = child + P[k:]
            vals = side_chains(cur, P, k, items, xl_c, xu_c)
            lo = np.where(lb, facts_c[k]["hl"], P[k][0]).astype(np.int64)
            hi = np.where(uc, P[k][1], facts_c[k]["hh"]).astype(np.int64)
            hi = np.where(P[k][2], P[k][1], hi)      # saturated: h = Q_HIGH, upper side unused; keep parent value
            for n, s, _ in items:
                (lo if s == "lower" else hi)[n] = vals[(n, s)]
            if (lo < P[k][0]).any() or (hi > P[k][1]).any():
                bad = int(np.nonzero((lo < P[k][0]) | (hi > P[k][1]))[0][0])
                raise AuditStop(f"layer {k}:{bad}: child bound looser than its parent (monotonicity broken)")
            child.append((lo, hi, P[k][2], P[k][3]))
        return child, facts_c

    def check_witnesses(bounds, facts_c, traces, xl_c, xu_c, tag):
        inside = {w: t for w, t in traces.items() if np.all(t[0] >= xl_c) and np.all(t[0] <= xu_c)}
        for wid, (_, tr) in inside.items():
            for k in range(top):
                r = tr[k]["r"]
                lo, hi, sat, _ = bounds[k]
                if (r < lo)[~(facts_c[k]["stable"] < 0)].any() or (r > hi)[~sat].any():
                    raise AuditStop(f"{tag}: layer-{k} bound contradicted at {wid}")
        return sorted(inside)

    def bounds_digest(bounds):
        return {f"L{k}": {"lo": sha_arr(b[0]), "hi": sha_arr(b[1]), "width_sum": int((b[1] - b[0]).sum())}
                for k, b in enumerate(bounds)}

    try:
        # ---------------- 1. architecture check (diagnostic only; never used by split selection)
        w3 = mats[3][0]
        csr = {}
        for o in ("013", "016", "017"):
            neurons = sorted({e["neuron"] for c in images[o]["focus"] for e in rankings[o][str(c)]["top16_global"]
                              if e["layer"] == 3})
            pred = pd.csr_predecessors(w3, neurons)
            rep = pd.overlap_report(pred)
            two_hop = {}
            w2 = mats[2][0].tocsr()
            for n in neurons:
                two_hop[n] = set(int(c) for p in pred[n] for c in w2.indices[w2.indptr[p]:w2.indptr[p + 1]])
            shared2 = [len(two_hop[a] & two_hop[b]) for i, a in enumerate(neurons) for b in neurons[i + 1:]]
            csr[o] = {"neurons": neurons, "immediate": {k: rep[k] for k in ("support_sizes", "pairs", "all_disjoint")},
                      "overlapping_pairs": rep["overlapping_pairs"],
                      "two_hop_layer1_shared": {"pairs": len(shared2), "pairs_sharing": sum(s > 0 for s in shared2),
                                                "max_shared": max(shared2, default=0)}}
        summary["csr_overlap"] = csr
        (out / "csr_overlap.json").write_text(json.dumps(csr, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"csr_disjoint": {o: v["immediate"]["all_disjoint"] for o, v in csr.items()}}), flush=True)

        # ---------------- 2. control 017: rebuild the frozen parent and replay it unchanged
        t0 = time.monotonic()
        par17 = reconstruct(CONTROL_ORDINAL)
        same, facts_same = build_child(par17, par17["data"]["xl"], par17["data"]["xu"], global_deadline)
        if not all(np.array_equal(a, b) for c, p in zip(same, par17["bounds"]) for a, b in zip(c, p)):
            raise AuditStop("017 unchanged replay changed a bound")
        ev_same = evaluate(same, par17["bounds"], facts_same, par17["data"]["xl"], par17["data"]["xu"], par17["pols"],
                           par17["comps"], par17["target"], par17["image"]["focus"])
        control_ok = ev_same["raw"] == par17["eval"]["raw"] and ev_same["closed"] == par17["eval"]["closed"]
        summary["control_017"] = {"reproduced": bool(control_ok),
                                  "primary": ev_same["raw"][par17["image"]["primary"]],
                                  "closed": len(ev_same["closed"]), "wall_seconds": round(time.monotonic() - t0, 1)}
        print(json.dumps({"control_017": summary["control_017"]}), flush=True)
        if not control_ok:
            raise AuditStop("017 unchanged replay does not reproduce its parent")

        # ---------------- 3. parent 016
        t0 = time.monotonic()
        par = reconstruct(TARGET_ORDINAL)
        c5 = TARGET_COMPETITOR
        if par["eval"]["raw"][c5] != PARENT_C5:
            raise AuditStop(f"parent c5 {par['eval']['raw'][c5]} != {PARENT_C5}")
        parent_closed = par["eval"]["closed"]
        data = par["data"]
        xl, xu, blo, bhi = data["xl"], data["xu"], data["blo"].reshape(-1).astype(np.int64), \
            data["bhi"].reshape(-1).astype(np.int64)
        enc_v = lambda v: np.minimum((1 << 15) - 1, (v * (1 << F) + 128) // 256)
        if not (np.array_equal(enc_v(blo), xl) and np.array_equal(enc_v(bhi), xu)):
            raise AuditStop("cached encoded box is not the encoding of the cached byte box")
        lam5, steps5, scaled5 = par["eval"]["recorded"][c5]
        final_a, const = steps5[-1]["a"][0], int(steps5[-1]["c_in"][0])
        parent_wall = round(time.monotonic() - t0, 1)
        print(json.dumps({"parent_016": {"c5": PARENT_C5, "closed": len(parent_closed), "wall_seconds": parent_wall}}),
              flush=True)

        # ---------------- 4. split selection (frozen before any child is evaluated)
        sel = pd.select_split(final_a, blo, bhi)
        if sel is None:
            raise Inconclusive("no varying byte coordinate with nonzero coefficient")
        if int(xu[sel["index"]] - xl[sel["index"]]) != int(bhi[sel["index"]] - blo[sel["index"]]):
            raise AuditStop("encoded width differs from byte width at the split coordinate")
        shape = tuple(int(v) for v in net.input_shape)
        sel["position_hwc"] = [int(v) for v in np.unravel_index(sel["index"], shape)]
        sel["input_shape"] = list(shape)
        kids_b = pd.split_bytes(blo, bhi, sel["index"])
        part = pd.check_partition((blo, bhi), kids_b, sel["index"])
        sel["partition_check"] = part
        (out / "split.json").write_text(json.dumps(sel, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"split": {k: sel[k] for k in ("index", "position_hwc", "a", "low", "high", "mid", "score")},
                          "partition_ok": part["passed"]}), flush=True)
        if not part["passed"]:
            raise Inconclusive("incomplete child coverage")
        kids = [(enc_v(lo).astype(np.int64), enc_v(hi).astype(np.int64)) for lo, hi in kids_b]
        worst = pd.worst_child_of_form(const, final_a, kids)
        if min(worst) != scaled5:
            raise pd.PartitionAccountingError(f"worst child of the parent form {min(worst)} != parent {scaled5}")

        # ---------------- 5. children
        traces = witnesses(TARGET_ORDINAL, data)
        parent_cost = graph(par["bounds"], par["bounds"], par["facts"], par["flags"], par["side_pol"], xl, xu,
                            par["pols"], par["comps"], par["target"], par["eval"]["closure"])
        ip_deadline = min(time.monotonic() + IP_WALL_CAP, global_deadline)
        children = []
        with cd.ResourceMonitor() as res:
            for ci, (cxl, cxu) in enumerate(kids):
                tc = time.monotonic()
                cb, cf = build_child(par, cxl, cxu, ip_deadline)
                inside = check_witnesses(cb, cf, traces, cxl, cxu, f"child {ci}")
                ev = evaluate(cb, par["bounds"], cf, cxl, cxu, par["pols"], par["comps"], par["target"],
                              par["image"]["focus"], record_rows=[c5])
                lamc, stc, scc = ev["recorded"][c5]
                for s_p, s_c in zip(steps5, stc):
                    if not np.array_equal(s_p["a"], s_c["a"]):
                        raise pd.ReplayError("child c5 chain left the parent's coefficient path")
                for wid in inside:
                    x, tr = traces[wid]
                    try:
                        cd.account_chain(stc, 0, lamc[0], tr, x, cxl, cxu, biases, scc)
                    except cd.ImplementationError as exc:
                        raise AuditStop(f"child {ci} c5 accounting at {wid}: {exc}")
                if time.monotonic() > ip_deadline:
                    raise Inconclusive("timeout before the child cost graph")
                cost = graph(cb, par["bounds"], cf, par["flags"], par["side_pol"], cxl, cxu, par["pols"], par["comps"],
                             par["target"], ev["closure"])
                children.append({"child": ci, "byte_values": part["child_values"][ci], "c5": ev["raw"][c5],
                                 "c5_scaled": str(ev["scaled"][c5]), "closed": ev["closed"],
                                 "closed_count": len(ev["closed"]), "focus": ev["focus"], "raw_t": ev["raw_t"],
                                 "clamp": {str(c): {"top": (e["top"] or {}).get("rule"),
                                                    "bottom": (e["bottom"] or {}).get("rule")}
                                           for c, e in ev["closure"].items()},
                                 "bounds": bounds_digest(cb), "witnesses_inside": inside, "cost": cost,
                                 "wall_seconds": round(time.monotonic() - tc, 1)})
                print(json.dumps({"child": ci, "c5": ev["raw"][c5], "closed": len(ev["closed"]),
                                  "wall_seconds": children[-1]["wall_seconds"]}), flush=True)
        metrics, _ = res.metrics()
        worst_c5 = min(ch["c5"] for ch in children)
        regress = {ci: sorted(set(parent_closed) - set(ch["closed"])) for ci, ch in enumerate(children)}
        newly = sorted(set.intersection(*(set(ch["closed"]) for ch in children)) - set(parent_closed))
        graphs_ok = parent_cost.get("status") == "OK" and all(ch["cost"].get("status") == "OK" for ch in children)
        combined_terms = sum(ch["cost"].get("multiply_terms", 0) for ch in children)
        bridge = {"partition_coverage_lemma": 1, "child_byte_encoding_contracts": 2, "child_nonvacuity_witnesses": 2,
                  "child_box_ledger_blocks_ASSUMED": 2 * sum(math.ceil(n / BOX_BLOCK) for n in sizes)}
        cost_gate = {"parent_multiply_terms": parent_cost.get("multiply_terms"),
                     "combined_child_multiply_terms": combined_terms,
                     "ratio": round(combined_terms / parent_cost["multiply_terms"], 3) if graphs_ok else None,
                     "graphs_status": [parent_cost.get("status")] + [ch["cost"].get("status") for ch in children],
                     "parent_chains": parent_cost.get("chains_planned"),
                     "combined_child_chains": sum(ch["cost"].get("chains_planned", 0) for ch in children),
                     "bridge_obligations_separate": bridge,
                     "peak_rss_bytes": metrics.get("peak_process_tree_rss_bytes"),
                     "peak_cgroup_bytes": metrics.get("peak_cgroup_memory_bytes"),
                     "planned_assertions": None, "planned_harnesses": None, "serialized_proof_bytes": None,
                     "not_measured_reason": "no exporter path for replayed child chains in this diagnostic; "
                                            "multiply terms and chains are the measured proxies"}
        mem_ok = (metrics.get("peak_process_tree_rss_bytes") or 0) < (8 << 30)
        cost_gate["passed"] = bool(cost_gate["ratio"] is not None and cost_gate["ratio"] <= 2.0 and mem_ok)
        all_regress = sorted({c for v in regress.values() for c in v})
        gain = worst_c5 - PARENT_C5
        if worst_c5 >= PASS_C5 and not all_regress and summary["control_017"]["reproduced"]:
            outcome = "STRONG_PASS" if gain >= STRONG_GAIN or newly else "DIAGNOSTIC_PASS"
        else:
            outcome = "FAIL"
            summary["reason"] = "worst-child c5 gain below 33" if gain < 33 else \
                "threshold met but a parent-closed competitor is not closed in both children"
        summary.update({
            "outcome": outcome, "split": {k: sel[k] for k in ("index", "position_hwc", "a", "low", "high", "mid",
                                                              "score")},
            "child_byte_values": part["child_values"],
            "parent": {"c5": PARENT_C5, "c5_scaled": str(scaled5), "closed": parent_closed,
                       "closed_count": len(parent_closed), "bounds": bounds_digest(par["bounds"]), "cost": parent_cost,
                       "wall_seconds": parent_wall,
                       "policy_sides": len(par["side_pol"]),
                       "margin_policies_searched": int((par["pols"] >= 0).any(axis=1).sum())},
            "children": children, "worst_child_c5": worst_c5, "worst_child_gain": gain,
            "closed_regressions": regress, "newly_closed_in_both": newly, "cost_gate": cost_gate,
            "partition_accounting": {"parent_form_child_minima": [str(v) for v in worst], "parent_scaled": str(scaled5),
                                     "passed": True},
            "ip_stage_wall_seconds": round(sum(ch["wall_seconds"] for ch in children), 1),
            "production_two_child_rule_justified": outcome in ("DIAGNOSTIC_PASS", "STRONG_PASS") and cost_gate["passed"],
        })
    except pd.PartitionAccountingError as exc:
        summary.update(outcome="AUDIT_STOP", reason=f"PARTITION_ACCOUNTING_ERROR: {exc}")
    except AuditStop as exc:
        summary.update(outcome="AUDIT_STOP", reason=str(exc))
    except (Inconclusive, pd.ReplayError) as exc:
        summary.update(outcome="INCONCLUSIVE", reason=f"{type(exc).__name__}: {exc}")
    except MemoryError as exc:
        summary.update(outcome="INCONCLUSIVE", reason=f"memory: {exc}")
    summary["wall_seconds"] = round(time.monotonic() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"outcome": summary["outcome"], "reason": summary.get("reason"),
                      "wall_seconds": summary["wall_seconds"]}), flush=True)
    return 0 if summary["outcome"] != "AUDIT_STOP" else 4


if __name__ == "__main__":
    sys.exit(main())
