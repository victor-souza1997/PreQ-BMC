#!/usr/bin/env python
"""P3c pilot: slope search in the layer-2 and layer-3 bound chains, cascaded (docs/ssv2026/P3c_cascade_pilot.md).

DIAGNOSTIC SCREEN ONLY. Untrusted finder arithmetic; no ESBMC run, no certificate, no VERIFIED label.

  cd /home/joao/code/preqbmc
  systemd-run --user --scope -q -p MemoryMax=8G /home/joao/miniconda3/envs/quad/bin/python -u \
    /home/joao/code/preqbmc-qformat/src/scripts/p3c_cascade_pilot.py \
    --run output/crown_diagnostics_20261031T161801Z --p3b output/p3b_bound_slope_pilot_20261031T173914Z \
    --output output/p3c_cascade_pilot_<UTC>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

WORKTREE = Path(__file__).resolve().parents[2]
SRC = WORKTREE / "src"
MAIN = Path("/home/joao/code/preqbmc")
_spec = importlib.util.spec_from_file_location("diagnostic_import_guard", SRC / "scripts" / "diagnostic_import_guard.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

WALL_CAP = 7200.0
GATE = {"013": -947, "016": -648}      # P3b primary + 5 % of the remaining gap (pre-registered)
FROZEN = {
    "src/scripts/crown_finder/crown_chain_export_v2.py": "1541896e43a01ada3d688d311f8c3db676b83ba371a88920cc5d3f2db94a9146",
    "src/scripts/crown_finder/crown_chain_export.py": "84e61a573f180e39a6e537caf3232405d77e2f48fac15eae178ffd6df4342c36",
    "src/scripts/crown_finder/affine_export.py": "7657897bac0420885f6825f1e15179ce498a77134e4c13065abe8a4ef4fdbfc0",
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditStop(RuntimeError):
    pass


class BudgetStop(RuntimeError):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--p3b", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="smoke test: these ordinals only (never the gate run)")
    args = ap.parse_args(argv)
    if Path.cwd().resolve() != MAIN.resolve():
        raise SystemExit(f"run from {MAIN}")
    finder = guard.install(SRC, MAIN / "src")
    import numpy as np
    import affine_export
    import crown_chain_export as cce
    import crown_chain_export_v2 as v2
    from verification import crown_diagnostics as cd
    audit = guard.audit(finder, MAIN / "src")
    if not audit["passed"]:
        raise SystemExit(f"import audit failed: {audit['violations']}")
    for path, sha in FROZEN.items():
        if sha256_file(WORKTREE / path) != sha:
            raise SystemExit(f"frozen source changed: {path}")
    run, p3b, out = args.run.resolve(), args.p3b.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; use a fresh directory")
    p3b_summary = json.loads((p3b / "summary.json").read_text())
    if (p3b / "VOID.txt").exists() or p3b_summary["gate"] != "PASS":
        raise SystemExit("P3b run is void or did not pass")
    out.mkdir(parents=True)
    start = time.monotonic()
    deadline = start + WALL_CAP
    wall_deadline = time.time() + WALL_CAP          # search_policies() compares against time.time()

    config = json.loads((run / "manifest.json").read_text())["config"]
    preflight = json.loads((run / "preflight.json").read_text())
    if not preflight["passed"]:
        raise SystemExit("source run preflight did not pass")
    dv = json.loads((run / "violations.json").read_text())
    manifest = {
        "schema": "preqbmc_p3c_pilot_v1", "spec": "docs/ssv2026/P3c_cascade_pilot.md",
        "spec_sha256": sha256_file(WORKTREE / "docs/ssv2026/P3c_cascade_pilot.md"),
        "script_sha256": sha256_file(__file__), "diagnostics_module_sha256": sha256_file(cd.__file__),
        "guard_sha256": sha256_file(SRC / "scripts" / "diagnostic_import_guard.py"), "frozen": FROZEN,
        "source_run": str(run), "source_run_manifest_sha256": sha256_file(run / "manifest.json"),
        "p3b_run": str(p3b), "p3b_summary_sha256": sha256_file(p3b / "summary.json"),
        "import_audit": {k: v for k, v in audit.items() if k != "modules"},
        "imported_modules": [{"module": m["module"], "file": m["file"], "sha256": m["sha256"]} for m in audit["modules"]],
        "wall_cap_seconds": WALL_CAP, "gate": GATE, "only": args.only,
        "evidence_kind": "SCREEN_BOUND_UNCHECKED", "esbmc_runs": 0,
    }
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
    per_side = int(config["slope_coordinates_per_root"])
    summary = {"images": {}, "gate": None}
    side_log = gzip.open(out / "sides.jsonl.gz", "wt", encoding="utf-8")

    def load_cache(ordinal):
        entry = preflight["caches"][ordinal]
        if sha256_file(entry["npz"]) != entry["npz_sha256"] or sha256_file(entry["json"]) != entry["json_sha256"]:
            raise AuditStop(f"{ordinal}: cache changed after preflight")
        data = dict(np.load(entry["npz"], allow_pickle=False))
        meta = json.loads(Path(entry["json"]).read_text())
        bounds = [(data[f"b{k}_lo"], data[f"b{k}_hi"], data[f"b{k}_sat"], data[f"b{k}_clip"]) for k in range(top)]
        info = [{key: data[f"i{k}_{key}"] for key in ("lo_chain", "hi_chain", "lower_box", "upper_chain")}
                for k in range(top)]
        facts = [{key: data[f"f{k}_{key}"] for key in ("hl", "hh", "stable", "upper_ok", "lower_ok")}
                 for k in range(top + 1)]
        return data, meta, bounds, info, facts

    def select(L, lo_c, hi_c, f):
        """bounds_v2 selection rule for one layer, given per-neuron chain values."""
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
        return (lo, hi, saturated, clipped), {"lo_chain": lo_c, "hi_chain": hi_c, "lower_box": lower_box,
                                              "upper_chain": upper_chain}

    def endpoint(side, scaled):
        return cd.ceil_div(int(scaled), ONE) if side == "lower" else (-int(scaled)) // ONE

    try:
        for image in [im for im in config["images"] if not args.only or im["ordinal"] in args.only]:
            o, target, primary = image["ordinal"], image["target"], image["primary"]
            t0 = time.monotonic()
            data, meta, bounds0, info0, facts = load_cache(o)
            xl, xu = data["xl"], data["xu"]
            comps = meta["comps"]
            x_nom = enc.encode(data["crop"]).astype(np.int64)
            traces = {f"{o}/nominal": (x_nom, cd.exact_forward(mats, x_nom, relu))}
            for w in dv["retained"]:
                if w["ordinal"] != o:
                    continue
                if sha256_file(w["encoded_path"]) != w["encoded_sha256"]:
                    raise AuditStop(f"{w['witness_id']}: witness file changed")
                x = np.load(w["encoded_path"], allow_pickle=False)
                if np.all(x >= xl) and np.all(x <= xu):
                    traces[w["witness_id"]] = (x, cd.exact_forward(mats, x, relu))
            for L in (2, 3):     # audit: the selection rule reproduces the cached bounds
                b, _ = select(L, info0[L]["lo_chain"], info0[L]["hi_chain"], facts[L])
                if not all(np.array_equal(p, q) for p, q in zip(b, bounds0[L])):
                    raise AuditStop(f"{o}: selection rule does not reproduce cached layer-{L} bounds")

            def chains(rows, L, bounds):
                out_sc = []
                cce.XU[0] = xu
                for s0 in range(0, rows.shape[0], 128):
                    sc, _, _ = v2.run_chain(rows[s0:s0 + 128], L, mats, bounds, xl)
                    out_sc += [int(v) for v in sc]
                return out_sc

            def unit_rows(idx, signs, width):
                E = np.zeros((len(idx), width), dtype=np.int64)
                E[np.arange(len(idx)), idx] = np.asarray(signs, dtype=np.int64) * ONE
                return E

            def tighten(L, bounds, info_L, recompute, arms, tag):
                """New layer-L bounds under `bounds` for layers < L. Returns (bounds_L, info_L, counts)."""
                lo, hi, sat, clip = bounds[L]
                n_all = lo.size
                if recompute:
                    idx = list(range(n_all))
                    lo_s = chains(unit_rows(idx, [1] * n_all, n_all), L, bounds)
                    hi_s = chains(unit_rows(idx, [-1] * n_all, n_all), L, bounds)
                    lo_c = np.array([endpoint("lower", v) for v in lo_s], dtype=np.int64)
                    hi_c = np.array([endpoint("upper", v) for v in hi_s], dtype=np.int64)
                else:
                    lo_c, hi_c = info_L["lo_chain"].copy(), info_L["hi_chain"].copy()
                targets = np.nonzero((lo < 0) & (hi > 0) & ~sat & ~clip)[0].tolist()
                sides = [(n, s) for n in targets for s in ("lower", "upper")]
                counts = {"sides": len(sides), "improved_A": 0, "improved_B": 0, "recomputed": bool(recompute)}
                if arms and sides:
                    rows = unit_rows([n for n, _ in sides], [1 if s == "lower" else -1 for _, s in sides], n_all)
                    base = chains(rows, L, bounds)
                    for i, (n, s) in enumerate(sides):
                        if endpoint(s, base[i]) != int((lo_c if s == "lower" else hi_c)[n]):
                            raise AuditStop(f"{o} L{L}:{n} {s}: unit chain differs from chain value")
                    a_sc = [None] * len(sides)
                    cce.XU[0] = xu
                    for s0 in range(0, len(sides), 64):
                        if time.monotonic() > deadline:
                            raise BudgetStop(f"{tag} arm A")
                        pol, best, tr = v2.search_policies(rows[s0:s0 + 64], L, mats, bounds, xl,
                                                           deadline=wall_deadline)
                        for k in range(best.shape[0]):
                            if tr[k] is None:
                                raise BudgetStop(f"{tag} arm A row unfinished")
                            a_sc[s0 + k] = (int(best[k]), [int(p) for p in pol[k]])
                    for i, (n, s) in enumerate(sides):
                        if time.monotonic() > deadline:
                            raise BudgetStop(f"{tag} arm B")
                        A = rows[i:i + 1]
                        cce.XU[0] = xu
                        sc, _, steps = v2.run_chain(A, L, mats, bounds, xl, record=True)
                        per_neuron = {}
                        for wid, (x, tr) in traces.items():
                            try:
                                acc = cd.account_chain(steps, 0, A[0], tr[:L + 1], x, xl, xu, biases, int(sc[0]))
                            except cd.ImplementationError as exc:
                                raise AuditStop(f"{o} L{L}:{n} {s} unit chain at {wid}: {exc}")
                            for rec in acc["steps"]:
                                for neuron, slack in rec.pop("_per_coordinate", {}).items():
                                    key = (rec["layer"], neuron)
                                    per_neuron[key] = max(per_neuron.get(key, 0), slack)
                        relu_a = {st["j"]: st["a"][0] for st in steps if st["kind"] == "relu"}
                        relu_out = {st["j"]: st["a_out"][0] for st in steps if st["kind"] == "relu"}
                        best_b = (int(sc[0]), None)
                        for _, _, j, k in cd.rank_coordinates(per_neuron, relu_a, bounds)[:per_side]:
                            a, a_base = int(relu_a[j][k]), int(relu_out[j][k])
                            for value in cd.slope_candidates(a, a_base):
                                with cd.SlopeOverride(v2, bounds[j], 0, k, a, value) as ov:
                                    tsc, _, _ = v2.run_chain(A, L, mats, bounds, xl)
                                if ov.applied != 1:
                                    raise AuditStop(f"override applied {ov.applied} times")
                                if int(tsc[0]) > best_b[0]:
                                    best_b = (int(tsc[0]), {"layer": j, "neuron": k, "a": a,
                                                            "baseline_a_out": a_base, "trial_a_out": value})
                        cands = {"baseline": (int(base[i]), None), "A": a_sc[i], "B": best_b}
                        name = max(("baseline", "A", "B"), key=lambda k2: (cands[k2][0], k2 == "baseline"))
                        if cands[name][0] > int(base[i]):
                            e = endpoint(s, cands[name][0])
                            (lo_c if s == "lower" else hi_c)[n] = e
                            counts[f"improved_{name}"] += 1
                        else:
                            name = "baseline"
                        side_log.write(json.dumps({"ordinal": o, "config": tag, "layer": L, "neuron": n, "side": s,
                                                   "baseline": endpoint(s, base[i]),
                                                   "arm_A": endpoint(s, a_sc[i][0]), "arm_A_policy": a_sc[i][1],
                                                   "arm_B": endpoint(s, best_b[0]), "arm_B_policy": best_b[1],
                                                   "chosen": name, "chain": int((lo_c if s == "lower"
                                                                                 else hi_c)[n])}) + "\n")
                for wid, (_, tr) in traces.items():      # chain values hold at every execution
                    r = tr[L]["r"]
                    if (r < lo_c).any() or (r > hi_c).any():
                        bad = int(np.nonzero((r < lo_c) | (r > hi_c))[0][0])
                        raise AuditStop(f"{o} {tag} L{L}:{bad}: chain value contradicted at {wid}")
                new_b, new_i = select(L, lo_c, hi_c, facts[L])
                for wid, (_, tr) in traces.items():
                    r = tr[L]["r"]
                    lo_s, hi_s, sat_s, _ = new_b
                    if (r < lo_s)[~(facts[L]["stable"] < 0)].any() or (r > hi_s)[~sat_s].any():
                        raise AuditStop(f"{o} {tag}: selected layer-{L} bound contradicted at {wid}")
                lo_n, hi_n = new_b[0], new_b[1]
                tgt = np.array(targets, dtype=np.int64)
                counts["width_sum_targets"] = {"before": int((hi[tgt] - lo[tgt]).sum()) if tgt.size else 0,
                                               "after": int((hi_n[tgt] - lo_n[tgt]).sum()) if tgt.size else 0}
                counts["unstable_after"] = int(((lo_n < 0) & (hi_n > 0) & ~new_b[2] & ~new_b[3]).sum())
                return new_b, new_i, counts

            def margins(bnds, search=False):
                lams = np.zeros((len(comps), sizes[-1]), dtype=np.int64)
                lams[:, target] = ONE
                lams[np.arange(len(comps)), comps] = -ONE
                e = unit_rows([target, target], [1, -1], sizes[-1])
                cce.XU[0] = xu
                sc, _, _ = v2.run_chain(lams, top, mats, bnds, xl)
                ts, _, _ = v2.run_chain(e, top, mats, bnds, xl)
                if search:
                    open_rows = [k for k in range(len(comps)) if cd.ceil_div(int(sc[k]), ONE) < 1]
                    if open_rows:
                        _, best, tr = v2.search_policies(lams[open_rows], top, mats, bnds, xl, deadline=wall_deadline)
                        if any(t is None for t in tr):
                            raise BudgetStop("margin search")
                        sc = sc.copy()
                        for r, k in enumerate(open_rows):
                            sc[k] = max(int(sc[k]), int(best[r]))
                raw = {c: cd.ceil_div(int(v), ONE) for c, v in zip(comps, sc.tolist())}
                rt = [cd.ceil_div(int(ts[0]), ONE), (-int(ts[1])) // ONE]
                clo = cd.closure_edges(raw, facts[top]["hh"], facts[top]["hl"], facts[top]["lower_ok"], rt[0], rt[1],
                                       target)
                return {"primary": raw[primary], "closed": sum(v["closed"] for v in clo.values()),
                        "margin_min": min(raw.values()), "focus": {str(c): raw[c] for c in image["focus"]},
                        "open": sorted((v, c) for c, v in raw.items() if v < 1)}

            saved_xu = cce.XU[0]
            res = {"counts": {}}
            try:
                res["baseline"] = margins(bounds0)
                if res["baseline"]["primary"] != int(meta["raw_lb"][str(primary)]):
                    raise AuditStop(f"{o}: baseline primary differs from cache")
                # P3b reproduction: layer 3 only, arms on, layers 0-2 cached
                b3, _, c3 = tighten(3, bounds0, info0[3], False, True, "p3b_repro")
                p3b_bounds = [b if k != 3 else b3 for k, b in enumerate(bounds0)]
                res["p3b_repro"] = margins(p3b_bounds)
                res["counts"]["p3b_repro_L3"] = c3
                want = p3b_summary["images"][o]["new_primary"]
                if res["p3b_repro"]["primary"] != want:
                    raise AuditStop(f"{o}: P3b reproduction {res['p3b_repro']['primary']} != {want}")
                # cascade: layer 2 first
                b2, i2, c2 = tighten(2, bounds0, info0[2], False, True, "cascade_L2")
                res["counts"]["cascade_L2"] = c2
                l2_bounds = [b if k != 2 else b2 for k, b in enumerate(bounds0)]
                b3o, _, c3o = tighten(3, l2_bounds, None, True, False, "l2_only_L3")
                res["counts"]["l2_only_L3"] = c3o
                res["l2_only"] = margins([b if k != 3 else b3o for k, b in enumerate(l2_bounds)])
                b3c, _, c3c = tighten(3, l2_bounds, None, True, True, "cascade_L3")
                res["counts"]["cascade_L3"] = c3c
                cascade = [b if k != 3 else b3c for k, b in enumerate(l2_bounds)]
                res["cascade"] = margins(cascade)
                # exact accounting of every focus margin chain in the cascade configuration
                lam_f = np.zeros((len(image["focus"]), sizes[-1]), dtype=np.int64)
                lam_f[:, target] = ONE
                for k, c in enumerate(image["focus"]):
                    lam_f[k, c] = -ONE
                cce.XU[0] = xu
                fsc, _, fsteps = v2.run_chain(lam_f, top, mats, cascade, xl, record=True)
                for k, c in enumerate(image["focus"]):
                    for wid, (x, tr) in traces.items():
                        try:
                            cd.account_chain(fsteps, k, lam_f[k], tr, x, xl, xu, biases, int(fsc[k]))
                        except cd.ImplementationError as exc:
                            raise AuditStop(f"{o} c{c} cascade margin at {wid}: {exc}")
                res["accounting_checked"] = True
                res["cascade_margin_search"] = margins(cascade, search=True)
            finally:
                cce.XU[0] = saved_xu
            res["wall_seconds"] = round(time.monotonic() - t0, 1)
            res["witnesses"] = sorted(traces)
            summary["images"][o] = res
            print(json.dumps({"image": o, **{k: {kk: res[k][kk] for kk in ("primary", "closed", "margin_min")}
                                             for k in ("baseline", "p3b_repro", "l2_only", "cascade",
                                                       "cascade_margin_search")},
                              "wall_seconds": res["wall_seconds"]}), flush=True)
            (out / f"{o}.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    except AuditStop as stop:
        summary["audit_stop"] = str(stop)
        print(json.dumps({"AUDIT_STOP": str(stop)}), flush=True)
    except BudgetStop as stop:
        summary["budget_stop"] = str(stop)
        print(json.dumps({"NOT_RUN_BUDGET": str(stop)}), flush=True)
    finally:
        side_log.close()
    imgs = summary["images"]
    if "audit_stop" in summary:
        summary["gate"] = "AUDIT_STOP"
    elif args.only:
        summary["gate"] = "SMOKE_TEST_NO_GATE"
    elif "budget_stop" in summary or not all(o in imgs for o in GATE):
        passed = [o for o in GATE if o in imgs and imgs[o]["cascade"]["primary"] >= GATE[o]]
        summary["gate"] = "PASS" if passed else "NOT_RUN_BUDGET"
    else:
        hit = [o for o in GATE if imgs[o]["cascade"]["primary"] >= GATE[o]]
        all_closed = [o for o, v in imgs.items() if v["cascade"]["closed"] == 42 and not
                      next(im for im in config["images"] if im["ordinal"] == o).get("control")]
        summary["gate"] = "PASS" if hit or all_closed else "FAIL"
        summary["gate_hits"] = {"margin": hit, "all_closed": all_closed}
    if "017" in imgs and imgs["017"]["cascade"]["closed"] < 42:
        summary["control_flag"] = "017 lost a closed competitor"
    summary["wall_seconds"] = round(time.monotonic() - start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"gate": summary["gate"], "wall_seconds": summary["wall_seconds"]}), flush=True)
    return 0 if summary["gate"] != "AUDIT_STOP" else 4


if __name__ == "__main__":
    sys.exit(main())
