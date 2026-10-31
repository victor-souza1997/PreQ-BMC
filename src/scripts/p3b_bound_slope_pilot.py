#!/usr/bin/env python
"""P3b pilot: ReLU-slope search inside the layer-3 bound chains (docs/ssv2026/P3b_bound_slope_pilot.md).

DIAGNOSTIC SCREEN ONLY. Untrusted finder arithmetic; no ESBMC run, no certificate, no VERIFIED label.
Reads the eps-1 caches, retained witnesses and preflight of a finished diagnostic run (hash-checked),
recomputes the unit chains of every unstable layer-3 neuron under three policies, and reruns the
margins with the tightened layer-3 bounds.

  cd /home/joao/code/preqbmc
  systemd-run --user --scope -q -p MemoryMax=8G /home/joao/miniconda3/envs/quad/bin/python -u \
    /home/joao/code/preqbmc-qformat/src/scripts/p3b_bound_slope_pilot.py \
    --run output/crown_diagnostics_20261031T161801Z --output output/p3b_bound_slope_pilot_<UTC>
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

LAYER = 3
WALL_CAP = 3600.0
FROZEN = {
    "src/scripts/crown_finder/crown_chain_export_v2.py": "1541896e43a01ada3d688d311f8c3db676b83ba371a88920cc5d3f2db94a9146",
    "src/scripts/crown_finder/crown_chain_export.py": "84e61a573f180e39a6e537caf3232405d77e2f48fac15eae178ffd6df4342c36",
    "src/scripts/crown_finder/affine_export.py": "7657897bac0420885f6825f1e15179ce498a77134e4c13065abe8a4ef4fdbfc0",
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditStop(RuntimeError):
    pass


class Pinned:
    """Pin crown_chain_export.XU[0] for a block of chain computations, then restore it."""

    def __init__(self, cce, xu):
        self.cce, self.xu = cce, xu

    def __enter__(self):
        self.saved, self.cce.XU[0] = self.cce.XU[0], self.xu

    def __exit__(self, *exc):
        if self.cce.XU[0] is not self.xu:
            raise RuntimeError("XU replaced inside a pinned block")
        self.cce.XU[0] = self.saved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
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
    run, out = args.run.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; use a fresh directory")
    out.mkdir(parents=True)
    start = time.monotonic()
    deadline = start + WALL_CAP
    wall_deadline = time.time() + WALL_CAP     # search_policies() compares against time.time()

    manifest_src = json.loads((run / "manifest.json").read_text())
    config = manifest_src["config"]
    preflight = json.loads((run / "preflight.json").read_text())
    if not preflight["passed"]:
        raise SystemExit("source run preflight did not pass")
    dv = json.loads((run / "violations.json").read_text())
    manifest = {
        "schema": "preqbmc_p3b_pilot_v1", "spec": "docs/ssv2026/P3b_bound_slope_pilot.md",
        "spec_sha256": sha256_file(WORKTREE / "docs/ssv2026/P3b_bound_slope_pilot.md"),
        "script_sha256": sha256_file(__file__), "diagnostics_module_sha256": sha256_file(cd.__file__),
        "guard_sha256": sha256_file(SRC / "scripts" / "diagnostic_import_guard.py"),
        "frozen": FROZEN, "source_run": str(run), "source_run_manifest_sha256": sha256_file(run / "manifest.json"),
        "source_run_source_manifest_sha256": manifest_src["source_manifest_sha256"],
        "preflight_sha256": sha256_file(run / "preflight.json"),
        "violations_sha256": sha256_file(run / "violations.json"),
        "import_audit": {k: v for k, v in audit.items() if k != "modules"},
        "imported_modules": [{"module": m["module"], "file": m["file"], "sha256": m["sha256"]} for m in audit["modules"]],
        "wall_cap_seconds": WALL_CAP, "only": args.only, "evidence_kind": "SCREEN_BOUND_UNCHECKED", "esbmc_runs": 0,
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
    per_side = int(config["slope_coordinates_per_root"])
    summary = {"images": {}, "gate": None}
    neuron_log = gzip.open(out / "sides.jsonl.gz", "wt", encoding="utf-8")

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

    try:
        for image in [im for im in config["images"] if not args.only or im["ordinal"] in args.only]:
            o, target, primary = image["ordinal"], image["target"], image["primary"]
            t0 = time.monotonic()
            data, meta, bounds, info, facts = load_cache(o)
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
            lo, hi, sat, clip = bounds[LAYER]
            targets = np.nonzero((lo < 0) & (hi > 0) & ~sat & ~clip)[0]
            sides = [(int(n), s) for n in targets.tolist() for s in ("lower", "upper")]
            rows = np.zeros((len(sides), lo.size), dtype=np.int64)
            for i, (n, s) in enumerate(sides):
                rows[i, n] = ONE if s == "lower" else -ONE

            def endpoint(side, scaled):
                return cd.ceil_div(int(scaled), ONE) if side == "lower" else (-int(scaled)) // ONE

            # ---- baseline unit chains: must equal the cached bounds_v2 chain values
            base_scaled = np.empty(len(sides), dtype=object)
            with Pinned(cce, xu):
                for s0 in range(0, len(sides), 128):
                    sc, _, _ = v2.run_chain(rows[s0:s0 + 128], LAYER, mats, bounds, xl)
                    base_scaled[s0:s0 + 128] = [int(v) for v in sc]
            for i, (n, s) in enumerate(sides):
                cached = int(info[LAYER]["lo_chain" if s == "lower" else "hi_chain"][n])
                if endpoint(s, base_scaled[i]) != cached:
                    raise AuditStop(f"{o} L{LAYER}:{n} {s}: unit chain {endpoint(s, base_scaled[i])} != cached {cached}")

            # ---- arm A: production per-layer scalar search on each unit root
            a_scaled = [None] * len(sides)
            a_policy = [None] * len(sides)
            with Pinned(cce, xu):
                for s0 in range(0, len(sides), 64):
                    if time.monotonic() > deadline:
                        break
                    pol, best, trace = v2.search_policies(rows[s0:s0 + 64], LAYER, mats, bounds, xl,
                                                          deadline=wall_deadline)
                    for k in range(best.shape[0]):
                        if trace[k] is None and time.monotonic() < deadline - 5:
                            raise AuditStop(f"{o}: search_policies stopped before the deadline")
                        if trace[k] is not None:
                            a_scaled[s0 + k], a_policy[s0 + k] = int(best[k]), [int(p) for p in pol[k]]

            # ---- arm B: I-S single-coordinate family on the top-ranked coordinates of each unit chain
            b_scaled, b_policy = [None] * len(sides), [None] * len(sides)
            for i, (n, s) in enumerate(sides):
                if time.monotonic() > deadline:
                    break
                A = rows[i:i + 1]
                with Pinned(cce, xu):
                    sc, _, steps = v2.run_chain(A, LAYER, mats, bounds, xl, record=True)
                per_neuron = {}
                for wid, (x, tr) in traces.items():
                    try:
                        acc = cd.account_chain(steps, 0, A[0], tr[:LAYER + 1], x, xl, xu, biases, int(sc[0]))
                    except cd.ImplementationError as exc:
                        raise AuditStop(f"{o} L{LAYER}:{n} {s} baseline unit chain at {wid}: {exc}")
                    for rec in acc["steps"]:
                        for neuron, slack in rec.pop("_per_coordinate", {}).items():
                            key = (rec["layer"], neuron)
                            per_neuron[key] = max(per_neuron.get(key, 0), slack)
                relu_a = {st["j"]: st["a"][0] for st in steps if st["kind"] == "relu"}
                relu_out = {st["j"]: st["a_out"][0] for st in steps if st["kind"] == "relu"}
                best_b = (int(sc[0]), 0, None)
                for rank, (slack, _, j, k) in enumerate(cd.rank_coordinates(per_neuron, relu_a, bounds)[:per_side]):
                    a, a_base = int(relu_a[j][k]), int(relu_out[j][k])
                    for value in cd.slope_candidates(a, a_base):
                        with Pinned(cce, xu), cd.SlopeOverride(v2, bounds[j], 0, k, a, value) as ov:
                            tsc, _, _ = v2.run_chain(A, LAYER, mats, bounds, xl)
                        if ov.applied != 1:
                            raise AuditStop(f"override applied {ov.applied} times")
                        if int(tsc[0]) > best_b[0]:
                            best_b = (int(tsc[0]), rank, {"layer": j, "neuron": k, "a": a, "baseline_a_out": a_base,
                                                          "trial_a_out": value})
                b_scaled[i], b_policy[i] = best_b[0], best_b[2]

            # ---- assemble the tightened layer-3 bounds (strictly tighter only), check at every execution
            new_lo, new_hi = lo.copy(), hi.copy()
            src_lo = np.zeros(lo.size, dtype=np.int8)
            src_hi = np.zeros(lo.size, dtype=np.int8)
            arm_only = {"A": (lo.copy(), hi.copy()), "B": (lo.copy(), hi.copy())}
            counts = {"sides": len(sides), "A_finished": sum(v is not None for v in a_scaled),
                      "B_finished": sum(v is not None for v in b_scaled), "tightened": 0, "tightened_by_A": 0,
                      "tightened_by_B": 0, "not_run_budget": 0}
            for i, (n, s) in enumerate(sides):
                if a_scaled[i] is None or b_scaled[i] is None:
                    counts["not_run_budget"] += 1
                cands = {"baseline": int(base_scaled[i])}
                if a_scaled[i] is not None:
                    cands["A"] = a_scaled[i]
                if b_scaled[i] is not None:
                    cands["B"] = b_scaled[i]
                for name, scv in cands.items():
                    e = endpoint(s, scv)
                    bad = [wid for wid, (_, tr) in traces.items() if cd.endpoint_refuted(s, e, tr[LAYER]["r"][n])]
                    if bad:
                        raise AuditStop(f"{o} L{LAYER}:{n} {s} arm {name} endpoint {e} contradicted at {bad[0]}")
                selected = int((lo if s == "lower" else hi)[n])
                arr, srcs = (new_lo, src_lo) if s == "lower" else (new_hi, src_hi)
                best_name, best_e = None, selected
                for name in ("A", "B"):
                    if name in cands:
                        e = endpoint(s, cands[name])
                        if cd.tighter(s, e, selected):
                            only = arm_only[name][0 if s == "lower" else 1]
                            only[n] = e
                        if cd.tighter(s, e, best_e):
                            best_name, best_e = name, e
                if best_name is not None:
                    arr[n] = best_e
                    srcs[n] = 1 if best_name == "A" else 2
                    counts["tightened"] += 1
                    counts[f"tightened_by_{best_name}"] += 1
                neuron_log.write(json.dumps({"ordinal": o, "layer": LAYER, "neuron": n, "side": s,
                                             "selected_baseline": selected,
                                             "baseline_unit_chain": endpoint(s, base_scaled[i]),
                                             "arm_A": None if a_scaled[i] is None else endpoint(s, a_scaled[i]),
                                             "arm_A_policy": a_policy[i],
                                             "arm_B": None if b_scaled[i] is None else endpoint(s, b_scaled[i]),
                                             "arm_B_policy": b_policy[i], "chosen": best_name, "new": best_e,
                                             "nominal_raw": int(traces[f"{o}/nominal"][1][LAYER]["r"][n])}) + "\n")
            if (new_lo > new_hi).any():
                raise AuditStop(f"{o}: empty tightened interval")
            width_base = int((hi[targets] - lo[targets]).sum())
            width_new = int((new_hi[targets] - new_lo[targets]).sum())

            def with_layer(nl, nh):
                return [b if k != LAYER else (nl, nh, sat, clip) for k, b in enumerate(bounds)]

            def margins(bnds, search=False):
                lams = np.zeros((len(comps), sizes[-1]), dtype=np.int64)
                lams[:, target] = ONE
                lams[np.arange(len(comps)), comps] = -ONE
                e = np.zeros((2, sizes[-1]), dtype=np.int64)
                e[0, target], e[1, target] = ONE, -ONE
                with Pinned(cce, xu):
                    sc, terms, _ = v2.run_chain(lams, top, mats, bnds, xl)
                    ts, _, _ = v2.run_chain(e, top, mats, bnds, xl)
                    if search:
                        open_rows = [k for k in range(len(comps)) if cd.ceil_div(int(sc[k]), ONE) < 1]
                        if open_rows:
                            _, best, tr = v2.search_policies(lams[open_rows], top, mats, bnds, xl,
                                                             deadline=wall_deadline)
                            if any(t is None for t in tr):
                                raise AuditStop(f"{o}: margin search did not finish")
                            sc = sc.copy()
                            for r, k in enumerate(open_rows):
                                sc[k] = max(int(sc[k]), int(best[r]))
                raw = {c: cd.ceil_div(int(v), ONE) for c, v in zip(comps, sc.tolist())}
                rt = [cd.ceil_div(int(ts[0]), ONE), (-int(ts[1])) // ONE]
                clo = cd.closure_edges(raw, facts[top]["hh"], facts[top]["hl"], facts[top]["lower_ok"], rt[0], rt[1],
                                       target)
                return {"raw": raw, "scaled": {c: int(v) for c, v in zip(comps, sc.tolist())},
                        "closed": sum(v["closed"] for v in clo.values()), "margin_min": min(raw.values()),
                        "raw_t": rt, "terms_sum": int(sum(int(t) for t in terms))}

            base_m = margins(bounds)
            if base_m["raw"] != {int(c): v for c, v in meta["raw_lb"].items()}:
                raise AuditStop(f"{o}: baseline margins differ from the cache")
            new_bounds = with_layer(new_lo, new_hi)
            new_m = margins(new_bounds)
            a_m = margins(with_layer(*arm_only["A"]))
            b_m = margins(with_layer(*arm_only["B"]))
            searched = margins(new_bounds, search=True) if time.monotonic() < deadline else None

            # ---- exact accounting of every focus margin chain at every execution, new bounds
            lam_f = np.zeros((len(image["focus"]), sizes[-1]), dtype=np.int64)
            lam_f[:, target] = ONE
            for k, c in enumerate(image["focus"]):
                lam_f[k, c] = -ONE
            with Pinned(cce, xu):
                fsc, _, fsteps = v2.run_chain(lam_f, top, mats, new_bounds, xl, record=True)
            for k, c in enumerate(image["focus"]):
                for wid, (x, tr) in traces.items():
                    try:
                        cd.account_chain(fsteps, k, lam_f[k], tr, x, xl, xu, biases, int(fsc[k]))
                    except cd.ImplementationError as exc:
                        raise AuditStop(f"{o} c{c} new-bounds margin at {wid}: {exc}")
            threshold = cd.material_gain_threshold(base_m["raw"][primary], bool(image.get("control")))
            gain = new_m["raw"][primary] - base_m["raw"][primary]
            summary["images"][o] = {
                "primary": primary, "baseline_primary": base_m["raw"][primary], "new_primary": new_m["raw"][primary],
                "primary_gain": gain, "material_threshold": threshold, "material": gain >= threshold,
                "arm_A_only_primary": a_m["raw"][primary], "arm_B_only_primary": b_m["raw"][primary],
                "searched_primary": None if searched is None else searched["raw"][primary],
                "focus": {str(c): {"baseline": base_m["raw"][c], "new": new_m["raw"][c],
                                   "searched": None if searched is None else searched["raw"][c]}
                          for c in image["focus"]},
                "closed": {"baseline": base_m["closed"], "new": new_m["closed"],
                           "searched": None if searched is None else searched["closed"]},
                "margin_min": {"baseline": base_m["margin_min"], "new": new_m["margin_min"],
                               "searched": None if searched is None else searched["margin_min"]},
                "raw_t": {"baseline": base_m["raw_t"], "new": new_m["raw_t"]},
                "layer3_unstable": int(targets.size), "layer3_width_sum": {"baseline": width_base, "new": width_new},
                "counts": counts, "witnesses": sorted(traces), "wall_seconds": round(time.monotonic() - t0, 1),
                "accounting_checked": True, "control": bool(image.get("control")),
            }
            print(json.dumps({"image": o, **{k: summary["images"][o][k] for k in
                                             ("baseline_primary", "new_primary", "primary_gain", "material", "closed",
                                              "wall_seconds")}}), flush=True)
            (out / f"{o}.json").write_text(json.dumps(summary["images"][o], indent=1, sort_keys=True) + "\n")
    except AuditStop as stop:
        summary["audit_stop"] = str(stop)
        print(json.dumps({"AUDIT_STOP": str(stop)}), flush=True)
    finally:
        neuron_log.close()
    imgs = summary["images"]
    opens = [o for o in ("013", "016") if o in imgs]
    budget_short = any(v["counts"]["not_run_budget"] for v in imgs.values())
    if "audit_stop" in summary:
        summary["gate"] = "AUDIT_STOP"
    elif args.only:
        summary["gate"] = "SMOKE_TEST_NO_GATE"
    elif len(opens) < 2:
        summary["gate"] = "NOT_RUN_BUDGET"
    elif any(imgs[o]["material"] for o in opens):
        summary["gate"] = "PASS"
    else:
        summary["gate"] = "FAIL" + (" (some sides NOT_RUN_BUDGET)" if budget_short else "")
    if "017" in imgs and imgs["017"]["closed"]["new"] < imgs["017"]["closed"]["baseline"]:
        summary["control_flag"] = "017 lost a closed competitor"
    summary["wall_seconds"] = round(time.monotonic() - start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"gate": summary["gate"], "wall_seconds": summary["wall_seconds"]}), flush=True)
    return 0 if summary["gate"] != "AUDIT_STOP" else 4


if __name__ == "__main__":
    sys.exit(main())
