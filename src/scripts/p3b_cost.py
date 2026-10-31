#!/usr/bin/env python
"""P3b cost measurement: planned dependency graph with and without the tightened layer-3 bounds.

DIAGNOSTIC SCREEN ONLY. No ESBMC, no export. Mirrors the dependency closure and the `--cost`
tally of crown_chain_export_v2.main(), extended with per-side policies for the layer-3 bound
chains found by p3b_bound_slope_pilot.py. The baseline configuration must reproduce the
production screen's chains_planned and multiply_terms exactly, or the run stops.

  cd /home/joao/code/preqbmc
  systemd-run --user --scope -q -p MemoryMax=8G /home/joao/miniconda3/envs/quad/bin/python -u \
    /home/joao/code/preqbmc-qformat/src/scripts/p3b_cost.py \
    --run output/crown_diagnostics_20261031T161801Z --pilot output/p3b_bound_slope_pilot_20261031T173914Z \
    --output output/p3b_cost_<UTC>
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
EXPECTED_BASELINE = {   # production screen, output/p3_finder_pilot_20261031
    "013": {"chains_planned": 6770, "multiply_terms": 331893368},
    "016": {"chains_planned": 7851, "multiply_terms": 365356538},
    "017": {"chains_planned": 3240, "multiply_terms": 157246375},
}
US_PER_TERM = 330      # MEASURED basis from the October positives (P3 doc): 298-330 us/term
REFERENCE_CPU_H = 22.63   # largest October positive (006); P3 cost cap was 1.5x this


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditStop(RuntimeError):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
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
    audit = guard.audit(finder, MAIN / "src")
    if not audit["passed"]:
        raise SystemExit(f"import audit failed: {audit['violations']}")
    run, pilot, out = args.run.resolve(), args.pilot.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    if (pilot / "VOID.txt").exists():
        raise SystemExit("pilot run is void")
    pilot_summary = json.loads((pilot / "summary.json").read_text())
    if pilot_summary["gate"] != "PASS":
        raise SystemExit("pilot gate did not pass")
    out.mkdir(parents=True)
    config = json.loads((run / "manifest.json").read_text())["config"]
    preflight = json.loads((run / "preflight.json").read_text())
    manifest = {"schema": "preqbmc_p3b_cost_v1", "script_sha256": sha256_file(__file__),
                "diagnostics_module_sha256": sha256_file(cd.__file__),
                "finder_sha256": sha256_file(v2.__file__), "source_run": str(run), "pilot_run": str(pilot),
                "pilot_summary_sha256": sha256_file(pilot / "summary.json"),
                "pilot_sides_sha256": sha256_file(pilot / "sides.jsonl.gz"),
                "import_audit": {k: v for k, v in audit.items() if k != "modules"},
                "evidence_kind": "SCREEN_BOUND_UNCHECKED", "esbmc_runs": 0,
                "cpu_estimate_basis": f"{US_PER_TERM} us/term (MEASURED on the October positives); ESTIMATE only"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    v2.configure_format(int(config["qif"]["total_bits"]), int(config["qif"]["fractional_bits"]))
    _, _, net, _, _ = affine_export.load()
    mats = [affine_export.lowered(layer) for layer in net.layers]
    sizes = [m[0].shape[0] for m in mats]
    top = len(mats) - 1
    ONE = 1 << cd.SCALE_BITS
    ADAPTIVE = v2.ADAPTIVE
    sides_by_image = {}
    with gzip.open(pilot / "sides.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            sides_by_image.setdefault(r["ordinal"], []).append(r)
    results = {}

    for image in config["images"]:
        o, target = image["ordinal"], image["target"]
        t0 = time.time()
        entry = preflight["caches"][o]
        if sha256_file(entry["npz"]) != entry["npz_sha256"]:
            raise AuditStop(f"{o}: cache changed")
        data = dict(np.load(entry["npz"], allow_pickle=False))
        xl, xu = data["xl"], data["xu"]
        bounds = [(data[f"b{k}_lo"], data[f"b{k}_hi"], data[f"b{k}_sat"], data[f"b{k}_clip"]) for k in range(top)]
        info = [{key: data[f"i{k}_{key}"] for key in ("lo_chain", "hi_chain", "lower_box", "upper_chain")}
                for k in range(top)]
        facts = [{key: data[f"f{k}_{key}"] for key in ("hl", "hh", "stable", "upper_ok", "lower_ok")}
                 for k in range(top + 1)]
        comps = [c for c in range(sizes[-1]) if c != target]

        # tightened layer-3 bounds and per-side policies from the pilot log
        lo, hi, sat, clip = bounds[LAYER]
        new_lo, new_hi = lo.copy(), hi.copy()
        side_policy = {}     # (neuron, side) -> ("A", row) | ("B", override dict)
        for r in sides_by_image[o]:
            if r["chosen"] is None:
                continue
            n, s = r["neuron"], r["side"]
            (new_lo if s == "lower" else new_hi)[n] = r["new"]
            side_policy[(n, s)] = ("A", r["arm_A_policy"]) if r["chosen"] == "A" else ("B", r["arm_B_policy"])
        tight_bounds = [b if k != LAYER else (new_lo, new_hi, sat, clip) for k, b in enumerate(bounds)]
        tight_upper_chain = info[LAYER]["upper_chain"].copy()
        for (n, s) in side_policy:
            if s == "upper":
                tight_upper_chain[n] = True     # a tightened upper side is chain-backed

        def unit_rows(idx, sign, width):
            E = np.zeros((len(idx), width), dtype=np.int64)
            E[np.arange(len(idx)), idx] = sign * ONE
            return E

        def graph(bnds, upper_chain3, search, policy_for):
            """Exporter dependency closure + cost tally for one configuration."""

            cce.XU[0] = xu
            lams = np.zeros((len(comps), sizes[-1]), dtype=np.int64)
            lams[:, target] = ONE
            lams[np.arange(len(comps)), comps] = -ONE
            pols = np.full((len(comps), top), ADAPTIVE, dtype=np.int64)
            scaled, _, _ = v2.run_chain(lams, top, mats, bnds, xl)
            if search:
                open_rows = np.nonzero(v2.ceil_div(scaled, ONE) < 1)[0]
                if open_rows.size:
                    p, sc, tr = v2.search_policies(lams[open_rows], top, mats, bnds, xl, None)
                    pols[open_rows] = p
                    scaled = scaled.copy()
                    scaled[open_rows] = sc
            margins = dict(zip(comps, v2.ceil_div(scaled, ONE).tolist()))
            tl, _, _ = v2.run_chain(unit_rows([target], 1, sizes[-1]), top, mats, bnds, xl)
            th, _, _ = v2.run_chain(unit_rows([target], -1, sizes[-1]), top, mats, bnds, xl)
            raw_t_low, raw_t_high = int(v2.ceil_div(tl, ONE)[0]), int(np.floor_divide(-th, ONE)[0])
            fl = facts[top]
            logit, closed = set(), 0
            for c in comps:
                if fl["hh"][c] <= cd.Q_HIGH - 1:
                    top_ok = True
                elif raw_t_high <= cd.Q_HIGH:
                    top_ok = True
                    logit.add(("upper", target))
                else:
                    top_ok = False
                if fl["lower_ok"][c] and fl["hl"][c] >= cd.Q_LOW:
                    bot_ok = True
                elif raw_t_low >= cd.Q_LOW + 1:
                    bot_ok = True
                    logit.add(("lower", target))
                else:
                    bot_ok = False
                closed += top_ok and bot_ok and margins[c] >= 1
            root_rows = [lams] + [unit_rows([t], 1 if s == "lower" else -1, sizes[-1]) for s, t in sorted(logit)]
            roots = np.concatenate(root_rows)
            table = np.concatenate([pols, np.full((len(logit), top), ADAPTIVE, dtype=np.int64)])
            cost = {"multiply_terms": 0, "chains": 0, "two_sided_relu": 0, "relu_coordinates": 0,
                    "layer3_policy_chains": 0}

            def tally(E, layer, policies=None, override=None):
                if override is not None:
                    with cd.SlopeOverride(v2, bnds[override["layer"]], 0, override["neuron"], override["a"],
                                          override["trial_a_out"]) as ov:
                        _, terms, steps = v2.run_chain(E, layer, mats, bnds, xl, record=True)
                        nd = v2.needs(E, layer, mats, bnds)
                    if ov.applied != 2:
                        raise AuditStop(f"override applied {ov.applied} times")
                else:
                    _, terms, steps = v2.run_chain(E, layer, mats, bnds, xl, record=True, policies=policies)
                    nd = v2.needs(E, layer, mats, bnds, policies=policies)
                cost["multiply_terms"] += int(terms.sum())
                cost["chains"] += E.shape[0]
                for st in steps:
                    if st["kind"] == "relu":
                        j = st["j"]
                        cls = v2.classify(st["a"], st["a_out"], bnds[j][2], bnds[j][3])
                        cost["two_sided_relu"] += int(((cls == 4) | (cls == 6)).sum())
                        cost["relu_coordinates"] += int((cls >= 0).sum())
                return nd, steps

            pending = {j: [np.zeros(sizes[j], bool) for _ in range(3)] for j in range(top)}

            def add(nd):
                for j, arrs in nd.items():
                    for p, a in zip(pending[j], arrs):
                        p |= a

            nd, _ = tally(roots, top, table)
            add(nd)
            chains_per_layer = {}
            for k in range(top - 1, -1, -1):
                need_l, need_uv, _ = pending[k]
                if (need_l & (facts[k]["stable"] < 0)).any():
                    return {"status": "FINDER_INCONCLUSIVE", "cause": f"layer {k}: stable-inactive lower needed"}
                upper_chain = upper_chain3 if k == LAYER else info[k]["upper_chain"]
                chain_l = need_l & ~info[k]["lower_box"]
                chain_u = need_uv & upper_chain
                chains_per_layer[str(k)] = int(chain_l.sum() + chain_u.sum())
                for sign, mask, side in ((1, chain_l, "lower"), (-1, chain_u, "upper")):
                    idx = np.nonzero(mask)[0].tolist()
                    plain = [n for n in idx if policy_for(k, n, side) is None]
                    for s0 in range(0, len(plain), 64):
                        nd, _ = tally(unit_rows(plain[s0:s0 + 64], sign, sizes[k]), k)
                        add(nd)
                    for n in idx:
                        pol = policy_for(k, n, side)
                        if pol is None:
                            continue
                        cost["layer3_policy_chains"] += 1
                        E = unit_rows([n], sign, sizes[k])
                        if pol[0] == "A":
                            nd, steps = tally(E, k, policies=np.array([pol[1]], dtype=np.int64))
                        else:
                            nd, steps = tally(E, k, override=pol[1])
                        got = steps[-1]["scaled"][0]
                        e = v2.ceil_div(got, ONE) if side == "lower" else (-int(got)) // ONE
                        want = int((bnds[k][0] if side == "lower" else bnds[k][1])[n])
                        if int(e) != want:
                            raise AuditStop(f"{o} L{k}:{n} {side}: policy chain gives {int(e)} != bound {want}")
                        add(nd)
            planned = sum(chains_per_layer.values()) + roots.shape[0]
            return {"status": "OK", "margin_min": int(min(margins.values())), "closed": int(closed),
                    "positive": int(sum(m >= 1 for m in margins.values())),
                    "chains_planned": int(planned), "chains_per_layer": chains_per_layer, **cost,
                    "estimated_esbmc_cpu_hours": round(cost["multiply_terms"] * US_PER_TERM / 3.6e9, 2)}

        no_policy = lambda k, n, side: None
        tight_policy = lambda k, n, side: side_policy.get((n, side)) if k == LAYER else None
        res = {"baseline": graph(bounds, info[LAYER]["upper_chain"], False, no_policy)}
        exp = EXPECTED_BASELINE[o]
        if (res["baseline"]["chains_planned"], res["baseline"]["multiply_terms"]) != \
                (exp["chains_planned"], exp["multiply_terms"]):
            raise AuditStop(f"{o}: baseline cost mirror {res['baseline']['chains_planned']}/"
                            f"{res['baseline']['multiply_terms']} != production {exp}")
        res["tightened"] = graph(tight_bounds, tight_upper_chain, False, tight_policy)
        res["tightened_margin_search"] = graph(tight_bounds, tight_upper_chain, True, tight_policy)
        res["wall_seconds"] = round(time.time() - t0, 1)
        b = res["baseline"]
        for key in ("tightened", "tightened_margin_search"):
            r = res[key]
            if r["status"] == "OK":
                r["delta_multiply_terms"] = r["multiply_terms"] - b["multiply_terms"]
                r["ratio_multiply_terms"] = round(r["multiply_terms"] / b["multiply_terms"], 3)
                r["delta_chains"] = r["chains_planned"] - b["chains_planned"]
        results[o] = res
        print(json.dumps({"image": o, **{k: {kk: v.get(kk) for kk in ("closed", "margin_min", "chains_planned",
                                                                         "multiply_terms", "layer3_policy_chains",
                                                                         "estimated_esbmc_cpu_hours")}
                                         for k, v in res.items() if isinstance(v, dict)},
                          "wall_seconds": res["wall_seconds"]}), flush=True)
        (out / f"{o}.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps({"images": results, "reference_cpu_hours_006": REFERENCE_CPU_H,
                                                  "cost_cap_cpu_hours": round(1.5 * REFERENCE_CPU_H, 2)},
                                                 indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditStop as stop:
        print(json.dumps({"AUDIT_STOP": str(stop)}), flush=True)
        sys.exit(4)
