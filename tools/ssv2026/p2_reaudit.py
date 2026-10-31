"""P2: re-audit the saved v2 exports under the P1 checker, without editing them.

For each export:
1. Load the whole certificate with the corrected structural checks (layer
   traversal, chain-fact identity, clamp references).
2. Per shard, compare the assertions the corrected coverage rule requires with
   the labels in the old, hash-checked VERIFIED harnesses.
3. Missing assertions that are exactly the affine columns whose products cancel
   to 0 (which the old renderer skipped) get supplementary harnesses, checked
   with the same ESBMC settings as the original shards.
4. Coverage is recomputed over old + supplementary records.

Writes only under --output. Old shard summaries are read, never modified.

  PYTHONPATH=<worktree>/src python tools/ssv2026/p2_reaudit.py --output output/p2_reaudit_20261030
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from scripts.benchmark_esbmc_concurrency import benchmark
from verification.crown_chain_v2 import _header, checker_sources, load_certificate_v2
from verification.crown_chain_v2_coverage import _LABEL, required_assertions, verify_shard_coverage

EXPORTS = {
    "test_06267": "output/crown_chain_v2_test_20260925/005",
    "test_04083": "output/crown_chain_v2_test_20260925/015",
    **{f"validation_{o}": f"output/crown_chain_v2_validation_20261014/{o}" for o in ("006", "008", "011", "012", "017")},
}
# The settings the original shards were checked with (check_summary.json).
ESBMC = {"jobs": 4, "timeout": 60, "memlimit": "4g", "profile": "paper-z3", "min_available_gib": 6.0}
ZERO_CHECKS = ("g", "floor", "residual")


def _observed(records):
    """Assertion labels in the old VERIFIED, unchanged harnesses."""

    seen: Counter[str] = Counter()
    for record in records:
        source = Path(record["harness"]).read_text(encoding="utf-8")
        if record["status"] != "VERIFIED" or hashlib.sha256(source.encode()).hexdigest() != record["harness_sha256"]:
            raise ValueError(f"{record['harness']} is not an unchanged VERIFIED harness")
        for line in source.splitlines():
            if "__ESBMC_assert(" in line and not line.lstrip().startswith("void"):
                seen[_LABEL.search(line).group(1)] += 1
    return seen


def _zero_columns(certificate):
    """step_id -> (step, columns the rows touch whose product g cancels to 0)."""

    result = {}
    for chain in certificate.chains:
        for step in chain["_normalized_steps"]:
            if step["kind"] != "affine":
                continue
            csr = certificate.layers[int(step["layer_index"])]
            touched = {
                csr["indices"][position]
                for row in step["_input_coefficients"]
                for position in range(csr["indptr"][row], csr["indptr"][row + 1])
            }
            zero = sorted(touched - set(step["_g"]))
            if zero:
                result[step["step_id"]] = (step, zero)
    return result


def _render_zero_columns(certificate, step, columns):
    """The old affine chunk assertions, for the columns it skipped."""

    csr = certificate.layers[int(step["layer_index"])]
    lines = _header() + ["int main(void) {"]
    for ordinal, column in enumerate(columns):
        expression = " + ".join(
            f"((__int128)({csr['data'][position]}) * ({coefficient}))"
            for row, coefficient in step["_input_coefficients"].items()
            for position in range(csr["indptr"][row], csr["indptr"][row + 1])
            if csr["indices"][position] == column
        )
        sid = step["step_id"]
        lines.extend([
            f"  __int128 g_{ordinal} = {expression};",
            f'  __ESBMC_assert(g_{ordinal} == (__int128)(0), "{sid} g {column}");',
            f"  __int128 q_{ordinal} = floor_div_i128(g_{ordinal}, 256);",
            f'  __ESBMC_assert(q_{ordinal} == (__int128)(0), "{sid} floor {column}");',
            f'  __ESBMC_assert(g_{ordinal} - 256 * q_{ordinal} == (__int128)(0), "{sid} residual {column}");',
        ])
    return "\n".join(lines + ["  return 0;", "}", ""])


def audit_export(name, base, output):
    base, out = Path(base), output / name
    out.mkdir(parents=True, exist_ok=False)
    certificate_dir, deployment = base / "certificate", base / "box" / "proof" / "qnn_conv_native.c"
    started = time.monotonic()
    report = {"export": name, "base": str(base), "checker_sources": checker_sources()}
    try:
        full = load_certificate_v2(certificate_dir, deployment_c=deployment)
        report["structural"] = {"status": "PASS", "margin_closure": full.preflight["margin_closure"],
                                "chains": full.preflight["chains_loaded"], "facts": full.preflight["facts_validated"]}
        del full
    except ValueError as error:
        report["structural"] = {"status": "REJECTED", "reason": str(error)}
        return report
    shards = []
    for layer in range(5):
        summary = json.loads((base / f"check_l{layer}" / "check_summary.json").read_text(encoding="utf-8"))
        certificate = load_certificate_v2(certificate_dir, selected_layers=[layer], deployment_c=deployment)
        required = required_assertions(certificate)
        seen = _observed(summary["records"])
        missing = required - set(seen)
        zero = _zero_columns(certificate)
        explained = {f"{sid} {check} {column}" for sid, (_, columns) in zero.items()
                     for column in columns for check in ZERO_CHECKS}
        row = {"layer": layer, "old_status": summary["status"], "old_harnesses": summary["harness_count"],
               "required_assertions": len(required), "missing_before": len(missing),
               "missing_zero_product": len(missing & explained), "missing_unexplained": sorted(missing - explained)[:20],
               "unexpected": sorted(set(seen) - required)[:20],
               "duplicated": sorted(label for label, count in seen.items() if count > 1)[:20],
               "zero_product_steps": len(zero), "zero_product_columns": sum(len(c) for _, c in zero.values())}
        records = list(summary["records"])
        if zero and not row["missing_unexplained"] and not row["unexpected"] and not row["duplicated"]:
            supplement = out / f"l{layer}_supplement"
            harness_dir = supplement / "harnesses"
            harness_dir.mkdir(parents=True)
            paths, metadata = [], {}
            for index, (sid, (step, columns)) in enumerate(sorted(zero.items())):
                path = harness_dir / f"zero_{index:05d}.c"
                path.write_text(_render_zero_columns(certificate, step, columns), encoding="utf-8")
                paths.append(path)
                metadata[str(path.resolve())] = {"step_id": sid, "kind": "affine_zero_columns",
                                                 "harness_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            result = benchmark(paths, supplement / "esbmc", **ESBMC)
            supplementary = [{**metadata[str(Path(r["harness"]).resolve())], **r} for r in result["records"]]
            (supplement / "records.json").write_text(json.dumps(supplementary, indent=1) + "\n", encoding="utf-8")
            counts = Counter(r["status"] for r in supplementary)
            row["supplement"] = {"harnesses": len(paths), "status_counts": dict(counts),
                                 "aborted_low_memory": result["aborted_low_memory"],
                                 "summed_esbmc_seconds": round(sum(float(r["elapsed_seconds"]) for r in supplementary), 1)}
            records += supplementary
        try:
            row["coverage"] = verify_shard_coverage(certificate, records)
        except ValueError as error:
            row["coverage"] = {"status": "INCOMPLETE", "reason": str(error)[:600]}
        shards.append(row)
        del certificate
        print(json.dumps({"export": name, "layer": layer, "coverage": row["coverage"]["status"],
                          "missing_before": row["missing_before"], "supplement": row.get("supplement")}), flush=True)
    report["shards"] = shards
    report["verdict"] = (
        "ACCEPTED_WITH_SUPPLEMENT" if all(s["coverage"]["status"] == "COMPLETE" for s in shards)
        and any("supplement" in s for s in shards) else
        "ACCEPTED" if all(s["coverage"]["status"] == "COMPLETE" for s in shards) else "UNRESOLVED"
    )
    report["wall_seconds"] = round(time.monotonic() - started, 1)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, base in EXPORTS.items():
        if args.only and name not in args.only:
            continue
        report = audit_export(name, base, args.output)
        (args.output / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"export": name, "verdict": report.get("verdict"), "structural": report["structural"]["status"]}), flush=True)


if __name__ == "__main__":
    main()
