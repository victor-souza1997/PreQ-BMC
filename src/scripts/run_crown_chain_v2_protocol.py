"""Pre-registered per-image protocol: screen -> box proof -> v2 export -> ESBMC check.

Subcommands:
  draft --split S --output P.json [--seed N --n K | --ids ID ...] --launch-cutoff ISO
      Pin every source, data and parameter hash and fix the image list. --ids is
      validation-only; a test protocol draws its images from --seed.
  run --protocol P.json --output DIR [--max-full K] [--screen-only]
      Resumable. Phase A screens every image; phase B runs the heavy pipeline on
      FINDER_POSITIVE images in draw order until the launch cutoff.
  tally --protocol P.json --output DIR
      Write DIR/results.json with every image's outcome, NOT_RUN included.

A test protocol runs only while its file is tracked by git and unmodified, and only
while every pinned hash still matches: the protocol is fixed before any test outcome.
Only an ESBMC-VERIFIED outcome certifies anything; every other outcome is not a proof
of non-robustness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from verification.crown_chain_v2 import checker_sources

REPO = Path(__file__).resolve().parents[2]
FINDER = "src/scripts/crown_finder/crown_chain_export_v2.py"
SEARCH = "output/sign_conv_depthwise_pool_search_20260924"
TEST_EVALUATION = "output/sign_conv_depthwise_pool_test_20260924/test_evaluation.json"
DEPLOYMENT_SHA256 = "dd535f5480bec45cdae7774f849afca49f7afb5d645f02037e86a44d16f45097"
Q16 = {"total_bits": 16, "integer_bits": 7, "fractional_bits": 8}
BOX_PROOF = {
    "proof_mode": "proof_carrying_interval",
    "exact_margin_refinement": True,
    "block_size": 24,
    "max_blocks_per_layer": None,
    "profile": "paper-z3",
    "timeout_seconds": 120,
    "memlimit": "6g",
    "fail_fast": False,
    "jobs": 4,
    "min_available_gib": 6.0,
}
CHECK = {"jobs": 4, "timeout": 60, "memlimit": "4g", "profile": "paper-z3", "layers": [0, 1, 2, 3, 4]}
MEMORY_MAX = {"screen": "8G", "box": "8G", "export": "8G", "check": "12G"}
OUTCOMES = {
    "VERIFIED": "every obligation of a complete-network certificate is ESBMC VERIFIED (PROVED)",
    "MISCLASSIFIED": "the deployed integer QNN mislabels the unperturbed image; nothing to certify",
    "FINDER_INCONCLUSIVE": "the untrusted finder cannot close every competitor; no ESBMC run",
    "BOX_LEDGER_INCOMPLETE": "the ESBMC box proof has a required block that is not VERIFIED",
    "EXPORT_INCONCLUSIVE": "the finder could not export a certificate from the VERIFIED box ledger",
    "CHECK_FAILED": "an ESBMC obligation FAILED: the certificate is wrong, not a counterexample",
    "CHECK_INCONCLUSIVE": "an ESBMC obligation hit TIMEOUT, MEMOUT or UNKNOWN, or aggregation refused",
    "NOT_RUN_BUDGET": "FINDER_POSITIVE, but the launch cutoff passed before its heavy pipeline began",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def pinned_sources() -> dict[str, str]:
    """Every non-test Python source under src/, keyed by repo-relative path."""
    return {
        str(path.relative_to(REPO)): _sha256(path)
        for path in sorted((REPO / "src").rglob("*.py"))
        if "tests" not in path.relative_to(REPO / "src").parts and "__pycache__" not in path.parts
    }


def pinned_data() -> dict[str, str]:
    search = json.loads((REPO / SEARCH / "search_summary.json").read_text(encoding="utf-8"))
    selected = search["selected_candidate_summary"]
    return {
        "search_summary_sha256": _sha256(REPO / SEARCH / "search_summary.json"),
        "base_study_sha256": _sha256(Path(search["base_study_path"])),
        "folded_model_sha256": _sha256(Path(selected["folded_model_path"])),
        "test_evaluation_sha256": _sha256(REPO / TEST_EVALUATION),
        "deployment_source_sha256": DEPLOYMENT_SHA256,
    }


def esbmc_version() -> str:
    done = subprocess.run(["esbmc", "--version"], capture_output=True, text=True, check=True)
    version = (done.stdout + done.stderr).strip()
    if not version:
        raise RuntimeError("esbmc --version printed nothing")
    return version


def _records(split: str) -> list[dict[str, Any]]:
    search = json.loads((REPO / SEARCH / "search_summary.json").read_text(encoding="utf-8"))
    base = json.loads(Path(search["base_study_path"]).read_text(encoding="utf-8"))
    return [row for row in base["records"] if row["split"] == split]


def draft(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite protocol {args.output}")
    rows = _records(args.split)
    if args.ids:
        if args.split != "validation":
            raise SystemExit("--ids is for validation dry runs; a test protocol draws from --seed")
        known = {row["id"] for row in rows}
        if not set(args.ids) <= known or len(set(args.ids)) != len(args.ids):
            raise SystemExit("--ids must name distinct records of the split")
        images, sampling = list(args.ids), {"method": "explicit_validation_ids"}
    else:
        if args.seed is None or args.n is None:
            raise SystemExit("a drawn protocol needs --seed and --n")
        order = np.random.default_rng(args.seed).permutation(len(rows))[: args.n]
        images = [rows[int(i)]["id"] for i in order]
        sampling = {
            "method": "uniform_without_replacement",
            "population": f"every {args.split} record of the base study, in file order",
            "population_size": len(rows),
            "rng": "numpy.random.default_rng(seed).permutation(population_size)[:n]",
            "seed": args.seed,
            "n": args.n,
        }
    cutoff = datetime.fromisoformat(args.launch_cutoff)
    if cutoff.tzinfo is None:
        raise SystemExit("--launch-cutoff needs a UTC offset")
    protocol = {
        "schema": "crown_chain_v2_protocol_v1",
        "drafted_at": _now(),
        "split": args.split,
        "epsilon_raw_bytes": args.epsilon,
        "sampling": sampling,
        "images_in_draw_order": images,
        "launch_cutoff": cutoff.isoformat(),
        "budget_rule": (
            "Phase A screens every image. Phase B takes FINDER_POSITIVE images in draw order and "
            "starts no box proof, export or check after launch_cutoff; every image it did not start "
            "is NOT_RUN_BUDGET. An image already started runs to its outcome."
        ),
        "outcomes": OUTCOMES,
        "reporting_rule": (
            "Report every drawn image's outcome. The certified rate is VERIFIED / n, with every "
            "non-VERIFIED outcome counted as not certified; also report VERIFIED / correctly "
            "classified. No outcome other than VERIFIED is reported as robust, and none is "
            "reported as non-robust."
        ),
        "box_proof": {"resize_mode": "nearest_center", "input_format": Q16, "qif": [Q16] * 5, "proof": BOX_PROOF},
        "check": CHECK,
        "memory_max": MEMORY_MAX,
        "data": pinned_data(),
        "checker_sources": checker_sources(),
        "sources": pinned_sources(),
        "esbmc_version": esbmc_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(args.output), "split": args.split, "images": len(images),
                      "sha256": _sha256(args.output)}))


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "crown_chain_v2_protocol_v1":
        raise ValueError("not a crown_chain_v2 protocol")
    if protocol["split"] == "test":
        if not path.resolve().is_relative_to(REPO):
            raise SystemExit("a test protocol must live in the repository")
        relative = str(path.resolve().relative_to(REPO))
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=REPO,
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO).returncode == 0
        if not (tracked and clean):
            raise SystemExit("a test protocol must be committed and unmodified before it runs")
    mismatched = [
        name for name, (pinned, current) in {
            "data": (protocol["data"], pinned_data()),
            "checker_sources": (protocol["checker_sources"], checker_sources()),
            "sources": (protocol["sources"], pinned_sources()),
            "esbmc_version": (protocol["esbmc_version"], esbmc_version()),
        }.items() if pinned != current
    ]
    if mismatched:
        detail = sorted(
            key for key in set(protocol["sources"]) | set(pinned_sources())
            if protocol["sources"].get(key) != pinned_sources().get(key)
        ) if "sources" in mismatched else []
        raise SystemExit(f"protocol pins no longer match: {mismatched} {detail[:10]}")
    return protocol


def _heavy(stage: str, protocol: dict[str, Any], argv: list[str], log: Path) -> int:
    """Run one heavy job under a memory scope; one heavy job runs at a time."""
    env = {**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/tmp/preqbmc-mpl",
           "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": ""}
    command = ["systemd-run", "--user", "--scope", "-q", "-p",
               f"MemoryMax={protocol['memory_max'][stage]}", sys.executable, *argv]
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"# {_now()} {' '.join(command)}\n")
        stream.flush()
        return subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode


def _set_aside(path: Path) -> None:
    """A stage output left by an interrupted run is kept, renamed, never deleted."""
    if path.exists():
        path.rename(path.with_name(f"{path.name}.interrupted.{int(time.time())}"))


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return payload


def _last_error(log: Path) -> str:
    lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-1][-400:] if lines else "no output"


def screen(protocol: dict[str, Any], image: str, work: Path) -> dict[str, Any]:
    report = work / "screen.json"
    if report.exists():
        return json.loads(report.read_text(encoding="utf-8"))
    code = _heavy("screen", protocol, [FINDER, "screen", str(protocol["epsilon_raw_bytes"]), image,
                                       "--report", str(report)], work / "screen.log")
    if code != 0 or not report.exists():
        return _write(work / "screen_error.json", {"status": "SCREEN_ERROR", "cause": _last_error(work / "screen.log")})
    return json.loads(report.read_text(encoding="utf-8"))


def full_pipeline(protocol: dict[str, Any], protocol_path: Path, image: str, work: Path) -> dict[str, Any]:
    timing: dict[str, float] = {}
    box = work / "box"
    if not (box / "pilot_summary.json").exists():
        _set_aside(box)
        config = {
            "schema": "ssv_conv_native_pilot_v1",
            "source_search_output": str(REPO / SEARCH),
            "test_evaluation": str(REPO / TEST_EVALUATION),
            "sample_id": image,
            "epsilon_raw_bytes": protocol["epsilon_raw_bytes"],
            **protocol["box_proof"],
            "claim_boundary": {
                "purpose": "box proof ledger for a CROWN-chain v2 certificate",
                "protocol": str(protocol_path),
                "protocol_sha256": _sha256(protocol_path),
            },
            "split": protocol["split"],
            "expected_deployment_source_sha256": protocol["data"]["deployment_source_sha256"],
        }
        _write(work / "box_config.json", config)
        start = time.time()
        _heavy("box", protocol, ["src/scripts/run_ssv_conv_native_pilot.py", "--config",
                                 str(work / "box_config.json"), "--output", str(box)], work / "box.log")
        timing["box_seconds"] = round(time.time() - start, 1)
    if not (box / "pilot_summary.json").exists():
        return {"outcome": "BOX_LEDGER_INCOMPLETE", "cause": _last_error(work / "box.log"), **timing}
    pilot = json.loads((box / "pilot_summary.json").read_text(encoding="utf-8"))
    if pilot["final_status"] == "VERIFIED" and pilot["certified"]:
        return {"outcome": "VERIFIED", "method": "esbmc_box_proof", "evidence": str(box / "pilot_summary.json"), **timing}
    if pilot["final_status"] != "ABSTRACTION_INCONCLUSIVE":
        return {"outcome": "BOX_LEDGER_INCOMPLETE", "cause": f"box proof {pilot['final_status']}", **timing}
    deployment = box / "proof" / "qnn_conv_native.c"
    if _sha256(deployment) != protocol["data"]["deployment_source_sha256"]:
        raise ValueError(f"{deployment} is not the pinned deployment")

    certificate = work / "certificate"
    if not (certificate / "certificate.json").exists():
        _set_aside(certificate)
        start = time.time()
        code = _heavy("export", protocol, [
            FINDER, "export", str(protocol["epsilon_raw_bytes"]), image, str(certificate),
            "--ledger", str(box / "proof" / "proof_summary.json"), "--protocol", str(protocol_path),
            "--report", str(work / "export.json"),
        ], work / "export.log")
        timing["export_seconds"] = round(time.time() - start, 1)
        if code != 0 or not (certificate / "certificate.json").exists():
            return {"outcome": "EXPORT_INCONCLUSIVE", "cause": _last_error(work / "export.log"), **timing}

    check = protocol["check"]
    shards = []
    for layer in check["layers"]:
        shard = work / f"check_l{layer}"
        if not (shard / "check_summary.json").exists():
            _set_aside(shard)
            start = time.time()
            _heavy("check", protocol, [
                "src/scripts/check_crown_chain_v2.py", str(certificate), "--deployment-c", str(deployment),
                "--output", str(shard), "--shard-layer", str(layer), "--jobs", str(check["jobs"]),
                "--timeout", str(check["timeout"]), "--memlimit", check["memlimit"], "--profile", check["profile"],
            ], work / "check.log")
            timing[f"check_l{layer}_seconds"] = round(time.time() - start, 1)
        if not (shard / "check_summary.json").exists():
            return {"outcome": "CHECK_INCONCLUSIVE", "cause": f"layer {layer}: {_last_error(work / 'check.log')}", **timing}
        summary = json.loads((shard / "check_summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "SHARD_VERIFIED":
            outcome = "CHECK_FAILED" if summary["status_counts"].get("FAILED") else "CHECK_INCONCLUSIVE"
            return {"outcome": outcome, "cause": f"layer {layer}: {summary['status']} {summary['status_counts']}", **timing}
        shards.append(shard)

    complete = work / "complete"
    if not (complete / "complete_check_summary.json").exists():
        _set_aside(complete)
        argv = ["src/scripts/aggregate_crown_chain_v2.py", str(certificate), "--deployment-c", str(deployment),
                "--output", str(complete)]
        for shard in shards:
            argv += ["--shard-check", str(shard)]
        _heavy("check", protocol, argv, work / "aggregate.log")
    if not (complete / "complete_check_summary.json").exists():
        return {"outcome": "CHECK_INCONCLUSIVE", "cause": f"aggregation: {_last_error(work / 'aggregate.log')}", **timing}
    summary = json.loads((complete / "complete_check_summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "VERIFIED" or summary["split"] != protocol["split"]:
        return {"outcome": "CHECK_INCONCLUSIVE", "cause": f"aggregate {summary['status']}", **timing}
    return {
        "outcome": "VERIFIED", "method": "esbmc_crown_chain_v2",
        "evidence": str(complete / "complete_check_summary.json"),
        "claim": summary["claim"], "guarantee_level": summary["guarantee_level"],
        "harnesses_verified": summary["harnesses_verified"],
        "minimum_raw_margin_lower_bound": summary["margin_closure"]["minimum_raw_margin_lower_bound"],
        **timing,
    }


def run(args: argparse.Namespace) -> None:
    protocol = validate_protocol(args.protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "protocol_binding.json"
    binding = {"protocol": str(args.protocol.resolve()), "protocol_sha256": _sha256(args.protocol)}
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != binding:
        raise SystemExit(f"{args.output} belongs to another protocol")
    _write(marker, binding)
    images = protocol["images_in_draw_order"]
    works = [args.output / f"{ordinal:03d}" for ordinal in range(len(images))]
    screens = []
    for image, work in zip(images, works):
        work.mkdir(exist_ok=True)
        screens.append(screen(protocol, image, work))
        if screens[-1]["status"] == "SCREEN_ERROR":
            pass  # retried on the next run; tally reports it with its cause
        elif screens[-1]["status"] != "FINDER_POSITIVE" and not (work / "outcome.json").exists():
            status = screens[-1]["status"]
            _write(work / "outcome.json", {
                "image": image,
                "outcome": status if status in OUTCOMES else "FINDER_INCONCLUSIVE",
                "cause": screens[-1].get("cause") or screens[-1].get("open_competitors") or status,
                "screen": screens[-1],
            })
        print(json.dumps({"phase": "screen", "image": image, "status": screens[-1]["status"]}), flush=True)
    if args.screen_only:
        return
    started = 0
    cutoff = datetime.fromisoformat(protocol["launch_cutoff"])
    for image, work, screened in zip(images, works, screens):
        if (work / "outcome.json").exists() or screened["status"] != "FINDER_POSITIVE":
            continue
        began = (work / "box_config.json").exists()
        if not began and (datetime.now(timezone.utc) > cutoff or (args.max_full is not None and started >= args.max_full)):
            continue  # tally reports it as NOT_RUN_BUDGET (after the cutoff) or pending
        started += 1
        start = time.time()
        result = full_pipeline(protocol, args.protocol, image, work)
        _write(work / "outcome.json", {"image": image, **result, "screen": screened,
                                       "pipeline_seconds_this_session": round(time.time() - start, 1),
                                       "finished_at": _now()})
        print(json.dumps({"phase": "full", "image": image, "outcome": result["outcome"]}), flush=True)


def tally(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    binding = json.loads((args.output / "protocol_binding.json").read_text(encoding="utf-8"))
    if binding["protocol_sha256"] != _sha256(args.protocol):
        raise SystemExit("output directory was produced under another protocol")
    after_cutoff = datetime.now(timezone.utc) > datetime.fromisoformat(protocol["launch_cutoff"])
    rows = []
    for ordinal, image in enumerate(protocol["images_in_draw_order"]):
        work = args.output / f"{ordinal:03d}"
        if (work / "outcome.json").exists():
            row = json.loads((work / "outcome.json").read_text(encoding="utf-8"))
            row.pop("screen", None)
        elif (work / "screen_error.json").exists():
            cause = json.loads((work / "screen_error.json").read_text(encoding="utf-8"))["cause"]
            row = {"image": image, "outcome": "FINDER_INCONCLUSIVE", "cause": f"screen error: {cause}"}
        else:
            row = {"image": image, "outcome": "NOT_RUN_BUDGET" if after_cutoff else "PENDING"}
        rows.append({"ordinal": ordinal, **row})
    counts = Counter(row["outcome"] for row in rows)
    n = len(rows)
    correct = n - counts["MISCLASSIFIED"]
    results = {
        "schema": "crown_chain_v2_protocol_results_v1",
        "protocol": str(args.protocol), "protocol_sha256": binding["protocol_sha256"],
        "split": protocol["split"], "epsilon_raw_bytes": protocol["epsilon_raw_bytes"],
        "complete": counts["PENDING"] == 0,
        "n": n, "counts": dict(sorted(counts.items())),
        "certified_rate": counts["VERIFIED"] / n if n else None,
        "certified_rate_among_correct": counts["VERIFIED"] / correct if correct else None,
        "evidence": {
            "PROVED": "each VERIFIED row: every ESBMC obligation of its certificate VERIFIED",
            "MEASURED": "the counts and rates, over the drawn sample",
            "NOT_CLAIMED": "no non-VERIFIED row is evidence of non-robustness",
        },
        "images": rows,
    }
    _write(args.output / "results.json", results)
    print(json.dumps({key: results[key] for key in ("split", "n", "complete", "counts", "certified_rate")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("draft")
    p.add_argument("--split", choices=("test", "validation"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epsilon", type=int, default=1)
    p.add_argument("--seed", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--launch-cutoff", required=True)
    p = sub.add_parser("run")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-full", type=int)
    p.add_argument("--screen-only", action="store_true")
    p = sub.add_parser("tally")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"draft": draft, "run": run, "tally": tally}[args.command](args)


if __name__ == "__main__":
    main()
