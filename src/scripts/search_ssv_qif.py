"""Search shared GTSRB layer formats using derived preimages and ESBMC.

The first format certified over every source-eligible calibration region is
frozen for independent evaluation. Candidate costs are nominal affine parameter
bits, not actual int64 C array bytes. No precision monotonicity is assumed.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from datasets.gtsrb_study import sha256
from models.ssv_artifact import load_artifact
from scripts.freeze_ssv_campaign import freeze, main_runs
from scripts.prepare_ssv_gtsrb import validate_config
from scripts.report_ssv_regions import write_csv
from scripts.run_ssv_cnn_gate import write_new_json
from scripts.run_ssv_gtsrb import build_region_context, run_region
from synthesis.preqbmc import GPEncoding


def validate(config):
    if config.get("schema") != "ssv_qif_search_v1":
        raise ValueError("Expected ssv_qif_search_v1")
    search = config["search"]
    low, high = search["fractional_bits_min"], search["fractional_bits_max"]
    if type(low) is not int or type(high) is not int or not 8 <= low <= high <= 16:
        raise ValueError("Search requires integer fractional-bit limits: 8 <= F_min <= F_max <= 16")
    padding = search["integer_bit_padding"]
    if (not isinstance(padding, list) or not padding or padding[0] != 0
            or any(type(i) is not int or not 0 <= i <= 8 for i in padding)
            or sorted(set(padding)) != padding):
        raise ValueError("Integer-bit padding must be sorted distinct integers starting at zero")
    if type(search["max_candidates"]) is not int or search["max_candidates"] < 1:
        raise ValueError("max_candidates must be a positive integer")
    if search.get("objective") != "min_affine_parameter_bits":
        raise ValueError("The search objective is min_affine_parameter_bits")
    return search


def integer_bit_floors(synth):
    """Range-derived floors, including exact float parameters and property preimages."""
    floors, counts = [], []
    for layer in [*synth.dense_layers, synth.output_layer]:
        weights, biases = layer.layer_paras
        values = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in (
            weights, biases, layer.lb, layer.ub, layer.relaxed_lb, layer.relaxed_ub)])
        if not np.all(np.isfinite(values)):
            raise ValueError("Cannot derive a search range from non-finite bounds")
        internal = GPEncoding._required_internal_integer_bits_for_interval(float(values.min()), float(values.max()))
        floors.append(internal - 1)
        counts.append(int(np.size(weights) + np.size(biases)))
    return floors, counts


def enumerate_candidates(floors, counts, search):
    """Enumerate the complete bounded domain in nondecreasing nominal storage cost."""
    formats = [[LayerQuantizationSpec(i + padding + f + 1, i + padding, f)
                for padding in search["integer_bit_padding"]
                for f in range(search["fractional_bits_min"], search["fractional_bits_max"] + 1)
                if i + padding + f + 1 <= 62] for i in floors]
    candidates = []
    for specs in itertools.product(*formats):
        qif = [asdict(s) for s in specs]
        candidates.append({"id": "_".join(f"l{l}_Q{s.total_bits}_F{s.fractional_bits}" for l, s in enumerate(specs)),
                           "qif": qif,
                           "affine_parameter_bits": sum(n * s.total_bits for n, s in zip(counts, specs, strict=True))})
    return sorted(candidates, key=lambda c: (c["affine_parameter_bits"], tuple(
        (s["total_bits"], s["integer_bits"], s["fractional_bits"]) for s in c["qif"])))


def _load(path):
    return json.loads(Path(path).read_text())


def _identity(config_path, config):
    root = Path(__file__).resolve().parents[1]
    modules = ["scripts/search_ssv_qif.py", "scripts/run_ssv_gtsrb.py", "synthesis/preqbmc.py",
               "verification/c_templates.py", "verification/arith_kernel.py", "backends/c_qnn_generator.py"]
    return {"config_sha256": sha256(config_path),
            "calibration_study_sha256": sha256(config["calibration_study"]),
            "evaluation_study_sha256": sha256(config["evaluation_study"]),
            "implementation_sha256": hashlib.sha256("".join(sha256(root / p) for p in modules).encode()).hexdigest()}


def _configured_study(study, config):
    updated = copy.deepcopy(study)
    settings = {**config["verification"], "bit_lb": config["search"]["fractional_bits_min"],
                "bit_ub": config["search"]["fractional_bits_max"]}
    updated["config"]["verification"].update(settings)
    validate_config(updated["config"])
    for run in updated["runs"]:
        run.update(settings)
        run.pop("fixed_qif", None)
        run.pop("verification_mode", None)
    updated.pop("fixed_artifact", None)
    updated.pop("fixed_artifact_sha256", None)
    return updated


def prepare(config_path, config, output, identity):
    calibration, evaluation = _load(config["calibration_study"]), _load(config["evaluation_study"])
    if (calibration["model_sha256"] != evaluation["model_sha256"]
            or any(calibration["config"][key] != evaluation["config"][key] for key in ("preprocessing", "model"))):
        raise ValueError("Calibration and evaluation must use the same source model and encoder")
    if ({s["id"] for s in calibration["selected_images"]}
            & {s["id"] for s in evaluation["selected_images"]}):
        raise ValueError("Calibration and evaluation cohorts overlap")
    calibration = _configured_study(calibration, config)
    evaluation = _configured_study(evaluation, config)
    runs = main_runs(calibration)
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "calibration_base.json", calibration)
    write_new_json(output / "evaluation_base.json", evaluation)
    sources, floors, counts = [], None, None
    unavailable = []
    for run in runs:
        region_root = output / "source" / run["run_id"]
        region_root.mkdir(parents=True)
        context = build_region_context(calibration, run, region_root)
        synth = context.synthesizer
        eligible = synth.check_source_region(context.low, context.high)
        source = {"run_id": run["run_id"], "sample": run["sample"], "epsilon": run["epsilon"],
                  "source_region": synth.source_region_summary(), "preimage": None, "integer_bit_floors": None}
        if eligible:
            synth.backward_preimage_computation()
            source["preimage"] = synth.preimage_provenance_summary()
            if not source["preimage"]["all_property_preimages_available"]:
                unavailable.append(run["run_id"])
            else:
                region_floors, region_counts = integer_bit_floors(synth)
                if counts is not None and region_counts != counts:
                    raise ValueError("Region-specific parameter shape drift")
                counts = region_counts
                floors = region_floors if floors is None else [max(a, b) for a, b in zip(floors, region_floors, strict=True)]
                source["integer_bit_floors"] = region_floors
        sources.append(source)
        write_new_json(region_root / "source_region.json", source)
        print(run["run_id"], source["source_region"]["status"], flush=True)

    eligible = sorted([s for s in sources if s["source_region"]["status"] == "VERIFIED"],
                      key=lambda s: (s["source_region"]["certified_margin_lower_bound"], s["run_id"]))
    candidates = enumerate_candidates(floors, counts, config["search"]) if eligible and not unavailable else []
    manifest = {"schema": "ssv_qif_search_materialized_v1", "identity": identity,
                "model_sha256": calibration["model_sha256"], "sources": sources,
                "eligible_run_ids": [s["run_id"] for s in eligible], "preimage_unavailable_run_ids": unavailable,
                "shared_integer_bit_floors": floors, "affine_parameter_counts": counts,
                "search": config["search"], "candidates": candidates,
                "calibration_base_sha256": sha256(output / "calibration_base.json"),
                "evaluation_base_sha256": sha256(output / "evaluation_base.json")}
    write_new_json(output / "search_manifest.json", manifest)
    write_csv(output / "source_regions.csv", [{"run_id": s["run_id"], "sample_id": s["sample"]["id"],
              "epsilon": s["epsilon"], "source_method": s["source_region"].get("method"),
              "source_status": s["source_region"]["status"],
              "deeppoly_certified_margin_lower_bound": s["source_region"].get(
                  "deeppoly_certified_margin_lower_bound", s["source_region"].get("certified_margin_lower_bound")),
              "certified_margin_lower_bound": s["source_region"]["certified_margin_lower_bound"],
              "preimage_status": s["preimage"]["status"] if s["preimage"] else "SKIPPED"} for s in sources])
    return manifest


def _valid_report(report, run_id, model_hash, artifact_hash, qif):
    if (report.get("run_id") != run_id or report.get("model_sha256") != model_hash
            or report.get("fixed_artifact_sha256") != artifact_hash
            or report.get("verification_mode") != "fixed_qif_check" or report.get("qif") != qif):
        raise ValueError("Candidate report has a different region, model or quantized program")


def _certified(report, artifact):
    return (report.get("final_status") == "VERIFIED"
            and report.get("source_region", {}).get("status") == "VERIFIED"
            and report.get("encoded_contract_verified") is True
            and report.get("input_bridge_checked") is True
            and report.get("byte_crop_property_verified") is True
            and report.get("chaining", {}).get("all_ok") is True
            and report.get("vacuity", {}).get("status") == "PASSED"
            and report.get("generated_source_sha256") == artifact["generated_source_sha256"])


def check_candidate(candidate, manifest, output):
    directory = output / "calibration" / candidate["id"]
    study_path = directory / "study.json"
    if not directory.exists():
        freeze(output / "calibration_base.json", directory, candidate["qif"], campaign="main")
    study = _load(study_path)
    artifact_path = Path(study["fixed_artifact"])
    artifact, _ = load_artifact(artifact_path, study)
    if artifact["qif"] != candidate["qif"]:
        raise ValueError("Resumed candidate Q/I/F changed")
    runs = {r["run_id"]: r for r in study["runs"]}
    records, failure = [], None
    runs_root = output / "runs" / candidate["id"]
    runs_root.mkdir(parents=True, exist_ok=True)
    for run_id in manifest["eligible_run_ids"]:
        destination = runs_root / run_id
        report_path = destination / "region_summary.json"
        if report_path.is_file():
            report = _load(report_path)
        else:
            try:
                report = run_region(study, runs[run_id], destination, fixed_artifact=artifact_path)
            except Exception as exc:
                destination.mkdir(parents=True, exist_ok=True)
                error_path = destination / "runner_error.json"
                if not error_path.exists():
                    write_new_json(error_path, {"status": "RUNNER_ERROR", "error_type": type(exc).__name__, "message": str(exc)})
                raise
        _valid_report(report, run_id, manifest["model_sha256"], sha256(artifact_path), candidate["qif"])
        passed = _certified(report, artifact)
        record = {"run_id": run_id, "status": report["final_status"], "certified": passed,
                  "report": str(report_path.resolve())}
        records.append(record)
        print(candidate["id"], run_id, record["status"], flush=True)
        if not passed:
            failure = record
            break
    return {**candidate, "status": "VERIFIED" if failure is None else failure["status"],
            "certified": failure is None and len(records) == len(manifest["eligible_run_ids"]),
            "first_failed_region": failure, "checked_regions": records,
            "skipped_run_ids": manifest["eligible_run_ids"][len(records):],
            "artifact_sha256": sha256(artifact_path),
            "generated_source_sha256": artifact["generated_source_sha256"],
            "study": str(study_path.resolve())}


def execute(manifest, output):
    outcomes, selected = [], None
    started = time.monotonic()
    for candidate in manifest["candidates"][:manifest["search"]["max_candidates"]]:
        summary_path = output / "calibration" / candidate["id"] / "candidate_summary.json"
        # Recheck report identity even when resuming a completed candidate.
        outcome = check_candidate(candidate, manifest, output)
        if summary_path.is_file():
            if _load(summary_path) != outcome:
                raise ValueError("Completed candidate outcome changed during resume")
        else:
            write_new_json(summary_path, outcome)
        outcomes.append(outcome)
        if outcome["certified"]:
            evaluation_dir = output / "evaluation"
            if not evaluation_dir.exists():
                freeze(output / "evaluation_base.json", evaluation_dir, candidate["qif"], campaign="full")
            evaluation = _load(evaluation_dir / "study.json")
            eval_artifact, _ = load_artifact(evaluation["fixed_artifact"], evaluation)
            if eval_artifact["generated_source_sha256"] != outcome["generated_source_sha256"]:
                raise ValueError("Evaluation artifact differs from the calibrated C program")
            selected = {**outcome, "evaluation_study": str((evaluation_dir / "study.json").resolve()),
                        "evaluation_status": "MATERIALIZED_NOT_RUN"}
            break
    n_eligible = len(manifest["eligible_run_ids"])
    exhausted = selected is None and len(outcomes) < len(manifest["candidates"])
    status = ("SELECTED" if selected else "SOURCE_PROPERTY_INCONCLUSIVE" if not n_eligible
              else "PREIMAGE_UNAVAILABLE" if manifest["preimage_unavailable_run_ids"]
              else "SEARCH_BUDGET_EXHAUSTED" if exhausted else "SEARCH_DOMAIN_EXHAUSTED")
    result = {"schema": "ssv_qif_search_result_v1", "status": status, "identity": manifest["identity"],
              "selected": selected, "candidate_outcomes": outcomes,
              "candidate_domain_size": len(manifest["candidates"]), "candidates_tested": len(outcomes),
              "n_calibration_regions": len(manifest["sources"]), "source_eligible_regions": n_eligible,
              "source_inconclusive_regions": sum(s["source_region"]["status"] in {"INCONCLUSIVE", "UNKNOWN"}
                                                 for s in manifest["sources"]),
              "source_refuted_regions": sum(s["source_region"]["status"] == "REFUTED"
                                            for s in manifest["sources"]),
              "shared_integer_bit_floors": manifest["shared_integer_bit_floors"],
              "objective": "min_affine_parameter_bits", "optimality": "first_certified_in_bounded_cost_order",
              "budget_exhausted": exhausted, "search_elapsed_seconds_this_invocation": time.monotonic() - started,
              "evaluation_status": "MATERIALIZED_NOT_RUN" if selected else "NOT_MATERIALIZED"}
    write_new_json(output / "search_summary.json", result)
    write_csv(output / "candidate_search.csv", [{"candidate": c["id"], "qif": json.dumps(c["qif"]),
              "affine_parameter_bits": c["affine_parameter_bits"], "status": c["status"],
              "certified": c["certified"], "regions_checked": len(c["checked_regions"]),
              "regions_skipped": len(c["skipped_run_ids"]),
              "first_failed_region": c["first_failed_region"]["run_id"] if c["first_failed_region"] else None} for c in outcomes])
    return result


def search(config_path, output, *, resume=False, prepare_only=False):
    config_path, output = Path(config_path), Path(output)
    config = _load(config_path)
    validate(config)
    identity = _identity(config_path, config)
    if resume:
        manifest = _load(output / "search_manifest.json")
        if (manifest["identity"] != identity
                or sha256(output / "calibration_base.json") != manifest["calibration_base_sha256"]
                or sha256(output / "evaluation_base.json") != manifest["evaluation_base_sha256"]):
            raise ValueError("Cannot resume: config, study or implementation identity changed")
    else:
        manifest = prepare(config_path, config, output, identity)
    if prepare_only:
        return {"status": "PREPARED_NOT_RUN", "candidate_domain_size": len(manifest["candidates"]), "selected": None}
    if (output / "search_summary.json").is_file():
        return _load(output / "search_summary.json")
    return execute(manifest, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Compute source checks, live MILP preimages and the domain without ESBMC")
    args = parser.parse_args()
    result = search(args.config, args.output, resume=args.resume, prepare_only=args.prepare_only)
    print(json.dumps({"status": result["status"], "candidate_domain_size": result["candidate_domain_size"],
                      "selected_qif": result["selected"]["qif"] if result["selected"] else None,
                      "output": str(args.output)}, indent=2))
    raise SystemExit(0 if args.prepare_only or result["selected"] else 2)


if __name__ == "__main__":
    main()
