"""Read existing reports for the methodology audit; never rewrite run artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_statuses(value, counts: Counter) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "final_status":
                counts[str(item)] += 1
            nested_statuses(item, counts)
    elif isinstance(value, list):
        for item in value:
            nested_statuses(item, counts)


def campaign(root: Path, name: str) -> dict:
    paths = sorted((root / "output" / name).glob("*/reports/pipeline_summary.json"))
    statuses, guarantees, soundness, experiment_soundness = (Counter() for _ in range(4))
    queries, query_statuses, query_times, nondeterminism = (Counter() for _ in range(4))
    cegar, e2e, saturation, exact_match = (Counter() for _ in range(4))
    margins = defaultdict(list)
    paired = defaultdict(dict)
    regions = set()
    rows = []
    for path in paths:
        pipeline = read(path)
        experiment = read(path.with_name("experiment_summary.json"))
        config = read(path.parent.parent / "run_config.json")
        run = path.parent.parent.name
        arm = run.split("_")[1]
        key = (pipeline["dataset"], pipeline["arch"], pipeline["sample_id"], pipeline["input_epsilon"])
        regions.add(key)
        statuses[str(experiment.get("final_status"))] += 1
        guarantees[str(experiment.get("guarantee_level"))] += 1
        soundness[str(pipeline.get("soundness"))] += 1
        experiment_soundness[str(experiment.get("soundness"))] += 1
        cegar[str(pipeline.get("cegar", {}).get("rounds_run"))] += 1
        e2e[str(pipeline.get("end_to_end_verification", {}).get("enabled"))] += 1
        saturation[str(experiment.get("quality_refined", {}).get("no_saturation_status"))] += 1
        exact_match[str(experiment.get("python_c_exact_match"))] += 1
        margins[pipeline["dataset"]].append((pipeline["clean_margin"], pipeline["source_region"]["certified_margin_lower_bound"]))
        for call in pipeline["esbmc_call_records"]:
            kind = call["property_type"]
            queries[kind] += 1
            query_statuses[f"{kind}:{call['status']}"] += 1
            query_times[kind] += float(call.get("elapsed_seconds", 0))
            harness = root / call["harness"]
            if harness.exists():
                symbolic = "nondet_longlong(" in harness.read_text(encoding="utf-8")
                nondeterminism[f"{kind}:{symbolic}"] += 1
        runtime = pipeline["timing_metrics"]["total_runtime_seconds"]
        qif = tuple(zip(pipeline["synthesis"]["total_bits"], pipeline["synthesis"]["integer_bits"], pipeline["synthesis"]["fractional_bits"]))
        row = {
            "run": run,
            "beta": config["esbmc_layer_block_size"],
            "timeout_seconds": config["esbmc_timeout_seconds"],
            "memlimit": config["esbmc_memlimit"],
            "total_runtime_seconds": runtime,
            "calls": len(pipeline["esbmc_call_records"]),
            "qif": qif,
        }
        rows.append(row)
        if config["esbmc_layer_block_size"] == 2:
            paired[key][arm] = row
    comparisons = {}
    for dataset in sorted(margins):
        pairs = [v for k, v in paired.items() if k[0] == dataset and "a" in v and "b" in v]
        if pairs:
            ratios = [v["a"]["total_runtime_seconds"] / v["b"]["total_runtime_seconds"] for v in pairs]
            comparisons[dataset] = {
                "matched_regions": len(pairs),
                "cuts_on_over_off_runtime_ratio_median": statistics.median(ratios),
                "same_selected_qif_in_every_pair": all(v["a"]["qif"] == v["b"]["qif"] for v in pairs),
            }
    time_sum = sum(query_times.values())
    return {
        "run_reports": len(paths),
        "distinct_model_sample_epsilon_regions": len(regions),
        "experiment_final_status": dict(statuses),
        "guarantee_level": dict(guarantees),
        "pipeline_soundness": dict(soundness),
        "experiment_soundness_get": dict(experiment_soundness),
        "no_saturation_status": dict(saturation),
        "python_c_exact_match": dict(exact_match),
        "e2e_enabled": dict(e2e),
        "cegar_rounds_run": dict(cegar),
        "query_counts": dict(queries),
        "query_status_counts": dict(query_statuses),
        "query_elapsed_seconds": dict(query_times),
        "query_time_fractions": {k: v / time_sum for k, v in query_times.items()} if time_sum else {},
        "harness_contains_nondet_by_query": dict(nondeterminism),
        "margin_ranges": {
            dataset: {
                "clean_min_median_max": [min(x[0] for x in values), statistics.median(x[0] for x in values), max(x[0] for x in values)],
                "source_min_median_max": [min(x[1] for x in values), statistics.median(x[1] for x in values), max(x[1] for x in values)],
            }
            for dataset, values in sorted(margins.items())
        },
        "matched_beta2_cuts_comparison": comparisons,
        "selected_runtime_rows": [
            row for row in rows
            if name.endswith("escalation_runs")
            or row["run"].startswith("ladc_a_iris_beta_sweep_sample0_")
            or row["run"].startswith("ladc_a_mnist_1blk25_beta_sweep_sample3_")
        ],
    }


def collect(root: Path) -> dict:
    paths = sorted((root / "output").rglob("experiment_summary.json"))
    top, nested = Counter(), Counter()
    for path in paths:
        report = read(path)
        top[str(report.get("final_status"))] += 1
        nested_statuses(report, nested)
    confirmed = []
    margin_refuted = []
    confirmed_reports_only = 0
    for path in sorted((root / "output").rglob("pipeline_summary.json")):
        report = read(path)
        if "MARGIN_REFUTED" in json.dumps(report):
            margin_refuted.append(str(path.relative_to(root)))
        for record in report.get("counterexamples", {}).get("records", []):
            if record.get("replay_confirmed"):
                confirmed.append(json.dumps(record, sort_keys=True))
                confirmed_reports_only += path.parent.name == "reports"
    return {
        "counting_policy": "Current files, including archives; nested statuses and duplicated reports are not independent regions.",
        "experiment_summary_files": len(paths),
        "top_level_final_status_counts": dict(top),
        "nested_final_status_occurrences": sum(nested.values()),
        "nested_final_status_counts": dict(nested),
        "pipeline_files_containing_margin_refuted": margin_refuted,
        "confirmed_cex_payload_occurrences_all_pipeline_files": len(confirmed),
        "confirmed_cex_payload_occurrences_reports_directory_only": confirmed_reports_only,
        "distinct_confirmed_cex_payloads_not_independent_region_count": len(set(confirmed)),
        "campaigns": {
            name: campaign(root, name)
            for name in ("ladc2026_matrix_runs", "ladc2026_mnist_timeout_escalation_runs")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(collect(args.repo_root.resolve()), indent=2))
