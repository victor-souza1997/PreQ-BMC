from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METHODS = ("formal_only", "quality_refined")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate paper experiment outputs.")
    parser.add_argument("--runs-root", type=Path, default=Path("output/paper_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/paper_results"))
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in fieldnames})
    return path


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _discover_run_dirs(runs_root: Path) -> list[Path]:
    candidates = {path.parent for path in runs_root.rglob("run_status.json")}
    candidates.update(path.parents[1] for path in runs_root.rglob("reports/experiment_summary.json"))
    return sorted(path for path in candidates if path.exists())


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _bool_or_empty(value: Any) -> Any:
    if value is None or value == "":
        return ""
    return bool(value)


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _accumulator_fits_int64(section: dict[str, Any]) -> Any:
    layers = section.get("accumulator_range", [])
    if not layers:
        return ""
    return bool(all(layer.get("fits_int64", False) for layer in layers))


def _semantics_available(section: dict[str, Any]) -> bool:
    return bool(section.get("fixed_point_semantics", {}).get("layers"))


def _method_row(
    run_dir: Path,
    status: dict[str, Any],
    experiment: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    benchmark = experiment.get("benchmark", {})
    reference = experiment.get("reference", {})
    section = experiment.get(method, {})
    deployment = section.get("deployment_metrics", {})
    resource = section.get("resource_metrics", {})
    stats = section.get("verification_stats", {})
    artifacts = experiment.get("artifacts", {})
    quality = experiment.get("quality_refined", {})
    run_name = status.get("name") or run_dir.name
    status_value = status.get("status") or ("success" if experiment else "failed")

    return {
        "run_name": run_name,
        "dataset": benchmark.get("dataset", status.get("dataset")),
        "arch": benchmark.get("arch", status.get("arch")),
        "sample_id": benchmark.get("sample_id", status.get("sample_id")),
        "eps": benchmark.get("eps", status.get("eps")),
        "method": method,
        "status": status_value,
        "success": status_value == "success",
        "verified": section.get("success") if method == "formal_only" else section.get("accepted"),
        "accepted": section.get("accepted") if method == "quality_refined" else "",
        "Q": section.get("Q"),
        "I": section.get("I"),
        "F": section.get("F"),
        "total_bits_sum": section.get("total_bits_sum"),
        "weighted_avg_bits_per_parameter": section.get("weighted_avg_bits_per_parameter"),
        "full_precision_keras_accuracy": reference.get("full_precision_keras_accuracy"),
        "quantized_keras_accuracy": deployment.get("quantized_keras_accuracy"),
        "python_fixed_accuracy": deployment.get("python_fixed_accuracy"),
        "c_fixed_accuracy": deployment.get("c_fixed_accuracy"),
        "python_c_exact_match": deployment.get("python_c_exact_match"),
        "mismatch_rate_vs_keras": deployment.get("mismatch_rate_vs_keras"),
        "max_saturation_rate": deployment.get("max_saturation_rate"),
        "mean_saturation_rate": deployment.get("mean_saturation_rate"),
        "max_abs_logit_error": deployment.get("max_abs_logit_error"),
        "mean_abs_logit_error": deployment.get("mean_abs_logit_error"),
        "num_layers": resource.get("num_layers"),
        "num_parameters": resource.get("num_parameters"),
        "float_parameter_memory_bytes": resource.get("float_parameter_memory_bytes"),
        "fixed_parameter_memory_bytes": resource.get("fixed_parameter_memory_bytes"),
        "compression_ratio_vs_float32": resource.get("compression_ratio_vs_float32"),
        "activation_memory_bytes_estimate": resource.get("activation_memory_bytes_estimate"),
        "c_source_lines": resource.get("c_source_lines"),
        "c_shared_library_size_bytes": resource.get("c_shared_library_size_bytes"),
        "backward_time": stats.get("backward_time"),
        "forward_time": stats.get("forward_time"),
        "total_time": stats.get("total_time"),
        "esbmc_calls": stats.get("esbmc_calls"),
        "refinement_steps": quality.get("refinement_steps") if method == "quality_refined" else "",
        "final_reason": quality.get("final_reason") if method == "quality_refined" else "",
        "fixed_point_semantics_available": _semantics_available(section),
        "no_saturation_verified_all_layers": section.get("no_saturation_verified_all_layers"),
        "accumulator_fits_int64_all_layers": _accumulator_fits_int64(section),
        "artifact_c_source": artifacts.get("c_source"),
        "artifact_c_shared_library": artifacts.get("c_shared_library"),
        "error_message": status.get("error_message"),
    }


ALL_EXPERIMENTS_FIELDS = [
    "run_name",
    "dataset",
    "arch",
    "sample_id",
    "eps",
    "method",
    "status",
    "success",
    "verified",
    "accepted",
    "Q",
    "I",
    "F",
    "total_bits_sum",
    "weighted_avg_bits_per_parameter",
    "full_precision_keras_accuracy",
    "quantized_keras_accuracy",
    "python_fixed_accuracy",
    "c_fixed_accuracy",
    "python_c_exact_match",
    "mismatch_rate_vs_keras",
    "max_saturation_rate",
    "mean_saturation_rate",
    "max_abs_logit_error",
    "mean_abs_logit_error",
    "num_layers",
    "num_parameters",
    "float_parameter_memory_bytes",
    "fixed_parameter_memory_bytes",
    "compression_ratio_vs_float32",
    "activation_memory_bytes_estimate",
    "c_source_lines",
    "c_shared_library_size_bytes",
    "backward_time",
    "forward_time",
    "total_time",
    "esbmc_calls",
    "refinement_steps",
    "final_reason",
    "fixed_point_semantics_available",
    "no_saturation_verified_all_layers",
    "accumulator_fits_int64_all_layers",
    "artifact_c_source",
    "artifact_c_shared_library",
    "error_message",
]


def _concat_report_csv(run_dirs: list[Path], report_name: str, output_path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = ["run_name"]
    for run_dir in run_dirs:
        status = _read_json(run_dir / "run_status.json")
        run_name = status.get("name") or run_dir.name
        for row in _read_csv_rows(run_dir / "reports" / report_name):
            merged = {"run_name": run_name, **row}
            rows.append(merged)
            for key in merged:
                if key not in fieldnames:
                    fieldnames.append(key)
    return _write_csv(output_path, rows, fieldnames)


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _latex_cell(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and len(stripped) > 28:
            try:
                parsed = json.loads(stripped)
                return _latex_escape("/".join(str(x) for x in parsed))
            except Exception:
                pass
    numeric = _num(value)
    if numeric is not None:
        return f"{numeric:.4f}".rstrip("0").rstrip(".")
    return _latex_escape(value)


def _write_latex_table(path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:limit] if limit is not None else rows
    alignment = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(_latex_escape(header) for header, _ in columns) + " \\\\",
        "\\hline",
    ]
    for row in selected:
        lines.append(" & ".join(_latex_cell(row.get(key)) for _, key in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_latex_tables(output_dir: Path, rows: list[dict[str, Any]], fixed_semantics_rows: list[dict[str, Any]]) -> dict[str, str]:
    tables_dir = output_dir / "tables"
    artifacts = {
        "table_main_formal_vs_refined_tex": str(
            _write_latex_table(
                tables_dir / "table_main_formal_vs_refined.tex",
                rows,
                [
                    ("Dataset", "dataset"),
                    ("Arch", "arch"),
                    ("Method", "method"),
                    ("Q", "Q"),
                    ("I", "I"),
                    ("F", "F"),
                    ("C Acc.", "c_fixed_accuracy"),
                    ("Max Sat.", "max_saturation_rate"),
                    ("Mismatch", "mismatch_rate_vs_keras"),
                    ("Verified", "verified"),
                    ("Time", "total_time"),
                ],
            )
        ),
        "table_deployment_gap_tex": str(
            _write_latex_table(
                tables_dir / "table_deployment_gap.tex",
                rows,
                [
                    ("Dataset", "dataset"),
                    ("Arch", "arch"),
                    ("Method", "method"),
                    ("Keras Q Acc.", "quantized_keras_accuracy"),
                    ("Python Fixed Acc.", "python_fixed_accuracy"),
                    ("C Fixed Acc.", "c_fixed_accuracy"),
                    ("Python/C Exact", "python_c_exact_match"),
                    ("Max Error", "max_abs_logit_error"),
                ],
            )
        ),
        "table_resource_metrics_tex": str(
            _write_latex_table(
                tables_dir / "table_resource_metrics.tex",
                rows,
                [
                    ("Dataset", "dataset"),
                    ("Arch", "arch"),
                    ("Method", "method"),
                    ("Params", "num_parameters"),
                    ("Avg Bits", "weighted_avg_bits_per_parameter"),
                    ("Fixed Bytes", "fixed_parameter_memory_bytes"),
                    ("Compression", "compression_ratio_vs_float32"),
                ],
            )
        ),
        "table_scalability_tex": str(
            _write_latex_table(
                tables_dir / "table_scalability.tex",
                [row for row in rows if row.get("method") == "formal_only"],
                [
                    ("Dataset", "dataset"),
                    ("Arch", "arch"),
                    ("Params", "num_parameters"),
                    ("ESBMC Calls", "esbmc_calls"),
                    ("Time", "total_time"),
                    ("Status", "status"),
                ],
            )
        ),
    }
    if fixed_semantics_rows:
        artifacts["table_fixed_point_semantics_tex"] = str(
            _write_latex_table(
                tables_dir / "table_fixed_point_semantics.tex",
                fixed_semantics_rows,
                [
                    ("Dataset", "dataset"),
                    ("Arch", "arch"),
                    ("Method", "method"),
                    ("Layer", "layer_index"),
                    ("Q", "Q"),
                    ("I", "I"),
                    ("F", "F"),
                    ("Overflow Mode", "overflow_mode"),
                    ("Rounding Mode", "rounding_mode"),
                    ("Acc Fits Int64", "fits_int64"),
                    ("No-Sat", "no_saturation_status"),
                ],
            )
        )
    return artifacts


def _no_saturation_status(row: dict[str, Any]) -> str:
    if row.get("no_saturation_verified_all_layers") in {True, "True", "true", "1"}:
        return "VERIFIED"
    value = row.get("no_saturation_verified_all_layers")
    if value in {False, "False", "false", "0"}:
        return "NOT_VERIFIED"
    return ""


def _enrich_fixed_semantics_rows(
    fixed_semantics_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    status_by_run_method = {
        (str(row.get("run_name", "")), str(row.get("method", ""))): _no_saturation_status(row)
        for row in experiment_rows
    }
    enriched: list[dict[str, Any]] = []
    for row in fixed_semantics_rows:
        merged = dict(row)
        key = (str(row.get("run_name", "")), str(row.get("method", "")))
        merged["no_saturation_status"] = status_by_run_method.get(key, "")
        enriched.append(merged)
    return enriched


def _best_row(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("method") == method and _num(row.get("c_fixed_accuracy")) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _num(row.get("c_fixed_accuracy")) or -1.0)


def _group_by_run(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("run_name", "")), {})[str(row.get("method", ""))] = row
    return grouped


def _write_summary_md(output_dir: Path, rows: list[dict[str, Any]], failed_rows: list[dict[str, Any]]) -> Path:
    grouped = _group_by_run(rows)
    succeeded = {name for name, methods in grouped.items() if any(row.get("status") == "success" for row in methods.values())}
    datasets = sorted({str(row.get("dataset")) for row in rows if row.get("dataset")})
    archs = sorted({str(row.get("arch")) for row in rows if row.get("arch")})
    verified_rows = [row for row in rows if str(row.get("status")) == "success"]
    largest = max(verified_rows, key=lambda row: _num(row.get("num_parameters")) or -1.0, default=None)
    best_formal = _best_row(rows, "formal_only")
    best_refined = _best_row(rows, "quality_refined")

    improved: list[str] = []
    saturation_reduced: list[str] = []
    exact_match = 0
    for run_name, methods in grouped.items():
        formal = methods.get("formal_only", {})
        refined = methods.get("quality_refined", {})
        formal_acc = _num(formal.get("c_fixed_accuracy")) or _num(formal.get("python_fixed_accuracy"))
        refined_acc = _num(refined.get("c_fixed_accuracy")) or _num(refined.get("python_fixed_accuracy"))
        keras_acc = _num(formal.get("quantized_keras_accuracy"))
        if formal_acc is not None and refined_acc is not None and keras_acc is not None:
            if keras_acc - formal_acc > 0.05 and refined_acc > formal_acc:
                improved.append(run_name)
        formal_sat = _num(formal.get("max_saturation_rate"))
        refined_sat = _num(refined.get("max_saturation_rate"))
        if formal_sat is not None and refined_sat is not None and refined_sat < formal_sat:
            saturation_reduced.append(run_name)
        if refined.get("python_c_exact_match") in {True, "True", "true", "1"}:
            exact_match += 1

    slowest = max(verified_rows, key=lambda row: _num(row.get("total_time")) or -1.0, default=None)
    lines = [
        "# Paper Experiment Summary",
        "",
        f"- Total runs configured: {len(grouped)}",
        f"- Total runs succeeded: {len(succeeded)}",
        f"- Total runs failed/skipped: {len(failed_rows)}",
        f"- Datasets covered: {', '.join(datasets) or '(none)'}",
        f"- Architectures covered: {', '.join(archs) or '(none)'}",
        f"- Largest verified model: {_describe_row(largest)}",
        f"- Best formal-only result: {_describe_row(best_formal, include_accuracy=True)}",
        f"- Best refined result: {_describe_row(best_refined, include_accuracy=True)}",
        f"- Formal-only deployment-poor cases improved by refinement: {len(improved)}",
        f"- Runs with saturation reduction: {len(saturation_reduced)}",
        f"- Runs with Python/C exact match in refined backend: {exact_match}",
        f"- Scalability observation: slowest successful row was {_describe_row(slowest, include_time=True)}",
        "",
        "## Improved Deployment-Poor Cases",
        *(f"- {name}" for name in improved[:25]),
        "",
        "## Saturation Reduction Cases",
        *(f"- {name}" for name in saturation_reduced[:25]),
    ]
    path = output_dir / "paper_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _describe_row(row: dict[str, Any] | None, *, include_accuracy: bool = False, include_time: bool = False) -> str:
    if not row:
        return "(none)"
    bits = f"{row.get('dataset')}/{row.get('arch')} ({row.get('run_name')})"
    params = row.get("num_parameters")
    if params not in (None, ""):
        bits += f", params={params}"
    if include_accuracy:
        bits += f", c_acc={row.get('c_fixed_accuracy')}"
    if include_time:
        bits += f", time={row.get('total_time')}"
    return bits


def aggregate(runs_root: Path, output_dir: Path) -> dict[str, Any]:
    run_dirs = _discover_run_dirs(runs_root)
    rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        status = _read_json(run_dir / "run_status.json")
        experiment = _read_json(run_dir / "reports" / "experiment_summary.json")
        pipeline = _read_json(run_dir / "reports" / "pipeline_summary.json")
        if not status:
            status = {
                "name": run_dir.name,
                "status": "success" if experiment else "failed",
                "output_dir": str(run_dir),
            }
        if status.get("status") != "success":
            failed_rows.append(status)
        for method in METHODS:
            rows.append(_method_row(run_dir, status, experiment, method))
        run_records.append(
            {
                "run_dir": str(run_dir),
                "status": status,
                "experiment_summary": experiment,
                "pipeline_summary_available": bool(pipeline),
            }
        )

    _write_csv(output_dir / "all_experiments.csv", rows, ALL_EXPERIMENTS_FIELDS)
    _write_json(
        output_dir / "all_experiments.json",
        {
            "runs_root": str(runs_root),
            "num_run_dirs": len(run_dirs),
            "num_rows": len(rows),
            "num_failed_or_skipped": len(failed_rows),
            "runs": run_records,
        },
    )
    _write_csv(
        output_dir / "failed_runs.csv",
        failed_rows,
        [
            "name",
            "dataset",
            "arch",
            "sample_id",
            "eps",
            "status",
            "return_code",
            "started_at",
            "finished_at",
            "elapsed_seconds",
            "output_dir",
            "error_message",
        ],
    )

    _concat_report_csv(run_dirs, "table_formal_vs_refined.csv", output_dir / "all_formal_vs_refined.csv")
    _concat_report_csv(run_dirs, "table_deployment_metrics.csv", output_dir / "all_deployment_metrics.csv")
    _concat_report_csv(run_dirs, "table_resource_metrics.csv", output_dir / "all_resource_metrics.csv")
    _concat_report_csv(run_dirs, "table_refinement_history.csv", output_dir / "all_refinement_history.csv")
    fixed_semantics_path = _concat_report_csv(
        run_dirs,
        "table_fixed_point_semantics.csv",
        output_dir / "all_fixed_point_semantics.csv",
    )
    fixed_semantics_rows = _enrich_fixed_semantics_rows(_read_csv_rows(fixed_semantics_path), rows)
    if fixed_semantics_rows:
        fixed_fields: list[str] = []
        for row in fixed_semantics_rows:
            for key in row:
                if key not in fixed_fields:
                    fixed_fields.append(key)
        _write_csv(fixed_semantics_path, fixed_semantics_rows, fixed_fields)

    latex_artifacts = _write_latex_tables(output_dir, rows, fixed_semantics_rows)
    summary_path = _write_summary_md(output_dir, rows, failed_rows)
    return {
        "all_experiments_csv": str(output_dir / "all_experiments.csv"),
        "all_experiments_json": str(output_dir / "all_experiments.json"),
        "paper_summary_md": str(summary_path),
        **latex_artifacts,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifacts = aggregate(args.runs_root, args.output_dir)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
