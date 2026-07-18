from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METHODS = ("formal_only", "quality_refined")
STATUS_ORDER = ("FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN", "SKIPPED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate article experiment outputs.")
    parser.add_argument("--input-root", "--runs-root", dest="input_root", type=Path, default=Path("output/article_runs"))
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, default=Path("output/article_results"))
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _cell(row.get(field)) for field in fieldnames})
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _base_dataset(dataset: Any) -> str:
    text = str(dataset or "")
    for base in ("fashion-mnist", "mnist_onnx", "mnist64", "mnist", "iris", "seeds"):
        if text == base or text.startswith(f"{base}_"):
            return base
    return text


def _input_scale(dataset: Any) -> float:
    return 255.0 if _base_dataset(dataset) in {"mnist", "mnist64", "mnist_onnx", "fashion-mnist"} else 1.0


def _input_epsilon(status: dict[str, Any], run_config: dict[str, Any], pipeline: dict[str, Any], benchmark: dict[str, Any]) -> Any:
    return (
        pipeline.get("input_epsilon")
        or pipeline.get("eps")
        or benchmark.get("input_epsilon")
        or benchmark.get("eps")
        or run_config.get("input_epsilon")
        or run_config.get("eps")
        or status.get("input_epsilon")
        or status.get("eps")
    )


def _normalized_input_epsilon(dataset: Any, input_epsilon: Any, pipeline: dict[str, Any]) -> Any:
    if pipeline.get("normalized_input_epsilon") is not None:
        return pipeline.get("normalized_input_epsilon")
    eps_value = _num(input_epsilon)
    if eps_value is None:
        return ""
    return eps_value / _input_scale(dataset)


def _discover_run_dirs(input_root: Path) -> list[Path]:
    if not input_root.exists():
        return []
    candidates = {path.parent for path in input_root.rglob("run_status.json")}
    candidates.update(path.parents[1] for path in input_root.rglob("reports/experiment_summary.json"))
    for path in input_root.rglob("pipeline_summary.json"):
        candidates.add(path.parent.parent if path.parent.name == "reports" else path.parent)
    return sorted(path for path in candidates if path.exists())


def _first_existing_json(*paths: Path) -> dict[str, Any]:
    for path in paths:
        payload = _read_json(path)
        if payload:
            return payload
    return {}


def _status_priority(statuses: list[str]) -> str:
    cleaned = [status for status in statuses if status]
    if not cleaned:
        return "UNKNOWN"
    if all(status == "VERIFIED" for status in cleaned):
        return "VERIFIED"
    if "PARTIAL_VERIFIED" in cleaned and not any(status in STATUS_ORDER[:3] for status in cleaned):
        return "PARTIAL_VERIFIED"
    for status in STATUS_ORDER:
        if status in cleaned:
            return status
    return cleaned[0]


def _layer_records_from_pipeline(pipeline: dict[str, Any], section: dict[str, Any]) -> list[dict[str, Any]]:
    layers = section.get("esbmc_status_per_layer")
    if isinstance(layers, list) and layers:
        return layers
    quality = pipeline.get("quality_refinement", {})
    for step in reversed(quality.get("steps", [])):
        step_layers = _get(step, "esbmc", "layers", default=[])
        if step_layers:
            return step_layers
    return _get(pipeline, "formal_saturation_verification", "layers", default=[])


def _method_row(
    *,
    run_dir: Path,
    run_status: dict[str, Any],
    run_config: dict[str, Any],
    experiment: dict[str, Any],
    pipeline: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    benchmark = experiment.get("benchmark", {})
    section = experiment.get(method, {})
    deployment = section.get("deployment_metrics", {})
    reference = experiment.get("reference", {})
    stats = section.get("verification_stats", {})
    timing = pipeline.get("timing_metrics", {})
    esbmc = pipeline.get("esbmc_status_counts", {})
    blockwise = pipeline.get("blockwise_verification", {})
    dataset = benchmark.get("dataset", pipeline.get("dataset", run_config.get("dataset", run_status.get("dataset"))))
    arch = benchmark.get("arch", pipeline.get("arch", run_config.get("arch", run_status.get("arch"))))
    input_epsilon = _input_epsilon(run_status, run_config, pipeline, benchmark)
    normalized_input_epsilon = _normalized_input_epsilon(dataset, input_epsilon, pipeline)
    float32_accuracy = reference.get("full_precision_keras_accuracy")
    quantized_acc = deployment.get("quantized_keras_accuracy")
    python_acc = deployment.get("python_fixed_accuracy")
    c_acc = deployment.get("c_fixed_accuracy")
    python_c_exact = section.get("python_c_exact_match", deployment.get("python_c_exact_match"))
    section_success = bool(section.get("success")) if method == "formal_only" else bool(section.get("accepted"))
    contract_status = section.get("contract_status") or (
        "VERIFIED" if section.get("contract_verified") else "UNKNOWN"
    )
    if contract_status == "UNKNOWN" and method == "formal_only" and section_success:
        verified_count = _num(esbmc.get("esbmc_verified_count")) or 0
        failed_count = sum(
            _num(esbmc.get(key)) or 0
            for key in ("esbmc_failed_count", "esbmc_timeout_count", "esbmc_memout_count", "esbmc_unknown_count")
        )
        if verified_count > 0 and failed_count == 0:
            contract_status = "VERIFIED"
    contract_verified_value = bool(section.get("contract_verified", contract_status == "VERIFIED") or contract_status == "VERIFIED")
    no_saturation_status = section.get("no_saturation_status") or "SKIPPED"
    final_status = section.get("final_status") or run_status.get("final_status") or run_status.get("status", "UNKNOWN")
    layer_records = _layer_records_from_pipeline(pipeline, section)
    failure_layer = ""
    failure_block = ""
    failure_property = ""
    for layer in layer_records:
        layer_statuses = [
            str(layer.get("contract_status", "")),
            str(layer.get("no_saturation_status", "")),
            str(layer.get("status", "")),
        ]
        if any(status in {"FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN"} for status in layer_statuses):
            failure_layer = layer.get("layer_index", "")
            failure_property = (
                "no_saturation"
                if str(layer.get("no_saturation_status", "")) in {"FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN"}
                else "contract"
            )
            blocks = layer.get("blocks") or layer.get("no_saturation_blocks") or []
            for block in blocks:
                if block.get("status") in {"FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN"}:
                    failure_block = block.get("block_index", "")
                    failure_property = block.get("property_type", failure_property)
                    break
            break

    formal_success = bool(contract_verified_value or section_success)
    deployment_success = bool(section.get("deployment_quality_accepted", False))
    full_success = bool(formal_success and deployment_success and _bool(python_c_exact) is True)
    return {
        "run_name": run_status.get("name") or run_config.get("name") or run_dir.name,
        "dataset": dataset,
        "arch": arch,
        "sample_id": benchmark.get("sample_id", pipeline.get("sample_id", run_config.get("sample_id", run_status.get("sample_id")))),
        "input_epsilon": input_epsilon,
        "normalized_input_epsilon": normalized_input_epsilon,
        "method": method,
        "mode": run_config.get("mode", run_config.get("ablation_mode", "full_pipeline")),
        "status": run_status.get("status", "success" if experiment else "failed"),
        "final_status": final_status,
        "contract_status": contract_status,
        "contract_verified": contract_verified_value,
        "no_saturation_status": no_saturation_status,
        "no_saturation_verified": section.get("no_saturation_verified"),
        "formal_success": formal_success,
        "deployment_quality_accepted": deployment_success,
        "deployment_success": deployment_success,
        "python_c_exact_match": python_c_exact,
        "full_success": full_success,
        "float32_accuracy": float32_accuracy,
        "quantized_keras_accuracy": quantized_acc,
        "python_fixed_accuracy": python_acc,
        "c_fixed_accuracy": c_acc,
        "accuracy_drop_float_to_keras_quantized": _drop(float32_accuracy, quantized_acc),
        "accuracy_drop_keras_quantized_to_python_fixed": _drop(quantized_acc, python_acc),
        "accuracy_drop_keras_quantized_to_c_fixed": _drop(quantized_acc, c_acc),
        "mismatch_rate_vs_keras": deployment.get("mismatch_rate_vs_keras"),
        "max_abs_logit_error": deployment.get("max_abs_logit_error"),
        "mean_abs_logit_error": deployment.get("mean_abs_logit_error"),
        "max_saturation_rate": deployment.get("max_saturation_rate"),
        "mean_saturation_rate": deployment.get("mean_saturation_rate"),
        "Q": section.get("Q"),
        "I": section.get("I"),
        "F": section.get("F"),
        "total_bits_sum": section.get("total_bits_sum"),
        "weighted_avg_bits_per_parameter": section.get("weighted_avg_bits_per_parameter"),
        "refinement_steps": section.get("refinement_steps") if method == "quality_refined" else 0,
        "failure_reason": section.get("final_reason") or run_status.get("error_message"),
        "failed_layer": failure_layer,
        "failed_block": failure_block,
        "failed_property": failure_property,
        "total_runtime_seconds": timing.get("total_runtime_seconds", run_status.get("elapsed_seconds")),
        "preimage_time_seconds": timing.get("preimage_time_seconds", stats.get("backward_time")),
        "bitwidth_search_time_seconds": timing.get("bitwidth_search_time_seconds", stats.get("forward_time")),
        "esbmc_contract_time_seconds": timing.get("esbmc_contract_time_seconds"),
        "esbmc_no_saturation_time_seconds": timing.get("esbmc_no_saturation_time_seconds"),
        "deployment_eval_time_seconds": timing.get("deployment_eval_time_seconds"),
        "refinement_time_seconds": timing.get("refinement_time_seconds"),
        "total_esbmc_time_seconds": timing.get("total_esbmc_time_seconds"),
        "max_esbmc_query_time_seconds": timing.get("max_esbmc_query_time_seconds"),
        "mean_esbmc_query_time_seconds": timing.get("mean_esbmc_query_time_seconds"),
        "number_of_esbmc_calls": timing.get("number_of_esbmc_calls", stats.get("esbmc_calls")),
        "esbmc_verified_count": esbmc.get("esbmc_verified_count"),
        "esbmc_failed_count": esbmc.get("esbmc_failed_count"),
        "esbmc_timeout_count": esbmc.get("esbmc_timeout_count"),
        "esbmc_memout_count": esbmc.get("esbmc_memout_count"),
        "esbmc_unknown_count": esbmc.get("esbmc_unknown_count"),
        "esbmc_total_count": esbmc.get("esbmc_total_count"),
        "timeout_rate": esbmc.get("timeout_rate"),
        "memout_rate": esbmc.get("memout_rate"),
        "unknown_rate": esbmc.get("unknown_rate"),
        "blockwise_enabled": blockwise.get("enabled", section.get("blockwise_verification_enabled")),
        "block_size": blockwise.get("block_size", section.get("blockwise_block_size")),
        "total_blocks": blockwise.get("total_blocks", section.get("blockwise_total_blocks")),
        "verified_blocks": blockwise.get("verified_blocks", section.get("blockwise_verified_blocks")),
        "failed_blocks": blockwise.get("failed_blocks", section.get("blockwise_failed_blocks")),
        "timeout_blocks": blockwise.get("timeout_blocks", section.get("blockwise_timeout_blocks")),
        "memout_blocks": blockwise.get("memout_blocks"),
        "unknown_blocks": blockwise.get("unknown_blocks"),
        "skipped_blocks_due_to_fail_fast": blockwise.get("skipped_blocks_due_to_fail_fast"),
        "largest_neurons_per_query": blockwise.get("largest_neurons_per_query"),
        "largest_input_dim_per_query": blockwise.get("largest_input_dim_per_query"),
        "largest_estimated_macs_per_query": blockwise.get("largest_estimated_macs_per_query"),
        "output_dir": str(run_dir),
    }


def _drop(source: Any, target: Any) -> Any:
    source_num = _num(source)
    target_num = _num(target)
    if source_num is None or target_num is None:
        return ""
    return source_num - target_num


ALL_FIELDS = [
    "run_name", "dataset", "arch", "sample_id", "input_epsilon", "normalized_input_epsilon",
    "method", "mode", "status", "final_status", "contract_status", "contract_verified",
    "no_saturation_status", "no_saturation_verified", "formal_success", "deployment_success",
    "full_success", "python_c_exact_match", "float32_accuracy", "quantized_keras_accuracy",
    "python_fixed_accuracy", "c_fixed_accuracy", "accuracy_drop_float_to_keras_quantized",
    "accuracy_drop_keras_quantized_to_python_fixed", "accuracy_drop_keras_quantized_to_c_fixed",
    "mismatch_rate_vs_keras", "max_abs_logit_error", "mean_abs_logit_error", "max_saturation_rate",
    "mean_saturation_rate", "Q", "I", "F", "total_bits_sum", "weighted_avg_bits_per_parameter",
    "refinement_steps", "failure_reason", "failed_layer", "failed_block", "failed_property",
    "total_runtime_seconds", "preimage_time_seconds", "bitwidth_search_time_seconds",
    "esbmc_contract_time_seconds", "esbmc_no_saturation_time_seconds", "deployment_eval_time_seconds",
    "refinement_time_seconds", "total_esbmc_time_seconds", "max_esbmc_query_time_seconds",
    "mean_esbmc_query_time_seconds", "number_of_esbmc_calls", "esbmc_verified_count",
    "esbmc_failed_count", "esbmc_timeout_count", "esbmc_memout_count", "esbmc_unknown_count",
    "esbmc_total_count", "timeout_rate", "memout_rate", "unknown_rate", "blockwise_enabled",
    "block_size", "total_blocks", "verified_blocks", "failed_blocks", "timeout_blocks",
    "memout_blocks", "unknown_blocks", "skipped_blocks_due_to_fail_fast", "largest_neurons_per_query",
    "largest_input_dim_per_query", "largest_estimated_macs_per_query", "output_dir",
]


def _bitwidth_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        q_values = _json_list(row.get("Q"))
        i_values = _json_list(row.get("I"))
        f_values = _json_list(row.get("F"))
        count = max(len(q_values), len(i_values), len(f_values))
        for index in range(count):
            q = _list_get(q_values, index)
            i = _list_get(i_values, index)
            f = _list_get(f_values, index)
            output.append(
                {
                    "run_name": row.get("run_name"),
                    "dataset": row.get("dataset"),
                    "arch": row.get("arch"),
                    "sample_id": row.get("sample_id"),
                    "input_epsilon": row.get("input_epsilon"),
                    "normalized_input_epsilon": row.get("normalized_input_epsilon"),
                    "method": row.get("method"),
                    "layer_index": index,
                    "Q": q,
                    "I": i,
                    "F": f,
                    "total_bits": q,
                    "integer_bits": i,
                    "fractional_bits": f,
                }
            )
    return output


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _list_get(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else ""


def _success_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset"),
            "arch": row.get("arch"),
            "sample_id": row.get("sample_id"),
            "method": row.get("method"),
            "input_epsilon": row.get("input_epsilon"),
            "formal_success": row.get("formal_success"),
            "deployment_success": row.get("deployment_success"),
            "full_success": row.get("full_success"),
            "no_saturation_status": row.get("no_saturation_status"),
            "final_status": row.get("final_status"),
            "failure_reason": row.get("failure_reason"),
            "failed_layer": row.get("failed_layer"),
            "failed_block": row.get("failed_block"),
            "failed_property": row.get("failed_property"),
        }
        for row in rows
    ]


def _mrr_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("arch"), row.get("sample_id"), row.get("method"))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (dataset, arch, sample_id, method), group in grouped.items():
        eps_values = sorted({_num(row.get("input_epsilon")) for row in group if _num(row.get("input_epsilon")) is not None})
        if len(eps_values) < 2:
            continue
        verified: list[float] = []
        failed: list[float] = []
        status_at_eps: dict[float, str] = {}
        runtime = 0.0
        for row in group:
            eps = _num(row.get("input_epsilon"))
            if eps is None:
                continue
            status = str(row.get("contract_status") or row.get("final_status") or "")
            status_at_eps[eps] = _status_priority([status_at_eps.get(eps, ""), status])
            runtime += _num(row.get("total_runtime_seconds")) or 0.0
            if row.get("contract_status") == "VERIFIED" or row.get("final_status") in {"VERIFIED", "PARTIAL_VERIFIED"}:
                verified.append(eps)
            else:
                failed.append(eps)
        largest_failed = max(failed) if failed else ""
        output.append(
            {
                "dataset": dataset,
                "arch": arch,
                "sample_id": sample_id,
                "method": method,
                "eps_values_tested": eps_values,
                "eps_verified": sorted(set(verified)),
                "eps_failed": sorted(set(failed)),
                "mrr_discrete": max(verified) if verified else "",
                "largest_failed_eps": largest_failed,
                "status_at_largest_eps": status_at_eps.get(largest_failed, "") if largest_failed != "" else "",
                "total_runtime_seconds": runtime,
            }
        )
    return output


def _implementation_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "run_name", "dataset", "arch", "sample_id", "method", "input_epsilon",
        "max_saturation_rate", "mean_saturation_rate", "mismatch_rate_vs_keras",
        "python_c_exact_match", "max_abs_logit_error", "mean_abs_logit_error",
        "no_saturation_status",
    ]
    return [{field: row.get(field) for field in fields} for row in rows]


def _ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        mode = row.get("mode") or "full_pipeline"
        output.append(
            {
                "dataset": row.get("dataset"),
                "arch": row.get("arch"),
                "sample_id": row.get("sample_id"),
                "input_epsilon": row.get("input_epsilon"),
                "mode": mode,
                "method": row.get("method"),
                "blockwise_enabled": row.get("blockwise_enabled"),
                "refinement_enabled": row.get("method") == "quality_refined",
                "formal_verification_enabled": mode != "naive_uniform_8bit_fixed",
                "status": row.get("final_status") or row.get("status"),
                "total_runtime_seconds": row.get("total_runtime_seconds"),
                "esbmc_time_seconds": row.get("total_esbmc_time_seconds"),
                "timeout_rate": row.get("timeout_rate"),
                "memout_rate": row.get("memout_rate"),
                "quantized_keras_accuracy": row.get("quantized_keras_accuracy"),
                "python_fixed_accuracy": row.get("python_fixed_accuracy"),
                "c_fixed_accuracy": row.get("c_fixed_accuracy"),
                "max_saturation_rate": row.get("max_saturation_rate"),
                "mismatch_rate_vs_keras": row.get("mismatch_rate_vs_keras"),
            }
        )
    return output


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    for old, new in {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
    }.items():
        text = text.replace(old, new)
    return text


def _latex_cell(value: Any) -> str:
    number = _num(value)
    if number is not None:
        return f"{number:.4f}".rstrip("0").rstrip(".")
    return _latex_escape(value)


def _write_latex(path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = 80) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:limit] if limit is not None else rows
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(_latex_escape(header) for header, _ in columns) + " \\\\",
        "\\hline",
    ]
    for row in selected:
        lines.append(" & ".join(_latex_cell(row.get(key)) for _, key in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_latex_tables(output_root: Path, tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    latex_dir = output_root / "latex"
    artifacts = {
        "table_quality_metrics_tex": _write_latex(
            latex_dir / "table_quality_metrics.tex",
            tables["quality"],
            [("Dataset", "dataset"), ("Arch", "arch"), ("Method", "method"), ("Keras Q", "quantized_keras_accuracy"), ("Python", "python_fixed_accuracy"), ("C", "c_fixed_accuracy")],
        ),
        "table_scalability_tex": _write_latex(
            latex_dir / "table_scalability.tex",
            tables["scalability"],
            [("Dataset", "dataset"), ("Arch", "arch"), ("Method", "method"), ("Status", "final_status"), ("ESBMC calls", "number_of_esbmc_calls"), ("ESBMC time", "total_esbmc_time_seconds")],
        ),
        "table_bitwidths_tex": _write_latex(
            latex_dir / "table_bitwidths.tex",
            tables["bitwidths"],
            [("Dataset", "dataset"), ("Arch", "arch"), ("Method", "method"), ("Layer", "layer_index"), ("Q", "Q"), ("I", "I"), ("F", "F")],
        ),
        "table_mrr_tex": _write_latex(
            latex_dir / "table_mrr.tex",
            tables["mrr"],
            [("Dataset", "dataset"), ("Arch", "arch"), ("Method", "method"), ("MRR", "mrr_discrete"), ("Verified eps", "eps_verified")],
        ),
        "table_ablation_tex": _write_latex(
            latex_dir / "table_ablation.tex",
            tables["ablation"],
            [("Dataset", "dataset"), ("Arch", "arch"), ("Mode", "mode"), ("Method", "method"), ("Status", "status"), ("C Acc.", "c_fixed_accuracy")],
        ),
        "table_implementation_gap_tex": _write_latex(
            latex_dir / "table_implementation_gap.tex",
            tables["implementation_gap"],
            [("Dataset", "dataset"), ("Arch", "arch"), ("Method", "method"), ("Sat.", "max_saturation_rate"), ("Mismatch", "mismatch_rate_vs_keras"), ("Exact", "python_c_exact_match")],
        ),
    }
    return {key: str(value) for key, value in artifacts.items()}


def _summarize_smt(input_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in input_rows:
        key = (
            row.get("run_name"),
            row.get("dataset"),
            row.get("arch"),
            row.get("method"),
            row.get("layer_index"),
            row.get("property_type"),
            row.get("mode"),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        sizes = [_num(row.get("smt_file_size_mb")) or 0.0 for row in group]
        bvmuls = [_num(row.get("bvmul_count")) or 0.0 for row in group]
        asserts = [_num(row.get("assert_count")) or 0.0 for row in group]
        depths = [_num(row.get("max_parenthesis_depth")) or 0.0 for row in group]
        times = [_num(row.get("query_time_seconds")) or 0.0 for row in group]
        statuses = [str(row.get("status", "")) for row in group]
        output.append(
            {
                "run_name": key[0],
                "dataset": key[1],
                "arch": key[2],
                "method": key[3],
                "layer_index": key[4],
                "property_type": key[5],
                "mode": key[6],
                "max_file_size_mb": max(sizes, default=0.0),
                "mean_file_size_mb": sum(sizes) / len(sizes) if sizes else 0.0,
                "max_bvmul_count": max(bvmuls, default=0.0),
                "mean_bvmul_count": sum(bvmuls) / len(bvmuls) if bvmuls else 0.0,
                "max_assert_count": max(asserts, default=0.0),
                "max_parenthesis_depth": max(depths, default=0.0),
                "max_query_time_seconds": max(times, default=0.0),
                "status": _status_priority(statuses),
            }
        )
    return output


def _write_summary(output_root: Path, rows: list[dict[str, Any]], failed_rows: list[dict[str, Any]]) -> Path:
    datasets = sorted({str(row.get("dataset")) for row in rows if row.get("dataset")})
    verified = [row for row in rows if row.get("final_status") in {"VERIFIED", "PARTIAL_VERIFIED"}]
    formal_success = [row for row in rows if _bool(row.get("formal_success")) is True]
    lines = [
        "# Article Experiment Summary",
        "",
        f"- Method rows: {len(rows)}",
        f"- Formal contract successful method rows: {len(formal_success)}",
        f"- Verified or partially verified method rows: {len(verified)}",
        f"- Failed/skipped run records: {len(failed_rows)}",
        f"- Datasets: {', '.join(datasets) or '(none)'}",
        f"- Output directory: {output_root}",
        "",
        "Scalability frontier runs are retained in the tables even when they fail, time out, or run out of memory.",
    ]
    path = output_root / "article_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def aggregate(input_root: Path, output_root: Path) -> dict[str, Any]:
    run_dirs = _discover_run_dirs(input_root)
    rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_status = _read_json(run_dir / "run_status.json")
        run_config = _read_json(run_dir / "run_config.json")
        pipeline = _first_existing_json(run_dir / "pipeline_summary.json", run_dir / "reports" / "pipeline_summary.json")
        experiment = _read_json(run_dir / "reports" / "experiment_summary.json")
        if not run_status:
            run_status = {
                "name": run_dir.name,
                "status": "success" if experiment or pipeline else "failed",
                "output_dir": str(run_dir),
            }
        if run_status.get("status") != "success":
            failed_rows.append(run_status)
        for method in METHODS:
            rows.append(
                _method_row(
                    run_dir=run_dir,
                    run_status=run_status,
                    run_config=run_config,
                    experiment=experiment,
                    pipeline=pipeline,
                    method=method,
                )
            )
        run_records.append(
            {
                "run_dir": str(run_dir),
                "run_status": run_status,
                "has_pipeline_summary": bool(pipeline),
                "has_experiment_summary": bool(experiment),
            }
        )

    quality_rows = [
        {field: row.get(field) for field in [
            "run_name", "dataset", "arch", "sample_id", "method", "input_epsilon",
            "float32_accuracy", "quantized_keras_accuracy", "python_fixed_accuracy",
            "c_fixed_accuracy", "accuracy_drop_float_to_keras_quantized",
            "accuracy_drop_keras_quantized_to_python_fixed",
            "accuracy_drop_keras_quantized_to_c_fixed", "mismatch_rate_vs_keras",
            "python_c_exact_match", "max_abs_logit_error", "mean_abs_logit_error",
            "max_saturation_rate", "mean_saturation_rate", "refinement_steps",
        ]}
        for row in rows
    ]
    bitwidth_rows = _bitwidth_rows(rows)
    success_rows = _success_rows(rows)
    scalability_rows = [{field: row.get(field) for field in ALL_FIELDS} for row in rows]
    esbmc_rows = [
        {field: row.get(field) for field in [
            "run_name", "dataset", "arch", "sample_id", "method", "input_epsilon",
            "esbmc_verified_count", "esbmc_failed_count", "esbmc_timeout_count",
            "esbmc_memout_count", "esbmc_unknown_count", "esbmc_total_count",
            "timeout_rate", "memout_rate", "unknown_rate",
        ]}
        for row in rows
    ]
    mrr_rows = _mrr_rows(rows)
    implementation_gap_rows = _implementation_gap_rows(rows)
    ablation_rows = _ablation_rows(rows)

    smt_path = output_root / "smt_complexity.csv"
    smt_rows = _read_csv(smt_path)
    smt_fields = SMT_COMPLEXITY_FIELDS
    if not smt_path.exists():
        _write_csv(smt_path, [], smt_fields)
    smt_summary_rows = _summarize_smt(smt_rows)

    _write_csv(output_root / "all_experiments.csv", rows, ALL_FIELDS)
    _write_json(
        output_root / "all_experiments.json",
        {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "num_run_dirs": len(run_dirs),
            "num_method_rows": len(rows),
            "runs": run_records,
            "rows": rows,
        },
    )
    _write_csv(output_root / "table_quality_metrics.csv", quality_rows, QUALITY_FIELDS)
    _write_csv(output_root / "table_bitwidths.csv", bitwidth_rows, BITWIDTH_FIELDS)
    _write_csv(output_root / "table_success_failure.csv", success_rows, SUCCESS_FIELDS)
    _write_csv(output_root / "table_scalability.csv", scalability_rows, ALL_FIELDS)
    _write_csv(output_root / "table_esbmc_status_counts.csv", esbmc_rows, ESBMC_FIELDS)
    _write_csv(output_root / "table_smt_complexity_summary.csv", smt_summary_rows, SMT_SUMMARY_FIELDS)
    _write_csv(output_root / "table_mrr.csv", mrr_rows, MRR_FIELDS)
    _write_csv(output_root / "table_implementation_gap.csv", implementation_gap_rows, IMPLEMENTATION_GAP_FIELDS)
    _write_csv(output_root / "table_ablation.csv", ablation_rows, ABLATION_FIELDS)
    _write_csv(output_root / "failed_runs.csv", failed_rows, FAILED_RUN_FIELDS)
    summary_path = _write_summary(output_root, rows, failed_rows)
    latex = _write_latex_tables(
        output_root,
        {
            "quality": quality_rows,
            "scalability": scalability_rows,
            "bitwidths": bitwidth_rows,
            "mrr": mrr_rows,
            "ablation": ablation_rows,
            "implementation_gap": implementation_gap_rows,
        },
    )
    return {
        "all_experiments_csv": str(output_root / "all_experiments.csv"),
        "all_experiments_json": str(output_root / "all_experiments.json"),
        "article_summary_md": str(summary_path),
        **latex,
    }


QUALITY_FIELDS = [
    "run_name", "dataset", "arch", "sample_id", "method", "input_epsilon",
    "float32_accuracy", "quantized_keras_accuracy", "python_fixed_accuracy", "c_fixed_accuracy",
    "accuracy_drop_float_to_keras_quantized", "accuracy_drop_keras_quantized_to_python_fixed",
    "accuracy_drop_keras_quantized_to_c_fixed", "mismatch_rate_vs_keras", "python_c_exact_match",
    "max_abs_logit_error", "mean_abs_logit_error", "max_saturation_rate", "mean_saturation_rate",
    "refinement_steps",
]
BITWIDTH_FIELDS = [
    "run_name", "dataset", "arch", "sample_id", "input_epsilon", "normalized_input_epsilon",
    "method", "layer_index", "Q", "I", "F", "total_bits", "integer_bits", "fractional_bits",
]
SUCCESS_FIELDS = [
    "dataset", "arch", "sample_id", "method", "input_epsilon", "formal_success",
    "deployment_success", "full_success", "no_saturation_status", "final_status", "failure_reason",
    "failed_layer", "failed_block", "failed_property",
]
ESBMC_FIELDS = [
    "run_name", "dataset", "arch", "sample_id", "method", "input_epsilon",
    "esbmc_verified_count", "esbmc_failed_count", "esbmc_timeout_count", "esbmc_memout_count",
    "esbmc_unknown_count", "esbmc_total_count", "timeout_rate", "memout_rate", "unknown_rate",
]
SMT_COMPLEXITY_FIELDS = [
    "run_name", "dataset", "arch", "method", "layer_index", "property_type", "mode", "smt_path",
    "smt_file_size_bytes", "smt_file_size_mb", "line_count", "char_count", "declare_fun_count",
    "define_fun_count", "assert_count", "bvmul_count", "bvadd_count", "bvsub_count", "bvshl_count",
    "bvlshr_count", "bvashr_count", "ite_count", "extract_count", "concat_count", "select_count",
    "store_count", "max_parenthesis_depth", "parse_balance_ok", "query_time_seconds", "status",
]
SMT_SUMMARY_FIELDS = [
    "run_name", "dataset", "arch", "method", "layer_index", "property_type", "mode",
    "max_file_size_mb", "mean_file_size_mb", "max_bvmul_count", "mean_bvmul_count",
    "max_assert_count", "max_parenthesis_depth", "max_query_time_seconds", "status",
]
MRR_FIELDS = [
    "dataset", "arch", "sample_id", "method", "eps_values_tested", "eps_verified", "eps_failed",
    "mrr_discrete", "largest_failed_eps", "status_at_largest_eps", "total_runtime_seconds",
]
IMPLEMENTATION_GAP_FIELDS = [
    "run_name", "dataset", "arch", "sample_id", "method", "input_epsilon", "max_saturation_rate",
    "mean_saturation_rate", "mismatch_rate_vs_keras", "python_c_exact_match", "max_abs_logit_error",
    "mean_abs_logit_error", "no_saturation_status",
]
ABLATION_FIELDS = [
    "dataset", "arch", "sample_id", "input_epsilon", "mode", "method", "blockwise_enabled",
    "refinement_enabled", "formal_verification_enabled", "status", "total_runtime_seconds",
    "esbmc_time_seconds", "timeout_rate", "memout_rate", "quantized_keras_accuracy",
    "python_fixed_accuracy", "c_fixed_accuracy", "max_saturation_rate", "mismatch_rate_vs_keras",
]
FAILED_RUN_FIELDS = [
    "name", "dataset", "arch", "sample_id", "eps", "input_epsilon", "status", "return_code",
    "started_at", "finished_at", "elapsed_seconds", "output_dir", "error_message",
]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifacts = aggregate(args.input_root, args.output_root)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
