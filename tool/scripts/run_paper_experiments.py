from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import re
import sys
import traceback
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthesis.pipeline import RobustnessPipelineConfig, run_robustness_pipeline
from utils.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch of paper experiment configurations.")
    parser.add_argument("--config", required=True, type=Path, help="JSON config with a top-level 'runs' list.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/paper_results"),
        help="Directory for aggregate paper results.",
    )
    parser.add_argument(
        "--run-output-root",
        type=Path,
        default=None,
        help="Directory for per-run pipeline outputs. Defaults to OUTPUT_DIR/runs.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("runs"), list):
        raise ValueError(f"Expected {path} to contain a top-level 'runs' list.")
    return payload


def _slug(value: Any) -> str:
    text = str(value).replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _iter_run_specs(batch_config: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for block_index, block in enumerate(batch_config["runs"]):
        sample_ids = block.get("sample_ids", [block.get("sample_id", 0)])
        eps_values = block.get("eps_values", [block.get("eps", 1.0)])
        for sample_id in sample_ids:
            for eps in eps_values:
                spec = dict(block)
                spec.pop("sample_ids", None)
                spec.pop("sample_id", None)
                spec.pop("eps_values", None)
                spec.pop("eps", None)
                spec["block_index"] = int(block_index)
                spec["sample_id"] = int(sample_id)
                spec["eps"] = float(eps)
                specs.append(spec)
    return specs


def _optional_threshold(raw_value: Any, default: float | None) -> float | None:
    if raw_value is None:
        return default
    value = float(raw_value)
    return None if value < 0 else value


def _run_output_dir(run_output_root: Path, spec: dict[str, Any], run_index: int) -> Path:
    name = "__".join(
        [
            f"{run_index:04d}",
            _slug(spec["dataset"]),
            _slug(spec["arch"]),
            f"id{spec['sample_id']}",
            f"eps{_slug(spec['eps'])}",
        ]
    )
    return run_output_root / name


def _pipeline_config(spec: dict[str, Any], output_dir: Path) -> RobustnessPipelineConfig:
    valid_labels = spec.get("valid_labels")
    if valid_labels is not None:
        valid_labels = tuple(int(label) for label in valid_labels)

    return RobustnessPipelineConfig(
        dataset=str(spec["dataset"]),
        arch=str(spec["arch"]),
        sample_id=int(spec["sample_id"]),
        eps=float(spec["eps"]),
        bit_lb=int(spec.get("bit_lb", 1)),
        bit_ub=int(spec.get("bit_ub", 16)),
        preimg_mode=str(spec.get("preimg_mode", "milp")),
        verify_mode=str(spec.get("verify_mode", "milp")),
        output_dir=output_dir,
        if_relax=bool(spec.get("if_relax", False)),
        target_label=(int(spec["target_label"]) if spec.get("target_label") is not None else None),
        valid_labels=valid_labels,
        compare_split=str(spec.get("compare_split", "test")),
        compare_limit=(None if int(spec.get("compare_limit", 100)) == 0 else int(spec.get("compare_limit", 100))),
        compile_c_backend=bool(spec.get("compile_c_backend", not bool(spec.get("skip_c_backend", False)))),
        compiler=str(spec.get("compiler", "gcc")),
        enable_diagnostics=bool(spec.get("enable_diagnostics", True)),
        accuracy_drop_threshold=_optional_threshold(spec.get("accuracy_drop_threshold"), 0.05),
        saturation_threshold=_optional_threshold(spec.get("saturation_threshold"), 0.01),
        mismatch_threshold=_optional_threshold(spec.get("mismatch_threshold"), 0.05),
        max_quality_refinement_steps=max(0, int(spec.get("max_quality_refinement_steps", 10))),
        no_gurobi=bool(spec.get("no_gurobi", False)),
        save_preimage_cache=bool(spec.get("save_preimage_cache", False)),
        preimage_cache_dir=Path(spec["preimage_cache_dir"]) if spec.get("preimage_cache_dir") else None,
        preimage_cache_key=spec.get("preimage_cache_key"),
        export_paper_tables=bool(spec.get("export_paper_tables", True)),
        baseline_results_json=Path(spec["baseline_results_json"]) if spec.get("baseline_results_json") else None,
    )


def _read_experiment_summary(pipeline_summary: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    artifact_path = pipeline_summary.get("artifacts", {}).get("experiment_summary")
    path = Path(artifact_path) if artifact_path else output_dir / "reports" / "experiment_summary.json"
    if not path.exists():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def _aggregate_row(result: dict[str, Any]) -> dict[str, Any]:
    run = result["run"]
    summary = result.get("experiment_summary") or {}
    formal = summary.get("formal_only", {})
    refined = summary.get("quality_refined", {})
    formal_deploy = formal.get("deployment_metrics", {})
    refined_deploy = refined.get("deployment_metrics", {})
    formal_stats = formal.get("verification_stats", {})
    return {
        "dataset": run.get("dataset"),
        "arch": run.get("arch"),
        "sample_id": run.get("sample_id"),
        "eps": run.get("eps"),
        "success": bool(result.get("success", False)),
        "formal_bits": json.dumps(formal.get("Q", [])),
        "refined_bits": json.dumps(refined.get("Q", [])),
        "formal_accuracy": formal_deploy.get("python_fixed_accuracy"),
        "refined_accuracy": refined_deploy.get("python_fixed_accuracy"),
        "max_saturation_formal": formal_deploy.get("max_saturation_rate"),
        "max_saturation_refined": refined_deploy.get("max_saturation_rate"),
        "total_time": formal_stats.get("total_time"),
        "refinement_steps": refined.get("refinement_steps"),
    }


def _write_aggregate_csv(path: Path, results: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "dataset",
        "arch",
        "sample_id",
        "eps",
        "success",
        "formal_bits",
        "refined_bits",
        "formal_accuracy",
        "refined_accuracy",
        "max_saturation_formal",
        "max_saturation_refined",
        "total_time",
        "refinement_steps",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(_aggregate_row(result))
    return path


def _write_aggregate_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    run_output_root = args.run_output_root or (output_dir / "runs")
    configure_logging(output_dir / "paper_experiments.log", getattr(logging, args.log_level))

    batch_config = _load_config(args.config)
    repo_root = Path(__file__).resolve().parent.parent
    run_specs = _iter_run_specs(batch_config)
    results: list[dict[str, Any]] = []

    for run_index, spec in enumerate(run_specs):
        output_for_run = _run_output_dir(run_output_root, spec, run_index)
        run_record = {
            "run_index": int(run_index),
            "dataset": spec.get("dataset"),
            "arch": spec.get("arch"),
            "sample_id": spec.get("sample_id"),
            "eps": spec.get("eps"),
            "output_dir": str(output_for_run),
        }
        try:
            pipeline_config = _pipeline_config(spec, output_for_run)
            pipeline_summary = run_robustness_pipeline(repo_root, pipeline_config)
            experiment_summary, experiment_summary_path = _read_experiment_summary(pipeline_summary, output_for_run)
            success = bool((experiment_summary or {}).get("quality_refined", {}).get("accepted", False))
            if experiment_summary is None:
                success = False
            results.append(
                {
                    "run": run_record,
                    "success": success,
                    "pipeline_summary_path": str(output_for_run / "reports" / "pipeline_summary.json"),
                    "experiment_summary_path": str(experiment_summary_path),
                    "experiment_summary": experiment_summary,
                }
            )
        except Exception as exc:  # noqa: BLE001 - batch runner must continue after failures.
            logging.exception("Paper experiment run failed: %s", run_record)
            results.append(
                {
                    "run": run_record,
                    "success": False,
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    "experiment_summary": None,
                }
            )

    aggregate = {
        "config": str(args.config),
        "num_runs": int(len(results)),
        "num_successful": int(sum(1 for result in results if result.get("success"))),
        "num_failed": int(sum(1 for result in results if not result.get("success"))),
        "results": results,
    }
    json_path = _write_aggregate_json(output_dir / "all_experiments.json", aggregate)
    csv_path = _write_aggregate_csv(output_dir / "all_experiments.csv", results)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "num_runs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
