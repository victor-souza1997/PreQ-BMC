from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configured paper experiments once.")
    parser.add_argument("--config", required=True, type=Path, help="JSON experiment plan.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--only", action="append", default=[], metavar="NAME_OR_PATTERN")
    parser.add_argument("--skip", action="append", default=[], metavar="NAME_OR_PATTERN")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--aggregate", action="store_true", help="Run aggregate_paper_results.py after experiments.")
    parser.add_argument("--plots", action="store_true", help="Run plot_paper_results.py after aggregation.")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--solver", choices=["cbc", "gurobi"], default="gurobi")
    parser.add_argument(
        "--discover-benchmarks",
        action="store_true",
        help="Print benchmark weights discovered under tool/benchmark and exit.",
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("runs"), list):
        raise ValueError(f"{path} must contain a top-level 'runs' list.")
    return payload


def _slug(value: Any) -> str:
    text = str(value).strip().replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _run_name(run: dict[str, Any]) -> str:
    if run.get("name"):
        return _slug(run["name"])
    return _slug(f"{run['dataset']}_{run['arch']}_sample{run['sample_id']}_eps{run['eps']}")


def _matches(patterns: list[str], run: dict[str, Any]) -> bool:
    haystacks = [
        str(run.get("name", "")),
        str(run.get("dataset", "")),
        str(run.get("arch", "")),
        f"{run.get('dataset', '')}_{run.get('arch', '')}",
    ]
    for pattern in patterns:
        lowered_pattern = pattern.lower()
        if lowered_pattern == "mnist":
            if str(run.get("dataset", "")).lower() == "mnist":
                return True
            continue
        for value in haystacks:
            lowered_value = value.lower()
            if fnmatch(lowered_value, lowered_pattern) or lowered_pattern in lowered_value:
                return True
    return False


def _iter_enabled_runs(config: dict[str, Any], only: list[str], skip: list[str]) -> list[dict[str, Any]]:
    defaults = dict(config.get("defaults", {}))
    runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(config["runs"]):
        run = {**defaults, **raw_run}
        run["name"] = _run_name(run)
        run["run_index"] = int(index)
        if not bool(run.get("enabled", True)):
            run["_skip_reason"] = "disabled"
        elif only and not _matches(only, run):
            run["_skip_reason"] = "not matched by --only"
        elif skip and _matches(skip, run):
            run["_skip_reason"] = "matched by --skip"
        runs.append(run)
    return runs


def _discover_cli_flags() -> set[str]:
    script = _tool_root() / "scripts" / "run_robustness_pipeline.py"
    source = script.read_text(encoding="utf-8")
    return set(re.findall(r"""["'](--[A-Za-z0-9_-]+)["']""", source))


def _add_if_supported(command: list[str], supported: set[str], flag: str, value: Any | None = None) -> None:
    if flag not in supported:
        return
    command.append(flag)
    if value is not None:
        command.append(str(value))


def _optional_threshold(value: Any) -> str:
    return "-1" if value is None else str(value)


def _build_pipeline_command(
    *,
    python_executable: str,
    run: dict[str, Any],
    output_dir: Path,
    supported_flags: set[str],
    default_solver: str = "cbc",
) -> list[str]:
    script = _tool_root() / "scripts" / "run_robustness_pipeline.py"
    command = [
        python_executable,
        str(script),
        "--dataset",
        str(run["dataset"]),
        "--arch",
        str(run["arch"]),
        "--sample-id",
        str(run.get("sample_id", 0)),
        "--eps",
        str(run.get("eps", 1.0)),
        "--bit-lb",
        str(run.get("bit_lb", 2)),
        "--bit-ub",
        str(run.get("bit_ub", 24)),
        "--preimage-mode",
        str(run.get("preimg_mode", "milp")),
        "--verify-mode",
        str(run.get("verify_mode", "esbmc")),
        "--solver",
        str(run.get("solver", default_solver)),
        "--output-dir",
        str(output_dir),
        "--compare-split",
        str(run.get("compare_split", "test")),
        "--compare-limit",
        str(run.get("compare_limit", 100)),
        "--accuracy-drop-threshold",
        _optional_threshold(run.get("accuracy_drop_threshold", 0.05)),
        "--saturation-threshold",
        _optional_threshold(run.get("saturation_threshold", 0.01)),
        "--mismatch-threshold",
        _optional_threshold(run.get("mismatch_threshold", 0.05)),
        "--max-quality-refinement-steps",
        str(run.get("max_quality_refinement_steps", 10)),
    ]

    if not bool(run.get("compile_c_backend", True)):
        _add_if_supported(command, supported_flags, "--skip-c-backend")
    if bool(run.get("enable_diagnostics", True)):
        _add_if_supported(command, supported_flags, "--enable-diagnostics")
    else:
        _add_if_supported(command, supported_flags, "--disable-diagnostics")
    if bool(run.get("export_paper_tables", True)):
        _add_if_supported(command, supported_flags, "--export-paper-tables")
    else:
        _add_if_supported(command, supported_flags, "--no-export-paper-tables")

    if "formal_saturation_check" in run:
        _add_if_supported(
            command,
            supported_flags,
            "--formal-saturation-check" if bool(run["formal_saturation_check"]) else "--no-formal-saturation-check",
        )
    if "empirical_saturation_check" in run:
        _add_if_supported(
            command,
            supported_flags,
            "--empirical-saturation-check"
            if bool(run["empirical_saturation_check"])
            else "--no-empirical-saturation-check",
        )

    if "incremental_bmc" in run:
        _add_if_supported(
            command,
            supported_flags,
            "--incremental-bmc" if bool(run["incremental_bmc"]) else "--no-incremental-bmc",
        )
    if "interval_analysis" in run:
        _add_if_supported(
            command,
            supported_flags,
            "--interval-analysis" if bool(run["interval_analysis"]) else "--no-interval-analysis",
        )

    for flag, key in (
        ("--compiler", "compiler"),
        ("--target-label", "target_label"),
        ("--valid-labels", "valid_labels"),
        ("--baseline-results-json", "baseline_results_json"),
        ("--preimage-cache-dir", "preimage_cache_dir"),
        ("--preimage-cache-key", "preimage_cache_key"),
        ("--esbmc-layer-block-size", "esbmc_layer_block_size"),
    ):
        if run.get(key) is not None:
            value = run[key]
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            _add_if_supported(command, supported_flags, flag, value)

    for flag, key in (
        ("--no-gurobi", "no_gurobi"),
        ("--save-preimage-cache", "save_preimage_cache"),
    ):
        if bool(run.get(key, False)):
            _add_if_supported(command, supported_flags, flag)

    if run.get("if_relax") is not None:
        _add_if_supported(command, supported_flags, "--if-relax", int(bool(run["if_relax"])))

    return command


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _status_payload(
    run: dict[str, Any],
    output_dir: Path,
    *,
    status: str,
    return_code: int | None,
    started_at: str | None,
    finished_at: str | None,
    elapsed_seconds: float | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    reports_dir = output_dir / "reports"
    return {
        "name": run["name"],
        "dataset": run.get("dataset"),
        "arch": run.get("arch"),
        "sample_id": run.get("sample_id"),
        "eps": run.get("eps"),
        "status": status,
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "output_dir": str(output_dir),
        "experiment_summary_path": str(reports_dir / "experiment_summary.json"),
        "pipeline_summary_path": str(reports_dir / "pipeline_summary.json"),
        "error_message": error_message,
    }


def _run_one(command: list[str], run: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", run)
    (output_dir / "command.txt").write_text(_command_text(command) + "\n", encoding="utf-8")

    started_at = _now()
    start = time.monotonic()
    timeout_seconds = run.get("timeout_seconds")
    stdout_path = output_dir / "run_stdout.log"
    stderr_path = output_dir / "run_stderr.log"
    status = "failed"
    return_code: int | None = None
    error_message: str | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                cwd=_repo_root(),
                timeout=float(timeout_seconds) if timeout_seconds is not None else None,
                check=False,
                text=True,
            )
        return_code = int(completed.returncode)
        status = "success" if completed.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        return_code = -1
        error_message = f"Timed out after {timeout_seconds} seconds"
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(f"\nTIMEOUT: {error_message}\n{exc}\n")
    except Exception as exc:  # noqa: BLE001 - experiment runner must persist status.
        status = "failed"
        return_code = -1
        error_message = f"{type(exc).__name__}: {exc}"
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(f"\nERROR: {error_message}\n")

    finished_at = _now()
    elapsed = time.monotonic() - start
    if status != "success" and error_message is None and stderr_path.exists():
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        error_message = stderr_text[-4000:] if stderr_text else None

    payload = _status_payload(
        run,
        output_dir,
        status=status,
        return_code=return_code,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
        error_message=error_message,
    )
    _write_json(output_dir / "run_status.json", payload)
    return payload


def _write_skipped_status(run: dict[str, Any], output_dir: Path, reason: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", run)
    payload = _status_payload(
        run,
        output_dir,
        status="skipped",
        return_code=None,
        started_at=None,
        finished_at=_now(),
        elapsed_seconds=0.0,
        error_message=reason,
    )
    _write_json(output_dir / "run_status.json", payload)
    return payload


def _arch_from_stem(base_name: str, stem: str) -> str:
    if stem == base_name:
        return "default"
    prefix = f"{base_name}_"
    suffix = stem[len(prefix) :] if stem.startswith(prefix) else stem
    match = re.fullmatch(r"(\d+)x(\d+)", suffix)
    if match:
        width = int(match.group(1))
        depth = int(match.group(2))
        return f"{depth}blk_" + "_".join(str(width) for _ in range(depth))
    return suffix


def discover_benchmarks() -> list[dict[str, Any]]:
    benchmark_root = _tool_root() / "benchmark"
    discovered: list[dict[str, Any]] = []
    for weight_path in sorted(benchmark_root.glob("*/*_weight.h5")):
        base_name = weight_path.parent.name
        stem = weight_path.name.removesuffix("_weight.h5")
        arch = _arch_from_stem(base_name, stem)
        if base_name == "mnist" and arch not in {"1blk_10", "1blk_25", "2blk_25_25"}:
            continue
        dataset = stem if base_name in {"iris", "seeds"} and stem != base_name else base_name
        discovered.append(
            {
                "dataset": dataset,
                "arch": arch,
                "weight_path": str(weight_path),
            }
        )
    return discovered


def _run_followup(command: list[str], dry_run: bool) -> None:
    print(_command_text(command))
    if not dry_run:
        subprocess.run(command, cwd=_repo_root(), check=False)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.discover_benchmarks:
        print(json.dumps(discover_benchmarks(), indent=2))
        return

    config = _load_json(args.config)
    metadata = config.get("metadata", {})
    output_root = args.output_root or Path(metadata.get("output_root", "output/paper_runs"))
    aggregate_output_dir = Path(metadata.get("aggregate_output_dir", "output/paper_results"))
    supported_flags = _discover_cli_flags()
    runs = _iter_enabled_runs(config, args.only, args.skip)

    executed = 0
    failed = False
    for run in runs:
        output_dir = output_root / run["name"]
        if run.get("_skip_reason"):
            if not args.dry_run:
                _write_skipped_status(run, output_dir, str(run["_skip_reason"]))
            continue
        if args.max_runs is not None and executed >= args.max_runs:
            if not args.dry_run:
                _write_skipped_status(run, output_dir, "skipped by --max-runs")
            continue

        command = _build_pipeline_command(
            python_executable=args.python_executable,
            run=run,
            output_dir=output_dir,
            supported_flags=supported_flags,
            default_solver=args.solver,
        )
        print(_command_text(command))
        executed += 1
        if args.dry_run:
            continue

        status = _run_one(command, run, output_dir)
        if status["status"] != "success":
            failed = True
            if not args.continue_on_error:
                break

    if args.aggregate:
        aggregate_cmd = [
            args.python_executable,
            str(_tool_root() / "scripts" / "aggregate_paper_results.py"),
            "--runs-root",
            str(output_root),
            "--output-dir",
            str(aggregate_output_dir),
        ]
        _run_followup(aggregate_cmd, args.dry_run)

    if args.plots:
        plots_cmd = [
            args.python_executable,
            str(_tool_root() / "scripts" / "plot_paper_results.py"),
            "--input",
            str(aggregate_output_dir / "all_experiments.csv"),
            "--output-dir",
            str(aggregate_output_dir / "figures"),
        ]
        _run_followup(plots_cmd, args.dry_run)

    if failed and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
