from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.esbmc_install import resolve_esbmc_executable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run article experiments for deployment-aware QNN verification.")
    parser.add_argument("--config", type=Path, default=Path("experiments/article_experiments.json"))
    parser.add_argument("--only", action="append", default=[], metavar="NAME_OR_PATTERN")
    parser.add_argument("--skip", action="append", default=[], metavar="NAME_OR_PATTERN")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("output/article_runs"))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--esbmc-profile", default="paper-fast")
    parser.add_argument("--esbmc-timeout-seconds", type=int, default=900)
    parser.add_argument("--esbmc-memlimit", default="6g")
    parser.add_argument("--esbmc-layer-block-size", type=int, default=10)
    parser.add_argument("--esbmc-jobs", type=int, default=1)
    parser.add_argument("--solver", choices=["cbc", "gurobi"], default="cbc")
    parser.add_argument("--gurobi-threads", type=int, default=4)
    parser.add_argument(
        "--unsound-contract-tolerance",
        "--unsound_contract_tolerance",
        dest="unsound_contract_tolerance",
        action="store_true",
        default=None,
        help="Force legacy non-zero hidden-contract tolerance for all runs.",
    )
    parser.add_argument(
        "--no-unsound-contract-tolerance",
        "--no_unsound_contract_tolerance",
        dest="unsound_contract_tolerance",
        action="store_false",
        help="Force strict zero hidden-contract tolerance for all runs.",
    )
    parser.add_argument(
        "--enforce-contract-chaining",
        "--enforce_contract_chaining",
        dest="enforce_contract_chaining",
        action="store_true",
        default=None,
        help="Force assume-guarantee chaining enforcement for all runs.",
    )
    parser.add_argument(
        "--no-enforce-contract-chaining",
        "--no_enforce_contract_chaining",
        dest="enforce_contract_chaining",
        action="store_false",
        help="Diagnostic only: accept runs even when chaining_ok is false.",
    )
    parser.add_argument(
        "--propagate-contract-tolerance",
        "--propagate_contract_tolerance",
        dest="propagate_contract_tolerance",
        action="store_true",
        default=None,
        help="Force sound propagation of widened hidden contracts for all runs.",
    )
    parser.add_argument(
        "--no-propagate-contract-tolerance",
        "--no_propagate_contract_tolerance",
        dest="propagate_contract_tolerance",
        action="store_false",
        help="Do not propagate widened hidden contracts for all runs.",
    )
    parser.add_argument("--esbmc-generate-smt-formula", action="store_true")
    parser.add_argument("--mrr-mode", choices=["none", "discrete", "binary"], default=None)
    parser.add_argument("--mrr-eps-values", default=None)
    parser.add_argument("--mrr-binary-low", type=float, default=None)
    parser.add_argument("--mrr-binary-high", type=float, default=None)
    parser.add_argument("--mrr-binary-iters", type=int, default=8)
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_config(path: Path) -> dict[str, Any]:
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
    return _slug(
        f"{run.get('dataset')}_{run.get('arch')}_sample{run.get('sample_id', 0)}_eps{run.get('input_epsilon', run.get('eps', 1.0))}"
    )


def _matches(patterns: list[str], run: dict[str, Any]) -> bool:
    haystacks = [
        str(run.get("name", "")),
        str(run.get("dataset", "")),
        str(run.get("arch", "")),
        f"{run.get('dataset', '')}_{run.get('arch', '')}",
        str(run.get("mode", "")),
    ]
    for pattern in patterns:
        lowered = pattern.lower()
        for value in haystacks:
            candidate = value.lower()
            if fnmatch(candidate, lowered) or lowered in candidate:
                return True
    return False


def _parse_eps_values(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _binary_probe_values(low: float | None, high: float | None, iters: int) -> list[float]:
    if low is None or high is None:
        return []
    values = {float(low), float(high)}
    left = float(low)
    right = float(high)
    for _ in range(max(0, int(iters))):
        mid = (left + right) / 2.0
        values.add(mid)
        right = mid
    return sorted(values)


def _clean_margin(logits: Any, predicted_class: int) -> float:
    import numpy as np

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return float("inf")
    other_logits = np.delete(values, int(predicted_class))
    return float(values[int(predicted_class)] - np.max(other_logits))


def _margin_records_for_benchmark(dataset_name: str, arch: str) -> list[dict[str, Any]]:
    import numpy as np

    from datasets.loaders import load_dataset
    from models.loading import (
        build_and_load_deep_model,
        infer_dense_architecture_from_h5,
        normalize_dataset_selection,
        parse_architecture,
        resolve_weight_path,
    )

    selection = normalize_dataset_selection(dataset_name)
    dataset = load_dataset(selection.base_name)
    weights_path = resolve_weight_path(_repo_root(), dataset_name, arch)
    inferred_arch = infer_dense_architecture_from_h5(weights_path)
    input_dim = dataset.input_dim
    num_classes = dataset.num_classes
    layer_units = parse_architecture(arch, num_classes)
    if inferred_arch:
        input_dim = inferred_arch[0]
        layer_units = inferred_arch[1:]
    model = build_and_load_deep_model(
        input_dim=input_dim,
        layer_units=layer_units,
        weights_path=weights_path,
        input_scale=dataset.input_scale,
    )
    logits = np.asarray(model(np.asarray(dataset.x_test, dtype=np.float32), training=False).numpy(), dtype=np.float64)
    predicted = np.argmax(logits, axis=1).astype(int)
    labels = np.asarray(dataset.y_test).reshape(-1).astype(int)
    records: list[dict[str, Any]] = []
    for sample_id, (prediction, label) in enumerate(zip(predicted, labels, strict=True)):
        margin = _clean_margin(logits[sample_id], int(prediction))
        records.append(
            {
                "sample_id": int(sample_id),
                "predicted_label": int(prediction),
                "sample_label": int(label),
                "clean_margin": margin,
                "correctly_classified": bool(prediction == label),
            }
        )
    return records


def _window_around(values: list[dict[str, Any]], center: int, count: int) -> list[dict[str, Any]]:
    if not values or count <= 0:
        return []
    left = max(0, int(center) - count // 2)
    right = min(len(values), left + count)
    left = max(0, right - count)
    return values[left:right]


def _select_stratified_by_margin(
    *,
    base: dict[str, Any],
    margin_cache: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    dataset_name = str(base["dataset"])
    arch = str(base["arch"])
    cache_key = (dataset_name, arch)
    if cache_key not in margin_cache:
        margin_cache[cache_key] = _margin_records_for_benchmark(dataset_name, arch)

    correct_only = bool(base.get("sample_selection_correct_only", True))
    candidates = [
        record for record in margin_cache[cache_key]
        if (record.get("correctly_classified") or not correct_only)
    ]
    if not candidates:
        raise ValueError(f"No candidate samples found for stratified_by_margin selection on {dataset_name}/{arch}.")

    per_stratum = int(base.get("samples_per_stratum", base.get("sample_selection_per_stratum", 1)))
    per_stratum = max(1, per_stratum)
    sorted_records = sorted(candidates, key=lambda item: (float(item["clean_margin"]), int(item["sample_id"])))
    n = len(sorted_records)
    selections: list[tuple[str, dict[str, Any]]] = []
    selections.extend(("low", record) for record in sorted_records[:per_stratum])
    selections.extend(("median", record) for record in _window_around(sorted_records, n // 2, per_stratum))
    selections.extend(("high", record) for record in sorted_records[-per_stratum:])

    selected_ids: list[int] = []
    selected_meta: dict[int, dict[str, Any]] = {}
    ranks = {int(record["sample_id"]): index for index, record in enumerate(sorted_records)}
    for stratum, record in selections:
        sample_id = int(record["sample_id"])
        if sample_id in selected_meta:
            continue
        rank = ranks[sample_id]
        quantile = float(rank / (n - 1)) if n > 1 else 0.0
        selected_ids.append(sample_id)
        selected_meta[sample_id] = {
            **record,
            "sample_selection": "stratified_by_margin",
            "sample_selection_stratum": stratum,
            "sample_selection_rank": int(rank),
            "sample_selection_quantile": quantile,
            "sample_selection_candidates": int(n),
            "samples_per_stratum": int(per_stratum),
        }
    return selected_ids, selected_meta


def _expand_runs(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    defaults = dict(config.get("defaults", {}))
    runs: list[dict[str, Any]] = []
    mrr_values = _parse_eps_values(args.mrr_eps_values)
    binary_values = _binary_probe_values(args.mrr_binary_low, args.mrr_binary_high, args.mrr_binary_iters)
    margin_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, raw in enumerate(config["runs"]):
        base = {**defaults, **raw}
        if args.unsound_contract_tolerance is not None:
            base["unsound_contract_tolerance"] = bool(args.unsound_contract_tolerance)
        if args.propagate_contract_tolerance is not None:
            base["propagate_contract_tolerance"] = bool(args.propagate_contract_tolerance)
            if args.propagate_contract_tolerance and args.enforce_contract_chaining is None:
                base["enforce_contract_chaining"] = True
        if args.enforce_contract_chaining is not None:
            base["enforce_contract_chaining"] = bool(args.enforce_contract_chaining)
        sample_ids = base.pop("sample_ids", None)
        sample_metadata: dict[int, dict[str, Any]] = {}
        if sample_ids is None:
            sample_selection = base.get("sample_selection")
            if sample_selection == "stratified_by_margin":
                sample_ids, sample_metadata = _select_stratified_by_margin(base=base, margin_cache=margin_cache)
            elif sample_selection in (None, "", "explicit"):
                sample_ids = [base.get("sample_id", 0)]
            else:
                raise ValueError(f"Unsupported sample_selection={sample_selection!r}.")
        eps_sweep = base.pop("eps_sweep", None)
        if mrr_values and bool(base.get("mrr_enabled", False)):
            eps_sweep = mrr_values
        elif args.mrr_mode == "binary" and binary_values and bool(base.get("mrr_enabled", False)):
            eps_sweep = binary_values
        eps_values = list(eps_sweep) if eps_sweep is not None else [base.get("input_epsilon", base.get("eps", 1.0))]

        for sample_id in sample_ids:
            for eps_value in eps_values:
                run = dict(base)
                run["sample_id"] = int(sample_id)
                if int(sample_id) in sample_metadata:
                    run.update(sample_metadata[int(sample_id)])
                run["eps"] = float(eps_value)
                run["input_epsilon"] = float(eps_value)
                run["perturbation_radius"] = float(eps_value)
                run["run_index"] = int(index)
                expanded_name = _run_name(run)
                if len(sample_ids) > 1 and f"sample{sample_id}".lower() not in expanded_name.lower():
                    expanded_name = _slug(f"{expanded_name}_sample{sample_id}")
                if eps_sweep is not None:
                    if f"eps{_slug(eps_value)}".lower() not in expanded_name.lower():
                        expanded_name = _slug(f"{expanded_name}_eps{_slug(eps_value)}")
                    run["mrr_mode"] = args.mrr_mode or "discrete"
                    run["eps_values_tested"] = [float(value) for value in eps_values]
                run["name"] = expanded_name
                if not bool(run.get("enabled", True)) and not args.include_disabled:
                    run["_skip_reason"] = "disabled"
                elif args.only and not _matches(args.only, run):
                    run["_skip_reason"] = "not matched by --only"
                elif args.skip and _matches(args.skip, run):
                    run["_skip_reason"] = "matched by --skip"
                runs.append(run)
    return runs


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _run_capture(command: list[str], timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=_repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - metadata capture is best effort.
        return f"unavailable: {type(exc).__name__}: {exc}"
    return completed.stdout.strip()


def _runtime_metadata(args: argparse.Namespace, run: dict[str, Any], command: list[str]) -> dict[str, Any]:
    esbmc_executable = resolve_esbmc_executable() or "esbmc"
    return {
        "git_commit": _run_capture(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_capture(["git", "branch", "--show-current"]),
        "command": _command_text(command),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": args.python_executable,
        "esbmc_executable": esbmc_executable,
        "esbmc_version": _run_capture([esbmc_executable, "--version"]),
        "solver_profile": run.get("esbmc_profile", args.esbmc_profile),
        "timeout": int(run.get("esbmc_timeout_seconds", args.esbmc_timeout_seconds)),
        "memlimit": str(run.get("esbmc_memlimit", args.esbmc_memlimit)),
        "block_size": int(run.get("esbmc_layer_block_size", args.esbmc_layer_block_size)),
        "solver": str(run.get("solver", args.solver)),
        "gurobi_threads": int(run.get("gurobi_threads", args.gurobi_threads)),
        "esbmc_jobs": int(run.get("esbmc_jobs", args.esbmc_jobs)),
        "blockwise_fail_fast": bool(run.get("blockwise_fail_fast", True)),
        "blockwise_run_all_blocks_on_failure": bool(run.get("blockwise_run_all_blocks_on_failure", False)),
        "no_saturation_continue_on_unknown": bool(run.get("no_saturation_continue_on_unknown", False)),
        "unsound_contract_tolerance": bool(run.get("unsound_contract_tolerance", False)),
        "propagate_contract_tolerance": bool(run.get("propagate_contract_tolerance", False)),
        "enforce_contract_chaining": bool(run.get("enforce_contract_chaining", True)),
        "dataset": run.get("dataset"),
        "architecture": run.get("arch"),
        "sample_id": run.get("sample_id"),
        "sample_selection": run.get("sample_selection"),
        "sample_selection_stratum": run.get("sample_selection_stratum"),
        "sample_selection_rank": run.get("sample_selection_rank"),
        "sample_selection_quantile": run.get("sample_selection_quantile"),
        "clean_margin": run.get("clean_margin"),
        "predicted_label": run.get("predicted_label"),
        "sample_label": run.get("sample_label"),
        "input_epsilon": run.get("input_epsilon", run.get("eps")),
        "normalized_input_epsilon": _normalized_eps(run),
        "random_seed": run.get("random_seed"),
    }


def _normalized_eps(run: dict[str, Any]) -> float:
    dataset = str(run.get("dataset", ""))
    scale = 255.0 if dataset.startswith(("mnist", "fashion-mnist")) else 1.0
    return float(run.get("input_epsilon", run.get("eps", 0.0))) / scale


def _status_payload(
    run: dict[str, Any],
    output_dir: Path,
    *,
    status: str,
    return_code: int | None,
    started_at: str | None,
    finished_at: str | None,
    elapsed_seconds: float | None,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "name": run.get("name") or output_dir.name,
        "dataset": run.get("dataset"),
        "arch": run.get("arch"),
        "sample_id": run.get("sample_id"),
        "sample_selection": run.get("sample_selection"),
        "sample_selection_stratum": run.get("sample_selection_stratum"),
        "sample_selection_rank": run.get("sample_selection_rank"),
        "sample_selection_quantile": run.get("sample_selection_quantile"),
        "clean_margin": run.get("clean_margin"),
        "predicted_label": run.get("predicted_label"),
        "sample_label": run.get("sample_label"),
        "eps": run.get("eps"),
        "input_epsilon": run.get("input_epsilon", run.get("eps")),
        "normalized_input_epsilon": _normalized_eps(run),
        "status": status,
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "output_dir": str(output_dir),
        "pipeline_summary_path": str(output_dir / "reports" / "pipeline_summary.json"),
        "error_message": error_message,
        "metadata": metadata or {},
    }


def _discover_cli_flags() -> set[str]:
    script = _tool_root() / "scripts" / "run_robustness_pipeline.py"
    source = script.read_text(encoding="utf-8")
    return set(re.findall(r"""["'](--[A-Za-z0-9_-]+)["']""", source))


def _add_flag(command: list[str], supported: set[str], flag: str, value: Any | None = None) -> None:
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
    args: argparse.Namespace,
    supported_flags: set[str],
) -> list[str]:
    script = _tool_root() / "scripts" / "run_robustness_pipeline.py"
    max_refinement = run.get("max_quality_refinement_steps", 10)
    if run.get("quality_refinement") is False:
        max_refinement = 0
    block_size = run.get("esbmc_layer_block_size", args.esbmc_layer_block_size)
    if run.get("mode") in {"full_layer", "full_layer_verification", "monolithic"}:
        block_size = 0
    command = [
        python_executable,
        str(script),
        "--dataset",
        str(run["dataset"]),
        "--arch",
        str(run["arch"]),
        "--sample-id",
        str(run.get("sample_id", 0)),
        "--input-epsilon",
        str(run.get("input_epsilon", run.get("eps", 1.0))),
        "--bit-lb",
        str(run.get("bit_lb", 3)),
        "--bit-ub",
        str(run.get("bit_ub", 16)),
        "--preimage-mode",
        str(run.get("preimg_mode", "milp")),
        "--verify-mode",
        str(run.get("verify_mode", "esbmc")),
        "--solver",
        str(run.get("solver", args.solver)),
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
        str(max_refinement),
        "--esbmc-profile",
        str(run.get("esbmc_profile", args.esbmc_profile)),
        "--esbmc-timeout-seconds",
        str(run.get("esbmc_timeout_seconds", args.esbmc_timeout_seconds)),
        "--esbmc-memlimit",
        str(run.get("esbmc_memlimit", args.esbmc_memlimit)),
        "--esbmc-layer-block-size",
        str(block_size),
        "--esbmc-jobs",
        str(run.get("esbmc_jobs", args.esbmc_jobs)),
        "--gurobi-threads",
        str(run.get("gurobi_threads", args.gurobi_threads)),
    ]

    if not bool(run.get("compile_c_backend", True)):
        _add_flag(command, supported_flags, "--skip-c-backend")
    if bool(run.get("enable_diagnostics", True)):
        _add_flag(command, supported_flags, "--enable-diagnostics")
    else:
        _add_flag(command, supported_flags, "--disable-diagnostics")
    if bool(run.get("export_paper_tables", True)):
        _add_flag(command, supported_flags, "--export-paper-tables")
    else:
        _add_flag(command, supported_flags, "--no-export-paper-tables")
    if bool(run.get("blockwise_fail_fast", True)):
        _add_flag(command, supported_flags, "--blockwise-fail-fast")
    else:
        _add_flag(command, supported_flags, "--no-blockwise-fail-fast")
    if bool(run.get("blockwise_run_all_blocks_on_failure", False)):
        _add_flag(command, supported_flags, "--blockwise-run-all-blocks-on-failure")
    else:
        _add_flag(command, supported_flags, "--no-blockwise-run-all-blocks-on-failure")
    if bool(run.get("no_saturation_continue_on_unknown", False)):
        _add_flag(command, supported_flags, "--no-saturation-continue-on-unknown")
    else:
        _add_flag(command, supported_flags, "--no-no-saturation-continue-on-unknown")
    if bool(run.get("unsound_contract_tolerance", False)):
        _add_flag(command, supported_flags, "--unsound-contract-tolerance")
    if bool(run.get("propagate_contract_tolerance", False)):
        _add_flag(command, supported_flags, "--propagate-contract-tolerance")
    if not bool(run.get("enforce_contract_chaining", True)):
        _add_flag(command, supported_flags, "--no-enforce-contract-chaining")

    formal_no_saturation = bool(run.get("formal_no_saturation", run.get("formal_saturation_check", False)))
    _add_flag(
        command,
        supported_flags,
        "--formal-no-saturation" if formal_no_saturation else "--no-formal-no-saturation",
    )
    require_no_saturation = bool(run.get("require_formal_no_saturation", False))
    _add_flag(
        command,
        supported_flags,
        "--require-formal-no-saturation" if require_no_saturation else "--no-require-formal-no-saturation",
    )
    empirical_saturation = bool(run.get("empirical_saturation_check", True))
    _add_flag(
        command,
        supported_flags,
        "--empirical-saturation-check" if empirical_saturation else "--no-empirical-saturation-check",
    )

    for flag, key in (
        ("--target-label", "target_label"),
        ("--valid-labels", "valid_labels"),
        ("--compiler", "compiler"),
        ("--baseline-results-json", "baseline_results_json"),
        ("--preimage-cache-dir", "preimage_cache_dir"),
        ("--preimage-cache-key", "preimage_cache_key"),
    ):
        value = run.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        _add_flag(command, supported_flags, flag, value)

    for flag, key in (
        ("--no-gurobi", "no_gurobi"),
        ("--save-preimage-cache", "save_preimage_cache"),
    ):
        if bool(run.get(key, False)):
            _add_flag(command, supported_flags, flag)

    if run.get("if_relax") is not None:
        _add_flag(command, supported_flags, "--if-relax", int(bool(run["if_relax"])))
    return command


def _copy_pipeline_summary(output_dir: Path) -> None:
    source = output_dir / "reports" / "pipeline_summary.json"
    target = output_dir / "pipeline_summary.json"
    if source.exists():
        shutil.copy2(source, target)


def _run_smt_formula_generation(output_dir: Path, timeout_seconds: int, memlimit: str) -> None:
    for c_file in sorted((output_dir / "layers").rglob("*.c")):
        smt_file = c_file.with_suffix(".smt2")
        if smt_file.exists():
            continue
        command = [
            "esbmc",
            str(c_file),
            "--function",
            "main",
            "--smt-formula-only",
            "--bitwuzla",
            "--bv",
            "--timeout",
            str(timeout_seconds),
            "--memlimit",
            str(memlimit),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=_repo_root(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds + 120,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - formula export is optional.
            (output_dir / "smt_generation_errors.log").open("a", encoding="utf-8").write(f"{c_file}: {exc}\n")
            continue
        if completed.stdout:
            smt_file.write_text(completed.stdout, encoding="utf-8")
        if completed.stderr:
            smt_file.with_suffix(".smt2.stderr.log").write_text(completed.stderr, encoding="utf-8")


def _run_one(command: list[str], run: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", run)
    (output_dir / "command.txt").write_text(_command_text(command) + "\n", encoding="utf-8")
    metadata = _runtime_metadata(args, run, command)
    _write_json(output_dir / "reproducibility.json", metadata)

    started_at = _now()
    start = time.monotonic()
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    status = "failed"
    return_code: int | None = None
    error_message: str | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=_repo_root(),
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=float(run["runner_timeout_seconds"]) if run.get("runner_timeout_seconds") else None,
                check=False,
            )
        return_code = int(completed.returncode)
        status = "success" if return_code == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        return_code = -1
        error_message = f"Runner timed out: {exc}"
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(f"\nTIMEOUT: {error_message}\n")
    except Exception as exc:  # noqa: BLE001 - experiment runners must persist failure state.
        status = "failed"
        return_code = -1
        error_message = f"{type(exc).__name__}: {exc}"
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(f"\nERROR: {error_message}\n")

    if error_message is None and status != "success" and stderr_path.exists():
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        error_message = stderr_text[-4000:] if stderr_text else None

    if args.esbmc_generate_smt_formula:
        _run_smt_formula_generation(
            output_dir,
            timeout_seconds=int(run.get("esbmc_timeout_seconds", args.esbmc_timeout_seconds)),
            memlimit=str(run.get("esbmc_memlimit", args.esbmc_memlimit)),
        )
    _copy_pipeline_summary(output_dir)

    payload = _status_payload(
        run,
        output_dir,
        status=status,
        return_code=return_code,
        started_at=started_at,
        finished_at=_now(),
        elapsed_seconds=time.monotonic() - start,
        metadata=metadata,
        error_message=error_message,
    )
    _write_json(output_dir / "run_status.json", payload)
    return payload


def _write_skipped(run: dict[str, Any], output_dir: Path, reason: str) -> dict[str, Any]:
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


def _resume_success(output_dir: Path) -> bool:
    status_path = output_dir / "run_status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return status.get("status") == "success"


def _has_existing_output(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def _followup(command: list[str], dry_run: bool) -> None:
    print(_command_text(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=_repo_root(), check=False)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    metadata = config.get("metadata", {})
    output_root = args.output_root or Path(metadata.get("output_root", "output/article_runs"))
    aggregate_output_root = Path(metadata.get("aggregate_output_root", "output/article_results"))
    supported_flags = _discover_cli_flags()
    runs = _expand_runs(config, args)

    executed = 0
    failed = False
    for run in runs:
        output_dir = output_root / str(run["name"])
        if run.get("_skip_reason"):
            if not args.dry_run:
                _write_skipped(run, output_dir, str(run["_skip_reason"]))
            continue
        if args.max_runs is not None and executed >= args.max_runs:
            if not args.dry_run:
                _write_skipped(run, output_dir, "skipped by --max-runs")
            continue
        if args.resume and not args.force and _resume_success(output_dir):
            print(f"resume-skip {run['name']}", flush=True)
            continue
        if not args.dry_run and not args.force and not args.resume and _has_existing_output(output_dir):
            message = f"existing-output-skip {run['name']} at {output_dir}; use --force to overwrite or --resume to reuse"
            print(message, flush=True)
            failed = True
            if not args.continue_on_error:
                break
            continue

        command = _build_pipeline_command(
            python_executable=args.python_executable,
            run=run,
            output_dir=output_dir,
            args=args,
            supported_flags=supported_flags,
        )
        print(_command_text(command), flush=True)
        executed += 1
        if args.dry_run:
            continue
        status = _run_one(command, run, output_dir, args)
        allowed_failure = bool(run.get("allowed_failure", run.get("scalability_frontier", False)))
        if status["status"] != "success" and not allowed_failure:
            failed = True
            if not args.continue_on_error:
                break

    if args.aggregate:
        aggregate_cmd = [
            args.python_executable,
            str(_tool_root() / "scripts" / "aggregate_article_results.py"),
            "--input-root",
            str(output_root),
            "--output-root",
            str(aggregate_output_root),
        ]
        _followup(aggregate_cmd, args.dry_run)

    if args.plots:
        plots_cmd = [
            args.python_executable,
            str(_tool_root() / "scripts" / "plot_article_results.py"),
            "--input-root",
            str(aggregate_output_root),
            "--output-root",
            str(aggregate_output_root / "plots"),
        ]
        _followup(plots_cmd, args.dry_run)

    if failed and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
