from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CASES: dict[str, dict[str, Any]] = {
    "iris": {
        "dataset": "iris",
        "arch": "1blk_10",
        "sample_id": 25,
        "eps": 0.05,
    },
    "seeds": {
        "dataset": "seeds",
        "arch": "1blk_10",
        "sample_id": 0,
        "eps": 0.01,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _run_pipeline(
    *,
    python_executable: str,
    case_name: str,
    case: dict[str, Any],
    solver: str,
    output_root: Path,
    timeout_seconds: int,
    quick: bool,
) -> dict[str, Any]:
    output_dir = output_root / case_name / solver
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(_tool_root() / "scripts" / "run_robustness_pipeline.py"),
        "--dataset",
        str(case["dataset"]),
        "--arch",
        str(case["arch"]),
        "--sample-id",
        str(case["sample_id"]),
        "--eps",
        str(case["eps"]),
        "--bit-lb",
        "3",
        "--bit-ub",
        "16",
        "--preimage-mode",
        "milp",
        "--verify-mode",
        "esbmc",
        "--solver",
        solver,
        "--esbmc-layer-block-size",
        "10",
        "--esbmc-profile",
        "paper-fast",
        "--esbmc-timeout",
        "900",
        "--esbmc-memlimit",
        "6g",
        "--gurobi-threads",
        "4",
        "--formal-no-saturation",
        "--no-require-formal-no-saturation",
        "--output-dir",
        str(output_dir),
    ]
    if quick:
        command.extend(
            [
                "--compare-limit",
                "10",
                "--max-quality-refinement-steps",
                "0",
                "--no-formal-no-saturation",
                "--no-empirical-saturation-check",
                "--accuracy-drop-threshold",
                "-1",
                "--saturation-threshold",
                "-1",
                "--mismatch-threshold",
                "-1",
                "--skip-c-backend",
                "--no-export-paper-tables",
            ]
        )
    (output_dir / "command.txt").write_text(_command_text(command) + "\n", encoding="utf-8")
    started = time.monotonic()
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=_repo_root(),
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    summary_path = output_dir / "reports" / "pipeline_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {
        "solver": solver,
        "return_code": int(completed.returncode),
        "elapsed_seconds": time.monotonic() - started,
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "summary": summary,
    }


def _extract_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"available": False}
    synthesis = summary.get("synthesis", {})
    quality = summary.get("quality_refinement", {})
    return {
        "available": True,
        "success": synthesis.get("success"),
        "total_bits": synthesis.get("total_bits"),
        "integer_bits": synthesis.get("integer_bits"),
        "fractional_bits": synthesis.get("fractional_bits"),
        "quality_accepted": quality.get("accepted"),
        "esbmc_status_counts": summary.get("esbmc_status_counts", {}),
        "blockwise_verification": summary.get("blockwise_verification", {}),
        "formal_saturation_verification": summary.get("formal_saturation_verification", {}),
    }


def _compare(case_name: str, cbc: dict[str, Any], gurobi: dict[str, Any]) -> dict[str, Any]:
    cbc_summary = _extract_summary(cbc.get("summary"))
    gurobi_summary = _extract_summary(gurobi.get("summary"))
    comparable_fields = ["success", "total_bits", "integer_bits", "fractional_bits", "quality_accepted"]
    differences = {
        field: {"cbc": cbc_summary.get(field), "gurobi": gurobi_summary.get(field)}
        for field in comparable_fields
        if cbc_summary.get(field) != gurobi_summary.get(field)
    }
    return {
        "case": case_name,
        "cbc": {key: value for key, value in cbc.items() if key != "summary"} | cbc_summary,
        "gurobi": {key: value for key, value in gurobi.items() if key != "summary"} | gurobi_summary,
        "matches": not differences,
        "differences": differences,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local CBC-vs-Gurobi parity checks for small robustness cases.")
    parser.add_argument("--case", action="append", choices=sorted(CASES), default=None)
    parser.add_argument("--output-root", type=Path, default=Path("output/solver_parity"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--quick", action="store_true", help="Skip deployment quality/C-table work for a faster smoke comparison.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_cases = args.case or ["iris", "seeds"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "cases": [],
        "output_root": str(args.output_root),
        "quick": bool(args.quick),
    }
    failed = False
    for case_name in selected_cases:
        case = CASES[case_name]
        cbc_result = _run_pipeline(
            python_executable=args.python_executable,
            case_name=case_name,
            case=case,
            solver="cbc",
            output_root=args.output_root,
            timeout_seconds=args.timeout_seconds,
            quick=bool(args.quick),
        )
        gurobi_result = _run_pipeline(
            python_executable=args.python_executable,
            case_name=case_name,
            case=case,
            solver="gurobi",
            output_root=args.output_root,
            timeout_seconds=args.timeout_seconds,
            quick=bool(args.quick),
        )
        comparison = _compare(case_name, cbc_result, gurobi_result)
        report["cases"].append(comparison)
        failed = failed or not comparison["matches"] or cbc_result["return_code"] != 0 or gurobi_result["return_code"] != 0

    report["matches"] = not failed
    report_path = args.output_root / "solver_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "matches": report["matches"]}, indent=2))
    return 0 if report["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
