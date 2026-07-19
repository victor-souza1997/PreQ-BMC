from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from verification.esbmc_install import install_esbmc, local_esbmc_path, resolve_esbmc_executable


DEMO_DATASET = "iris_15x2"
DEMO_ARCH = "2blk_15_15"
DEMO_SAMPLE_ID = 27
DEMO_EPS = 0.05
DEMO_CACHE_KEY = "iris_15x2__2blk_15_15__sample27__eps0.05__milp__e904e4f9a08a7a5f"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tool_root() -> Path:
    return _repo_root() / "src"


def _script_path(*parts: str) -> Path:
    return _tool_root().joinpath("scripts", *parts)


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _tail(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _python_package_report() -> list[dict[str, Any]]:
    modules = [
        ("numpy", "required for numerical arrays"),
        ("tensorflow", "required for Keras benchmark models"),
        ("h5py", "required for HDF5 benchmark weights"),
        ("sklearn", "required for Iris/Seeds dataset loading"),
        ("matplotlib", "optional for plots"),
        ("pandas", "optional for result analysis"),
        ("torch", "optional for legacy conversion utilities"),
        ("onnx", "optional for ONNX conversion utilities"),
        ("mip", "optional; provides the default CBC MILP backend"),
        ("gurobipy", "optional; provides the Gurobi reference MILP backend"),
    ]
    return [
        {
            "module": module,
            "available": _module_available(module),
            "purpose": purpose,
        }
        for module, purpose in modules
    ]


def _ensure_esbmc_installed_if_requested(install_missing: bool) -> None:
    if not install_missing or resolve_esbmc_executable() is not None:
        return
    print("ESBMC is missing; installing repo-local ESBMC...", flush=True)
    install_esbmc(root=_repo_root())


def _print_environment_report(*, install_missing_esbmc: bool = False) -> dict[str, Any]:
    _ensure_esbmc_installed_if_requested(install_missing_esbmc)
    esbmc = resolve_esbmc_executable()
    repo_local_esbmc = local_esbmc_path()
    cbc_available = _module_available("mip")
    gurobi_available = _module_available("gurobipy")
    packages = _python_package_report()
    report = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "python_ok": sys.version_info >= (3, 10),
        "esbmc_path": esbmc,
        "esbmc_available": esbmc is not None,
        "repo_local_esbmc_path": str(repo_local_esbmc),
        "repo_local_esbmc_available": repo_local_esbmc.exists(),
        "cbc_available": cbc_available,
        "gurobipy_available": gurobi_available,
        "packages": packages,
    }
    print(json.dumps(report, indent=2))
    if esbmc is None:
        print("\nESBMC was not found. Run `preqbmc install-esbmc`, or install ESBMC and ensure `esbmc --version` works.")
    if not cbc_available:
        print("\nCBC/python-mip is the default license-free MILP backend. Install it with `pip install -e '.[cbc]'`.")
    if not gurobi_available:
        print("\nGurobi/gurobipy is optional and only needed for `--solver gurobi` reference runs.")
    missing_demo = [
        item["module"]
        for item in packages
        if item["module"] in {"numpy", "tensorflow", "h5py", "sklearn"} and not item["available"]
    ]
    if missing_demo:
        print(
            "\nMissing packages for the full demo/pipeline: "
            + ", ".join(missing_demo)
            + ". Install with `pip install -e '.[full]'`; Gurobi licensing is only needed for `--solver gurobi`."
        )
    return report


def cmd_verify_environment(args: argparse.Namespace, extra: list[str]) -> int:
    del extra
    _print_environment_report(install_missing_esbmc=bool(args.install_missing_esbmc))
    return 0


def cmd_install_esbmc(args: argparse.Namespace, extra: list[str]) -> int:
    del extra
    result = install_esbmc(
        root=_repo_root(),
        asset_pattern=args.asset_pattern,
        force=bool(args.force),
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"ESBMC path: {result['esbmc_path']}", flush=True)
        if result.get("installed"):
            print(f"Installed {result['release']} from {result['asset']}", flush=True)
            version_output = str(result.get("version_output", "")).strip()
            if version_output:
                print(version_output, flush=True)
        else:
            print(f"Skipped: {result.get('reason')}", flush=True)
    return 0


def _default_cache_dir() -> Path:
    return _repo_root() / "examples" / "preimage_cache"


def _demo_cache_available(cache_dir: Path) -> bool:
    return (cache_dir / DEMO_CACHE_KEY / "metadata.json").exists() and (
        cache_dir / DEMO_CACHE_KEY / "preimage.npz"
    ).exists()


def _demo_required_modules_available() -> tuple[bool, list[str]]:
    required = ["numpy", "tensorflow", "h5py", "sklearn"]
    missing = [module for module in required if not _module_available(module)]
    return not missing, missing


def cmd_demo(args: argparse.Namespace, extra: list[str]) -> int:
    _ensure_esbmc_installed_if_requested(bool(args.install_missing_esbmc))
    esbmc = resolve_esbmc_executable()
    cbc_available = _module_available("mip")
    gurobi_available = _module_available("gurobipy")
    cache_dir = Path(args.preimage_cache_dir)
    output_dir = Path(args.output)
    print(f"ESBMC: {esbmc or 'not found'}", flush=True)
    print(f"CBC/python-mip: {'available' if cbc_available else 'not available'}", flush=True)
    print(f"Gurobi/gurobipy: {'available' if gurobi_available else 'not available'}", flush=True)
    print(f"MILP solver: {args.solver}", flush=True)
    print(f"no-gurobi mode: {bool(args.no_gurobi)}", flush=True)
    if args.no_gurobi:
        print(f"preimage cache: {cache_dir}", flush=True)

    if esbmc is None:
        print("Cannot run the ESBMC demo because `esbmc` was not found.")
        print("Run `preqbmc install-esbmc`, then rerun `preqbmc verify-environment`.")
        return 2

    modules_ok, missing_modules = _demo_required_modules_available()
    if not modules_ok:
        print("Cannot run the demo because required Python packages are missing: " + ", ".join(missing_modules))
        print("Install them with `pip install -e '.[full]'`.")
        return 2

    if args.no_gurobi and not _demo_cache_available(cache_dir):
        print(f"Cannot run --no-gurobi demo: expected cache key `{DEMO_CACHE_KEY}` under {cache_dir}.")
        print("Regenerate it with Gurobi or restore examples/preimage_cache from the artifact.")
        return 2

    command = [
        sys.executable,
        str(_script_path("run_robustness_pipeline.py")),
        "--dataset",
        args.dataset,
        "--arch",
        args.arch,
        "--sample-id",
        str(args.sample_id),
        "--eps",
        str(args.eps),
        "--bit-lb",
        str(args.bit_lb),
        "--bit-ub",
        str(args.bit_ub),
        "--preimage-mode",
        args.preimage_mode,
        "--verify-mode",
        "esbmc",
        "--solver",
        args.solver,
        "--compare-limit",
        str(args.compare_limit),
        "--output-dir",
        str(output_dir),
    ]
    if args.no_gurobi:
        command.extend(
            [
                "--no-gurobi",
                "--preimage-cache-dir",
                str(cache_dir),
                "--preimage-cache-key",
                str(args.preimage_cache_key),
            ]
        )
    command.extend(extra)

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "run_stdout.log"
    stderr_path = output_dir / "run_stderr.log"
    (output_dir / "command.txt").write_text(_command_text(command) + "\n", encoding="utf-8")

    print("Running: " + _command_text(command), flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=_repo_root(), stdout=stdout, stderr=stderr, text=True, check=False)
    print(f"Results directory: {output_dir}")
    print(f"Reports: {output_dir / 'reports'}")
    print(f"Verification harnesses: {output_dir / 'layers'}")
    print(f"Deployment C output: {output_dir / 'c_export' / 'qnn_model.c'}")
    print(f"Run stdout log: {stdout_path}")
    print(f"Run stderr log: {stderr_path}")
    print(f"Used cache/no-gurobi mode: {bool(args.no_gurobi)}")
    if completed.returncode != 0:
        print(f"Demo failed with return code {completed.returncode}. Last stderr lines:")
        print(_tail(stderr_path) or "(stderr was empty)")
    return int(completed.returncode)


def cmd_reproduce(args: argparse.Namespace, extra: list[str]) -> int:
    command = [
        sys.executable,
        str(_script_path("run_article_experiments.py")),
        "--config",
        str(args.config),
        "--solver",
        args.solver,
    ]
    for value in args.only or []:
        command.extend(["--only", value])
    if args.max_runs is not None:
        command.extend(["--max-runs", str(args.max_runs)])
    if args.output_root is not None:
        command.extend(["--output-root", str(args.output_root)])
    if args.dry_run:
        command.append("--dry-run")
    if args.aggregate:
        command.append("--aggregate")
    if args.plots:
        command.append("--plots")
    if args.continue_on_error:
        command.append("--continue-on-error")
    command.extend(extra)
    print("Running: " + _command_text(command))
    completed = subprocess.run(command, cwd=_repo_root(), check=False)
    return int(completed.returncode)


def cmd_aggregate(args: argparse.Namespace, extra: list[str]) -> int:
    aggregate_command = [
        sys.executable,
        str(_script_path("aggregate_article_results.py")),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(args.output_root),
    ]
    aggregate_command.extend(extra)
    print("Running: " + _command_text(aggregate_command))
    completed = subprocess.run(aggregate_command, cwd=_repo_root(), check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    if args.plots:
        plot_command = [
            sys.executable,
            str(_script_path("plot_article_results.py")),
            "--input-root",
            str(args.output_root),
            "--output-root",
            str(args.output_root / "plots"),
        ]
        print("Running: " + _command_text(plot_command))
        completed = subprocess.run(plot_command, cwd=_repo_root(), check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preqbmc", description="Public PreQ-BMC artifact CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a small Iris artifact demo.")
    demo.add_argument("--output", type=Path, default=Path("output/demo_run"))
    demo.add_argument("--solver", default="cbc", choices=["cbc", "gurobi"])
    demo.add_argument("--no-gurobi", action="store_true", help="Use cached preimage contracts instead of solving the preimage MILP.")
    demo.add_argument("--preimage-cache-dir", type=Path, default=_default_cache_dir())
    demo.add_argument("--preimage-cache-key", default=DEMO_CACHE_KEY)
    demo.add_argument("--dataset", default=DEMO_DATASET)
    demo.add_argument("--arch", default=DEMO_ARCH)
    demo.add_argument("--sample-id", type=int, default=DEMO_SAMPLE_ID)
    demo.add_argument("--eps", type=float, default=DEMO_EPS)
    demo.add_argument("--bit-lb", type=int, default=1)
    demo.add_argument("--bit-ub", type=int, default=16)
    demo.add_argument("--preimage-mode", default="milp", choices=["milp", "abstr", "comp"])
    demo.add_argument("--compare-limit", type=int, default=10)
    demo.add_argument(
        "--install-missing-esbmc",
        action="store_true",
        help="Download repo-local ESBMC automatically if no ESBMC executable is found.",
    )
    demo.set_defaults(func=cmd_demo)

    reproduce = subparsers.add_parser("reproduce", help="Run article experiment configurations.")
    reproduce.add_argument("--config", type=Path, default=Path("experiments/article_experiments.json"))
    reproduce.add_argument("--solver", default="cbc", choices=["cbc", "gurobi"])
    reproduce.add_argument("--only", action="append", default=[])
    reproduce.add_argument("--max-runs", type=int, default=None)
    reproduce.add_argument("--output-root", type=Path, default=None)
    reproduce.add_argument("--dry-run", action="store_true")
    reproduce.add_argument("--aggregate", action="store_true")
    reproduce.add_argument("--plots", action="store_true")
    reproduce.add_argument("--continue-on-error", action="store_true")
    reproduce.set_defaults(func=cmd_reproduce)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate article experiment outputs.")
    aggregate.add_argument("--input-root", type=Path, default=Path("output/article_runs"))
    aggregate.add_argument("--output-root", type=Path, default=Path("output/article_results"))
    aggregate.add_argument("--plots", action="store_true")
    aggregate.set_defaults(func=cmd_aggregate)

    verify = subparsers.add_parser("verify-environment", help="Report solver and Python package availability.")
    verify.add_argument(
        "--install-missing-esbmc",
        action="store_true",
        help="Download repo-local ESBMC automatically if no ESBMC executable is found.",
    )
    verify.set_defaults(func=cmd_verify_environment)

    install = subparsers.add_parser("install-esbmc", help="Download ESBMC into this repository under .local/.")
    install.add_argument("--asset-pattern", default=None, help="Release asset glob, e.g. '*linux*.zip'.")
    install.add_argument("--force", action="store_true", help="Replace an existing repo-local ESBMC install.")
    install.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    install.set_defaults(func=cmd_install_esbmc)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    return int(args.func(args, extra))


if __name__ == "__main__":
    raise SystemExit(main())
