from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


COUNT_PATTERNS = {
    "declare_fun_count": "(declare-fun",
    "define_fun_count": "(define-fun",
    "assert_count": "(assert",
    "bvmul_count": "bvmul",
    "bvadd_count": "bvadd",
    "bvsub_count": "bvsub",
    "bvshl_count": "bvshl",
    "bvlshr_count": "bvlshr",
    "bvashr_count": "bvashr",
    "ite_count": "ite",
    "extract_count": "extract",
    "concat_count": "concat",
    "select_count": "select",
    "store_count": "store",
}

FIELDS = [
    "run_name",
    "dataset",
    "arch",
    "method",
    "layer_index",
    "property_type",
    "mode",
    "smt_path",
    "smt_file_size_bytes",
    "smt_file_size_mb",
    "line_count",
    "char_count",
    "declare_fun_count",
    "define_fun_count",
    "assert_count",
    "bvmul_count",
    "bvadd_count",
    "bvsub_count",
    "bvshl_count",
    "bvlshr_count",
    "bvashr_count",
    "ite_count",
    "extract_count",
    "concat_count",
    "select_count",
    "store_count",
    "max_parenthesis_depth",
    "parse_balance_ok",
    "query_time_seconds",
    "status",
]

SUMMARY_FIELDS = [
    "run_name",
    "dataset",
    "arch",
    "method",
    "layer_index",
    "property_type",
    "mode",
    "max_file_size_mb",
    "mean_file_size_mb",
    "max_bvmul_count",
    "mean_bvmul_count",
    "max_assert_count",
    "max_parenthesis_depth",
    "max_query_time_seconds",
    "status",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze SMT-LIB syntactic complexity for generated ESBMC formulas.")
    parser.add_argument("--input-root", type=Path, default=Path("output/article_runs"))
    parser.add_argument("--output", type=Path, default=Path("output/article_results/smt_complexity.csv"))
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _run_dir_for(path: Path, input_root: Path) -> Path:
    rel = path.relative_to(input_root)
    if not rel.parts:
        return input_root
    return input_root / rel.parts[0]


def _metadata_for(path: Path, input_root: Path) -> dict[str, Any]:
    run_dir = _run_dir_for(path, input_root)
    status = _read_json(run_dir / "run_status.json")
    config = _read_json(run_dir / "run_config.json")
    pipeline = _read_json(run_dir / "pipeline_summary.json") or _read_json(run_dir / "reports" / "pipeline_summary.json")
    return {
        "run_name": status.get("name") or config.get("name") or run_dir.name,
        "dataset": pipeline.get("dataset", config.get("dataset", status.get("dataset"))),
        "arch": pipeline.get("arch", config.get("arch", status.get("arch"))),
        "method": config.get("method", "both"),
    }


def _infer_from_name(path: Path) -> dict[str, Any]:
    name = path.name
    layer_match = re.search(r"layer_(\d+)", name)
    if "no_sat" in name or "no_saturation" in name:
        property_type = "no_saturation"
    elif "monolithic" in name:
        property_type = "monolithic"
    elif "output" in name:
        property_type = "output"
    else:
        property_type = "preimage"
    mode = "blockwise" if "block" in name else "monolithic" if "monolithic" in name else "full_layer"
    return {
        "layer_index": int(layer_match.group(1)) if layer_match else "",
        "property_type": property_type,
        "mode": mode,
    }


def _parenthesis_depth(text: str) -> tuple[int, bool]:
    depth = 0
    max_depth = 0
    ok = True
    for char in text:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth -= 1
            if depth < 0:
                ok = False
                depth = 0
    return max_depth, ok and depth == 0


def _query_metadata(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_suffix(".c.stdout.log"),
        Path(str(path).removesuffix(".smt2") + ".stdout.log"),
    ]
    stderr_candidates = [
        path.with_suffix(".c.stderr.log"),
        Path(str(path).removesuffix(".smt2") + ".stderr.log"),
    ]
    text = ""
    for candidate in candidates + stderr_candidates:
        if candidate.exists():
            text += "\n" + candidate.read_text(encoding="utf-8", errors="replace")[-20000:]
    status = ""
    if "VERIFICATION SUCCESSFUL" in text:
        status = "VERIFIED"
    elif "VERIFICATION FAILED" in text:
        status = "FAILED"
    elif "timeout" in text.lower():
        status = "TIMEOUT"
    return {"query_time_seconds": "", "status": status}


def analyze_file(path: Path, input_root: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    size = path.stat().st_size
    max_depth, balance_ok = _parenthesis_depth(text)
    metadata = _metadata_for(path, input_root)
    metadata.update(_infer_from_name(path))
    metadata.update(_query_metadata(path))
    row: dict[str, Any] = {
        **metadata,
        "smt_path": str(path),
        "smt_file_size_bytes": int(size),
        "smt_file_size_mb": float(size / (1024 * 1024)),
        "line_count": int(text.count("\n") + (1 if text else 0)),
        "char_count": int(len(text)),
        "max_parenthesis_depth": int(max_depth),
        "parse_balance_ok": bool(balance_ok),
    }
    for field, pattern in COUNT_PATTERNS.items():
        row[field] = int(text.count(pattern))
    return row


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _status(statuses: list[str]) -> str:
    statuses = [status for status in statuses if status]
    if not statuses:
        return ""
    if all(status == "VERIFIED" for status in statuses):
        return "VERIFIED"
    for status in ("FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN"):
        if status in statuses:
            return status
    return statuses[0]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
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
        sizes = [_num(row.get("smt_file_size_mb")) for row in group]
        bvmul = [_num(row.get("bvmul_count")) for row in group]
        asserts = [_num(row.get("assert_count")) for row in group]
        depths = [_num(row.get("max_parenthesis_depth")) for row in group]
        query_times = [_num(row.get("query_time_seconds")) for row in group]
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
                "max_bvmul_count": max(bvmul, default=0.0),
                "mean_bvmul_count": sum(bvmul) / len(bvmul) if bvmul else 0.0,
                "max_assert_count": max(asserts, default=0.0),
                "max_parenthesis_depth": max(depths, default=0.0),
                "max_query_time_seconds": max(query_times, default=0.0),
                "status": _status([str(row.get("status", "")) for row in group]),
            }
        )
    return output


def analyze(input_root: Path, output: Path, summary_output: Path | None = None) -> dict[str, str]:
    rows = [analyze_file(path, input_root) for path in sorted(input_root.rglob("*.smt2"))]
    _write_csv(output, rows, FIELDS)
    summary_path = summary_output or output.parent / "table_smt_complexity_summary.csv"
    _write_csv(summary_path, summarize(rows), SUMMARY_FIELDS)
    return {"smt_complexity_csv": str(output), "smt_complexity_summary_csv": str(summary_path)}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.input_root, args.output, args.summary_output), indent=2))


if __name__ == "__main__":
    main()
