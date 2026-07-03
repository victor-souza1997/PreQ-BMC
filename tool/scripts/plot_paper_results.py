from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper figures from all_experiments.csv.")
    parser.add_argument("--input", type=Path, default=Path("output/paper_results/all_experiments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/paper_results/figures"))
    return parser


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_list(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    except Exception:
        return []
    return []


def _group_by_run(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("run_name", ""), {})[row.get("method", "")] = row
    return {key: value for key, value in grouped.items() if key}


def _label(run_name: str, methods: dict[str, dict[str, str]]) -> str:
    formal = methods.get("formal_only") or methods.get("quality_refined") or {}
    return f"{formal.get('dataset', '')}/{formal.get('arch', '')}" or run_name


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=200)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _grouped_bar(
    grouped: dict[str, dict[str, dict[str, str]]],
    output_dir: Path,
    name: str,
    ylabel: str,
    series: list[tuple[str, Callable[[dict[str, dict[str, str]]], float | None]]],
) -> None:
    categories = list(grouped.keys())
    if not categories:
        _warn(f"Skipping {name}: no rows")
        return
    labels = [_label(run_name, grouped[run_name]) for run_name in categories]
    x = list(range(len(categories)))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.55), 4.8))
    plotted = False
    for idx, (series_name, getter) in enumerate(series):
        values = []
        for run_name in categories:
            value = getter(grouped[run_name])
            values.append(float("nan") if value is None else value)
            plotted = plotted or value is not None
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        ax.bar(offsets, values, width=width, label=series_name)
    if not plotted:
        _warn(f"Skipping {name}: all values missing")
        plt.close(fig)
        return
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir, name)


def _scatter(
    rows: list[dict[str, str]],
    output_dir: Path,
    name: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    annotate_key: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    markers = {"formal_only": "o", "quality_refined": "s"}
    plotted = False
    for method in ("formal_only", "quality_refined"):
        xs: list[float] = []
        ys: list[float] = []
        labels: list[str] = []
        for row in rows:
            if row.get("method") != method:
                continue
            x = _float(row.get(x_key))
            y = _float(row.get(y_key))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
            labels.append(row.get(annotate_key or "arch", ""))
        if xs:
            plotted = True
            ax.scatter(xs, ys, marker=markers.get(method, "o"), label=method)
            if annotate_key:
                for x, y, label in zip(xs, ys, labels, strict=False):
                    ax.annotate(label, (x, y), fontsize=7, alpha=0.75)
    if not plotted:
        _warn(f"Skipping {name}: no numeric points")
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output_dir, name)


def _verification_time(rows: list[dict[str, str]], output_dir: Path) -> None:
    _scatter(
        rows,
        output_dir,
        "fig_verification_time",
        "num_parameters",
        "total_time",
        "num_parameters",
        "total_time",
        annotate_key="arch",
    )


def _scalability(rows: list[dict[str, str]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    status_marker = {"success": "o", "failed": "x", "timeout": "^", "skipped": "."}
    plotted = False
    for row in rows:
        if row.get("method") != "formal_only":
            continue
        x = _float(row.get("num_parameters"))
        y = _float(row.get("total_time"))
        if x is None or y is None:
            continue
        plotted = True
        marker = status_marker.get(row.get("status", ""), "o")
        ax.scatter([x], [y], marker=marker, label=row.get("status") if row.get("status") else None)
        ax.annotate(row.get("arch", ""), (x, y), fontsize=7, alpha=0.75)
    if not plotted:
        _warn("Skipping fig_scalability: no numeric points")
        plt.close(fig)
        return
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.set_xlabel("num_parameters")
    ax.set_ylabel("total_time")
    ax.grid(alpha=0.25)
    _save(fig, output_dir, "fig_scalability")


def _refinement_steps(grouped: dict[str, dict[str, dict[str, str]]], output_dir: Path) -> None:
    _grouped_bar(
        grouped,
        output_dir,
        "fig_refinement_steps",
        "refinement_steps",
        [("quality_refined", lambda methods: _float((methods.get("quality_refined") or {}).get("refinement_steps")))],
    )


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _bitwidth_allocation(grouped: dict[str, dict[str, dict[str, str]]], output_dir: Path) -> None:
    for run_name, methods in grouped.items():
        method_rows = [(method, methods.get(method, {})) for method in ("formal_only", "quality_refined")]
        if not any(_json_list(row.get("Q")) for _, row in method_rows):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for axis, (method, row) in zip(axes, method_rows, strict=False):
            q_values = _json_list(row.get("Q"))
            i_values = _json_list(row.get("I"))
            f_values = _json_list(row.get("F"))
            if not q_values:
                axis.set_title(f"{method} missing")
                continue
            x = list(range(len(q_values)))
            width = 0.25
            axis.bar([v - width for v in x], q_values, width=width, label="Q")
            axis.bar(x, i_values, width=width, label="I")
            axis.bar([v + width for v in x], f_values, width=width, label="F")
            axis.set_title(method)
            axis.set_xlabel("layer index")
            axis.set_xticks(x)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("bits")
        axes[0].legend(fontsize=8)
        _save(fig, output_dir, f"fig_bitwidth_allocation_{_sanitize_filename(run_name)}")


def generate_plots(input_csv: Path, output_dir: Path) -> None:
    rows = _read_rows(input_csv)
    grouped = _group_by_run(rows)

    def safe(name: str, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 - plotting should continue.
            _warn(f"Skipping {name}: {type(exc).__name__}: {exc}")
            plt.close("all")

    safe(
        "fig_accuracy_comparison",
        lambda: _grouped_bar(
            grouped,
            output_dir,
            "fig_accuracy_comparison",
            "accuracy",
            [
                ("quantized Keras", lambda methods: _float((methods.get("formal_only") or methods.get("quality_refined") or {}).get("quantized_keras_accuracy"))),
                ("formal C", lambda methods: _float((methods.get("formal_only") or {}).get("c_fixed_accuracy"))),
                ("refined C", lambda methods: _float((methods.get("quality_refined") or {}).get("c_fixed_accuracy"))),
            ],
        ),
    )
    safe(
        "fig_saturation_comparison",
        lambda: _grouped_bar(
            grouped,
            output_dir,
            "fig_saturation_comparison",
            "max_saturation_rate",
            [
                ("formal_only", lambda methods: _float((methods.get("formal_only") or {}).get("max_saturation_rate"))),
                ("quality_refined", lambda methods: _float((methods.get("quality_refined") or {}).get("max_saturation_rate"))),
            ],
        ),
    )
    safe(
        "fig_mismatch_comparison",
        lambda: _grouped_bar(
            grouped,
            output_dir,
            "fig_mismatch_comparison",
            "mismatch_rate_vs_keras",
            [
                ("formal_only", lambda methods: _float((methods.get("formal_only") or {}).get("mismatch_rate_vs_keras"))),
                ("quality_refined", lambda methods: _float((methods.get("quality_refined") or {}).get("mismatch_rate_vs_keras"))),
            ],
        ),
    )
    safe(
        "fig_bitwidth_total",
        lambda: _grouped_bar(
            grouped,
            output_dir,
            "fig_bitwidth_total",
            "weighted_avg_bits_per_parameter",
            [
                ("formal_only", lambda methods: _float((methods.get("formal_only") or {}).get("weighted_avg_bits_per_parameter"))),
                ("quality_refined", lambda methods: _float((methods.get("quality_refined") or {}).get("weighted_avg_bits_per_parameter"))),
            ],
        ),
    )
    safe(
        "fig_resource_compression",
        lambda: _grouped_bar(
            grouped,
            output_dir,
            "fig_resource_compression",
            "compression_ratio_vs_float32",
            [
                ("formal_only", lambda methods: _float((methods.get("formal_only") or {}).get("compression_ratio_vs_float32"))),
                ("quality_refined", lambda methods: _float((methods.get("quality_refined") or {}).get("compression_ratio_vs_float32"))),
            ],
        ),
    )
    safe("fig_verification_time", lambda: _verification_time(rows, output_dir))
    safe("fig_scalability", lambda: _scalability(rows, output_dir))
    safe("fig_refinement_steps", lambda: _refinement_steps(grouped, output_dir))
    safe(
        "fig_accuracy_vs_saturation",
        lambda: _scatter(
            rows,
            output_dir,
            "fig_accuracy_vs_saturation",
            "max_saturation_rate",
            "c_fixed_accuracy",
            "max_saturation_rate",
            "c_fixed_accuracy",
            annotate_key="method",
        ),
    )
    safe("fig_bitwidth_allocation_per_run", lambda: _bitwidth_allocation(grouped, output_dir))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_plots(args.input, args.output_dir)


if __name__ == "__main__":
    main()
