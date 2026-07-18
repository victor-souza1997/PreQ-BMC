from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PLOT_NAMES = [
    "fig_accuracy_formal_vs_refined",
    "fig_accuracy_drop",
    "fig_bitwidth_allocation",
    "fig_refinement_steps",
    "fig_scalability_status",
    "fig_esbmc_time_by_arch",
    "fig_mrr_discrete",
    "fig_smt_size_full_vs_block",
    "fig_smt_depth_full_vs_block",
    "fig_bvmul_full_vs_block",
    "fig_saturation_formal_vs_refined",
    "fig_mismatch_formal_vs_refined",
    "fig_accuracy_vs_saturation",
    "fig_python_c_logit_error",
    "fig_ablation_accuracy",
    "fig_ablation_esbmc_time",
    "fig_ablation_status",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate article plots from aggregated CSV tables.")
    parser.add_argument("--input-root", type=Path, default=Path("output/article_results"))
    parser.add_argument("--output-root", type=Path, default=Path("output/article_results/plots"))
    return parser


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


def _json_list(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    values: list[float] = []
    for item in parsed:
        number = _num(item)
        if number is not None:
            values.append(number)
    return values


def _save(fig: plt.Figure, output_root: Path, name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_root / f"{name}.png", dpi=200)
    plt.close(fig)


def _placeholder(output_root: Path, name: str, message: str = "No data") -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    _save(fig, output_root, name)


def _label(row: dict[str, str]) -> str:
    dataset = row.get("dataset", "")
    arch = row.get("arch", "")
    sample = row.get("sample_id", "")
    return f"{dataset}/{arch}/s{sample}".strip("/")


def _grouped_method_bar(
    rows: list[dict[str, str]],
    output_root: Path,
    name: str,
    value_key: str,
    ylabel: str,
    methods: tuple[str, ...] = ("formal_only", "quality_refined"),
) -> None:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        run = row.get("run_name") or _label(row)
        grouped.setdefault(run, {})[row.get("method", "")] = row
    if not grouped:
        _placeholder(output_root, name)
        return
    labels = list(grouped.keys())
    x = list(range(len(labels)))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.8))
    plotted = False
    for index, method in enumerate(methods):
        values = []
        for run in labels:
            value = _num((grouped[run].get(method) or {}).get(value_key))
            values.append(float("nan") if value is None else value)
            plotted = plotted or value is not None
        offsets = [pos - 0.4 + width / 2 + index * width for pos in x]
        ax.bar(offsets, values, width=width, label=method)
    if not plotted:
        plt.close(fig)
        _placeholder(output_root, name)
        return
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output_root, name)


def _scatter(
    rows: list[dict[str, str]],
    output_root: Path,
    name: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    plotted = False
    markers = {"formal_only": "o", "quality_refined": "s"}
    for method in ("formal_only", "quality_refined"):
        xs: list[float] = []
        ys: list[float] = []
        labels: list[str] = []
        for row in rows:
            if row.get("method") != method:
                continue
            x = _num(row.get(x_key))
            y = _num(row.get(y_key))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
            labels.append(row.get("arch", ""))
        if xs:
            plotted = True
            ax.scatter(xs, ys, marker=markers.get(method, "o"), label=method)
            for x, y, label in zip(xs, ys, labels, strict=False):
                ax.annotate(label, (x, y), fontsize=7, alpha=0.7)
    if not plotted:
        plt.close(fig)
        _placeholder(output_root, name)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output_root, name)


def _status_bar(rows: list[dict[str, str]], output_root: Path, name: str, status_key: str = "final_status") -> None:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get(status_key) or row.get("status") or "UNKNOWN"
        counts[status] = counts.get(status, 0) + 1
    if not counts:
        _placeholder(output_root, name)
        return
    labels = list(counts.keys())
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    positions = range(len(labels))
    ax.bar(positions, [counts[label] for label in labels])
    ax.set_ylabel("count")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_root, name)


def _bitwidth_plot(rows: list[dict[str, str]], output_root: Path) -> None:
    rows = [row for row in rows if _num(row.get("Q")) is not None]
    if not rows:
        _placeholder(output_root, "fig_bitwidth_allocation")
        return
    labels = [f"{row.get('dataset')}/{row.get('arch')}/L{row.get('layer_index')}" for row in rows]
    x = list(range(len(rows)))
    q = [_num(row.get("Q")) or 0 for row in rows]
    i_values = [_num(row.get("I")) or 0 for row in rows]
    f_values = [_num(row.get("F")) or 0 for row in rows]
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.35), 4.8))
    ax.bar([v - width for v in x], q, width=width, label="Q")
    ax.bar(x, i_values, width=width, label="I")
    ax.bar([v + width for v in x], f_values, width=width, label="F")
    ax.set_ylabel("bits")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output_root, "fig_bitwidth_allocation")


def _mrr_plot(rows: list[dict[str, str]], output_root: Path) -> None:
    values = [(row, _num(row.get("mrr_discrete"))) for row in rows]
    values = [(row, value) for row, value in values if value is not None]
    if not values:
        _placeholder(output_root, "fig_mrr_discrete")
        return
    labels = [_label(row) for row, _ in values]
    fig, ax = plt.subplots(figsize=(max(7, len(values) * 0.5), 4.4))
    ax.bar(range(len(values)), [value for _, value in values])
    ax.set_ylabel("mrr_discrete")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_root, "fig_mrr_discrete")


def _smt_plot(
    rows: list[dict[str, str]],
    output_root: Path,
    name: str,
    key: str,
    ylabel: str,
) -> None:
    values = [(row, _num(row.get(key))) for row in rows]
    values = [(row, value) for row, value in values if value is not None]
    if not values:
        _placeholder(output_root, name)
        return
    labels = [f"{row.get('mode', '')}/{row.get('property_type', '')}" for row, _ in values]
    fig, ax = plt.subplots(figsize=(max(7, len(values) * 0.45), 4.4))
    ax.bar(range(len(values)), [value for _, value in values])
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_root, name)


def _run_plot(name: str, callback: Callable[[], None], output_root: Path) -> None:
    try:
        callback()
    except Exception as exc:  # noqa: BLE001 - plotting should be best effort.
        plt.close("all")
        _placeholder(output_root, name, f"Plot unavailable\n{type(exc).__name__}: {exc}")


def generate_plots(input_root: Path, output_root: Path) -> None:
    all_rows = _read_csv(input_root / "all_experiments.csv")
    quality = _read_csv(input_root / "table_quality_metrics.csv")
    bitwidths = _read_csv(input_root / "table_bitwidths.csv")
    mrr = _read_csv(input_root / "table_mrr.csv")
    smt = _read_csv(input_root / "table_smt_complexity_summary.csv")
    ablation = _read_csv(input_root / "table_ablation.csv")

    plotters: dict[str, Callable[[], None]] = {
        "fig_accuracy_formal_vs_refined": lambda: _grouped_method_bar(quality, output_root, "fig_accuracy_formal_vs_refined", "c_fixed_accuracy", "c_fixed_accuracy"),
        "fig_accuracy_drop": lambda: _grouped_method_bar(quality, output_root, "fig_accuracy_drop", "accuracy_drop_keras_quantized_to_c_fixed", "accuracy drop"),
        "fig_bitwidth_allocation": lambda: _bitwidth_plot(bitwidths, output_root),
        "fig_refinement_steps": lambda: _grouped_method_bar(all_rows, output_root, "fig_refinement_steps", "refinement_steps", "refinement steps", methods=("quality_refined",)),
        "fig_scalability_status": lambda: _status_bar(all_rows, output_root, "fig_scalability_status"),
        "fig_esbmc_time_by_arch": lambda: _grouped_method_bar(all_rows, output_root, "fig_esbmc_time_by_arch", "total_esbmc_time_seconds", "ESBMC time (s)"),
        "fig_mrr_discrete": lambda: _mrr_plot(mrr, output_root),
        "fig_smt_size_full_vs_block": lambda: _smt_plot(smt, output_root, "fig_smt_size_full_vs_block", "max_file_size_mb", "max SMT size (MB)"),
        "fig_smt_depth_full_vs_block": lambda: _smt_plot(smt, output_root, "fig_smt_depth_full_vs_block", "max_parenthesis_depth", "approx. nesting depth"),
        "fig_bvmul_full_vs_block": lambda: _smt_plot(smt, output_root, "fig_bvmul_full_vs_block", "max_bvmul_count", "bvmul count"),
        "fig_saturation_formal_vs_refined": lambda: _grouped_method_bar(quality, output_root, "fig_saturation_formal_vs_refined", "max_saturation_rate", "max saturation rate"),
        "fig_mismatch_formal_vs_refined": lambda: _grouped_method_bar(quality, output_root, "fig_mismatch_formal_vs_refined", "mismatch_rate_vs_keras", "mismatch rate"),
        "fig_accuracy_vs_saturation": lambda: _scatter(quality, output_root, "fig_accuracy_vs_saturation", "max_saturation_rate", "c_fixed_accuracy", "max saturation rate", "c_fixed_accuracy"),
        "fig_python_c_logit_error": lambda: _grouped_method_bar(quality, output_root, "fig_python_c_logit_error", "max_abs_logit_error", "max abs logit error"),
        "fig_ablation_accuracy": lambda: _grouped_method_bar(ablation, output_root, "fig_ablation_accuracy", "c_fixed_accuracy", "c_fixed_accuracy"),
        "fig_ablation_esbmc_time": lambda: _grouped_method_bar(ablation, output_root, "fig_ablation_esbmc_time", "esbmc_time_seconds", "ESBMC time (s)"),
        "fig_ablation_status": lambda: _status_bar(ablation, output_root, "fig_ablation_status", status_key="status"),
    }
    for name in PLOT_NAMES:
        _run_plot(name, plotters[name], output_root)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_plots(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
