"""Tables and separate runtime/memory figures for a real tiny-CNN gate report."""
import argparse
import csv
import json
from pathlib import Path
import hashlib

from reports.ssv_measurements import missing_device_report
from scripts.run_ssv_cnn_gate import write_new_json


def export(report_path, output):
    report = json.loads(report_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for r in report["derived_runs"]:
        measured = [c["peak_memory_bytes"] for c in r["calls"] if c.get("peak_memory_bytes") is not None]
        rows.append({"model": "enumerable_2x2_conv", "beta": r["beta"], "regions": 1,
                     "status": r["final_status"], "input_bridge": r["input_bridge_checked"],
                     "Q_per_layer": "/".join(map(str, r["total_bits"])),
                     "F_per_layer": "/".join(map(str, r["fractional_bits"])),
                     "calls": len(r["calls"]),
                     "esbmc_seconds": sum(c["elapsed_seconds"] for c in r["calls"]),
                     "max_query_seconds": max((c["elapsed_seconds"] for c in r["calls"]), default=0),
                     "sampled_peak_rss_mib": max(measured) / 2**20 if measured else None})
    if not rows:
        raise ValueError("No derived preimage runs to tabulate")
    with (output / "tiny_cnn.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    latex = [r"\begin{tabular}{rrrrr}", r"\hline", r"$\beta$ & Verified regions & Calls & ESBMC (s) & RSS (MiB) \\", r"\hline"]
    for row in rows:
        memory = "--" if row["sampled_peak_rss_mib"] is None else f"{row['sampled_peak_rss_mib']:.1f}"
        verified = int(row["status"] == "VERIFIED" and row["input_bridge"])
        latex.append(f"{row['beta']} & {verified}/1 & {row['calls']} & {row['esbmc_seconds']:.2f} & {memory} " + r"\\")
    latex.extend([r"\hline", r"\end{tabular}", "% One toy region, one timing repetition per beta; not GTSRB data."])
    (output / "tiny_cnn.tex").write_text("\n".join(latex) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, label in (("esbmc_seconds", "Total ESBMC time (s)"), ("sampled_peak_rss_mib", "Sampled query peak RSS (MiB)")):
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        known = [row for row in rows if row[metric] is not None]
        ax.bar([str(row["beta"]) for row in known], [row[metric] for row in known], color="#287d70")
        ax.set(xlabel="Neuron block size (0 = monolithic layer)", ylabel=label, title="Tiny CNN feasibility, one region")
        fig.tight_layout()
        fig.savefig(output / f"{metric}.pdf")
        plt.close(fig)
    write_new_json(output / "android_measurements.json", missing_device_report())
    write_new_json(output / "provenance.json", {"source": str(report_path.resolve()),
                   "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                   "distinct_images": 1, "distinct_regions": 1, "variants": len(rows),
                   "timing_repetitions_per_variant": 1, "dataset": "synthetic_not_GTSRB"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.report, args.output)


if __name__ == "__main__":
    main()
