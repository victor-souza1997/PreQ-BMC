#!/usr/bin/env python3
"""Markdown tables for the 2026-09-28 scalability runs (all MEASURED, validation 3652 only)."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output/ssv_scalability_20260928")


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


print("## Whole network, direct ESBMC on the compact C\n")
print("| bits | free bytes | interval flags | status | s | peak RSS GB | last phase |\n|---|---|---|---|---|---|---|")
for r in lines(root / "int16_calib" / "curve.jsonl"):
    print(f"| {r['bits']} | {r['free_bytes']} | {r['interval_analysis']} | {r['status']} | {r['elapsed_seconds']:.0f} | "
          f"{r['peak_process_tree_rss_bytes'] / 1e9:.2f} | {r['last_phase']} |")

print("\n## Layer 0, first R output rows (384 outputs per row)\n")
print("| bits | R | outputs | mode | free bytes | box width sum | status | s | peak RSS GB | last phase |\n"
      "|---|---|---|---|---|---|---|---|---|---|")
for folder in sorted(root.glob("rows_*")):
    for r in lines(folder / "layer0_rows_curve.jsonl"):
        print(f"| {r['bits']} | {r['output_rows']} | {r['outputs']} | {r['mode']} | {r['free_bytes']} | "
              f"{r['box_width_sum']} | {r['status']} | {r.get('elapsed_seconds', 0):.0f} | "
              f"{r.get('peak_process_tree_rss_bytes', 0) / 1e9:.2f} | {r.get('last_phase', '')} |")

print("\n## Layer 0, row 0, first C output columns (24 outputs per column)\n")
print("| bits | C | outputs | mode | solver profile | free bytes | box width sum | status | s | peak RSS GB | last phase |\n"
      "|---|---|---|---|---|---|---|---|---|---|---|")
for folder in sorted(root.glob("cols_*")):
    for r in lines(folder / "layer0_cols_curve.jsonl"):
        print(f"| {r['bits']} | {r['output_columns']} | {r['outputs']} | {r['mode']} (eps {r.get('epsilon', 1)}) | {r.get('profile', 'paper-z3')} | "
              f"{r['free_bytes']} | {r['box_width_sum']} | {r['status']} | {r.get('elapsed_seconds', 0):.0f} | "
              f"{r.get('peak_process_tree_rss_bytes', 0) / 1e9:.2f} | {r.get('last_phase', '')} |")

print("\n## Decomposed method (CROWN chain v2 + ESBMC), cumulative by layers included\n")
print("| image | layer | chains | harnesses | CPU-h | cumulative CPU-h | peak RSS MiB | jobs |\n"
      "|---|---|---|---|---|---|---|---|")
cumulative = {}
for r in json.loads((root / "layers_included.json").read_text()):
    cpu = r["summed_esbmc_seconds"] / 3600
    cumulative[r["image"]] = cumulative.get(r["image"], 0) + cpu
    print(f"| {r['image']} | L{r['layer']} | {r['chains_checked']} | {r['harness_count']} | {cpu:.2f} | "
          f"{cumulative[r['image']]:.2f} | {r['peak_aggregate_rss_mib']:.0f} | {r['jobs']} |")
