#!/usr/bin/env python3
"""MEASURED only: per-layer clamp saturation of a quantization candidate on one validation image.

CROWN chain v2 needs every hidden box strictly inside the signed format (hh < Q_HIGH) and
the stable-active lower sides above Q_LOW. This counts how many neurons violate that, at
the centre point and over the eps byte box (plain interval propagation, untrusted).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from datasets.gtsrb_study import load_crop  # noqa: E402
from quantization_candidates import load_candidate  # noqa: E402
from verification.conv_interval_fast import propagate_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--image", type=int, default=3652)
    ap.add_argument("--epsilon", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    folder, manifest, network, encoder, study, rows = load_candidate(args.artifact)
    row = rows[args.image]
    bits = manifest["bits"]
    q_low, q_high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    image = load_crop(study["dataset_root"], row)
    centre = encoder.encode(image).astype(np.int64)
    report = {"artifact": str(folder), "image": row["id"], "split": row["split"], "bits": bits,
              "target": int(row["class_id"]), "q_range": [q_low, q_high], "regions": []}
    boxes = [("centre", centre, centre)]
    for eps in args.epsilon:
        blo, bhi = encoder.byte_box(image, eps)
        boxes.append((f"eps{eps}", encoder.encode(blo).astype(np.int64), encoder.encode(bhi).astype(np.int64)))
    for name, xl, xu in boxes:
        box = propagate_box(network, xl, xu)
        layers = []
        for k in range(len(network.layers)):
            pl, ph = np.asarray(box.pre_low[k], dtype=np.int64), np.asarray(box.pre_high[k], dtype=np.int64)
            layers.append({
                "layer": k, "neurons": int(pl.size),
                "top_at_q_high": int((ph >= q_high).sum()),
                "stably_top_saturated": int((pl >= q_high).sum()),
                "bottom_at_q_low": int((pl <= q_low).sum()),
                "unstable_relu": int(((pl < 0) & (ph > 0)).sum()),
                "median_width": float(np.median(ph - pl)),
            })
        report["regions"].append({"region": name, "free_inputs": int((xu > xl).sum()),
                                  "predicted": int(np.argmax(box.low[-1])) if name == "centre" else None,
                                  "layers": layers})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
