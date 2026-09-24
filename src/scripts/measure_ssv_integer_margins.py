"""Measure deployed-integer box margins of a selected candidate on validation data.

For each image and radius this propagates the exact integer interval semantics
(the same boxes the ESBMC transition certificates prove) and reports
- ``box``: the fraction whose final box margin is positive, which is the check the
  current margin harness makes; and
- ``difference_row``: the fraction whose difference-row lower bound is positive
  with clamp side conditions met, a candidate future obligation.

These are MEASUREMENTS for choosing candidates.  No region is certified here.
Only the validation split is read, so the one-shot test policy is untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backends.conv_fixed_point import quantize_restricted_sequential
from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop
from scripts.evaluate_ssv_deep_source_model import restricted_from_npz
from scripts.search_ssv_source_model import _sha256
from verification.conv_interval_fast import box_margin, difference_row_margin, propagate_box

Q16 = LayerQuantizationSpec(total_bits=16, integer_bits=7, fractional_bits=8)


def measure_network(network, encoder, images, labels, epsilons):
    predictions, per_epsilon = [], {eps: [] for eps in epsilons}
    for image, label in zip(images, labels, strict=True):
        center = encoder.encode(image)
        logits = propagate_box(network, center, center).low[-1]
        prediction = int(np.argmax(logits))
        predictions.append(prediction)
        for eps in epsilons:
            low, high = encoder.byte_box(image, eps)
            trace = propagate_box(network, encoder.encode(low), encoder.encode(high))
            bound, clamp_safe = difference_row_margin(network, trace, int(label))
            hidden = trace.high[-2] - trace.low[-2]
            per_epsilon[eps].append({
                "correct": prediction == label,
                "box_margin": box_margin(trace, int(label)),
                "difference_row_bound": bound,
                "clamp_safe": clamp_safe,
                "last_hidden_median_width": float(np.median(hidden)),
                "last_hidden_saturated": int(np.sum(trace.high[-2] == 32767)),
            })
    labels = np.asarray(labels)
    result = {"fixed_accuracy": float(np.mean(np.asarray(predictions) == labels)), "epsilons": {}}
    for eps, rows in per_epsilon.items():
        correct = np.asarray([r["correct"] for r in rows])
        box_ok = correct & (np.asarray([r["box_margin"] for r in rows]) > 0)
        bounds = np.asarray([r["difference_row_bound"] for r in rows])
        row_ok = correct & (bounds > 0) & np.asarray([r["clamp_safe"] for r in rows])
        result["epsilons"][str(eps)] = {
            "box_certified_fraction": float(np.mean(box_ok)),
            "difference_row_certified_fraction": float(np.mean(row_ok)),
            "median_difference_row_bound": float(np.median(bounds)),
            "median_last_hidden_width": float(np.median(
                [r["last_hidden_median_width"] for r in rows])),
            "mean_last_hidden_saturated": float(np.mean(
                [r["last_hidden_saturated"] for r in rows])),
        }
    return result


def load_validation(base, input_shape, resize_mode, count, seed):
    rows = [row for row in base["records"] if row["split"] == "validation"]
    chosen = np.random.default_rng(seed).choice(len(rows), size=min(count, len(rows)), replace=False)
    encoder = ByteImageEncoder(input_shape, fractional_bits=8, total_bits=16, resize_mode=resize_mode)
    root = Path(base["dataset_root"])
    # Raw crops, so byte_box/encode follow the exact path the proof runner uses.
    images = [load_crop(root, rows[i]) for i in sorted(chosen)]
    labels = [int(rows[i]["class_id"]) for i in sorted(chosen)]
    return encoder, images, labels


def measure(search_output: Path, count: int, epsilons, seed: int, resize_mode: str):
    search = json.loads((Path(search_output) / "search_summary.json").read_text())
    selected = search["selected_candidate_summary"]
    if selected is None:
        raise ValueError("Search output has no selected candidate")
    params = Path(selected["folded_model_path"])
    if _sha256(params) != selected["folded_model_sha256"]:
        raise ValueError("Candidate identity changed")
    base = json.loads(Path(search["base_study_path"]).read_text())
    model = restricted_from_npz(selected, params)
    network = quantize_restricted_sequential(
        model, [Q16] * (len(model.conv_kernels) + len(model.dense_kernels)),
        input_fractional_bits=8, input_total_bits=16,
    )
    encoder, images, labels = load_validation(base, model.input_shape, resize_mode, count, seed)
    return {
        "schema": "ssv_integer_margin_measurement_v1",
        "claim": "measured_not_certified",
        "split": "validation",
        "candidate_id": selected["candidate_id"],
        "folded_model_sha256": selected["folded_model_sha256"],
        "images": len(images),
        "sample_seed": seed,
        **measure_network(network, encoder, images, labels, epsilons),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--epsilons", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resize-mode", default="nearest_center")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.search_output, args.images, args.epsilons, args.seed, args.resize_mode)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
