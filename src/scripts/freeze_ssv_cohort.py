"""Derive a disjoint, outcome-blind GTSRB cohort from an existing frozen study.

The source weights and all verification settings remain immutable. This command
only evaluates the already-trained float32 Keras model on the untouched test
split and selects a different, predeclared rank within each clean-margin tertile.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop, select_regions, sha256
from scripts.prepare_ssv_gtsrb import validate_config
from scripts.run_ssv_cnn_gate import write_new_json


def freeze(study_path, output, *, rank_offset_per_stratum):
    import tensorflow as tf
    study_path, output = Path(study_path), Path(output)
    study = json.loads(study_path.read_text())
    geom = validate_config(study["config"])
    if type(rank_offset_per_stratum) is not int or rank_offset_per_stratum <= 0:
        raise ValueError("A nonzero integer rank offset is required for a new cohort")
    if sha256(study["model_path"]) != study["model_sha256"]:
        raise ValueError("Source model identity mismatch")
    keras_path = Path(study["model_path"]).parent / "source.keras"
    if not keras_path.is_file():
        raise ValueError("Existing study has no frozen Keras source model for cohort selection")
    test = [row for row in study["records"] if row["split"] == "test"]
    encoder = ByteImageEncoder(geom.input_shape)
    inputs = np.stack([encoder.resize(load_crop(study["dataset_root"], row)) for row in test]).astype(np.float32) / 256
    model = tf.keras.models.load_model(keras_path, compile=False)
    logits = np.asarray(model.predict(inputs, batch_size=128, verbose=0))
    selected = select_regions(test, logits, per_stratum=study["config"]["selection"]["per_stratum"],
                              rank_offset_per_stratum=rank_offset_per_stratum)
    old = {row["id"] for row in study["selected_images"]}
    new = {row["id"] for row in selected}
    if old & new:
        raise ValueError("Requested rank offset overlaps the parent cohort; select a different offset")
    output.mkdir(parents=True, exist_ok=False)
    updated = copy.deepcopy(study)
    updated.update({
        "selected_images": selected,
        "runs": [],
        "cohort_kind": "outcome_blind_disjoint_margin_tertiles",
        "selection_rank_offset_per_stratum": rank_offset_per_stratum,
        "parent_study": str(study_path.resolve()),
        "parent_study_sha256": sha256(study_path),
        "parent_selected_image_ids": sorted(old),
        "selection_source_model": str(keras_path.resolve()),
        "selection_source_model_sha256": sha256(keras_path),
        "proof_status": "NOT_RUN",
    })
    # Recreate the same candidate matrix, with selected images replaced but no
    # verification outcome consulted. `freeze_ssv_campaign` narrows this to 36.
    for index, row in enumerate(selected):
        for epsilon in study["config"]["epsilon_raw_bytes"]:
            for beta in study["config"]["block_sizes"]:
                for cuts in study["config"]["margin_cuts"]:
                    updated["runs"].append({
                        "run_id": f"cohort2_image{index}_eps{epsilon}_beta{beta}_cuts{int(cuts)}",
                        "sample": row, "epsilon": epsilon, "block_size": beta,
                        "margin_cuts": cuts, **study["config"]["verification"],
                    })
    write_new_json(output / "study.json", updated)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank-offset-per-stratum", type=int, default=3)
    args = parser.parse_args()
    freeze(args.study, args.output, rank_offset_per_stratum=args.rank_offset_per_stratum)
    print(args.output / "study.json")


if __name__ == "__main__":
    main()
