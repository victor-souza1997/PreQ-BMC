"""Local official GTSRB files, track-safe validation, outcome-blind region selection."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_tracks(records, *, seed=2026, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be in (0,1)")
    result = [dict(row) for row in records]
    for label in sorted({row["class_id"] for row in result}):
        tracks = {row["track_id"] for row in result if row["class_id"] == label}
        if len(tracks) < 2:
            raise ValueError("Need at least two physical-sign tracks per class")
        ordered = sorted(tracks, key=lambda track: hashlib.sha256(f"{seed}:{label}:{track}".encode()).hexdigest())
        count = min(len(tracks) - 1, max(1, round(len(tracks) * validation_fraction)))
        validation = set(ordered[:count])
        for row in result:
            if row["class_id"] == label:
                row["split"] = "validation" if row["track_id"] in validation else "train"
    return result


def official_records(root: Path):
    """Paths are relative to an extracted official archive root; no train/test mixing."""
    root = Path(root).resolve()
    train = root / "Final_Training" / "Images"
    records = []
    for label in range(43):
        folder = train / f"{label:05d}"
        with (folder / f"GT-{label:05d}.csv").open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                path = folder / row["Filename"]
                if int(row["ClassId"]) != label or path.parent != folder or "_" not in path.stem:
                    raise ValueError("Unexpected official training row")
                records.append({"id": str(path.relative_to(root)), "class_id": label,
                                "track_id": f"{label}:{path.stem.split('_')[0]}",
                                "sha256": sha256(path), "split": "train"})
    records = split_tracks(records)
    with (root / "GT-final_test.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            path = root / "Final_Test" / "Images" / row["Filename"]
            if path.parent != root / "Final_Test" / "Images" or not 0 <= int(row["ClassId"]) < 43:
                raise ValueError("Unexpected official test row")
            records.append({"id": str(path.relative_to(root)), "class_id": int(row["ClassId"]),
                            "track_id": None, "sha256": sha256(path), "split": "test"})
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate image identity")
    return records


def load_crop(root, row):
    from PIL import Image
    root = Path(root).resolve()
    path = (root / row["id"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != row["sha256"]:
        raise ValueError("Dataset identity mismatch")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("Expected official RGB crop; implicit color conversion is unsupported")
        return np.asarray(image, dtype=np.uint8)


def select_regions(records, logits, *, per_stratum=3):
    """Freeze distinct, correctly classified test images before any solver call."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != len(records) or logits.shape[1] < 2 or not np.all(np.isfinite(logits)):
        raise ValueError("Invalid clean float logits")
    if per_stratum <= 0 or any(row["split"] != "test" for row in records):
        raise ValueError("Selection must use the untouched test split")
    eligible = []
    for row, values in zip(records, logits):
        prediction = int(np.argmax(values))
        if prediction == row["class_id"]:
            margin = float(values[prediction] - np.max(np.delete(values, prediction)))
            eligible.append({**row, "predicted_label": prediction, "clean_margin": margin})
    eligible.sort(key=lambda row: (row["clean_margin"], row["id"]))
    if len(eligible) < 3 * per_stratum:
        raise ValueError("Insufficient correctly classified test images; do not replace by proof outcomes")
    selected = []
    for name, indices in zip(("low", "median", "high"), np.array_split(np.arange(len(eligible)), 3)):
        # Within each tertile, prefer the lowest-margin distinct classes first.
        candidates = [eligible[int(i)] for i in indices]
        chosen, classes = [], set()
        for row in candidates:
            if row["class_id"] not in classes:
                chosen.append(row)
                classes.add(row["class_id"])
            if len(chosen) == per_stratum:
                break
        for row in candidates:
            if len(chosen) == per_stratum:
                break
            if row not in chosen:
                chosen.append(row)
        selected.extend({**row, "stratum": name} for row in chosen)
    if len({row["class_id"] for row in selected}) < 2:
        raise ValueError("Selected regions must cover multiple classes")
    return selected
