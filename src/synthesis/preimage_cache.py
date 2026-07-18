from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


PREIMAGE_CACHE_FORMAT = "quadapter-preimage-cache-v1"


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _slug(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return text or "value"


def fingerprint_file(path: Path) -> str:
    """Return a stable content hash for cache identity."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preimage_cache_identity(
    *,
    dataset: str,
    arch: str,
    sample_id: int,
    eps: float,
    preimg_mode: str,
    if_relax: bool,
    target_label: int,
    valid_labels: tuple[int, ...] | None,
    weights_path: Path,
) -> tuple[str, dict[str, Any]]:
    """Build the default cache key and metadata used by export/load paths."""

    payload = {
        "dataset": dataset,
        "arch": arch,
        "sample_id": int(sample_id),
        "eps": float(eps),
        "preimg_mode": preimg_mode,
        "if_relax": bool(if_relax),
        "target_label": int(target_label),
        "valid_labels": list(valid_labels) if valid_labels is not None else None,
        "weights_sha256": fingerprint_file(weights_path),
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]
    key = "__".join(
        [
            _slug(dataset),
            _slug(arch),
            f"sample{int(sample_id)}",
            f"eps{float(eps):g}",
            _slug(preimg_mode),
            digest,
        ]
    )
    return key, payload


def save_preimage_cache(
    *,
    cache_root: Path,
    cache_key: str,
    layers: list[dict[str, Any]],
    scale_values: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    """Persist preimage bounds in a fast portable format."""

    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "scale_values": np.asarray(scale_values, dtype=np.float64),
        "layer_indices": np.asarray([layer["layer_index"] for layer in layers], dtype=np.int64),
        "layer_sizes": np.asarray([layer["layer_size"] for layer in layers], dtype=np.int64),
    }
    for offset, layer in enumerate(layers):
        arrays[f"relaxed_lb_{offset}"] = np.asarray(layer["relaxed_lb"], dtype=np.float64)
        arrays[f"relaxed_ub_{offset}"] = np.asarray(layer["relaxed_ub"], dtype=np.float64)

    np.savez(cache_dir / "preimage.npz", **arrays)
    metadata_payload = {
        "format": PREIMAGE_CACHE_FORMAT,
        "array_format": "npz",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "metadata": metadata,
        "layers": [
            {
                "layer_index": int(layer["layer_index"]),
                "layer_size": int(layer["layer_size"]),
            }
            for layer in layers
        ],
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")
    return cache_dir


def load_preimage_cache(*, cache_root: Path, cache_key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_dir = cache_root / cache_key
    metadata_path = cache_dir / "metadata.json"
    arrays_path = cache_dir / "preimage.npz"
    if not metadata_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(
            f"Preimage cache '{cache_key}' was not found under {cache_root}. "
            f"Expected {metadata_path} and {arrays_path}."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != PREIMAGE_CACHE_FORMAT:
        raise ValueError(
            f"Unsupported preimage cache format in {metadata_path}: {metadata.get('format')!r}. "
            f"Expected {PREIMAGE_CACHE_FORMAT!r}."
        )

    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    return metadata, arrays
