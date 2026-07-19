from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_external_baselines(path: str | Path | None) -> list[dict[str, Any]]:
    """Load optional external baseline results from JSON or CSV."""

    if path is None:
        return []

    baseline_path = Path(path)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline results file does not exist: {baseline_path}")

    if baseline_path.suffix.lower() == ".json":
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                return [dict(item) for item in payload["results"]]
            return [payload]
        raise ValueError(f"Unsupported JSON baseline payload in {baseline_path}")

    if baseline_path.suffix.lower() == ".csv":
        with baseline_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    raise ValueError(f"Unsupported baseline file extension: {baseline_path.suffix}")
