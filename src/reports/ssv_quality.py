"""Streaming empirical accuracy comparisons, independent of formal status."""
from dataclasses import dataclass

import numpy as np


@dataclass
class QualityComparison:
    n: int = 0
    source_correct: int = 0
    candidate_correct: int = 0
    mismatches: int = 0
    regressions: int = 0
    recoveries: int = 0
    error_sum: float = 0.0
    error_count: int = 0
    max_error: float = 0.0

    def add(self, source, candidate, label):
        source, candidate = np.asarray(source), np.asarray(candidate)
        if (source.ndim != 1 or source.size < 2 or source.shape != candidate.shape
                or not np.all(np.isfinite(source)) or not np.all(np.isfinite(candidate))
                or not 0 <= label < source.size):
            raise ValueError("Need finite matching logit vectors and an in-range label")
        source_label, candidate_label = int(source.argmax()), int(candidate.argmax())
        source_ok, candidate_ok = source_label == label, candidate_label == label
        self.n += 1
        self.source_correct += int(source_ok)
        self.candidate_correct += int(candidate_ok)
        self.mismatches += int(source_label != candidate_label)
        self.regressions += int(source_ok and not candidate_ok)
        self.recoveries += int(not source_ok and candidate_ok)
        errors = np.abs(source.astype(np.float64) - candidate)
        self.error_sum += float(errors.sum())
        self.error_count += errors.size
        self.max_error = max(self.max_error, float(errors.max()))

    def summary(self):
        if not self.n:
            raise ValueError("Cannot report accuracy without measurements")
        return {"n_images": self.n, "source_accuracy": self.source_correct / self.n,
                "candidate_accuracy": self.candidate_correct / self.n,
                "accuracy_drop_percentage_points": 100 * (self.source_correct - self.candidate_correct) / self.n,
                "prediction_mismatch_rate": self.mismatches / self.n,
                "source_correct_candidate_wrong_count": self.regressions,
                "source_wrong_candidate_correct_count": self.recoveries,
                "max_abs_logit_error": self.max_error,
                "mean_abs_logit_error": self.error_sum / self.error_count}
