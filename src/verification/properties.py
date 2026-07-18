from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationProperty:
    """Output classification property used during ESBMC verification."""

    target_label: int | None = None
    valid_labels: tuple[int, ...] | None = None

    def validate(self, num_classes: int) -> None:
        if self.target_label is None and not self.valid_labels:
            raise ValueError("Either target_label or valid_labels must be provided.")
        if self.target_label is not None and not (0 <= self.target_label < num_classes):
            raise ValueError(f"Invalid target label {self.target_label} for {num_classes} classes.")
        if self.valid_labels:
            invalid = [label for label in self.valid_labels if not (0 <= label < num_classes)]
            if invalid:
                raise ValueError(f"Invalid valid-label set {invalid} for {num_classes} classes.")

    @property
    def mode(self) -> str:
        return "valid_set" if self.valid_labels else "target"
