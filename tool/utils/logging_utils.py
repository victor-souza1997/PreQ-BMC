from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    """Configure a simple root logger once for CLI tools and scripts."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""

    return logging.getLogger(name)
