from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ESBMCConfig:
    """Configuration for the ESBMC command-line runner."""

    executable: str = "esbmc"
    timeout_seconds: int = 900
    verbosity: int = 10


@dataclass(frozen=True)
class ESBMCResult:
    """Normalized ESBMC execution result."""

    status: str
    command: tuple[str, ...]
    stdout: str
    stderr: str
    return_code: int


class ESBMCRunner:
    """Run ESBMC using the legacy command line used by the original pipeline."""

    def __init__(self, config: ESBMCConfig | None = None) -> None:
        self.config = config or ESBMCConfig()

    def infer_unwind(self, source: str) -> int:
        unwind = 0
        match_input = re.search(r"#define\s+INPUT_SIZE\s+(\d+)", source)
        match_layer = re.search(r"#define\s+LAYER_SIZE\s+(\d+)", source)
        if match_input:
            unwind = max(unwind, int(match_input.group(1)))
        if match_layer:
            unwind = max(unwind, int(match_layer.group(1)))
        return max(unwind, 1) + 1

    def run_file(self, c_file: Path) -> ESBMCResult:
        source = c_file.read_text(encoding="utf-8", errors="replace")
        unwind = self.infer_unwind(source)
        command = (
            self.config.executable,
            str(c_file),
            "--loop-invariant",
            "--function",
            "main",
            "--interval-analysis",
            "--unwind",
            str(unwind),
            "--incremental-bmc",
            "--state-hashing",
            "--force-malloc-success",
            "--overflow-check",
            "--timeout",
            str(self.config.timeout_seconds),
            "--verbosity",
            str(self.config.verbosity),
            "--print-stack-traces",
        )

        LOGGER.info("Running ESBMC on %s with unwind=%s", c_file, unwind)
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.config.timeout_seconds + 300,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ESBMCResult(
                status="TIMEOUT",
                command=command,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                return_code=-1,
            )
        except Exception as exc:
            return ESBMCResult(
                status="ERROR",
                command=command,
                stdout="",
                stderr=str(exc),
                return_code=-1,
            )

        LOGGER.debug("ESBMC return code: %s", completed.returncode)
        LOGGER.debug("--- STDOUT (tail) ---\n%s", (completed.stdout or "")[-20000:])
        LOGGER.debug("--- STDERR (tail) ---\n%s", (completed.stderr or "")[-20000:])

        if "VERIFICATION SUCCESSFUL" in completed.stderr:
            status = "VERIFIED"
        elif "VERIFICATION FAILED" in completed.stdout:
            status = "FAILED"
        else:
            status = "UNKNOWN"

        return ESBMCResult(
            status=status,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )
