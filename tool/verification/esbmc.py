from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from pathlib import Path

from utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ESBMCConfig:
    """Configuration for the ESBMC command-line runner."""

    executable: str = "esbmc"
    timeout_seconds: int = 900
    verbosity: int = 2


@dataclass(frozen=True)
class ESBMCResult:
    """Normalized ESBMC execution result."""

    status: str
    command: tuple[str, ...]
    stdout: str
    stderr: str
    return_code: int


class ESBMCRunner:
    """Run ESBMC with an unwind bound derived from the generated C program."""

    def __init__(self, config: ESBMCConfig | None = None) -> None:
        self.config = config or ESBMCConfig()

    def infer_unwind(self, source: str) -> int:
        matches = re.findall(r"#define\s+(?:INPUT_SIZE|LAYER_SIZE)\s+(\d+)", source)
        sizes = [int(match) for match in matches]
        return max(sizes, default=1) + 1

    def run_file(self, c_file: Path) -> ESBMCResult:
        source = c_file.read_text(encoding="utf-8", errors="replace")
        unwind = self.infer_unwind(source)
        command = (
            self.config.executable,
            str(c_file),
            "--function",
            "main",
            "--unwind",
            str(unwind),
            "--timeout",
            str(self.config.timeout_seconds),
            "--verbosity",
            str(self.config.verbosity),
        )

        LOGGER.info("Running ESBMC on %s with unwind=%s", c_file, unwind)
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.timeout_seconds + 60,
            )
        except subprocess.TimeoutExpired as exc:
            return ESBMCResult(
                status="TIMEOUT",
                command=command,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                return_code=-1,
            )
        except FileNotFoundError as exc:
            return ESBMCResult(
                status="ERROR",
                command=command,
                stdout="",
                stderr=str(exc),
                return_code=-1,
            )

        combined_output = f"{completed.stdout}\n{completed.stderr}"
        if "VERIFICATION SUCCESSFUL" in combined_output:
            status = "VERIFIED"
        elif "VERIFICATION FAILED" in combined_output:
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
