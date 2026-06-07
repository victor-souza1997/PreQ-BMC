from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import re
import subprocess
import time

from utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

ESBMCProfile = Literal[
    "fast",
    "preimage",
    "safety",
    "overflow",
]


@dataclass(frozen=True)
class ESBMCConfig:
    """Configuration for the ESBMC command-line runner."""

    executable: str = "esbmc"
    timeout_seconds: int = 900
    verbosity: int = 10
    default_profile: ESBMCProfile = "preimage"


@dataclass(frozen=True)
class ESBMCResult:
    """Normalized ESBMC execution result."""

    status: str
    command: tuple[str, ...]
    stdout: str
    stderr: str
    return_code: int
    elapsed_seconds: float = 0.0
    blocks: tuple[dict[str, Any], ...] = ()


class ESBMCRunner:
    """
    Run ESBMC on generated C harnesses.

    Profiles:
      fast:
        Debug-oriented profile. Keeps the command smaller.
      preimage:
        Main profile for layer-wise preimage contract checking.
      safety:
        Enables safety-oriented checks useful for the paper.
      overflow:
        Focuses on arithmetic overflow checks.
    """

    def __init__(self, config: ESBMCConfig | None = None) -> None:
        self.config = config or ESBMCConfig()

    def infer_unwind(self, source: str) -> int:
        """
        Infer a conservative unwind bound from generated C constants.

        The old implementation only considered INPUT_SIZE and LAYER_SIZE.
        This version also considers constants from full-network generated C,
        such as LAYER_0_IN and LAYER_0_OUT.
        """
        values: list[int] = []

        patterns = [
            r"#define\s+INPUT_SIZE\s+(\d+)",
            r"#define\s+LAYER_SIZE\s+(\d+)",
            r"#define\s+OUTPUT_SIZE\s+(\d+)",
            r"#define\s+NUM_CLASSES\s+(\d+)",
            r"#define\s+NUM_VALID_CLASSES\s+(\d+)",
            r"static\s+const\s+int\s+LAYER_\d+_IN\s*=\s*(\d+)",
            r"static\s+const\s+int\s+LAYER_\d+_OUT\s*=\s*(\d+)",
        ]

        for pattern in patterns:
            values.extend(int(match) for match in re.findall(pattern, source))

        # +1 is useful because ESBMC needs enough unwinding to cover loop exit.
        return max(values, default=1) + 1

    def build_command(
        self,
        c_file: Path,
        unwind: int,
        profile: ESBMCProfile,
    ) -> tuple[str, ...]:
        command: list[str] = [
            self.config.executable,
            str(c_file),
            "--function",
            "main",
            "--unwind",
            str(unwind),
            "--state-hashing",
            "--bitwuzla",
            "--bv",
            "--timeout",
            str(self.config.timeout_seconds),
            "--verbosity",
            str(self.config.verbosity),
            "--print-stack-traces",
        ]

        # Keep these only if they are stable in your ESBMC version.
        # They were already used in your original runner.
        if profile in ("preimage", "safety", "overflow"):
            command.extend(
                [
                   "--interval-analysis",
                   "--interval-analysis-simplify"
                ]
            )

        # Important for the paper:
        # Do NOT add --no-bounds-check, --no-div-by-zero-check or --no-pointer-check.
        # ESBMC checks several of these properties by default.
        if profile in ("safety", "overflow"):
            command.append("--overflow-check")

        # Your generated programs do not appear to use malloc.
        # Keeping this does not hurt, but it is not central to the paper.
        if profile in ("fast", "preimage", "safety", "overflow"):
            command.append("--force-malloc-success")

        return tuple(command)

    def run_file(
        self,
        c_file: Path,
        profile: ESBMCProfile | None = None,
    ) -> ESBMCResult:
        source = c_file.read_text(encoding="utf-8", errors="replace")
        unwind = self.infer_unwind(source)
        selected_profile = profile or self.config.default_profile

        command = self.build_command(
            c_file=c_file,
            unwind=unwind,
            profile=selected_profile,
        )

        LOGGER.info(
            "Running ESBMC on %s with profile=%s and unwind=%s",
            c_file,
            selected_profile,
            unwind,
        )
        print(command)
        start_time = time.monotonic()
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
                elapsed_seconds=time.monotonic() - start_time,
            )
        except Exception as exc:
            return ESBMCResult(
                status="ERROR",
                command=command,
                stdout="",
                stderr=str(exc),
                return_code=-1,
                elapsed_seconds=time.monotonic() - start_time,
            )
        elapsed_seconds = time.monotonic() - start_time

        LOGGER.debug("ESBMC return code: %s", completed.returncode)
        LOGGER.debug("--- STDOUT tail ---\n%s", (completed.stdout or "")[-20000:])
        LOGGER.debug("--- STDERR tail ---\n%s", (completed.stderr or "")[-20000:])

        combined_output = f"{completed.stdout}\n{completed.stderr}"

        if "VERIFICATION SUCCESSFUL" in combined_output:
            status = "VERIFIED"
        elif "VERIFICATION FAILED" in combined_output:
            status = "FAILED"
        elif completed.returncode == 124:
            status = "TIMEOUT"
        else:
            status = "UNKNOWN"

        return ESBMCResult(
            status=status,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
            elapsed_seconds=elapsed_seconds,
        )
