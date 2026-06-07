from __future__ import annotations

from collections import deque
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
    "paper-fast",
    "preimage",
    "safety",
    "overflow",
    "debug",
]


@dataclass(frozen=True)
class ESBMCConfig:
    """Configuration for the ESBMC command-line runner."""

    executable: str = "esbmc"
    timeout_seconds: int = 900
    memlimit: str = "6g"
    verbosity: int = 10
    default_profile: ESBMCProfile = "paper-fast"
    tail_lines: int = 100


@dataclass(frozen=True)
class ESBMCResult:
    """Normalized ESBMC execution result."""

    status: str
    command: tuple[str, ...]
    stdout: str
    stderr: str
    return_code: int
    elapsed_seconds: float = 0.0
    timeout_seconds: int = 900
    memlimit: str = "6g"
    stdout_log_path: str = ""
    stderr_log_path: str = ""
    resource_control: dict[str, Any] | None = None
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
            "--bitwuzla",
            "--bv",
            "--timeout",
            str(self.config.timeout_seconds),
        ]

        if self.config.memlimit:
            command.extend(["--memlimit", str(self.config.memlimit)])

        if profile == "paper-fast":
            command.extend(
                [
                    "--interval-analysis",
                    "--interval-analysis-simplify",
                    "--result-only",
                ]
            )
        elif profile in ("preimage", "safety", "overflow"):
            command.extend(
                [
                   "--interval-analysis",
                   "--interval-analysis-simplify"
                ]
            )
        elif profile == "debug":
            command.extend(
                [
                    "--verbosity",
                    str(self.config.verbosity),
                    "--print-stack-traces",
                    "--memstats",
                    "--show-claims",
                ]
            )

        # Important for the paper:
        # Do NOT add --no-bounds-check, --no-div-by-zero-check or --no-pointer-check.
        # ESBMC checks several of these properties by default.
        if profile in ("safety", "overflow"):
            command.append("--overflow-check")

        if profile in ("fast", "preimage", "safety", "overflow"):
            command.append("--force-malloc-success")

        return tuple(command)

    def _tail_file(self, path: Path) -> str:
        if not path.exists():
            return ""
        lines: deque[str] = deque(maxlen=max(1, int(self.config.tail_lines)))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lines.append(line)
        return "".join(lines)

    @staticmethod
    def _log_path(c_file: Path, stream_name: str) -> Path:
        return Path(f"{c_file}.{stream_name}.log")

    def _resource_control(
        self,
        *,
        stdout_log_path: Path,
        stderr_log_path: Path,
        elapsed_seconds: float,
        return_code: int,
        status: str,
        command: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "command": list(command),
            "timeout": f"{int(self.config.timeout_seconds)}s",
            "memlimit": str(self.config.memlimit),
            "elapsed_seconds": float(elapsed_seconds),
            "return_code": int(return_code),
            "status": status,
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
        }

    @staticmethod
    def _classify_status(combined_output: str, return_code: int) -> str:
        lower_output = combined_output.lower()
        memory_markers = (
            "memory limit",
            "out of memory",
            "std::bad_alloc",
            "bad_alloc",
            "cannot allocate memory",
            "killed",
        )
        timeout_markers = (
            "timed out",
            "timeout",
            "time limit",
        )

        if "VERIFICATION SUCCESSFUL" in combined_output:
            return "VERIFIED"
        if "VERIFICATION FAILED" in combined_output:
            return "FAILED"
        if any(marker in lower_output for marker in memory_markers) or return_code in {-9, 137}:
            return "MEMOUT"
        if return_code == 124 or any(marker in lower_output for marker in timeout_markers):
            return "TIMEOUT"
        return "UNKNOWN"

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
        stdout_log_path = self._log_path(c_file, "stdout")
        stderr_log_path = self._log_path(c_file, "stderr")

        LOGGER.info(
            "Running ESBMC on %s with profile=%s and unwind=%s",
            c_file,
            selected_profile,
            unwind,
        )
        start_time = time.monotonic()
        try:
            with stdout_log_path.open("w", encoding="utf-8", errors="replace") as stdout_log, stderr_log_path.open(
                "w",
                encoding="utf-8",
                errors="replace",
            ) as stderr_log:
                completed = subprocess.run(
                    command,
                    stdout=stdout_log,
                    stderr=stderr_log,
                    text=True,
                    timeout=self.config.timeout_seconds + 300,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            del exc
            elapsed_seconds = time.monotonic() - start_time
            stdout_tail = self._tail_file(stdout_log_path)
            stderr_tail = self._tail_file(stderr_log_path)
            status = "TIMEOUT"
            resource_control = self._resource_control(
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                elapsed_seconds=elapsed_seconds,
                return_code=-1,
                status=status,
                command=command,
            )
            return ESBMCResult(
                status=status,
                command=command,
                stdout=stdout_tail,
                stderr=stderr_tail,
                return_code=-1,
                elapsed_seconds=elapsed_seconds,
                timeout_seconds=int(self.config.timeout_seconds),
                memlimit=str(self.config.memlimit),
                stdout_log_path=str(stdout_log_path),
                stderr_log_path=str(stderr_log_path),
                resource_control=resource_control,
            )
        except Exception as exc:
            elapsed_seconds = time.monotonic() - start_time
            status = "ERROR"
            stdout_tail = self._tail_file(stdout_log_path)
            stderr_tail = f"{self._tail_file(stderr_log_path)}\n{exc}"
            resource_control = self._resource_control(
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                elapsed_seconds=elapsed_seconds,
                return_code=-1,
                status=status,
                command=command,
            )
            return ESBMCResult(
                status=status,
                command=command,
                stdout=stdout_tail,
                stderr=stderr_tail,
                return_code=-1,
                elapsed_seconds=time.monotonic() - start_time,
                timeout_seconds=int(self.config.timeout_seconds),
                memlimit=str(self.config.memlimit),
                stdout_log_path=str(stdout_log_path),
                stderr_log_path=str(stderr_log_path),
                resource_control=resource_control,
            )
        elapsed_seconds = time.monotonic() - start_time
        stdout_tail = self._tail_file(stdout_log_path)
        stderr_tail = self._tail_file(stderr_log_path)

        LOGGER.debug("ESBMC return code: %s", completed.returncode)
        LOGGER.debug("--- STDOUT tail ---\n%s", stdout_tail[-20000:])
        LOGGER.debug("--- STDERR tail ---\n%s", stderr_tail[-20000:])

        combined_output = f"{stdout_tail}\n{stderr_tail}"
        status = self._classify_status(combined_output, int(completed.returncode))
        resource_control = self._resource_control(
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            elapsed_seconds=elapsed_seconds,
            return_code=int(completed.returncode),
            status=status,
            command=command,
        )

        return ESBMCResult(
            status=status,
            command=command,
            stdout=stdout_tail,
            stderr=stderr_tail,
            return_code=int(completed.returncode),
            elapsed_seconds=elapsed_seconds,
            timeout_seconds=int(self.config.timeout_seconds),
            memlimit=str(self.config.memlimit),
            stdout_log_path=str(stdout_log_path),
            stderr_log_path=str(stderr_log_path),
            resource_control=resource_control,
        )
