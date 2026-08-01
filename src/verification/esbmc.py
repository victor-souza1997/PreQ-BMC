from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Literal
import os
import re
import signal
import subprocess
import time

from utils.logging_utils import get_logger
from verification.esbmc_install import resolve_esbmc_executable

LOGGER = get_logger(__name__)

ESBMCProfile = Literal[
    "fast",
    "paper-fast",
    "paper-z3",
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
    memory_poll_interval_seconds: float = 0.05


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
    peak_memory_bytes: int | None = None
    memory_measurement: str = "unavailable"
    resource_control: dict[str, Any] | None = None
    blocks: tuple[dict[str, Any], ...] = ()
    counterexample_inputs: list[int] | None = None
    counterexample_neuron: int | None = None
    counterexample_preclamp: int | None = None
    trace_command: tuple[str, ...] = ()
    trace_stdout_log_path: str = ""
    trace_stderr_log_path: str = ""


def _parse_counterexample_lines(
    lines: Iterable[str],
) -> tuple[list[int] | None, int | None, int | None]:
    input_values: dict[int, int] = {}
    neuron: int | None = None
    preclamp_values: list[int] = []

    input_pattern = re.compile(
        r"^\s*(?:input|in_|counterexample_input)\[(\d+)\]\s*=\s*(-?\d+)(?:\s|$)"
    )
    neuron_pattern = re.compile(
        r"^\s*(?:counterexample_neuron|violating_neuron|max_other_index)\s*=\s*(-?\d+)(?:\s|$)"
    )
    preclamp_pattern = re.compile(
        r"^\s*(?:counterexample_preclamp|pre_clamp|value)\s*=\s*(-?\d+)(?:\s|$)"
    )

    for line in lines:
        input_match = input_pattern.match(line)
        if input_match:
            input_values[int(input_match.group(1))] = int(input_match.group(2))
            continue
        neuron_match = neuron_pattern.match(line)
        if neuron_match:
            neuron = int(neuron_match.group(1))
            continue
        preclamp_match = preclamp_pattern.match(line)
        if preclamp_match:
            preclamp_values.append(int(preclamp_match.group(1)))

    inputs: list[int] | None = None
    if input_values:
        maximum_index = max(input_values)
        if set(input_values) == set(range(maximum_index + 1)):
            inputs = [input_values[index] for index in range(maximum_index + 1)]

    preclamp = (
        max(preclamp_values, key=lambda value: abs(int(value)))
        if preclamp_values
        else None
    )
    return inputs, neuron, preclamp


def parse_counterexample_trace(
    trace: str,
) -> tuple[list[int] | None, int | None, int | None]:
    """Defensively extract generated-harness values from an ESBMC trace."""

    return _parse_counterexample_lines(trace.splitlines())


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
      paper-z3:
        Low-noise paper profile using Z3 for formulas where Bitwuzla stalls.
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
        executable = resolve_esbmc_executable(self.config.executable) or self.config.executable
        solver_flag = "--z3" if profile == "paper-z3" else "--bitwuzla"
        command: list[str] = [
            executable,
            str(c_file),
            "--function",
            "main",
            "--unwind",
            str(unwind),
            solver_flag,
            "--bv",
            "--timeout",
            str(self.config.timeout_seconds),
        ]

        if self.config.memlimit:
            command.extend(["--memlimit", str(self.config.memlimit)])

        if profile in ("paper-fast", "paper-z3"):
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
        peak_memory_bytes: int | None,
        memory_measurement: str,
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
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_mib": (
                float(peak_memory_bytes / (1024 * 1024))
                if peak_memory_bytes is not None
                else None
            ),
            "memory_measurement": memory_measurement,
        }

    def _capture_counterexample(
        self,
        *,
        c_file: Path,
        base_command: tuple[str, ...],
    ) -> tuple[
        list[int] | None,
        int | None,
        int | None,
        tuple[str, ...],
        Path,
        Path,
    ]:
        """Re-run a failed query with ESBMC's supported state-trace output."""

        trace_command_values = [
            argument for argument in base_command if argument != "--result-only"
        ]
        trace_command = tuple(trace_command_values)
        stdout_path = self._log_path(c_file, "trace.stdout")
        stderr_path = self._log_path(c_file, "trace.stderr")
        process: subprocess.Popen[Any] | None = None

        try:
            with stdout_path.open(
                "w", encoding="utf-8", errors="replace"
            ) as stdout_log, stderr_path.open(
                "w", encoding="utf-8", errors="replace"
            ) as stderr_log:
                process = subprocess.Popen(
                    trace_command,
                    stdout=stdout_log,
                    stderr=stderr_log,
                    text=True,
                    start_new_session=os.name == "posix",
                )
                self._wait_with_peak_memory(
                    process,
                    self.config.timeout_seconds + 300,
                )

            # ESBMC 7.11 emits the state trace on stderr, while some older
            # releases use stdout. Parse both streamed files defensively.
            with stdout_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as stdout_trace, stderr_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as stderr_trace:
                inputs, neuron, preclamp = _parse_counterexample_lines(
                    chain(stdout_trace, stderr_trace)
                )
            if inputs is None:
                LOGGER.warning(
                    "ESBMC trace for %s did not contain a complete generated input vector.",
                    c_file,
                )
            return (
                inputs,
                neuron,
                preclamp,
                trace_command,
                stdout_path,
                stderr_path,
            )
        except Exception as exc:  # noqa: BLE001 - trace extraction is best effort.
            if process is not None:
                self._terminate_process_tree(process)
            LOGGER.warning(
                "Could not extract ESBMC counterexample from %s: %s",
                c_file,
                exc,
            )
            return None, None, None, trace_command, stdout_path, stderr_path

    @staticmethod
    def _process_tree_rss_bytes(root_pid: int) -> tuple[int | None, str]:
        """Return current Linux RSS for a process and all visible descendants."""

        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return None, "unavailable_non_linux"

        total_bytes = 0
        measured = False
        pending = [int(root_pid)]
        visited: set[int] = set()
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)

            status_path = proc_root / str(pid) / "status"
            try:
                for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("VmRSS:"):
                        fields = line.split()
                        if len(fields) >= 2:
                            total_bytes += int(fields[1]) * 1024
                            measured = True
                        break
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                pass

            children_path = proc_root / str(pid) / "task" / str(pid) / "children"
            try:
                pending.extend(int(value) for value in children_path.read_text().split())
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                pass

        return (total_bytes if measured else None), "linux_procfs_process_tree_rss"

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _wait_with_peak_memory(
        self,
        process: subprocess.Popen[Any],
        timeout_seconds: float,
    ) -> tuple[int, int | None, str, bool]:
        deadline = time.monotonic() + float(timeout_seconds)
        peak_memory_bytes: int | None = None
        measurement = "unavailable"
        poll_interval = max(0.01, float(self.config.memory_poll_interval_seconds))

        while True:
            current_memory, current_measurement = self._process_tree_rss_bytes(process.pid)
            measurement = current_measurement
            if current_memory is not None:
                peak_memory_bytes = max(peak_memory_bytes or 0, current_memory)

            return_code = process.poll()
            if return_code is not None:
                return int(return_code), peak_memory_bytes, measurement, False

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_process_tree(process)
                return -1, peak_memory_bytes, measurement, True
            time.sleep(min(poll_interval, remaining))

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
        *,
        extract_counterexample: bool = False,
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
        peak_memory_bytes: int | None = None
        memory_measurement = "unavailable"
        process: subprocess.Popen[Any] | None = None
        try:
            with stdout_log_path.open("w", encoding="utf-8", errors="replace") as stdout_log, stderr_log_path.open(
                "w",
                encoding="utf-8",
                errors="replace",
            ) as stderr_log:
                process = subprocess.Popen(
                    command,
                    stdout=stdout_log,
                    stderr=stderr_log,
                    text=True,
                    start_new_session=os.name == "posix",
                )
                return_code, peak_memory_bytes, memory_measurement, timed_out = self._wait_with_peak_memory(
                    process,
                    self.config.timeout_seconds + 300,
                )
        except Exception as exc:
            if process is not None:
                self._terminate_process_tree(process)
            elapsed_seconds = time.monotonic() - start_time
            stdout_tail = self._tail_file(stdout_log_path)
            stderr_tail = f"{self._tail_file(stderr_log_path)}\n{exc}"
            status = "ERROR"
            resource_control = self._resource_control(
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                elapsed_seconds=elapsed_seconds,
                return_code=-1,
                status=status,
                command=command,
                peak_memory_bytes=peak_memory_bytes,
                memory_measurement=memory_measurement,
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
                peak_memory_bytes=peak_memory_bytes,
                memory_measurement=memory_measurement,
                resource_control=resource_control,
            )

        if timed_out:
            elapsed_seconds = time.monotonic() - start_time
            status = "TIMEOUT"
            stdout_tail = self._tail_file(stdout_log_path)
            stderr_tail = self._tail_file(stderr_log_path)
            resource_control = self._resource_control(
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                elapsed_seconds=elapsed_seconds,
                return_code=-1,
                status=status,
                command=command,
                peak_memory_bytes=peak_memory_bytes,
                memory_measurement=memory_measurement,
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
                peak_memory_bytes=peak_memory_bytes,
                memory_measurement=memory_measurement,
                resource_control=resource_control,
            )

        elapsed_seconds = time.monotonic() - start_time
        stdout_tail = self._tail_file(stdout_log_path)
        stderr_tail = self._tail_file(stderr_log_path)

        LOGGER.debug("ESBMC return code: %s", return_code)
        LOGGER.debug("--- STDOUT tail ---\n%s", stdout_tail[-20000:])
        LOGGER.debug("--- STDERR tail ---\n%s", stderr_tail[-20000:])

        combined_output = f"{stdout_tail}\n{stderr_tail}"
        status = self._classify_status(combined_output, int(return_code))
        resource_control = self._resource_control(
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            elapsed_seconds=elapsed_seconds,
            return_code=int(return_code),
            status=status,
            command=command,
            peak_memory_bytes=peak_memory_bytes,
            memory_measurement=memory_measurement,
        )
        counterexample_inputs: list[int] | None = None
        counterexample_neuron: int | None = None
        counterexample_preclamp: int | None = None
        trace_command: tuple[str, ...] = ()
        trace_stdout_log_path = ""
        trace_stderr_log_path = ""
        if status == "FAILED" and extract_counterexample:
            (
                counterexample_inputs,
                counterexample_neuron,
                counterexample_preclamp,
                trace_command,
                trace_stdout_path,
                trace_stderr_path,
            ) = self._capture_counterexample(
                c_file=c_file,
                base_command=command,
            )
            trace_stdout_log_path = str(trace_stdout_path)
            trace_stderr_log_path = str(trace_stderr_path)
            resource_control["counterexample_trace"] = {
                "command": list(trace_command),
                "stdout_log_path": trace_stdout_log_path,
                "stderr_log_path": trace_stderr_log_path,
            }

        return ESBMCResult(
            status=status,
            command=command,
            stdout=stdout_tail,
            stderr=stderr_tail,
            return_code=int(return_code),
            elapsed_seconds=elapsed_seconds,
            timeout_seconds=int(self.config.timeout_seconds),
            memlimit=str(self.config.memlimit),
            stdout_log_path=str(stdout_log_path),
            stderr_log_path=str(stderr_log_path),
            peak_memory_bytes=peak_memory_bytes,
            memory_measurement=memory_measurement,
            resource_control=resource_control,
            counterexample_inputs=counterexample_inputs,
            counterexample_neuron=counterexample_neuron,
            counterexample_preclamp=counterexample_preclamp,
            trace_command=trace_command,
            trace_stdout_log_path=trace_stdout_log_path,
            trace_stderr_log_path=trace_stderr_log_path,
        )
