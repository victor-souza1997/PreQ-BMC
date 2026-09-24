"""Proof coordinator for convolution-native compositional verification.

The coordinator is intentionally strict: a network result is VERIFIED only
after every block, every chaining bridge, and the final margin obligation has
an ESBMC VERIFIED result.  Bounded feasibility pilots are always reported as
PARTIAL_NOT_CERTIFIED.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

from backends.conv_fixed_point import (
    ConvFixedPointNetwork,
    QuantizedConv2D,
    QuantizedDense,
    QuantizedOperator,
    generate_conv_qnn_source,
)
from scripts.benchmark_esbmc_concurrency import benchmark
from scripts.run_ssv_cnn_gate import write_new_json
from verification.arith_kernel import render_arith_kernel
from verification.conv_cegar import SparseRelationalCut
from verification.conv_contracts import (
    IntegerInvariant,
    LayerIntervalCertificate,
    obligation_sha256,
    dense_output_terms,
    propagate_network,
    render_conv_block_contract,
    render_invariant_bridge,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.esbmc_install import resolve_esbmc_executable
from verification.interval_lemma_parts import (
    render_clamp_relu_monotonicity_lemma,
    render_rounding_monotonicity_lemma,
)
from verification.interval_lemmas import (
    render_conv_interval_certificate,
    render_dense_interval_certificate,
)


@dataclass(frozen=True)
class ConvProofConfig:
    block_size: int = 1
    proof_mode: str = "explicit_transition"
    exact_margin_refinement: bool = False
    profile: str = "paper-z3"
    fail_fast: bool = True
    max_blocks_per_layer: int | None = None
    jobs: int = 1
    min_available_gib: float = 6.0

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("Convolution-native proofs require a positive block size")
        if type(self.exact_margin_refinement) is not bool:
            raise ValueError("exact_margin_refinement must be boolean")
        if self.proof_mode not in {"explicit_transition", "proof_carrying_interval"}:
            raise ValueError("Unsupported convolution proof mode")
        if self.max_blocks_per_layer is not None and self.max_blocks_per_layer <= 0:
            raise ValueError("max_blocks_per_layer must be positive when set")
        if self.jobs <= 0:
            raise ValueError("jobs must be positive")
        if self.min_available_gib < 0:
            raise ValueError("min_available_gib must be nonnegative")


def _c_array(values: Iterable[int]) -> str:
    return "{" + ", ".join(str(int(value)) for value in values) + "}"


def render_dense_block_contract(
    layer: QuantizedDense,
    certificate: LayerIntervalCertificate,
    output_indices: Iterable[int],
) -> str:
    outputs = tuple(int(index) for index in output_indices)
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("A dense block requires unique output indices")
    terms_by_output = [dense_output_terms(layer, index) for index in outputs]
    global_inputs = sorted({index for terms in terms_by_output for index, _ in terms}) or [0]
    local = {global_index: local_index for local_index, global_index in enumerate(global_inputs)}
    max_terms = max(1, *(len(terms) for terms in terms_by_output))
    term_counts, term_inputs, term_weights = [], [], []
    for terms in terms_by_output:
        padding = max_terms - len(terms)
        term_counts.append(len(terms))
        term_inputs.extend([local[index] for index, _ in terms] + [0] * padding)
        term_weights.extend([weight for _, weight in terms] + [0] * padding)
    states = []
    for index in outputs:
        if index in certificate.stable_active:
            states.append(1)
        elif index in certificate.stable_inactive:
            states.append(-1)
        else:
            states.append(0)
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
{render_arith_kernel()}
#define BLOCK_SIZE {len(outputs)}
#define LOCAL_INPUT_SIZE {len(global_inputs)}
#define MAX_TERMS {max_terms}
#define TOTAL_BITS {layer.spec.total_bits}
#define INPUT_FRACTIONAL_BITS {layer.input_fractional_bits}
static const int64_t INPUT_LOW[LOCAL_INPUT_SIZE] = {_c_array(certificate.input_invariant.low[i] for i in global_inputs)};
static const int64_t INPUT_HIGH[LOCAL_INPUT_SIZE] = {_c_array(certificate.input_invariant.high[i] for i in global_inputs)};
static const int TERM_COUNT[BLOCK_SIZE] = {_c_array(term_counts)};
static const int TERM_INPUT[BLOCK_SIZE * MAX_TERMS] = {_c_array(term_inputs)};
static const int64_t TERM_WEIGHT[BLOCK_SIZE * MAX_TERMS] = {_c_array(term_weights)};
static const int64_t BIAS[BLOCK_SIZE] = {_c_array(layer.bias_int[index] for index in outputs)};
static const int64_t OUTPUT_LOW[BLOCK_SIZE] = {_c_array(certificate.output_invariant.low[index] for index in outputs)};
static const int64_t OUTPUT_HIGH[BLOCK_SIZE] = {_c_array(certificate.output_invariant.high[index] for index in outputs)};
static const int STABLE_RELU[BLOCK_SIZE] = {_c_array(states)};
int main(void) {{
    int64_t input[LOCAL_INPUT_SIZE];
    for (int i = 0; i < LOCAL_INPUT_SIZE; ++i) {{
        input[i] = nondet_long_long();
        __ESBMC_assume(input[i] >= INPUT_LOW[i] && input[i] <= INPUT_HIGH[i]);
    }}
    for (int out = 0; out < BLOCK_SIZE; ++out) {{
        __int128 acc = 0;
        for (int term = 0; term < TERM_COUNT[out]; ++term) {{
            const int offset = out * MAX_TERMS + term;
            acc = mac_i128(acc, TERM_WEIGHT[offset], input[TERM_INPUT[offset]]);
        }}
        __int128 pre_relu = div_round_half_away_from_zero_i128(
            acc, ((__int128)1 << INPUT_FRACTIONAL_BITS)
        ) + (__int128)BIAS[out];
        pre_relu = clamp_to_signed_range_i128(pre_relu, TOTAL_BITS);
        if (STABLE_RELU[out] > 0)
            __ESBMC_assert(pre_relu >= 0, "stable active ReLU classification");
        if (STABLE_RELU[out] < 0)
            __ESBMC_assert(pre_relu <= 0, "stable inactive ReLU classification");
        __int128 value = pre_relu;
        if ({1 if layer.apply_relu else 0} && value < 0) value = 0;
        value = clamp_to_signed_range_i128(value, TOTAL_BITS);
        __ESBMC_assert(
            value >= OUTPUT_LOW[out] && value <= OUTPUT_HIGH[out],
            "dense output outside proposed integer invariant"
        );
    }}
    return 0;
}}
"""




def render_dense_relational_cut_validation(
    layer: QuantizedDense,
    certificate: LayerIntervalCertificate,
    cut: SparseRelationalCut,
) -> str:
    """Prove a sparse relation over exact outputs of one dense transition."""

    if any(index >= layer.output_size for index in cut.indices):
        raise ValueError("Relational cut output index is outside the dense layer")
    outputs = tuple(cut.indices)
    weights = [
        int(layer.weights_int[index, column])
        for index in outputs
        for column in range(layer.input_size)
    ]
    biases = [int(layer.bias_int[index]) for index in outputs]
    output_limit = 1 << (layer.spec.total_bits - 1)
    relation_limit = sum(abs(int(value)) * output_limit for value in cut.coefficients)
    bound_limit = max(abs(cut.lower * cut.scale), abs(cut.upper * cut.scale))
    if max(relation_limit, bound_limit) >= 1 << 127:
        raise ValueError("Relational cut arithmetic does not fit signed __int128")
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
{render_arith_kernel()}
#define LOCAL_INPUT_SIZE {layer.input_size}
#define CUT_SIZE {len(outputs)}
#define TOTAL_BITS {layer.spec.total_bits}
#define INPUT_FRACTIONAL_BITS {layer.input_fractional_bits}
static const int64_t INPUT_LOW[LOCAL_INPUT_SIZE] = {_c_array(certificate.input_invariant.low)};
static const int64_t INPUT_HIGH[LOCAL_INPUT_SIZE] = {_c_array(certificate.input_invariant.high)};
static const int64_t WEIGHT[CUT_SIZE * LOCAL_INPUT_SIZE] = {_c_array(weights)};
static const int64_t BIAS[CUT_SIZE] = {_c_array(biases)};
static const int64_t COEFFICIENT[CUT_SIZE] = {_c_array(cut.coefficients)};
int main(void) {{
    int64_t input[LOCAL_INPUT_SIZE];
    int64_t selected_output[CUT_SIZE];
    for (int i = 0; i < LOCAL_INPUT_SIZE; ++i) {{
        input[i] = nondet_long_long();
        __ESBMC_assume(input[i] >= INPUT_LOW[i] && input[i] <= INPUT_HIGH[i]);
    }}
    for (int out = 0; out < CUT_SIZE; ++out) {{
        __int128 acc = 0;
        for (int i = 0; i < LOCAL_INPUT_SIZE; ++i)
            acc = mac_i128(acc, WEIGHT[out * LOCAL_INPUT_SIZE + i], input[i]);
        __int128 value = div_round_half_away_from_zero_i128(
            acc, ((__int128)1 << INPUT_FRACTIONAL_BITS)) + BIAS[out];
        value = clamp_to_signed_range_i128(value, TOTAL_BITS);
        if ({1 if layer.apply_relu else 0} && value < 0) value = 0;
        selected_output[out] = (int64_t)clamp_to_signed_range_i128(value, TOTAL_BITS);
    }}
    __int128 relation = 0;
    for (int i = 0; i < CUT_SIZE; ++i)
        relation += (__int128)COEFFICIENT[i] * selected_output[i];
    __ESBMC_assert(relation >= (__int128){cut.lower} * {cut.scale},
                   "relational cut lower bound");
    __ESBMC_assert(relation <= (__int128){cut.upper} * {cut.scale},
                   "relational cut upper bound");
    return 0;
}}
"""

def render_dense_margin_contract(
    layer: QuantizedDense,
    invariant: IntegerInvariant,
    target: int,
    competitor: int,
) -> str:
    """Check one exact fixed-point output comparison over shared hidden inputs."""

    if layer.apply_relu or len(invariant.low) != layer.input_size:
        raise ValueError("Final margin requires a non-ReLU dense output layer")
    if not 0 <= target < layer.output_size or not 0 <= competitor < layer.output_size \
            or target == competitor:
        raise ValueError("Invalid target/competitor pair")
    weights = [
        *[int(value) for value in layer.weights_int[target]],
        *[int(value) for value in layer.weights_int[competitor]],
    ]
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
{render_arith_kernel()}
#define LOCAL_INPUT_SIZE {layer.input_size}
#define OUTPUT_SIZE 2
#define TOTAL_BITS {layer.spec.total_bits}
#define INPUT_FRACTIONAL_BITS {layer.input_fractional_bits}
static const int64_t INPUT_LOW[LOCAL_INPUT_SIZE] = {_c_array(invariant.low)};
static const int64_t INPUT_HIGH[LOCAL_INPUT_SIZE] = {_c_array(invariant.high)};
static const int64_t WEIGHT[OUTPUT_SIZE * LOCAL_INPUT_SIZE] = {_c_array(weights)};
static const int64_t BIAS[OUTPUT_SIZE] = {{{int(layer.bias_int[target])}, {int(layer.bias_int[competitor])}}};
int main(void) {{
    int64_t input[LOCAL_INPUT_SIZE];
    int64_t output[OUTPUT_SIZE];
    for (int i = 0; i < LOCAL_INPUT_SIZE; ++i) {{
        input[i] = nondet_long_long();
        __ESBMC_assume(input[i] >= INPUT_LOW[i] && input[i] <= INPUT_HIGH[i]);
    }}
    for (int out = 0; out < OUTPUT_SIZE; ++out) {{
        __int128 acc = 0;
        for (int i = 0; i < LOCAL_INPUT_SIZE; ++i)
            acc = mac_i128(acc, WEIGHT[out * LOCAL_INPUT_SIZE + i], input[i]);
        __int128 value = div_round_half_away_from_zero_i128(
            acc, ((__int128)1 << INPUT_FRACTIONAL_BITS)
        ) + (__int128)BIAS[out];
        value = clamp_to_signed_range_i128(value, TOTAL_BITS);
        output[out] = (int64_t)clamp_to_signed_range_i128(value, TOTAL_BITS);
    }}
    __ESBMC_assert(output[0] > output[1], "strict target/competitor margin");
    return 0;
}}
"""


def dense_margin_endpoint_witness(
    layer: QuantizedDense,
    invariant: IntegerInvariant,
    target: int,
    competitor: int,
) -> tuple[int, ...]:
    """Choose a deterministic box endpoint adverse to the raw output margin.

    This is only a candidate witness for abstraction imprecision.  It never
    represents a concrete network input unless a separate prefix proof says
    so, and callers must replay it through ESBMC before using it diagnostically.
    """

    if layer.apply_relu or len(invariant.low) != layer.input_size:
        raise ValueError("Final margin requires a non-ReLU dense output layer")
    if not 0 <= target < layer.output_size or not 0 <= competitor < layer.output_size \
            or target == competitor:
        raise ValueError("Invalid target/competitor pair")
    difference = layer.weights_int[target] - layer.weights_int[competitor]
    return tuple(
        int(low if int(coefficient) >= 0 else high)
        for coefficient, low, high in zip(
            difference, invariant.low, invariant.high, strict=True
        )
    )


def render_dense_margin_witness_replay(
    layer: QuantizedDense,
    invariant: IntegerInvariant,
    target: int,
    competitor: int,
    witness: Iterable[int],
) -> str:
    """Replay one abstract-box margin witness with deployment arithmetic."""

    values = tuple(int(value) for value in witness)
    if len(values) != layer.input_size:
        raise ValueError("Margin witness size does not match the dense input")
    if any(value < low or value > high for value, low, high in zip(
            values, invariant.low, invariant.high, strict=True)):
        raise ValueError("Margin witness is outside the certified hidden box")
    weights = [
        *[int(value) for value in layer.weights_int[target]],
        *[int(value) for value in layer.weights_int[competitor]],
    ]
    return f"""#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
{render_arith_kernel()}
#define LOCAL_INPUT_SIZE {layer.input_size}
#define OUTPUT_SIZE 2
#define TOTAL_BITS {layer.spec.total_bits}
#define INPUT_FRACTIONAL_BITS {layer.input_fractional_bits}
static const int64_t INPUT[LOCAL_INPUT_SIZE] = {_c_array(values)};
static const int64_t WEIGHT[OUTPUT_SIZE * LOCAL_INPUT_SIZE] = {_c_array(weights)};
static const int64_t BIAS[OUTPUT_SIZE] = {{{int(layer.bias_int[target])}, {int(layer.bias_int[competitor])}}};
int main(void) {{
    int64_t output[OUTPUT_SIZE];
    for (int out = 0; out < OUTPUT_SIZE; ++out) {{
        __int128 acc = 0;
        for (int i = 0; i < LOCAL_INPUT_SIZE; ++i)
            acc = mac_i128(acc, WEIGHT[out * LOCAL_INPUT_SIZE + i], INPUT[i]);
        __int128 value = div_round_half_away_from_zero_i128(
            acc, ((__int128)1 << INPUT_FRACTIONAL_BITS)
        ) + (__int128)BIAS[out];
        output[out] = (int64_t)clamp_to_signed_range_i128(value, TOTAL_BITS);
    }}
    __ESBMC_assert(output[0] > output[1],
                   "abstract hidden-box witness violates strict margin");
    return 0;
}}
"""



def render_invariant_competitor_margin(
    invariant: IntegerInvariant,
    target: int,
    competitor: int,
) -> str:
    """Check one strict class margin over a certified logit invariant."""

    if (len(invariant.shape) != 1 or not 0 <= target < len(invariant.low)
            or not 0 <= competitor < len(invariant.low) or target == competitor):
        raise ValueError("Invalid logit invariant or class pair")
    return f"""#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
#define TARGET_LOW {invariant.low[target]}
#define COMPETITOR_HIGH {invariant.high[competitor]}
int main(void) {{
    __ESBMC_assert(TARGET_LOW > COMPETITOR_HIGH,
                   "strict target/competitor invariant margin");
    return 0;
}}
"""

def render_byte_encoding_contract(
    byte_low: Iterable[int],
    byte_high: Iterable[int],
    encoded: IntegerInvariant,
    *,
    fractional_bits: int,
    total_bits: int,
) -> str:
    """Prove encoder monotonicity and replay every concrete endpoint."""

    low = tuple(int(value) for value in byte_low)
    high = tuple(int(value) for value in byte_high)
    if len(low) != len(high) or len(low) != len(encoded.low):
        raise ValueError("Byte encoding bounds do not match the encoded invariant")
    if any(a < 0 or b > 255 or a > b for a, b in zip(low, high, strict=True)):
        raise ValueError("Invalid uint8 perturbation box")
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
unsigned char nondet_uchar(void);
{render_arith_kernel()}
#define INVARIANT_SIZE {len(low)}
#define FRACTIONAL_BITS {fractional_bits}
#define TOTAL_BITS {total_bits}
static const uint8_t BYTE_LOW[INVARIANT_SIZE] = {_c_array(low)};
static const uint8_t BYTE_HIGH[INVARIANT_SIZE] = {_c_array(high)};
static const int64_t ENCODED_LOW[INVARIANT_SIZE] = {_c_array(encoded.low)};
static const int64_t ENCODED_HIGH[INVARIANT_SIZE] = {_c_array(encoded.high)};
static __int128 encode_byte(uint8_t raw) {{
    __int128 value = div_round_half_away_from_zero_i128(
        (__int128)raw * (((__int128)1) << FRACTIONAL_BITS), 256);
    return clamp_to_signed_range_i128(value, TOTAL_BITS);
}}
int main(void) {{
    uint8_t symbolic_low = nondet_uchar();
    uint8_t symbolic_value = nondet_uchar();
    uint8_t symbolic_high = nondet_uchar();
    __ESBMC_assume(symbolic_low <= symbolic_value);
    __ESBMC_assume(symbolic_value <= symbolic_high);
    __ESBMC_assert(
        encode_byte(symbolic_low) <= encode_byte(symbolic_value) &&
        encode_byte(symbolic_value) <= encode_byte(symbolic_high),
        "uint8 fixed-point encoder is monotone");
    for (int i = 0; i < INVARIANT_SIZE; ++i) {{
        __ESBMC_assert(encode_byte(BYTE_LOW[i]) == ENCODED_LOW[i],
                       "encoded lower endpoint mismatch");
        __ESBMC_assert(encode_byte(BYTE_HIGH[i]) == ENCODED_HIGH[i],
                       "encoded upper endpoint mismatch");
    }}
    return 0;
}}
"""

def render_nonvacuity_contract(invariant: IntegerInvariant, witness: Iterable[int]) -> str:
    concrete = tuple(int(value) for value in witness)
    if len(concrete) != len(invariant.low):
        raise ValueError("Non-vacuity witness size mismatch")
    return f"""#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
#define INVARIANT_SIZE {len(concrete)}
static const int64_t LOW[INVARIANT_SIZE] = {_c_array(invariant.low)};
static const int64_t HIGH[INVARIANT_SIZE] = {_c_array(invariant.high)};
static const int64_t WITNESS[INVARIANT_SIZE] = {_c_array(concrete)};
int main(void) {{
    for (int i = 0; i < INVARIANT_SIZE; ++i) {{
        __ESBMC_assert(LOW[i] <= HIGH[i], "nonempty invariant coordinate");
        __ESBMC_assert(LOW[i] <= WITNESS[i] && WITNESS[i] <= HIGH[i],
                       "concrete witness outside invariant");
    }}
    return 0;
}}
"""


class ESBMCProofStore:
    """Content-addressed cache containing only successful ESBMC obligations."""

    def __init__(
        self, output: Path, runner: ESBMCRunner, profile: str,
        cache_directory: Path | None = None,
    ):
        self.output = Path(output)
        self.harnesses = self.output / "harnesses"
        self.cache = (Path(cache_directory) if cache_directory is not None
                      else self.output / "verified_cache")
        self.harnesses.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.profile = profile
        self._parallel_batch = 0
        executable = resolve_esbmc_executable(runner.config.executable) or runner.config.executable
        try:
            version = subprocess.run(
                [executable, "--version"], check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10,
            ).stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            version = "UNKNOWN"
        self.checker_identity = {"executable": executable, "version": version}

    def _prepare(self, source: str, name: str, metadata: dict[str, Any]):
        identity = {
            **metadata,
            "profile": self.profile,
            "checker": self.checker_identity,
            "timeout_seconds": self.runner.config.timeout_seconds,
            "memlimit": self.runner.config.memlimit,
        }
        digest = obligation_sha256(source, identity)
        harness = self.harnesses / f"{name}_{digest[:16]}.c"
        if not harness.exists():
            harness.write_text(source, encoding="utf-8")
        cache_path = self.cache / f"{digest}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("digest") == digest and cached.get("status") == "VERIFIED":
                return ({**cached, "cache_reused": True, "harness": str(harness)}, None)
        return (None, {
            "digest": digest,
            "harness": harness,
            "cache_path": cache_path,
            "identity": identity,
        })

    def check(self, source: str, name: str, metadata: dict[str, Any]):
        cached, pending = self._prepare(source, name, metadata)
        if cached is not None:
            return cached
        assert pending is not None
        result = self.runner.run_file(pending["harness"], profile=self.profile)
        record = {
            "digest": pending["digest"],
            "status": result.status,
            "harness": str(pending["harness"]),
            "cache_reused": False,
            "identity": pending["identity"],
            "command": list(result.command),
            "elapsed_seconds": result.elapsed_seconds,
            "return_code": result.return_code,
            "resource_control": result.resource_control,
        }
        if result.status == "VERIFIED":
            write_new_json(pending["cache_path"], record)
        return record

    def check_many(
        self,
        obligations: list[tuple[str, str, dict[str, Any]]],
        *,
        jobs: int,
        min_available_gib: float,
    ) -> list[dict[str, Any]]:
        """Check independent obligations with bounded workers and a RAM guard."""

        records: list[dict[str, Any] | None] = [None] * len(obligations)
        pending_by_path: dict[str, tuple[int, dict[str, Any]]] = {}
        for index, (source, name, metadata) in enumerate(obligations):
            cached, pending = self._prepare(source, name, metadata)
            if cached is not None:
                records[index] = cached
            else:
                assert pending is not None
                pending_by_path[str(pending["harness"].resolve())] = (index, pending)
        if pending_by_path:
            batch_output = self.output / "parallel_batches" / f"batch_{self._parallel_batch:04d}"
            self._parallel_batch += 1
            result = benchmark(
                [entry[1]["harness"] for entry in pending_by_path.values()],
                batch_output,
                jobs=jobs,
                timeout=self.runner.config.timeout_seconds,
                memlimit=self.runner.config.memlimit,
                profile=self.profile,
                min_available_gib=min_available_gib,
            )
            completed = set()
            for row in result["records"]:
                key = str(Path(row["harness"]).resolve())
                index, pending = pending_by_path[key]
                completed.add(key)
                resource_control = {
                    "command": list(row["command"]),
                    "timeout": f"{self.runner.config.timeout_seconds}s",
                    "memlimit": self.runner.config.memlimit,
                    "elapsed_seconds": row["elapsed_seconds"],
                    "return_code": row["return_code"],
                    "status": row["status"],
                    "stdout_log_path": row["stdout_log_path"],
                    "stderr_log_path": row["stderr_log_path"],
                    "peak_memory_bytes": row["peak_memory_bytes"],
                    "peak_memory_mib": row["peak_memory_mib"],
                    "memory_measurement": "linux_procfs_process_tree_rss",
                }
                record = {
                    "digest": pending["digest"],
                    "status": row["status"],
                    "harness": str(pending["harness"]),
                    "cache_reused": False,
                    "identity": pending["identity"],
                    "command": list(row["command"]),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "return_code": row["return_code"],
                    "resource_control": resource_control,
                }
                records[index] = record
                if record["status"] == "VERIFIED":
                    write_new_json(pending["cache_path"], record)
            for key, (index, pending) in pending_by_path.items():
                if key in completed:
                    continue
                records[index] = {
                    "digest": pending["digest"],
                    "status": "UNKNOWN",
                    "harness": str(pending["harness"]),
                    "cache_reused": False,
                    "identity": pending["identity"],
                    "command": [],
                    "elapsed_seconds": 0.0,
                    "return_code": None,
                    "skipped_due_to_low_memory_guard": bool(result["aborted_low_memory"]),
                    "resource_control": {
                        "timeout": f"{self.runner.config.timeout_seconds}s",
                        "memlimit": self.runner.config.memlimit,
                        "status": "UNKNOWN",
                        "min_available_stop_gib": min_available_gib,
                    },
                }
        if any(record is None for record in records):
            raise RuntimeError("Parallel proof scheduler lost an obligation")
        return [record for record in records if record is not None]


class ConvProofCoordinator:
    def __init__(
        self,
        network: ConvFixedPointNetwork,
        output: Path,
        *,
        esbmc: ESBMCConfig | None = None,
        config: ConvProofConfig | None = None,
        cache_directory: Path | None = None,
        encoder_source: str = "",
    ):
        self.network = network
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.config = config or ConvProofConfig()
        self.runner = ESBMCRunner(esbmc)
        self.store = ESBMCProofStore(
            self.output, self.runner, self.config.profile, cache_directory=cache_directory
        )
        deployment_source = generate_conv_qnn_source(
            network, encoder_source=encoder_source
        )
        self.deployment_source_sha256 = hashlib.sha256(deployment_source.encode()).hexdigest()
        (self.output / "qnn_conv_native.c").write_text(deployment_source, encoding="utf-8")

    def _block_source(self, layer, certificate, indices):
        if self.config.proof_mode == "proof_carrying_interval":
            if isinstance(layer, QuantizedConv2D):
                return render_conv_interval_certificate(layer, certificate, indices)
            return render_dense_interval_certificate(layer, certificate, indices)
        if isinstance(layer, QuantizedConv2D):
            return render_conv_block_contract(layer, certificate, indices)
        return render_dense_block_contract(layer, certificate, indices)

    def verify(
        self,
        input_invariant: IntegerInvariant,
        *,
        input_witness: Iterable[int],
        target_class: int,
        input_byte_low: Iterable[int] | None = None,
        input_byte_high: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        certificates = propagate_network(self.network.layers, input_invariant)
        records: list[dict[str, Any]] = []
        partial = False
        stopped = False
        nonvacuity = self.store.check(
            render_nonvacuity_contract(input_invariant, input_witness),
            "input_nonvacuity",
            {"property_type": "input_nonvacuity", "deployment": self.deployment_source_sha256},
        )
        records.append(nonvacuity)
        if nonvacuity["status"] != "VERIFIED":
            stopped = True

        if (input_byte_low is None) != (input_byte_high is None):
            raise ValueError("Both raw byte bounds are required for input encoding proof")
        if not stopped and input_byte_low is not None:
            encoding = self.store.check(
                render_byte_encoding_contract(
                    input_byte_low,
                    input_byte_high,
                    input_invariant,
                    fractional_bits=self.network.input_fractional_bits,
                    total_bits=self.network.input_total_bits,
                ),
                "raw_byte_input_encoding",
                {
                    "property_type": "input_encoding",
                    "deployment": self.deployment_source_sha256,
                    "fractional_bits": self.network.input_fractional_bits,
                    "total_bits": self.network.input_total_bits,
                },
            )
            records.append(encoding)
            if encoding["status"] != "VERIFIED":
                stopped = True

        final_layer = self.network.layers[-1]
        if not isinstance(final_layer, QuantizedDense) or final_layer.apply_relu:
            raise ValueError("Convolution-native proof requires a dense non-ReLU output layer")

        transition_layers = (
            self.network.layers
            if self.config.proof_mode == "proof_carrying_interval"
            else self.network.layers[:-1]
        )
        transition_certificates = (
            certificates
            if self.config.proof_mode == "proof_carrying_interval"
            else certificates[:-1]
        )
        for layer_index, (layer, certificate) in enumerate(zip(
            transition_layers, transition_certificates, strict=True
        )):
            if stopped:
                break
            if self.config.proof_mode == "proof_carrying_interval":
                input_total_bits = (
                    self.network.input_total_bits
                    if layer_index == 0
                    else self.network.layers[layer_index - 1].spec.total_bits
                )
                arithmetic_sources = (
                    (
                        "rounding_monotonicity",
                        render_rounding_monotonicity_lemma(layer, input_total_bits),
                    ),
                    (
                        "clamp_relu_monotonicity",
                        render_clamp_relu_monotonicity_lemma(layer),
                    ),
                )
                for lemma_name, lemma_source in arithmetic_sources:
                    lemma = self.store.check(
                        lemma_source,
                        f"layer_{layer_index}_{lemma_name}",
                        {
                            "property_type": "fixed_point_arithmetic_lemma",
                            "lemma": lemma_name,
                            "deployment": self.deployment_source_sha256,
                            "layer_index": layer_index,
                            "input_total_bits": input_total_bits,
                            "shared_qif": asdict(layer.spec),
                        },
                    )
                    records.append(lemma)
                    if lemma["status"] != "VERIFIED":
                        stopped = True
                        break
                if stopped:
                    break
            blocks = [tuple(range(start, min(start + self.config.block_size, layer.output_size)))
                      for start in range(0, layer.output_size, self.config.block_size)]
            selected = blocks
            if self.config.max_blocks_per_layer is not None:
                selected = blocks[: self.config.max_blocks_per_layer]
                partial = partial or len(selected) < len(blocks)
            block_obligations = []
            for block_index, indices in enumerate(selected):
                block_obligations.append((
                    self._block_source(layer, certificate, indices),
                    f"layer_{layer_index}_block_{block_index}",
                    {
                        "property_type": (
                            "integer_interval_certificate"
                            if self.config.proof_mode == "proof_carrying_interval"
                            else "integer_layer_contract"
                        ),
                        "deployment": self.deployment_source_sha256,
                        "layer_index": layer_index,
                        "block_index": block_index,
                        "output_indices": list(indices),
                        "shared_qif": asdict(layer.spec),
                    },
                ))
            if self.config.jobs > 1 and not self.config.fail_fast:
                block_records = self.store.check_many(
                    block_obligations,
                    jobs=self.config.jobs,
                    min_available_gib=self.config.min_available_gib,
                )
            else:
                block_records = [
                    self.store.check(source, name, metadata)
                    for source, name, metadata in block_obligations
                ]
            for block_index, (indices, record) in enumerate(zip(
                selected, block_records, strict=True
            )):
                record.update({
                    "layer_index": layer_index,
                    "block_index": block_index,
                    "output_start": indices[0],
                    "output_end": indices[-1] + 1,
                    "shared_qif": asdict(layer.spec),
                })
                records.append(record)
                if record["status"] != "VERIFIED" and self.config.fail_fast:
                    stopped = True
                    break
            if stopped or partial:
                break
            if layer_index + 1 < len(certificates):
                bridge = self.store.check(
                    render_invariant_bridge(
                        certificate.output_invariant,
                        certificates[layer_index + 1].input_invariant,
                    ),
                    f"bridge_{layer_index}_{layer_index + 1}",
                    {
                        "property_type": "contract_chaining",
                        "deployment": self.deployment_source_sha256,
                        "from_layer": layer_index,
                        "to_layer": layer_index + 1,
                    },
                )
                records.append(bridge)
                if bridge["status"] != "VERIFIED":
                    stopped = True
                    break

        margin_records = []
        if not stopped and not partial:
            for competitor in range(final_layer.output_size):
                if competitor == target_class:
                    continue
                interval_mode = self.config.proof_mode == "proof_carrying_interval"
                margin_source = (
                    render_invariant_competitor_margin(
                        certificates[-1].output_invariant, target_class, competitor
                    )
                    if interval_mode
                    else render_dense_margin_contract(
                        final_layer, certificates[-1].input_invariant,
                        target_class, competitor,
                    )
                )
                margin = self.store.check(
                    margin_source,
                    f"final_margin_target_{target_class}_competitor_{competitor}",
                    {
                        "property_type": (
                            "strict_output_margin_interval"
                            if interval_mode else "strict_output_margin"
                        ),
                        "deployment": self.deployment_source_sha256,
                        "target_class": target_class,
                        "competitor_class": competitor,
                        "shared_qif": asdict(final_layer.spec),
                    },
                )
                records.append(margin)
                effective_margin = margin
                if (interval_mode and margin["status"] != "VERIFIED"
                        and self.config.exact_margin_refinement):
                    margin["required_for_certificate"] = False
                    witness = dense_margin_endpoint_witness(
                        final_layer, certificates[-1].input_invariant,
                        target_class, competitor,
                    )
                    witness_record = self.store.check(
                        render_dense_margin_witness_replay(
                            final_layer, certificates[-1].input_invariant,
                            target_class, competitor, witness,
                        ),
                        f"abstract_margin_witness_target_{target_class}_competitor_{competitor}",
                        {
                            "property_type": "strict_output_margin_abstraction_witness",
                            "deployment": self.deployment_source_sha256,
                            "target_class": target_class,
                            "competitor_class": competitor,
                            "shared_qif": asdict(final_layer.spec),
                            "trigger": margin["status"],
                            "witness_provenance": "adverse_raw_margin_box_endpoint",
                        },
                    )
                    witness_record["required_for_certificate"] = False
                    witness_record["diagnostic_only"] = True
                    witness_record["does_not_imply_real_input_counterexample"] = True
                    records.append(witness_record)
                    if witness_record["status"] == "FAILED":
                        effective_margin = witness_record
                    else:
                        effective_margin = self.store.check(
                            render_dense_margin_contract(
                                final_layer, certificates[-1].input_invariant,
                                target_class, competitor,
                            ),
                            f"exact_margin_target_{target_class}_competitor_{competitor}",
                            {
                                "property_type": "strict_output_margin_exact_refinement",
                                "deployment": self.deployment_source_sha256,
                                "target_class": target_class,
                                "competitor_class": competitor,
                                "shared_qif": asdict(final_layer.spec),
                                "trigger": margin["status"],
                            },
                        )
                        records.append(effective_margin)
                margin_records.append(effective_margin)
                if effective_margin["status"] != "VERIFIED" and self.config.fail_fast:
                    break

        required_block_count = sum(
            math.ceil(layer.output_size / self.config.block_size)
            for layer in transition_layers
        )
        executed_block_count = sum(
            record.get("identity", {}).get("property_type") in {
                "integer_layer_contract", "integer_interval_certificate"
            }
            for record in records
        )
        required_bridge_count = max(0, len(self.network.layers) - 1)
        executed_bridge_count = sum(
            record.get("identity", {}).get("property_type") == "contract_chaining"
            for record in records
        )
        all_required_blocks_executed = executed_block_count == required_block_count
        all_required_bridges_executed = executed_bridge_count == required_bridge_count

        required_records = [
            record for record in records
            if record.get("required_for_certificate", True)
        ]
        all_verified = bool(required_records) and all(
            record["status"] == "VERIFIED" for record in required_records
        )
        if partial:
            final_status = "PARTIAL_NOT_CERTIFIED"
        elif (all_verified and all_required_blocks_executed and all_required_bridges_executed
              and len(margin_records) == final_layer.output_size - 1):
            final_status = "VERIFIED"
        elif any(record["status"] == "FAILED"
                 and record.get("identity", {}).get("property_type")
                 == "strict_output_margin_abstraction_witness"
                 for record in records):
            final_status = "ABSTRACTION_INCONCLUSIVE"
        elif any(record["status"] == "FAILED"
                 and record.get("identity", {}).get("property_type", "").startswith("strict_output_margin")
                 for record in required_records):
            final_status = "ABSTRACTION_INCONCLUSIVE"
        elif any(record["status"] == "FAILED" for record in required_records):
            final_status = "FAILED"
        elif any(record["status"] == "TIMEOUT" for record in required_records):
            final_status = "TIMEOUT"
        elif any(record["status"] == "MEMOUT" for record in required_records):
            final_status = "MEMOUT"
        else:
            final_status = "UNKNOWN"
        summary = {
            "schema": "preqbmc_conv_native_proof_v1",
            "final_status": final_status,
            "certified": final_status == "VERIFIED",
            "proof_scope": "complete_network" if not partial else "bounded_feasibility_pilot",
            "input_domain": (
                "raw_uint8_box_with_formal_encoder"
                if input_byte_low is not None
                else "preencoded_integer_box"
            ),
            "deployment_source_sha256": self.deployment_source_sha256,
            "verified_cache_directory": str(self.store.cache),
            "shared_qif_per_layer": [asdict(layer.spec) for layer in self.network.layers],
            "accumulator_safety": {
                "implementation": "signed___int128",
                "conservative_absolute_envelopes": list(self.network.accumulator_envelopes),
                "signed_limit_exclusive": 1 << 127,
                "status": "PROVED_BY_STATIC_INTEGER_BOUND",
            },
            "block_size": self.config.block_size,
            "proof_mode": self.config.proof_mode,
            "resource_scheduling": {
                "jobs": self.config.jobs,
                "min_available_gib": self.config.min_available_gib,
                "parallel_blocks_only": self.config.jobs > 1,
                "dependency_barriers_preserved": True,
            },
            "refinement": {
                "strategy": "demand_driven_abstract_witness_then_exact_final_layer",
                "enabled": self.config.exact_margin_refinement,
                "triggered_competitors": [
                    record.get("identity", {}).get("competitor_class")
                    for record in records
                    if record.get("identity", {}).get("property_type")
                    == "strict_output_margin_exact_refinement"
                ],
                "abstract_witness_competitors": [
                    record.get("identity", {}).get("competitor_class")
                    for record in records
                    if record.get("identity", {}).get("property_type")
                    == "strict_output_margin_abstraction_witness"
                    and record.get("status") == "FAILED"
                ],
                "abstract_witnesses_are_not_network_counterexamples": True,
                "unverified_relations_never_assumed": True,
            },
            "proof_basis": {
                "signed_product_endpoint_rule": (
                    "mathematical ordered-integer lemma; concrete operands and endpoints "
                    "are replayed by ESBMC certificates"
                ),
                "rounding_clamp_relu_monotonicity": (
                    "ESBMC obligations in proof_carrying_interval mode"
                ),
                "certificate_policy": (
                    "no network certificate unless every generated certificate, bridge, "
                    "and final margin obligation is ESBMC VERIFIED"
                ),
            },
            "max_blocks_per_layer": self.config.max_blocks_per_layer,
            "required_transition_blocks": required_block_count,
            "executed_transition_blocks": executed_block_count,
            "required_hidden_blocks": required_block_count,
            "executed_hidden_blocks": executed_block_count,
            "all_required_blocks_executed": all_required_blocks_executed,
            "required_bridges": required_bridge_count,
            "executed_bridges": executed_bridge_count,
            "all_required_bridges_executed": all_required_bridges_executed,
            "invariants": [
                {
                    "layer_index": certificate.layer_index,
                    "shape": list(certificate.output_invariant.shape),
                    "stable_active": len(certificate.stable_active),
                    "stable_inactive": len(certificate.stable_inactive),
                    "unstable": certificate.unstable_count,
                }
                for certificate in certificates
            ],
            "obligations": records,
            "esbmc_calls_executed": sum(not record.get("cache_reused", False) for record in records),
            "esbmc_verified": sum(record["status"] == "VERIFIED" for record in records),
            "proof_relevant_obligations_verified": sum(
                record["status"] == "VERIFIED" for record in required_records
            ),
            "formal_rule": (
                "VERIFIED requires every required arithmetic lemma, transition certificate or "
                "exact transition block, chaining bridge, and final competitor margin obligation to be "
                "VERIFIED by ESBMC"
            ),
        }
        write_new_json(self.output / "proof_summary.json", summary)
        return summary
