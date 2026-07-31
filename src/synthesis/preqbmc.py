from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from backends.c_qnn_generator import (
    compile_c_qnn_shared_library,
    write_c_qnn_source,
)
from backends.fixed_point import (
    FixedPointNetwork,
    LayerQuantizationSpec,
    QuantizedLayer,
)
from symbolic_pp.DeepPoly_preqbmc import DP_DNN_network
from synthesis.preimage_cache import load_preimage_cache, save_preimage_cache
from synthesis.solver_backend import BackendConstants as GRB
from synthesis.solver_backend import SolverBackendName, build_model
from utils.fixed_point import int_get_min_max, quantize_int
from utils.logging_utils import get_logger
from verification.c_templates import (
    render_assumption_sentinel_program,
    render_hidden_affine_bounds_block_program,
    render_hidden_affine_bounds_program,
    render_no_saturation_block_program,
    render_no_saturation_program,
    render_network_end_to_end_program,
    render_output_target_program,
    render_output_valid_set_program,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner, ESBMCResult
from verification.invariants import propagate_exact_interval_details
from verification.properties import ClassificationProperty
from verification.replay import LayerReplayFormat, replay_on_python, replay_on_so

LOGGER = get_logger(__name__)

def _export_integer_bits(internal_integer_bits: int) -> int:
    """Convert Quadapter's internal sign-inclusive integer width to exported magnitude bits."""

    return max(int(internal_integer_bits) - 1, 0)


@dataclass(frozen=True)
class QuadapterConfig:
    """Configuration for the robustness quantization search."""

    bit_lb: int
    bit_ub: int
    preimg_mode: str
    verify_mode: str
    sample_id: int
    eps: float
    output_dir: Path
    if_relax: bool = False
    esbmc: ESBMCConfig = ESBMCConfig()
    no_gurobi: bool = False
    save_preimage_cache: bool = False
    preimage_cache_dir: Path | None = None
    preimage_cache_key: str | None = None
    preimage_cache_metadata: dict[str, Any] | None = None
    esbmc_layer_block_size: int = 0
    blockwise_fail_fast: bool = True
    blockwise_run_all_blocks_on_failure: bool = False
    no_saturation_continue_on_unknown: bool = False
    esbmc_jobs: int = 1
    gurobi_threads: int = 4
    solver: SolverBackendName = "cbc"
    unsound_contract_tolerance: bool = False
    propagate_contract_tolerance: bool = False
    enforce_contract_chaining: bool = True
    error_budget_mode: str = "heuristic"
    vacuity_check: bool | None = None
    cex_feedback: str = "off"
    harness_scope: str = "layer"
    e2e_invariants: bool = True

    @classmethod
    def from_namespace(cls, args: Any) -> "QuadapterConfig":
        solver = str(getattr(args, "solver", "cbc")).lower()
        if solver not in {"cbc", "gurobi"}:
            raise ValueError(f"Unsupported solver backend: {solver!r}. Expected 'cbc' or 'gurobi'.")
        return cls(
            bit_lb=int(args.bit_lb),
            bit_ub=int(args.bit_ub),
            preimg_mode=str(args.preimg_mode),
            verify_mode=str(args.verify_mode),
            sample_id=int(args.sample_id),
            eps=float(args.eps),
            output_dir=Path(getattr(args, "output_dir", getattr(args, "outputPath", "output"))),
            if_relax=bool(int(getattr(args, "if_relax", getattr(args, "ifRelax", 0)))),
            esbmc=ESBMCConfig(
                timeout_seconds=max(1, int(getattr(args, "esbmc_timeout_seconds", 900))),
                memlimit=str(getattr(args, "esbmc_memlimit", "6g")),
                default_profile=getattr(args, "esbmc_profile", "paper-fast"),
            ),
            no_gurobi=bool(getattr(args, "no_gurobi", False)),
            save_preimage_cache=bool(getattr(args, "save_preimage_cache", False)),
            preimage_cache_dir=(
                Path(getattr(args, "preimage_cache_dir"))
                if getattr(args, "preimage_cache_dir", None) is not None
                else None
            ),
            preimage_cache_key=getattr(args, "preimage_cache_key", None),
            esbmc_layer_block_size=max(0, int(getattr(args, "esbmc_layer_block_size", 0))),
            blockwise_fail_fast=bool(getattr(args, "blockwise_fail_fast", True)),
            blockwise_run_all_blocks_on_failure=bool(
                getattr(args, "blockwise_run_all_blocks_on_failure", False)
            ),
            no_saturation_continue_on_unknown=bool(getattr(args, "no_saturation_continue_on_unknown", False)),
            esbmc_jobs=max(1, int(getattr(args, "esbmc_jobs", 1))),
            gurobi_threads=max(1, int(getattr(args, "gurobi_threads", 4))),
            solver=solver,  # type: ignore[arg-type]
            unsound_contract_tolerance=bool(getattr(args, "unsound_contract_tolerance", False)),
            propagate_contract_tolerance=bool(getattr(args, "propagate_contract_tolerance", False)),
            enforce_contract_chaining=bool(getattr(args, "enforce_contract_chaining", True)),
            error_budget_mode=str(getattr(args, "error_budget_mode", "heuristic")).lower(),
            vacuity_check=getattr(args, "vacuity_check", None),
            cex_feedback=str(getattr(args, "cex_feedback", "off")).lower(),
            harness_scope=str(getattr(args, "harness_scope", "layer")).lower(),
            e2e_invariants=bool(getattr(args, "e2e_invariants", True)),
        )


@dataclass(frozen=True)
class SynthesisResult:
    """Stable result object for the robustness pipeline."""

    success: bool
    total_bits: list[int]
    fractional_bits: list[int]
    integer_bits: list[int]
    stats: dict[str, float]
    final_status: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total_bits": self.total_bits,
            "fractional_bits": self.fractional_bits,
            "integer_bits": self.integer_bits,
            "stats": self.stats,
            "final_status": self.final_status,
        }


class LayerEncoding:
    """Per-layer state used by the MILP/DeepPoly synthesis algorithm."""

    def __init__(
        self,
        gp_model: Any | None,
        preimg_mode: str,
        layer_index: int,
        layer_size: int,
        layer_paras: Any,
        bit_lb: int,
        bit_ub: int,
        if_hid: bool,
    ) -> None:
        self.layer_index = layer_index
        self.layer_size = layer_size
        self.layer_paras = layer_paras
        self.bit_lb = bit_lb
        self.bit_ub = bit_ub
        self.frac_bit: int | None = None
        self.grad = None
        self.realVal = None

        self.lb = np.zeros(layer_size, dtype=np.float32)
        self.ub = np.zeros(layer_size, dtype=np.float32)
        self.clipped_lb = np.zeros(layer_size, dtype=np.float32)
        self.clipped_ub = np.zeros(layer_size, dtype=np.float32)
        self.qu_lb = np.zeros(layer_size, dtype=np.float32)
        self.qu_ub = np.zeros(layer_size, dtype=np.float32)
        self.qu_clipped_lb = np.zeros(layer_size, dtype=np.float32)
        self.qu_clipped_ub = np.zeros(layer_size, dtype=np.float32)
        self.verified_activation_lb: np.ndarray | None = None
        self.verified_activation_ub: np.ndarray | None = None
        self.verified_activation_source: str = "deeppoly_clipped"
        self.error_budget_int: np.ndarray | None = None

        if layer_index > 0:
            self.max_weight = np.round(max(np.max(layer_paras[0]), np.max(layer_paras[1])))
            self.min_weight = np.round(min(np.min(layer_paras[0]), np.min(layer_paras[1])))
            self.max_int = max(abs(self.max_weight), abs(self.min_weight))
            if self.max_int == 0:
                self.int_bit = 1
            elif self.max_int == 1:
                self.int_bit = 2
            else:
                self.int_bit = int(np.ceil(math.log(self.max_int, 2)) + 1)
        else:
            self.int_bit = None

        self.relaxed_lb = np.zeros(layer_size, dtype=np.float32)
        self.relaxed_lb_expression = [1 for _ in range(layer_size)]
        self.relaxed_ub = np.zeros(layer_size, dtype=np.float32)
        self.relaxed_ub_expression = [1 for _ in range(layer_size)]
        self.actMode = np.zeros(layer_size, dtype=np.float32)
        self.bit_vars: list[Any] = []
        self.gp_vars_before: list[Any] = []
        self.gp_vars_after: list[Any] = []
        self.alpha: list[Any] = []
        self.beta: list[Any] = []
        self.gp_vars_lb_before: list[Any] = []
        self.gp_vars_ub_before: list[Any] = []
        self.alpha_before: list[Any] = []
        self.alpha_after: list[Any] = []
        self.beta_before: list[Any] = []
        self.beta_after: list[Any] = []

        if gp_model is None:
            return

        neuron_lb_after = 0 if if_hid else -GRB.MAXINT
        neuron_lb_before = -GRB.MAXINT

        self.bit_vars = [gp_model.addVar(vtype=GRB.BINARY) for _ in range(self.bit_ub - self.bit_lb + 1)]

        if preimg_mode in {"milp", "comp"}:
            self.gp_vars_before = [
                gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for _ in range(layer_size)
            ]
            self.gp_vars_after = [
                gp_model.addVar(lb=0 if if_hid else neuron_lb_after, ub=1000, vtype=GRB.CONTINUOUS)
                for _ in range(layer_size)
            ]
            self.alpha = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for _ in range(layer_size)]
            self.beta = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for _ in range(layer_size)]

        if preimg_mode in {"abstr", "comp"}:
            self.gp_vars_lb_before = [
                gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for _ in range(layer_size)
            ]
            self.gp_vars_ub_before = [
                gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for _ in range(layer_size)
            ]
            self.alpha_before = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for _ in range(layer_size)]
            self.alpha_after = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for _ in range(layer_size)]
            self.beta_before = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for _ in range(layer_size)]
            self.beta_after = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for _ in range(layer_size)]

        gp_model.update()

    def set_input_bounds(self, low: np.ndarray, high: np.ndarray) -> None:
        self.lb = low
        self.ub = high

    def set_realVal(self, realVal: Any) -> None:
        self.realVal = realVal


class GPEncoding:
    """Main Quadapter robustness synthesizer."""

    # Keeps lightweight test doubles and older integrations backward compatible.
    error_budget_mode = "zero"
    cex_feedback = "off"
    harness_scope = "layer"

    def __init__(
        self,
        arch: list[int],
        model: Any,
        config: QuadapterConfig | Any,
        original_prediction: int,
        x_low_real: np.ndarray,
        x_high_real: np.ndarray,
        property_spec: ClassificationProperty | None = None,
    ) -> None:
        self.config = config if isinstance(config, QuadapterConfig) else QuadapterConfig.from_namespace(config)
        self.tole = 1e-6
        self.bit_lb = self.config.bit_lb
        self.bit_ub = self.config.bit_ub
        self.preimg_mode = self.config.preimg_mode
        self.verify_mode = self.config.verify_mode
        self.solver: SolverBackendName = self.config.solver
        self.x_low_real = x_low_real
        self.x_high_real = x_high_real
        self.sample_id = self.config.sample_id
        self.eps = self.config.eps
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ifRelax = int(self.config.if_relax)
        self.scaleValueSet: list[float] = []
        self.esbmc_layer_block_size = max(0, int(self.config.esbmc_layer_block_size))
        self.blockwise_fail_fast = bool(self.config.blockwise_fail_fast)
        self.blockwise_run_all_blocks_on_failure = bool(self.config.blockwise_run_all_blocks_on_failure)
        self.no_saturation_continue_on_unknown = bool(self.config.no_saturation_continue_on_unknown)
        self.esbmc_jobs = max(1, int(self.config.esbmc_jobs))
        self.unsound_contract_tolerance = bool(self.config.unsound_contract_tolerance)
        self.propagate_contract_tolerance = bool(self.config.propagate_contract_tolerance)
        self.enforce_contract_chaining = bool(self.config.enforce_contract_chaining)
        self.error_budget_mode = str(self.config.error_budget_mode).lower()
        if self.error_budget_mode not in {"heuristic", "derived", "zero"}:
            raise ValueError(
                "error_budget_mode must be one of: 'heuristic', 'derived', 'zero'"
            )
        self.vacuity_check = (
            self.error_budget_mode == "derived"
            if self.config.vacuity_check is None
            else bool(self.config.vacuity_check)
        )
        self.cex_feedback = str(self.config.cex_feedback).lower()
        if self.cex_feedback not in {"off", "filter", "filter+jump"}:
            raise ValueError("cex_feedback must be one of: off, filter, filter+jump")
        self.harness_scope = str(self.config.harness_scope).lower()
        if self.harness_scope not in {"layer", "network"}:
            raise ValueError("harness_scope must be one of: layer, network")
        self.e2e_invariants = bool(self.config.e2e_invariants)
        self.esbmc_call_records: list[dict[str, Any]] = []
        self.esbmc_block_records: list[dict[str, Any]] = []
        self.esbmc_no_saturation_block_records: list[dict[str, Any]] = []
        self.chaining_records: list[dict[str, Any]] = []
        self.output_margin_records: list[dict[str, Any]] = []
        self.vacuity_records: list[dict[str, Any]] = []
        self.synthesis_final_status = "UNKNOWN"
        self.cex_pool: dict[int, list[dict[str, Any]]] = {}
        self.counterexample_records: list[dict[str, Any]] = []
        self.cex_filtered_counts: dict[int, int] = {}
        self.cex_bit_jumps: dict[int, list[dict[str, int]]] = {}
        self.end_to_end_record: dict[str, Any] = {
            "enabled": self.harness_scope == "network",
            "invariants_injected": self.e2e_invariants,
            "status": "NOT_RUN",
        }
        self.blockwise_skipped_blocks_due_to_fail_fast = 0
        self.blockwise_first_failed_block: dict[str, Any] | None = None

        needs_milp_model = (not self.config.no_gurobi) or self.verify_mode == "milp"
        if not needs_milp_model:
            self.gp_model = None
        else:
            self.gp_model = build_model(
                self.solver,
                "gp_encoding",
                threads=max(1, int(self.config.gurobi_threads)),
                output_flag=0,
            )
            self.gp_model.setParam("IntFeasTol", 1e-9)
            self.gp_model.setParam("FeasibilityTol", self.tole)
            self.gp_model.setParam(GRB.Param.Threads, max(1, int(self.config.gurobi_threads)))
            self.gp_model.setParam(GRB.Param.OutputFlag, 0)

        self._stats = {
            "encoding_time": 0.0,
            "solving_time": 0.0,
            "backward_time": 0.0,
            "forward_time": 0.0,
            "total_time": 0.0,
            "esbmc_calls": 0.0,
            "esbmc_block_calls": 0.0,
        }

        self.dense_layers: list[LayerEncoding] = []
        self.nnparas: list[Any] = []
        self.deep_model = model
        self.layerNum = len(model.dense_layers)
        self.targetCls = int(original_prediction)
        self.property_spec = property_spec or ClassificationProperty(target_label=self.targetCls)
        self.property_spec.validate(arch[-1])
        self.deepPolyNets_DNN = DP_DNN_network(True)
        self.esbmc_runner = ESBMCRunner(self.config.esbmc)

        self.input_gp_vars: list[Any] = []
        for i, _ in enumerate(model.dense_layers):
            tf_layer = model.dense_layers[i]
            w_cont, b_cont = tf_layer.get_weights()
            self.nnparas.append([w_cont.T, b_cont])

        self.output_layer = LayerEncoding(
            self.gp_model,
            preimg_mode=self.preimg_mode,
            layer_index=len(self.nnparas),
            layer_size=arch[-1],
            layer_paras=self.nnparas[-1],
            bit_lb=self.bit_lb,
            bit_ub=self.bit_ub,
            if_hid=False,
        )

        for layer in range(len(arch) - 2):
            self.dense_layers.append(
                LayerEncoding(
                    self.gp_model,
                    preimg_mode=self.preimg_mode,
                    layer_index=layer + 1,
                    layer_size=arch[layer + 1],
                    layer_paras=self.nnparas[layer],
                    bit_lb=self.bit_lb,
                    bit_ub=self.bit_ub,
                    if_hid=True,
                )
            )
            self.scaleValueSet.append(0)

        input_size = arch[0]
        self.input_layer = LayerEncoding(
            self.gp_model,
            preimg_mode=self.preimg_mode,
            layer_index=0,
            layer_size=input_size,
            layer_paras=None,
            bit_lb=self.bit_lb,
            bit_ub=self.bit_ub,
            if_hid=False,
        )

        self.deepPolyNets_DNN.load_dnn(model)

        for input_index in range(self.input_layer.layer_size):
            x_lb = x_low_real[input_index]
            x_ub = x_high_real[input_index]
            if self.gp_model is not None:
                self.input_gp_vars.append(self.gp_model.addVar(lb=x_lb, ub=x_ub, vtype=GRB.CONTINUOUS))

    @staticmethod
    def _linear_combination(coefficients: Any, variables: list[Any], constant: Any = 0.0) -> Any:
        if isinstance(constant, np.ndarray) and constant.shape == ():
            expr: Any = constant.item()
        elif isinstance(constant, np.generic):
            expr = constant.item()
        else:
            expr = constant
        for coefficient, variable in zip(np.asarray(coefficients, dtype=np.float64).ravel(), variables):
            coeff = float(coefficient)
            if coeff != 0.0:
                expr = expr + coeff * variable
        return expr

    @staticmethod
    def _interval_linear_combination(
        coefficients: Any,
        variable_bounds: list[tuple[float, float]],
        constant: float = 0.0,
    ) -> tuple[float, float]:
        lower = float(constant)
        upper = float(constant)
        for coefficient, (var_lb, var_ub) in zip(np.asarray(coefficients, dtype=np.float64).ravel(), variable_bounds):
            coeff = float(coefficient)
            if coeff >= 0:
                lower += coeff * float(var_lb)
                upper += coeff * float(var_ub)
            else:
                lower += coeff * float(var_ub)
                upper += coeff * float(var_lb)
        return lower, upper

    @staticmethod
    def _pre_relu_bounds(neuron_lb: float, neuron_ub: float, alpha_k: float, beta_k: float, relax_ub: float) -> tuple[float, float]:
        return float(neuron_lb - alpha_k * relax_ub), float(neuron_ub + beta_k * relax_ub)

    @staticmethod
    def _required_internal_integer_bits_for_interval(lower: float, upper: float) -> int:
        """Return sign-inclusive integer bits needed to represent a real interval.

        `LayerEncoding.int_bit` follows Quadapter's internal convention: it
        includes the sign bit. A value of 2 therefore corresponds to exported
        I=1 and real range [-2, 2) for any fractional precision.
        """

        lo = float(lower)
        hi = float(upper)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return 1
        if lo > hi:
            lo, hi = hi, lo

        internal_bits = 1
        while lo < -(2 ** (internal_bits - 1)) or hi >= (2 ** (internal_bits - 1)):
            internal_bits += 1
        return internal_bits

    def _required_internal_integer_bits_for_layer_contract(self, layer: LayerEncoding) -> int:
        lower_candidates: list[float] = []
        upper_candidates: list[float] = []

        for lower_values, upper_values in (
            (layer.lb, layer.ub),
            (layer.relaxed_lb, layer.relaxed_ub),
        ):
            lower_array = np.asarray(lower_values, dtype=np.float64)
            upper_array = np.asarray(upper_values, dtype=np.float64)
            finite_lower = lower_array[np.isfinite(lower_array)]
            finite_upper = upper_array[np.isfinite(upper_array)]
            if finite_lower.size:
                lower_candidates.append(float(np.min(finite_lower)))
            if finite_upper.size:
                upper_candidates.append(float(np.max(finite_upper)))

        if not lower_candidates and not upper_candidates:
            return 1
        lower = min(lower_candidates) if lower_candidates else 0.0
        upper = max(upper_candidates) if upper_candidates else 0.0
        return self._required_internal_integer_bits_for_interval(lower, upper)

    def _widen_internal_integer_bits_for_fixed_point_contracts(self) -> None:
        """Ensure layer formats can represent the contracts ESBMC must prove.

        The synthesis search varies fractional precision while keeping a single
        shared integer width per layer. After the ESBMC harness started modeling
        deployed saturation exactly, too-small integer widths could make every
        candidate fail for the same real-range reason. This pass keeps the
        shared-QIF policy, but raises the layer-level integer-width floor to
        cover the DeepPoly/preimage contract range before searching F.
        """

        for layer in [*self.dense_layers, self.output_layer]:
            if layer.int_bit is None:
                continue
            current = int(layer.int_bit)
            required = self._required_internal_integer_bits_for_layer_contract(layer)
            if required > current:
                LOGGER.info(
                    "Increasing layer %s internal integer bits from %s to %s to cover fixed-point contract range.",
                    layer.layer_index,
                    current,
                    required,
                )
                layer.int_bit = required

    def verified_quant(self, lb: np.ndarray, ub: np.ndarray) -> tuple[bool, Any, Any, Any]:
        result = self.run(lb, ub)
        if not result.success:
            return False, None, None, None
        return True, result.total_bits, result.fractional_bits, result.integer_bits

    def _network_lower_bound_configuration(
        self,
    ) -> tuple[list[int], list[int], list[int]]:
        layers = [*self.dense_layers, self.output_layer]
        fractional_bits = [int(self.bit_lb) for _ in layers]
        integer_bits = [
            _export_integer_bits(int(layer.int_bit))
            for layer in layers
        ]
        total_bits = [
            int(fractional) + int(integer) + 1
            for fractional, integer in zip(fractional_bits, integer_bits)
        ]
        return total_bits, fractional_bits, integer_bits

    def _fixed_point_network_from_configuration(
        self,
        total_bits: list[int],
        fractional_bits: list[int],
        integer_bits: list[int],
    ) -> tuple[FixedPointNetwork, list[LayerQuantizationSpec]]:
        encoded_layers = [*self.dense_layers, self.output_layer]
        specs = [
            LayerQuantizationSpec(
                total_bits=int(q_bits),
                integer_bits=int(i_bits),
                fractional_bits=int(f_bits),
            )
            for q_bits, f_bits, i_bits in zip(
                total_bits,
                fractional_bits,
                integer_bits,
            )
        ]
        layers = tuple(
            QuantizedLayer(
                weights_int=np.asarray(
                    quantize_int(
                        encoded.layer_paras[0],
                        spec.total_bits,
                        spec.fractional_bits,
                    ),
                    dtype=np.int64,
                ),
                biases_int=np.asarray(
                    quantize_int(
                        encoded.layer_paras[1],
                        spec.total_bits,
                        spec.fractional_bits,
                    ),
                    dtype=np.int64,
                ),
                spec=spec,
                is_output_layer=index == len(encoded_layers) - 1,
            )
            for index, (encoded, spec) in enumerate(
                zip(encoded_layers, specs)
            )
        )
        return (
            FixedPointNetwork(
                input_fractional_bits=specs[0].fractional_bits,
                input_total_bits=specs[0].total_bits,
                layers=layers,
            ),
            specs,
        )

    def _replay_network_integer_input(
        self,
        network: FixedPointNetwork,
        inputs_int: list[int] | np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(inputs_int, dtype=np.int64)
        input_fractional_bits = int(network.input_fractional_bits)
        for layer in network.layers:
            values = replay_on_python(
                values,
                layer,
                LayerReplayFormat(
                    input_fractional_bits=input_fractional_bits,
                    total_bits=layer.spec.total_bits,
                    apply_relu=not layer.is_output_layer,
                ),
            )
            input_fractional_bits = int(layer.spec.fractional_bits)
        return values

    def _network_output_violates(self, outputs: np.ndarray) -> bool:
        if self.property_spec.valid_labels:
            valid = set(int(value) for value in self.property_spec.valid_labels)
            valid_values = [
                int(outputs[index]) for index in sorted(valid)
            ]
            invalid_values = [
                int(outputs[index])
                for index in range(outputs.size)
                if index not in valid
            ]
            return bool(invalid_values and max(valid_values) <= max(invalid_values))
        target = int(
            self.property_spec.target_label
            if self.property_spec.target_label is not None
            else self.targetCls
        )
        return any(
            int(outputs[index]) >= int(outputs[target])
            for index in range(outputs.size)
            if index != target
        )

    def _verify_network_end_to_end(
        self,
        total_bits: list[int],
        fractional_bits: list[int],
        integer_bits: list[int],
    ) -> bool:
        network, specs = self._fixed_point_network_from_configuration(
            total_bits,
            fractional_bits,
            integer_bits,
        )
        input_scale = 1 << int(network.input_fractional_bits)
        input_q_min, input_q_max = specs[0].signed_range
        x_low = np.maximum(
            np.floor(np.asarray(self.x_low_real, dtype=np.float64) * input_scale).astype(np.int64),
            input_q_min,
        )
        x_high = np.minimum(
            np.ceil(np.asarray(self.x_high_real, dtype=np.float64) * input_scale).astype(np.int64),
            input_q_max,
        )
        assumption_box_cardinality, assumption_box_valid = (
            self._assumption_box_cardinality(x_low, x_high)
        )
        if not assumption_box_valid:
            self.end_to_end_record = {
                "enabled": True,
                "scope": "network",
                "status": "VACUOUS",
                "invariants_injected": bool(self.e2e_invariants),
                "selection_policy": "layer_integer_width_with_fractional_lower_bound",
                "total_bits": [int(value) for value in total_bits],
                "integer_bits": [int(value) for value in integer_bits],
                "fractional_bits": [int(value) for value in fractional_bits],
                "assumption_box_cardinality": assumption_box_cardinality,
                "esbmc_calls": 0,
            }
            return False
        interval_details = propagate_exact_interval_details(
            network.layers,
            specs,
            x_low,
            x_high,
        )

        layer_payloads: list[dict[str, object]] = []
        current_input_fractional_bits = int(network.input_fractional_bits)
        for layer, spec, intervals in zip(
            network.layers,
            specs,
            interval_details,
        ):
            layer_payloads.append(
                {
                    "input_size": int(layer.weights_int.shape[1]),
                    "output_size": int(layer.weights_int.shape[0]),
                    "total_bits": int(spec.total_bits),
                    "fractional_bits": int(spec.fractional_bits),
                    "input_fractional_bits": current_input_fractional_bits,
                    "weights_c_int": self.numpy_to_c_int_array(layer.weights_int),
                    "biases_c_int": self.numpy_to_c_int_array(layer.biases_int),
                    "invariant_low_c_int": self.numpy_to_c_int_array(intervals.output_low),
                    "invariant_high_c_int": self.numpy_to_c_int_array(intervals.output_high),
                    "accumulator_c_type": intervals.accumulator_c_type,
                }
            )
            current_input_fractional_bits = int(spec.fractional_bits)

        source = render_network_end_to_end_program(
            input_size=int(x_low.size),
            input_bounds_low_c_int=self.numpy_to_c_int_array(x_low),
            input_bounds_high_c_int=self.numpy_to_c_int_array(x_high),
            layers=layer_payloads,
            target_label=(
                int(self.property_spec.target_label)
                if self.property_spec.target_label is not None
                else int(self.targetCls)
            ),
            valid_classes=(
                tuple(int(value) for value in self.property_spec.valid_labels)
                if self.property_spec.valid_labels
                else None
            ),
            inject_invariants=bool(self.e2e_invariants),
        )
        layers_dir = self.output_dir / "layers"
        layers_dir.mkdir(parents=True, exist_ok=True)
        format_name = "_".join(
            f"Q{q}_F{f}" for q, f in zip(total_bits, fractional_bits)
        )
        harness = layers_dir / f"network_e2e_{format_name}.c"
        harness.write_text(source, encoding="utf-8")
        result = self._run_esbmc_file(
            harness,
            extract_counterexample=True,
        )
        self._stats["esbmc_calls"] += 1.0
        call_record = self._esbmc_call_record(
            result=result,
            layer_index=-1,
            block_index=None,
            start_neuron=None,
            end_neuron=None,
            all_bit=max(total_bits),
            frac_bit=max(fractional_bits),
            harness=harness,
            property_type="network_end_to_end",
            mode="network",
            input_dim=int(x_low.size),
            output_neurons=int(network.layers[-1].biases_int.size),
        )
        self.esbmc_call_records.append(call_record)

        replay_record: dict[str, Any] | None = None
        if result.status == "FAILED" and result.counterexample_inputs is not None:
            python_outputs = self._replay_network_integer_input(
                network,
                result.counterexample_inputs,
            )
            python_confirmed = self._network_output_violates(python_outputs)
            replay_record = {
                "inputs_int": [
                    int(value) for value in result.counterexample_inputs
                ],
                "python_outputs_int": [
                    int(value) for value in python_outputs
                ],
                "python_replay_confirmed": bool(python_confirmed),
                "so_replay_confirmed": None,
                "so_path": None,
            }
            try:
                c_export_dir = self.output_dir / "c_export"
                source_path = write_c_qnn_source(
                    network,
                    c_export_dir / "qnn_model.c",
                )
                so_path = compile_c_qnn_shared_library(
                    source_path,
                    c_export_dir / "qnn_model.so",
                )
                so_outputs = replay_on_so(
                    result.counterexample_inputs,
                    so_path,
                )
                replay_record.update(
                    {
                        "so_outputs_int": [
                            int(value) for value in so_outputs
                        ],
                        "so_replay_confirmed": bool(
                            self._network_output_violates(so_outputs)
                        ),
                        "so_path": str(so_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report unavailable C replay.
                replay_record["so_replay_error"] = str(exc)

        self.end_to_end_record = {
            "enabled": True,
            "scope": "network",
            "status": result.status,
            "invariants_injected": bool(self.e2e_invariants),
            "selection_policy": "layer_integer_width_with_fractional_lower_bound",
            "total_bits": [int(value) for value in total_bits],
            "integer_bits": [int(value) for value in integer_bits],
            "fractional_bits": [int(value) for value in fractional_bits],
            "assumption_box_cardinality": assumption_box_cardinality,
            "esbmc_calls": 1,
            "harness": str(harness),
            "command": list(result.command),
            "resource_control": result.resource_control,
            "intervals": [
                {
                    "layer_index": int(index),
                    "low": [int(value) for value in details.output_low],
                    "high": [int(value) for value in details.output_high],
                    "accumulator_c_type": details.accumulator_c_type,
                }
                for index, details in enumerate(interval_details)
            ],
            "counterexample_replay": replay_record,
        }
        if replay_record is not None:
            self.counterexample_records.append(
                {
                    "layer_index": -1,
                    "Q": list(total_bits),
                    "I": list(integer_bits),
                    "F": list(fractional_bits),
                    "inputs_int": replay_record["inputs_int"],
                    "replay_confirmed": bool(
                        replay_record["python_replay_confirmed"]
                    ),
                    "so_replay_confirmed": replay_record.get(
                        "so_replay_confirmed"
                    ),
                }
            )
        return result.status == "VERIFIED"

    def run(self, lb: np.ndarray, ub: np.ndarray) -> SynthesisResult:
        self.assert_input_box(lb, ub)
        self.symbolic_propagate()

        out_bounds_lb = self.output_layer.lb
        other_max = -1000.0
        for index, value in enumerate(self.output_layer.ub):
            if index == self.targetCls:
                continue
            other_max = max(other_max, value)

        if out_bounds_lb[self.targetCls] < other_max:
            raise ValueError("The property does not hold in the original DNN for the selected input region.")

        backward_start_time = time.time()
        if self.verify_mode == "esbmc" and self.harness_scope == "network":
            # Direct network verification does not consume layer preimages.
            # Symbolic propagation above is sufficient to size the candidate
            # integer formats; the generated harness proves the deployed
            # integer program itself.
            pass
        elif self.config.no_gurobi:
            self.load_cached_preimage()
        else:
            self.backward_preimage_computation()
            if self.config.save_preimage_cache:
                self.save_cached_preimage()
        self._widen_internal_integer_bits_for_fixed_point_contracts()
        backward_end_time = time.time()

        if self.verify_mode == "esbmc" and self.harness_scope == "network":
            total_bits, fractional_bits, integer_bits = (
                self._network_lower_bound_configuration()
            )
            if_success = self._verify_network_end_to_end(
                total_bits,
                fractional_bits,
                integer_bits,
            )
            if not if_success:
                total_bits = None
                fractional_bits = None
                integer_bits = None
                self.synthesis_final_status = str(
                    self.end_to_end_record.get("status", "UNKNOWN")
                )
        elif self.verify_mode == "esbmc":
            if_success, total_bits, fractional_bits, integer_bits = self.forward_quantization_with_esbmc()
        else:
            if_success, total_bits, fractional_bits, integer_bits = self.forward_quantization()
        forward_end_time = time.time()

        self._stats["backward_time"] = backward_end_time - backward_start_time
        self._stats["forward_time"] = forward_end_time - backward_end_time
        self._stats["total_time"] = self._stats["backward_time"] + self._stats["forward_time"]

        return SynthesisResult(
            success=bool(if_success),
            total_bits=total_bits or [],
            fractional_bits=fractional_bits or [],
            integer_bits=integer_bits or [],
            stats={key: float(value) for key, value in self._stats.items()},
            final_status="VERIFIED" if if_success else self.synthesis_final_status,
        )

    def assert_input_box(self, x_lb: np.ndarray, x_ub: np.ndarray) -> None:
        low = np.array(x_lb, dtype=np.float32) * np.ones(self.input_layer.layer_size, dtype=np.float32)
        high = np.array(x_ub, dtype=np.float32) * np.ones(self.input_layer.layer_size, dtype=np.float32)
        self.input_layer.set_input_bounds(low, high)
        self.deepPolyNets_DNN.property_region = 1

        for i in range(self.deepPolyNets_DNN.layerSizes[0]):
            neuron = self.deepPolyNets_DNN.layers[0].neurons[i]
            neuron.concrete_lower = low[i]
            neuron.concrete_upper = high[i]
            self.deepPolyNets_DNN.property_region *= high[i] - low[i]
            neuron.concrete_algebra_lower = np.array([low[i]])
            neuron.concrete_algebra_upper = np.array([high[i]])
            neuron.algebra_lower = np.array([low[i]])
            neuron.algebra_upper = np.array([high[i]])

    def symbolic_propagate(self) -> None:
        self.deepPolyNets_DNN.deeppoly()
        for i, layer in enumerate(self.dense_layers):
            for out_index in range(layer.layer_size):
                neuron = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index]
                layer.lb[out_index] = neuron.concrete_lower_noClip
                layer.ub[out_index] = neuron.concrete_upper_noClip
                layer.clipped_lb[out_index] = max(neuron.concrete_lower, 0)
                layer.clipped_ub[out_index] = max(neuron.concrete_upper, 0)
                if self.preimg_mode in {"abstr", "comp"}:
                    layer.actMode[out_index] = neuron.actMode

        for out_index in range(self.output_layer.layer_size):
            neuron = self.deepPolyNets_DNN.layers[-1].neurons[out_index]
            self.output_layer.lb[out_index] = neuron.concrete_lower_noClip
            self.output_layer.ub[out_index] = neuron.concrete_upper_noClip

    def _preimage_cache_root(self) -> Path:
        return self.config.preimage_cache_dir or (self.output_dir / "preimage_cache")

    def _preimage_cache_key(self) -> str:
        if not self.config.preimage_cache_key:
            raise ValueError(
                "A preimage cache key is required. Use --preimage-cache-key or run through "
                "scripts/run_robustness_pipeline.py so the key can be derived from the benchmark."
            )
        return self.config.preimage_cache_key

    def save_cached_preimage(self) -> Path:
        layers = [
            {
                "layer_index": int(layer.layer_index),
                "layer_size": int(layer.layer_size),
                "relaxed_lb": np.asarray(layer.relaxed_lb, dtype=np.float64),
                "relaxed_ub": np.asarray(layer.relaxed_ub, dtype=np.float64),
            }
            for layer in self.dense_layers
        ]
        cache_path = save_preimage_cache(
            cache_root=self._preimage_cache_root(),
            cache_key=self._preimage_cache_key(),
            layers=layers,
            scale_values=np.asarray(self.scaleValueSet, dtype=np.float64),
            metadata=self.config.preimage_cache_metadata or {},
        )
        LOGGER.info("Saved MILP preimage cache to %s", cache_path)
        return cache_path

    def load_cached_preimage(self) -> None:
        metadata, arrays = load_preimage_cache(
            cache_root=self._preimage_cache_root(),
            cache_key=self._preimage_cache_key(),
        )
        layer_indices = arrays["layer_indices"].astype(np.int64)
        layer_sizes = arrays["layer_sizes"].astype(np.int64)
        if len(layer_indices) != len(self.dense_layers):
            raise ValueError(
                f"Preimage cache has {len(layer_indices)} hidden layer(s), "
                f"but this model has {len(self.dense_layers)}."
            )

        for offset, layer in enumerate(self.dense_layers):
            cached_index = int(layer_indices[offset])
            cached_size = int(layer_sizes[offset])
            if cached_index != int(layer.layer_index) or cached_size != int(layer.layer_size):
                raise ValueError(
                    "Preimage cache does not match this model: "
                    f"cache layer {offset} has index/size {cached_index}/{cached_size}, "
                    f"model has {layer.layer_index}/{layer.layer_size}."
                )
            layer.relaxed_lb = arrays[f"relaxed_lb_{offset}"].astype(np.float32)
            layer.relaxed_ub = arrays[f"relaxed_ub_{offset}"].astype(np.float32)

        self.scaleValueSet = arrays["scale_values"].astype(np.float64).tolist()
        LOGGER.info("Loaded MILP preimage cache %s (%s)", self._preimage_cache_key(), metadata.get("format"))

    def backward_preimage_computation(self) -> None:
        if self.gp_model is None:
            raise RuntimeError("Cannot compute a MILP preimage without a solver model. Use load_cached_preimage() instead.")
        cur_layer = self.output_layer
        in_layer_index = len(self.dense_layers)
        for in_layer in reversed(self.dense_layers):
            in_layer_index -= 1
            scale_value = 0.0
            if self.preimg_mode in {"milp", "comp"}:
                scale_value = self.underPreImageMILP(in_layer_index, in_layer, cur_layer)
            if self.preimg_mode == "abstr" or (self.preimg_mode == "comp" and scale_value <= 0):
                scale_value = self.underPreImageAbstr(in_layer_index, in_layer, cur_layer)
            self.scaleValueSet[in_layer.layer_index - 1] = scale_value
            cur_layer = in_layer

    def underPreImageMILP(self, in_layer_index: int, in_layer: LayerEncoding, cur_layer: LayerEncoding) -> float:
        enc_start_time = time.time()
        var_ll: list[Any] = []
        prop_cstr_ll: list[Any] = []
        model_cstr_ll: list[Any] = []
        w = cur_layer.layer_paras[0]
        b = cur_layer.layer_paras[1]
        relaxScale = self.gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS)
        relaxScale_LL = [relaxScale]
        relu_after_bounds: list[tuple[float, float]] = []

        for in_index in range(in_layer.layer_size):
            neuron_val = in_layer.realVal[in_index]
            neuron_lb = in_layer.lb[in_index]
            neuron_ub = in_layer.ub[in_index]

            alpha_K = max(neuron_val - neuron_lb, 1e-3)
            beta_K = max(neuron_ub - neuron_val, 1e-3)

            model_cstr_ll.append(self.gp_model.addConstr(in_layer.alpha[in_index] == (alpha_K * relaxScale)))
            model_cstr_ll.append(self.gp_model.addConstr(in_layer.beta[in_index] == (beta_K * relaxScale)))
            model_cstr_ll.append(
                self.gp_model.addConstr(
                    in_layer.ub[in_index] + beta_K * relaxScale >= in_layer.lb[in_index] - alpha_K * relaxScale
                )
            )

            in_lb_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                in_index
            ].concrete_algebra_lower
            in_ub_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                in_index
            ].concrete_algebra_upper

            relaxed_lb_bias = in_lb_algebra[-1] - in_layer.alpha[in_index]
            relaxed_ub_bias = in_ub_algebra[-1] + in_layer.beta[in_index]

            symbolic_lb_expression = self._linear_combination(in_lb_algebra[:-1], self.input_gp_vars, relaxed_lb_bias)
            symbolic_ub_expression = self._linear_combination(in_ub_algebra[:-1], self.input_gp_vars, relaxed_ub_bias)
            pre_relu_bounds = self._pre_relu_bounds(float(neuron_lb), float(neuron_ub), float(alpha_K), float(beta_K), 100.0)
            post_relu_bounds = (max(0.0, pre_relu_bounds[0]), max(0.0, pre_relu_bounds[1]))
            relu_after_bounds.append(post_relu_bounds)

            model_cstr_ll.append(self.gp_model.addConstr(in_layer.gp_vars_before[in_index] <= symbolic_ub_expression))
            model_cstr_ll.append(self.gp_model.addConstr(in_layer.gp_vars_before[in_index] >= symbolic_lb_expression))
            model_cstr_ll.append(
                self.gp_model.addGenConstrMax(
                    in_layer.gp_vars_after[in_index],
                    [in_layer.gp_vars_before[in_index], 0],
                    operand_bounds=[pre_relu_bounds, (0.0, 0.0)],
                )
            )

        self.gp_model.update()

        cur_layer_before_bounds: list[tuple[float, float]] = []
        for out_index in range(cur_layer.layer_size):
            accumulation = self._linear_combination(w[out_index], in_layer.gp_vars_after, b[out_index])
            cur_layer_before_bounds.append(self._interval_linear_combination(w[out_index], relu_after_bounds, b[out_index]))
            model_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_before[out_index] == accumulation))

        enc_finish_time = time.time()
        self._stats["encoding_time"] += enc_finish_time - enc_start_time

        if cur_layer.layer_index == (len(self.dense_layers) + 1):
            other_vars = [
                cur_layer.gp_vars_before[i] for i in range(cur_layer.layer_size) if i != int(self.targetCls)
            ]
            other_bounds = [
                cur_layer_before_bounds[i] for i in range(cur_layer.layer_size) if i != int(self.targetCls)
            ]
            other_maximal = self.gp_model.addVar(lb=-1000, vtype=GRB.CONTINUOUS)
            prop_cstr_ll.append(
                self.gp_model.addGenConstrMax(other_maximal, other_vars, operand_bounds=other_bounds)
            )
            prop_cstr_ll.append(
                self.gp_model.addConstr(other_maximal >= cur_layer.gp_vars_before[self.targetCls] + self.tole)
            )
        else:
            sumOfK = 0
            for i in range(cur_layer.layer_size):
                k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                relaxScale_LL.append(k_i_lb)
                prop_cstr_ll.append(
                    self.gp_model.addConstr(
                        cur_layer.gp_vars_before[i] <= cur_layer.relaxed_lb_expression[i] - 1000 * (k_i_lb - 1) - 2 * self.tole
                    )
                )
                prop_cstr_ll.append(
                    self.gp_model.addConstr(
                        cur_layer.gp_vars_before[i] >= cur_layer.relaxed_lb_expression[i] - 1000 * k_i_lb + 2 * self.tole
                    )
                )
                sumOfK += k_i_lb

                k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                relaxScale_LL.append(k_i_ub)
                prop_cstr_ll.append(
                    self.gp_model.addConstr(
                        cur_layer.gp_vars_before[i] >= cur_layer.relaxed_ub_expression[i] + 1000 * (k_i_ub - 1) + 2 * self.tole
                    )
                )
                prop_cstr_ll.append(
                    self.gp_model.addConstr(
                        cur_layer.gp_vars_before[i] <= cur_layer.relaxed_ub_expression[i] + 1000 * k_i_ub - 2 * self.tole
                    )
                )
                sumOfK += k_i_ub

            prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

        self.gp_model.update()
        self.gp_model.setObjective(relaxScale, GRB.MINIMIZE)
        self.gp_model.update()
        self.gp_model.setParam("DualReductions", 0)
        opt_start_time = time.time()
        self.gp_model.optimize()
        opt_finish_time = time.time()
        self._stats["solving_time"] += opt_finish_time - opt_start_time

        scaleValue = -10000.0
        if self.gp_model.status == GRB.OPTIMAL:
            scaleValue = float(self.gp_model.value(relaxScale))
            for in_index in range(in_layer.layer_size):
                alpha = self.gp_model.value(in_layer.alpha[in_index])
                beta = self.gp_model.value(in_layer.beta[in_index])
                in_layer.relaxed_ub[in_index] = in_layer.ub[in_index] + beta
                in_layer.relaxed_lb[in_index] = in_layer.lb[in_index] - alpha

                in_lb_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                    in_index
                ].concrete_algebra_lower
                in_ub_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                    in_index
                ].concrete_algebra_upper

                relaxed_lb_bias = in_lb_algebra[-1] - alpha
                relaxed_ub_bias = in_ub_algebra[-1] + beta
                in_layer.relaxed_lb_expression[in_index] = self._linear_combination(
                    in_lb_algebra[:-1], self.input_gp_vars, relaxed_lb_bias
                )
                in_layer.relaxed_ub_expression[in_index] = self._linear_combination(
                    in_ub_algebra[:-1], self.input_gp_vars, relaxed_ub_bias
                )
                if in_layer.relaxed_ub[in_index] <= 0:
                    in_layer.relaxed_ub_expression[in_index] = 0

            self.gp_model.remove(prop_cstr_ll)
            self.gp_model.remove(model_cstr_ll)
            self.gp_model.remove(relaxScale_LL)
            self.gp_model.remove(var_ll)
            self.gp_model.update()

        return scaleValue

    def underPreImageAbstr(self, in_layer_index: int, in_layer: LayerEncoding, cur_layer: LayerEncoding) -> float:
        model_cstr_ll: list[Any] = []
        prop_cstr_ll: list[Any] = []
        w = cur_layer.layer_paras[0]
        relaxScale = self.gp_model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS)
        relaxScale_LL = [relaxScale]

        for in_index in range(in_layer.layer_size):
            neuron_val = in_layer.realVal[in_index]
            actMode = in_layer.actMode[in_index]
            neuron_lb = in_layer.lb[in_index]
            neuron_ub = in_layer.ub[in_index]

            if actMode == 1:
                alpha_K = neuron_val - neuron_lb
                beta_K = neuron_ub - neuron_val
                model_cstr_ll.append(self.gp_model.addConstr(in_layer.alpha_before[in_index] == (alpha_K * relaxScale)))
                model_cstr_ll.append(self.gp_model.addConstr(in_layer.beta_after[in_index] == (beta_K * relaxScale)))
                alpha_before_bounds = (0.0, max(0.0, float(alpha_K) * 1000.0))
                model_cstr_ll.append(
                    self.gp_model.addGenConstrMin(
                        in_layer.alpha_after[in_index],
                        [in_layer.alpha_before[in_index], float(in_layer.lb[in_index])],
                        operand_bounds=[alpha_before_bounds, (float(in_layer.lb[in_index]), float(in_layer.lb[in_index]))],
                    )
                )
            elif actMode == 2:
                continue
            else:
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.alpha_after[in_index] == (-neuron_lb * relaxScale))
                )
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.beta_after[in_index] == (neuron_ub * relaxScale))
                )

        self.gp_model.update()

        for out_index in range(cur_layer.layer_size):
            weights = w[out_index]
            tmp_add_lower = 0
            tmp_add_upper = 0

            for in_index in range(in_layer.layer_size):
                actMode = in_layer.actMode[in_index]
                if actMode == 1:
                    if weights[in_index] >= 0:
                        tmp_add_lower -= weights[in_index] * in_layer.alpha_after[in_index]
                        tmp_add_upper += weights[in_index] * in_layer.beta_after[in_index]
                    else:
                        tmp_add_lower += weights[in_index] * in_layer.beta_after[in_index]
                        tmp_add_upper -= weights[in_index] * in_layer.alpha_after[in_index]
                elif actMode == 2:
                    continue
                elif actMode == 3:
                    K = in_layer.ub[in_index] / (in_layer.ub[in_index] - in_layer.lb[in_index])
                    if weights[in_index] >= 0:
                        tmp_add_upper += weights[in_index] * K * (
                            in_layer.beta_after[in_index] + in_layer.alpha_after[in_index]
                        )
                    else:
                        tmp_add_lower += weights[in_index] * K * (
                            in_layer.beta_after[in_index] + in_layer.alpha_after[in_index]
                        )
                else:
                    K = in_layer.ub[in_index] / (in_layer.ub[in_index] - in_layer.lb[in_index])
                    if weights[in_index] >= 0:
                        tmp_add_lower -= weights[in_index] * in_layer.alpha_after[in_index]
                        tmp_add_upper += weights[in_index] * K * (
                            in_layer.beta_after[in_index] + in_layer.alpha_after[in_index]
                        )
                    else:
                        tmp_add_lower += weights[in_index] * K * (
                            in_layer.beta_after[in_index] + in_layer.alpha_after[in_index]
                        )
                        tmp_add_upper -= weights[in_index] * in_layer.alpha_after[in_index]

            model_cstr_ll.append(
                self.gp_model.addConstr((tmp_add_lower + cur_layer.lb[out_index]) == cur_layer.gp_vars_lb_before[out_index])
            )
            model_cstr_ll.append(
                self.gp_model.addConstr((tmp_add_upper + cur_layer.ub[out_index]) == cur_layer.gp_vars_ub_before[out_index])
            )

        if cur_layer.layer_index == (len(self.dense_layers) + 1):
            for var_index, var in enumerate(cur_layer.gp_vars_ub_before):
                if var_index == self.targetCls:
                    continue
                prop_cstr_ll.append(
                    self.gp_model.addConstr(cur_layer.gp_vars_lb_before[self.targetCls] >= (var + 2 * self.tole))
                )
        else:
            for var_index, _ in enumerate(cur_layer.gp_vars_lb_before):
                if cur_layer.actMode[var_index] == 1:
                    prop_cstr_ll.append(
                        self.gp_model.addConstr(cur_layer.gp_vars_ub_before[var_index] <= cur_layer.relaxed_ub[var_index])
                    )
                    prop_cstr_ll.append(
                        self.gp_model.addConstr(cur_layer.gp_vars_lb_before[var_index] >= cur_layer.relaxed_lb[var_index])
                    )
                elif cur_layer.actMode[var_index] == 2:
                    prop_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_ub_before[var_index] <= 0))
                else:
                    prop_cstr_ll.append(
                        self.gp_model.addConstr(cur_layer.gp_vars_ub_before[var_index] <= cur_layer.relaxed_ub[var_index])
                    )
                    prop_cstr_ll.append(
                        self.gp_model.addConstr(cur_layer.gp_vars_lb_before[var_index] >= cur_layer.relaxed_lb[var_index])
                    )

        self.gp_model.update()
        self.gp_model.setObjective(relaxScale, GRB.MAXIMIZE)
        self.gp_model.update()
        self.gp_model.setParam("DualReductions", 0)
        self.gp_model.optimize()

        if self.gp_model.status != GRB.OPTIMAL:
            return 0.0

        scaleValue = float(self.gp_model.value(relaxScale))
        for in_index in range(in_layer.layer_size):
            alpha_after = self.gp_model.value(in_layer.alpha_after[in_index])
            beta_after = self.gp_model.value(in_layer.beta_after[in_index])

            if in_layer.ub[in_index] <= 0:
                in_layer.relaxed_ub[in_index] = 0
                in_layer.relaxed_lb[in_index] = in_layer.lb[in_index] - alpha_after
            else:
                in_layer.relaxed_ub[in_index] = np.float32(in_layer.ub[in_index] + beta_after)
                in_layer.relaxed_lb[in_index] = np.float32(in_layer.lb[in_index] - alpha_after)

            in_lb_algebra = deepcopy(
                self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[in_index].concrete_algebra_lower
            )
            in_ub_algebra = deepcopy(
                self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[in_index].concrete_algebra_upper
            )

            relaxed_lb_bias = in_lb_algebra[-1] - alpha_after
            relaxed_ub_bias = in_ub_algebra[-1] + beta_after
            in_layer.relaxed_lb_expression[in_index] = self._linear_combination(
                in_lb_algebra[:-1], self.input_gp_vars, relaxed_lb_bias
            )
            in_layer.relaxed_ub_expression[in_index] = self._linear_combination(
                in_ub_algebra[:-1], self.input_gp_vars, relaxed_ub_bias
            )
            if in_layer.ub[in_index] <= 0:
                in_layer.relaxed_ub_expression[in_index] = 0

        self.gp_model.remove(prop_cstr_ll)
        self.gp_model.remove(model_cstr_ll)
        self.gp_model.remove(relaxScale_LL)
        self.gp_model.update()
        return scaleValue

    def forward_quantization_with_esbmc(self) -> tuple[bool, Any, Any, Any]:
        non_input_layers = [*self.dense_layers, self.output_layer]
        selected_q = [0] * len(non_input_layers)
        selected_f = [0] * len(non_input_layers)
        selected_i = [0] * len(non_input_layers)
        terminal_statuses: list[str] = []

        def search(layer_index: int) -> bool:
            if layer_index >= len(non_input_layers):
                return True

            cur_layer = non_input_layers[layer_index]
            in_layer = (
                self.input_layer
                if cur_layer.layer_index == 1
                else self.dense_layers[cur_layer.layer_index - 2]
            )
            weights = cur_layer.layer_paras[0]
            biases = cur_layer.layer_paras[1]
            frac_bit = int(self.bit_lb)

            while frac_bit <= int(self.bit_ub):
                int_bit = int(cur_layer.int_bit)
                all_bit = frac_bit + int_bit
                qu_w_int = np.asarray(
                    quantize_int(weights, all_bit, frac_bit),
                    dtype=np.int64,
                )
                qu_b_int = np.asarray(
                    quantize_int(biases, all_bit, frac_bit),
                    dtype=np.int64,
                )

                is_output_layer = layer_index == len(non_input_layers) - 1
                if self.error_budget_mode == "derived" and is_output_layer:
                    margin_record = self._record_output_margin_check(
                        cur_layer=cur_layer,
                        in_layer=in_layer,
                        weights_int=qu_w_int,
                        layer_index=layer_index,
                        all_bit=all_bit,
                        frac_bit=frac_bit,
                    )
                    if not margin_record["margin_ok"]:
                        terminal_statuses.append("MARGIN_TOO_SMALL")
                        LOGGER.warning(
                            "Rejecting output bits(Q=%s,F=%s): derived classification margin is too small.",
                            all_bit,
                            frac_bit,
                        )
                        frac_bit += 1
                        continue

                filtered = self._candidate_filtered_by_counterexample(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    qu_w_int=qu_w_int,
                    qu_b_int=qu_b_int,
                    frac_bit=frac_bit,
                    all_bit=all_bit,
                    layer_index=layer_index,
                )
                if filtered is not None:
                    terminal_statuses.append("FAILED")
                    frac_bit += 1
                    continue

                esbmc_result = self.verify_layer_with_esbmc(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    qu_w_int=qu_w_int,
                    qu_b_int=qu_b_int,
                    frac_bit=frac_bit,
                    all_bit=all_bit,
                    layer_index=layer_index,
                )
                if esbmc_result.status != "VERIFIED":
                    terminal_statuses.append(str(esbmc_result.status))
                counterexample = self._record_failed_counterexample(
                    result=esbmc_result,
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    qu_w_int=qu_w_int,
                    qu_b_int=qu_b_int,
                    frac_bit=frac_bit,
                    all_bit=all_bit,
                    layer_index=layer_index,
                )

                if (
                    self.cex_feedback == "filter+jump"
                    and counterexample is not None
                    and counterexample.get("counterexample_preclamp") is not None
                ):
                    preclamp = abs(int(counterexample["counterexample_preclamp"]))
                    real_magnitude = max(
                        float(preclamp) / float(1 << frac_bit),
                        1.0 / float(1 << frac_bit),
                    )
                    needed_int_bit = max(
                        int_bit,
                        int(math.ceil(math.log2(real_magnitude))) + 1,
                    )
                    needed_int_bit = min(needed_int_bit, int(self.bit_ub))
                    if needed_int_bit > int_bit:
                        jump = {
                            "from": int(int_bit),
                            "to": int(needed_int_bit),
                            "F": int(frac_bit),
                        }
                        self.cex_bit_jumps.setdefault(layer_index, []).append(jump)
                        LOGGER.info(
                            "CEGIS integer-bit jump layer=%s F=%s I=%s->%s",
                            layer_index,
                            frac_bit,
                            int_bit,
                            needed_int_bit,
                        )
                        cur_layer.int_bit = needed_int_bit
                        continue

                if esbmc_result.status != "VERIFIED":
                    frac_bit += 1
                    continue

                if not is_output_layer:
                    chaining_record = self._record_hidden_chaining_check(
                        cur_layer=cur_layer,
                        layer_index=layer_index,
                        all_bit=all_bit,
                        frac_bit=frac_bit,
                        in_layer=in_layer,
                        weights_int=qu_w_int,
                    )
                    if (
                        not chaining_record["chaining_ok"]
                        and self.enforce_contract_chaining
                    ):
                        terminal_statuses.append("FAILED")
                        frac_bit += 1
                        continue

                vacuity_record = self._run_vacuity_sentinel(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    layer_index=layer_index,
                    all_bit=all_bit,
                    frac_bit=frac_bit,
                )
                if vacuity_record["status"] not in {"NONVACUOUS", "SKIPPED"}:
                    terminal_statuses.append(str(vacuity_record["status"]))
                    frac_bit += 1
                    continue

                cur_layer.frac_bit = frac_bit
                selected_q[layer_index] = all_bit
                selected_f[layer_index] = frac_bit
                selected_i[layer_index] = _export_integer_bits(int_bit)
                self.update_quantized_weights_affine(
                    in_layer,
                    cur_layer,
                    all_bit,
                    frac_bit,
                    frac_bit,
                    layer_index,
                )

                if search(layer_index + 1):
                    return True

                # A downstream rejection can be caused by this layer's
                # inherited real error. Increase this shared layer precision
                # and re-establish every downstream contract.
                frac_bit += 1

            return False

        if search(0):
            self.synthesis_final_status = "VERIFIED"
            return True, selected_q, selected_f, selected_i

        if terminal_statuses and set(terminal_statuses) <= {"MARGIN_TOO_SMALL"}:
            self.synthesis_final_status = "MARGIN_TOO_SMALL"
        elif "TIMEOUT" in terminal_statuses:
            self.synthesis_final_status = "TIMEOUT"
        elif "MEMOUT" in terminal_statuses:
            self.synthesis_final_status = "MEMOUT"
        elif "UNKNOWN" in terminal_statuses or "ERROR" in terminal_statuses:
            self.synthesis_final_status = "UNKNOWN"
        elif terminal_statuses:
            self.synthesis_final_status = "FAILED"
        return False, None, None, None

    def verify_exported_quantization_with_esbmc(
        self,
        total_bits: list[int],
        fractional_bits: list[int],
        integer_bits: list[int],
        *,
        formal_saturation_check: bool = False,
        require_formal_saturation_check: bool = True,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Run the existing ESBMC layer checks for an explicit exported Q/I/F configuration.

        `integer_bits` follows the backend/export convention and excludes the sign bit.
        This method does not change the preimage methodology; it reuses the same generated
        layer contracts used by `forward_quantization_with_esbmc`. When requested, it also
        checks the fixed-point affine layer for formal no-saturation before clamp.
        """

        non_input_layers = self.dense_layers.copy()
        non_input_layers.append(self.output_layer)
        if not (len(total_bits) == len(fractional_bits) == len(integer_bits) == len(non_input_layers)):
            raise ValueError("Expected one Q/I/F entry per non-input layer.")

        records: list[dict[str, Any]] = []
        self.deepPolyNets_DNN.load_dnn(self.deep_model)

        for layer_index, cur_layer in enumerate(non_input_layers):
            in_layer = self.input_layer if cur_layer.layer_index == 1 else self.dense_layers[cur_layer.layer_index - 2]
            q_bits = int(total_bits[layer_index])
            f_bits = int(fractional_bits[layer_index])
            i_bits = int(integer_bits[layer_index])
            if q_bits != i_bits + f_bits + 1:
                records.append(
                    {
                        "layer_index": int(layer_index),
                        "total_bits": q_bits,
                        "integer_bits": i_bits,
                        "fractional_bits": f_bits,
                        "status": "INVALID_QIF",
                        "contract_status": "INVALID_QIF",
                        "contract_verified": False,
                        "no_saturation_formally_checked": False,
                        "no_saturation_status": "SKIPPED",
                        "no_saturation_verified": False,
                        "deployment_quality_accepted": True,
                        "final_status": "UNKNOWN",
                        "failure_type": "invalid_qif",
                    }
                )
                return False, records

            qu_w_int = quantize_int(cur_layer.layer_paras[0], q_bits, f_bits)
            qu_b_int = quantize_int(cur_layer.layer_paras[1], q_bits, f_bits)
            is_output_layer = cur_layer.layer_index == len(self.dense_layers) + 1
            if self.error_budget_mode == "derived" and is_output_layer:
                margin_record = self._record_output_margin_check(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    weights_int=np.asarray(qu_w_int),
                    layer_index=layer_index,
                    all_bit=q_bits,
                    frac_bit=f_bits,
                )
                if not margin_record["margin_ok"]:
                    records.append(
                        {
                            "layer_index": int(layer_index),
                            "total_bits": q_bits,
                            "integer_bits": i_bits,
                            "fractional_bits": f_bits,
                            "status": "MARGIN_TOO_SMALL",
                            "contract_status": "SKIPPED",
                            "contract_verified": False,
                            "no_saturation_formally_checked": False,
                            "no_saturation_status": "SKIPPED",
                            "no_saturation_verified": False,
                            "deployment_quality_accepted": True,
                            "final_status": "MARGIN_TOO_SMALL",
                            "failure_type": "derived_output_margin_too_small",
                            "output_margin": margin_record,
                        }
                    )
                    return False, records

            contract_result = self.verify_layer_with_esbmc(
                cur_layer=cur_layer,
                in_layer=in_layer,
                qu_w_int=np.asarray(qu_w_int),
                qu_b_int=np.asarray(qu_b_int),
                frac_bit=f_bits,
                all_bit=q_bits,
                layer_index=layer_index,
            )
            record: dict[str, Any] = {
                "layer_index": int(layer_index),
                "total_bits": q_bits,
                "integer_bits": i_bits,
                "fractional_bits": f_bits,
                "status": contract_result.status,
                "contract_status": contract_result.status,
                "contract_verified": contract_result.status == "VERIFIED",
                "no_saturation_formally_checked": False,
                "no_saturation_status": "PENDING" if formal_saturation_check else "SKIPPED",
                "no_saturation_verified": False,
                "deployment_quality_accepted": True,
                "final_status": "UNKNOWN",
                "blocks": [dict(block) for block in contract_result.blocks],
                "resource_control": contract_result.resource_control,
            }
            if contract_result.status != "VERIFIED":
                record["status"] = contract_result.status
                record["final_status"] = "FAILED" if contract_result.status == "FAILED" else "UNKNOWN"
                record["no_saturation_status"] = "SKIPPED"
                records.append(record)
                return False, records

            if cur_layer.layer_index < (len(self.dense_layers) + 1):
                chaining_record = self._record_hidden_chaining_check(
                    cur_layer=cur_layer,
                    layer_index=layer_index,
                    all_bit=q_bits,
                    frac_bit=f_bits,
                    in_layer=in_layer,
                    weights_int=np.asarray(qu_w_int),
                )
                record["chaining_ok"] = bool(chaining_record["chaining_ok"])
                record["chaining_enforced"] = bool(self.enforce_contract_chaining)
                record["chaining"] = chaining_record
                if not chaining_record["chaining_ok"] and self.enforce_contract_chaining:
                    record["status"] = "FAILED"
                    record["final_status"] = "FAILED"
                    record["failure_type"] = "assume_guarantee_chain_failed"
                    record["no_saturation_status"] = "SKIPPED"
                    records.append(record)
                    return False, records
            vacuity_record = self._run_vacuity_sentinel(
                cur_layer=cur_layer,
                in_layer=in_layer,
                layer_index=layer_index,
                all_bit=q_bits,
                frac_bit=f_bits,
            )
            record["vacuity_check"] = dict(vacuity_record)
            record["assumption_box_cardinality"] = vacuity_record.get(
                "assumption_box_cardinality"
            )
            if vacuity_record["status"] not in {"NONVACUOUS", "SKIPPED"}:
                record["status"] = vacuity_record["status"]
                record["final_status"] = vacuity_record["status"]
                record["failure_type"] = "vacuous_or_inconclusive_assumption_box"
                record["no_saturation_status"] = "SKIPPED"
                records.append(record)
                return False, records

            if formal_saturation_check:
                no_saturation_result = self.verify_layer_no_saturation_with_esbmc(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    qu_w_int=np.asarray(qu_w_int),
                    qu_b_int=np.asarray(qu_b_int),
                    frac_bit=f_bits,
                    all_bit=q_bits,
                    layer_index=layer_index,
                )
                no_saturation_blocks = [dict(block) for block in no_saturation_result.blocks]
                no_saturation_checked = bool(no_saturation_blocks) or no_saturation_result.status != "SKIPPED"
                no_saturation_verified = no_saturation_result.status == "VERIFIED"
                record["no_saturation_formally_checked"] = no_saturation_checked
                record["no_saturation_status"] = no_saturation_result.status
                record["no_saturation_verified"] = no_saturation_verified
                record["no_saturation_blocks"] = no_saturation_blocks
                record["no_saturation_resource_control"] = no_saturation_result.resource_control
                if no_saturation_result.status != "VERIFIED":
                    record["status"] = no_saturation_result.status
                    record["final_status"] = "FAILED" if require_formal_saturation_check else "PARTIAL_VERIFIED"
                    record["failure_type"] = (
                        "formal_saturation_possible"
                        if no_saturation_result.status == "FAILED"
                        else "formal_saturation_inconclusive"
                    )
                    records.append(record)
                    if require_formal_saturation_check:
                        return False, records
                    cur_layer.frac_bit = f_bits
                    self.update_quantized_weights_affine(in_layer, cur_layer, q_bits, f_bits, f_bits, layer_index)
                    continue

            if not formal_saturation_check:
                record["final_status"] = "PARTIAL_VERIFIED"
            else:
                record["final_status"] = "VERIFIED"
            record["status"] = record["final_status"]
            records.append(record)
            cur_layer.frac_bit = f_bits
            self.update_quantized_weights_affine(in_layer, cur_layer, q_bits, f_bits, f_bits, layer_index)

        return True, records

    def _uses_hidden_block_verification(self, cur_layer: LayerEncoding) -> bool:
        return (
            self.esbmc_layer_block_size > 0
            and cur_layer.layer_index < (len(self.dense_layers) + 1)
        )

    def _hidden_block_ranges(self, output_size: int) -> list[tuple[int, int]]:
        if self.esbmc_layer_block_size <= 0:
            return [(0, output_size)]
        return [
            (start, min(start + self.esbmc_layer_block_size, output_size))
            for start in range(0, output_size, self.esbmc_layer_block_size)
        ]

    @staticmethod
    def _record_block_status(status: str) -> str:
        if status in {"VERIFIED", "FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN"}:
            return status
        return "UNKNOWN"

    @staticmethod
    def _candidate_rejection_status(records: list[dict[str, Any]]) -> str:
        statuses = {str(record.get("status")) for record in records}
        if statuses == {"VERIFIED"}:
            return "VERIFIED"
        for status in ("MEMOUT", "TIMEOUT", "FAILED", "UNKNOWN"):
            if status in statuses:
                return status
        return "UNKNOWN"

    @staticmethod
    def _aggregate_no_saturation_status(records: list[dict[str, Any]]) -> str:
        statuses = {str(record.get("status")) for record in records if record.get("status") != "SKIPPED"}
        if statuses == {"VERIFIED"}:
            return "VERIFIED"
        if "FAILED" in statuses:
            return "FAILED"
        if "TIMEOUT" in statuses:
            return "TIMEOUT"
        if "MEMOUT" in statuses:
            return "MEMOUT"
        return "UNKNOWN"

    def _should_fail_fast_blocks(self) -> bool:
        return bool(
            self.blockwise_fail_fast
            and not self.blockwise_run_all_blocks_on_failure
        )

    def _should_stop_no_saturation_blocks(self, status: str) -> bool:
        if status == "FAILED":
            return True
        if status in {"TIMEOUT", "MEMOUT", "UNKNOWN"}:
            return not self.no_saturation_continue_on_unknown
        return False

    def _esbmc_call_record(
        self,
        *,
        result: ESBMCResult,
        layer_index: int,
        block_index: int | None,
        start_neuron: int | None,
        end_neuron: int | None,
        all_bit: int,
        frac_bit: int,
        harness: Path | None,
        property_type: str = "preimage",
        mode: str = "full_layer",
        input_dim: int | None = None,
        output_neurons: int | None = None,
        status: str | None = None,
        reason: str | None = None,
        skipped_due_to_fail_fast: bool = False,
    ) -> dict[str, Any]:
        record_status = status or self._record_block_status(result.status)
        record: dict[str, Any] = {
            "layer_index": int(layer_index),
            "Q": int(all_bit),
            "I": int(max(all_bit - frac_bit - 1, 0)),
            "F": int(frac_bit),
            "status": record_status,
            "time": float(result.elapsed_seconds),
            "elapsed_seconds": float(result.elapsed_seconds),
            "return_code": int(result.return_code),
            "timeout": f"{int(result.timeout_seconds)}s",
            "memlimit": str(result.memlimit),
            "command": list(result.command),
            "stdout_log_path": result.stdout_log_path,
            "stderr_log_path": result.stderr_log_path,
            "peak_memory_bytes": result.peak_memory_bytes,
            "peak_memory_mib": (
                float(result.peak_memory_bytes / (1024 * 1024))
                if result.peak_memory_bytes is not None
                else None
            ),
            "memory_measurement": result.memory_measurement,
            "resource_control": result.resource_control,
            "harness": str(harness) if harness is not None else None,
            "property_type": property_type,
            "mode": mode,
            "counterexample_inputs": (
                [int(value) for value in result.counterexample_inputs]
                if result.counterexample_inputs is not None
                else None
            ),
            "counterexample_neuron": result.counterexample_neuron,
            "counterexample_preclamp": result.counterexample_preclamp,
        }
        if input_dim is not None:
            record["input_dim"] = int(input_dim)
        if output_neurons is not None:
            record["neurons_per_query"] = int(output_neurons)
        if input_dim is not None and output_neurons is not None:
            record["estimated_macs"] = int(input_dim) * int(output_neurons)
        if block_index is not None:
            record["block_index"] = int(block_index)
        if start_neuron is not None:
            record["start_neuron"] = int(start_neuron)
        if end_neuron is not None:
            record["end_neuron"] = int(end_neuron)
        if result.status != record_status:
            record["raw_status"] = result.status
        if reason:
            record["reason"] = reason
        if skipped_due_to_fail_fast:
            record["skipped_due_to_fail_fast"] = True
        return record

    def _layer_input_bounds_int(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        scale: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cur_layer.layer_index == 1:
            fallback_low = getattr(
                in_layer,
                "lb",
                np.zeros(in_layer.layer_size, dtype=np.float64),
            )
            fallback_high = getattr(
                in_layer,
                "ub",
                np.zeros(in_layer.layer_size, dtype=np.float64),
            )
            x_lo = np.array(getattr(self, "x_low_real", fallback_low), dtype=np.float64)
            x_hi = np.array(getattr(self, "x_high_real", fallback_high), dtype=np.float64)
        elif (
            (self.error_budget_mode == "derived" or self.propagate_contract_tolerance)
            and getattr(in_layer, "verified_activation_lb", None) is not None
            and getattr(in_layer, "verified_activation_ub", None) is not None
        ):
            x_lo = np.array(in_layer.verified_activation_lb, dtype=np.float64)
            x_hi = np.array(in_layer.verified_activation_ub, dtype=np.float64)
        else:
            x_lo = np.array(in_layer.clipped_lb, dtype=np.float64)
            x_hi = np.array(in_layer.clipped_ub, dtype=np.float64)
        return (
            np.floor(x_lo * scale).astype(np.int64),
            np.ceil(x_hi * scale).astype(np.int64),
        )

    def _input_fractional_bits(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        output_fractional_bits: int,
    ) -> int:
        """Return the input scale used by the deployed fixed-point layer."""

        if self.error_budget_mode == "derived" and cur_layer.layer_index > 1:
            if in_layer.frac_bit is None:
                raise RuntimeError(
                    "Derived error budgets require the accepted previous-layer fractional width."
                )
            return int(in_layer.frac_bit)
        return int(output_fractional_bits)

    def _assumption_box_int(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        output_fractional_bits: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        frac_in = self._input_fractional_bits(
            cur_layer,
            in_layer,
            output_fractional_bits,
        )
        input_scale = 1 << frac_in
        assumed_lo, assumed_hi = self._layer_input_bounds_int(
            cur_layer,
            in_layer,
            input_scale,
        )
        return assumed_lo, assumed_hi, frac_in

    @staticmethod
    def _assumption_box_cardinality(
        assumed_lo_int: np.ndarray,
        assumed_hi_int: np.ndarray,
    ) -> tuple[str, bool]:
        low = np.asarray(assumed_lo_int, dtype=object).reshape(-1)
        high = np.asarray(assumed_hi_int, dtype=object).reshape(-1)
        valid = all(int(hi) >= int(lo) for lo, hi in zip(low, high))
        if not valid:
            return "0", False
        cardinality = 1
        for lo, hi in zip(low, high):
            cardinality *= int(hi) - int(lo) + 1
        return str(cardinality), True

    def _layer_preimage_bounds_int(
        self,
        cur_layer: LayerEncoding,
        scale: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        pre_lo = np.array(cur_layer.relaxed_lb if cur_layer.relaxed_lb is not None else cur_layer.lb, dtype=np.float64)
        pre_hi = np.array(cur_layer.relaxed_ub if cur_layer.relaxed_ub is not None else cur_layer.ub, dtype=np.float64)
        return (
            np.floor(pre_lo * scale).astype(np.int64),
            np.ceil(pre_hi * scale).astype(np.int64),
        )

    def _contract_tolerance_int(
        self,
        pre_lo_int: np.ndarray,
        pre_hi_int: np.ndarray,
        scale: int,
        *,
        cur_layer: LayerEncoding | None = None,
        weights_int: np.ndarray | None = None,
        assumed_lo_int: np.ndarray | None = None,
        assumed_hi_int: np.ndarray | None = None,
        frac_in: int | None = None,
        delta_in_int: np.ndarray | int = 0,
    ) -> np.ndarray:
        """Return the integer slack emitted in hidden contract harnesses."""

        pre_lo = np.asarray(pre_lo_int, dtype=np.int64)
        pre_hi = np.asarray(pre_hi_int, dtype=np.int64)
        mode = self.__dict__.get("error_budget_mode")
        if mode is None:
            mode = (
                "heuristic"
                if bool(getattr(self, "unsound_contract_tolerance", False))
                else "zero"
            )
        if mode == "derived":
            if (
                cur_layer is None
                or weights_int is None
                or assumed_lo_int is None
                or assumed_hi_int is None
                or frac_in is None
            ):
                raise ValueError("Derived error budgets require layer, weights, input box, and frac_in.")
            return self._derived_error_budget_int(
                cur_layer=cur_layer,
                weights_int=weights_int,
                assumed_lo_int=assumed_lo_int,
                assumed_hi_int=assumed_hi_int,
                frac_in=frac_in,
                delta_in_int=delta_in_int,
                frac_out=int(scale).bit_length() - 1,
            )
        if mode == "zero":
            return np.zeros_like(pre_lo, dtype=np.int64)

        abs_tol = int(scale) // 1000
        rel_tol_num = 1
        rel_tol_den = 100
        ranges = np.abs(pre_hi - pre_lo)
        return (abs_tol + (rel_tol_num * ranges) // rel_tol_den).astype(np.int64)

    def _derived_error_budget_int(
        self,
        cur_layer: LayerEncoding,
        weights_int: np.ndarray,
        assumed_lo_int: np.ndarray,
        assumed_hi_int: np.ndarray,
        frac_in: int,
        delta_in_int: np.ndarray | int,
        frac_out: int | None = None,
    ) -> np.ndarray:
        """Bound implementation/real-affine deviation in output-integer ULPs.

        Let ``S_in = 2**frac_in`` and ``S_out`` be the weight/output scale.
        An input integer ``A_j`` denotes ``A_j / S_in`` real units. Rounding a
        real weight to ``W_int / S_out`` introduces at most
        ``0.5 / S_out`` real error. Multiplication by ``|A_j| / S_in`` and
        conversion back to output ULPs (multiplication by ``S_out``) therefore
        contributes ``0.5 * |A_j| / S_in`` output ULPs. Summing and taking the
        ceiling gives ``delta_weights``.

        An inherited input error ``delta_in_j`` is measured in input ULPs.
        Its contribution is
        ``S_out * |W_real_ij| * delta_in_j / S_in`` output ULPs. Using
        ``|W_int|`` alone is not conservative when a weight rounds toward
        zero. Production calls therefore use the exact stored float weights;
        callers without them use the safe inequality
        ``S_out*|W_real| <= |W_int| + 1/2``.

        RHAZ plus quantized-bias error contributes at most one output ULP in
        total. ReLU is 1-Lipschitz, so no additional activation factor is
        needed.
        """

        weights = np.asarray(weights_int, dtype=object)
        low = np.asarray(assumed_lo_int, dtype=object).reshape(-1)
        high = np.asarray(assumed_hi_int, dtype=object).reshape(-1)
        if weights.ndim != 2 or weights.shape[1] != low.size or high.size != low.size:
            raise ValueError("Derived error-budget dimensions do not match the affine layer.")
        if int(frac_in) < 0:
            raise ValueError("frac_in must be non-negative.")

        if np.isscalar(delta_in_int):
            delta_vec = np.full(low.size, int(delta_in_int), dtype=object)
        else:
            delta_vec = np.asarray(delta_in_int, dtype=object).reshape(-1)
            if delta_vec.size != low.size:
                raise ValueError("Inherited error budget size does not match the layer input.")
        if any(int(value) < 0 for value in delta_vec):
            raise ValueError("Inherited error budgets must be non-negative.")

        scale_in = 1 << int(frac_in)
        max_abs_input = np.asarray(
            [max(abs(int(lo)), abs(int(hi))) for lo, hi in zip(low, high)],
            dtype=object,
        )
        max_abs_sum = sum(int(value) for value in max_abs_input)
        delta_weights_scalar = (max_abs_sum + (2 * scale_in) - 1) // (2 * scale_in)

        real_weights: np.ndarray | None = None
        layer_parameters = getattr(cur_layer, "layer_paras", None)
        if (
            frac_out is not None
            and layer_parameters is not None
            and len(layer_parameters) >= 1
        ):
            candidate = np.asarray(layer_parameters[0])
            if candidate.shape == weights.shape:
                real_weights = candidate

        budget_values: list[int] = []
        for neuron, row in enumerate(weights):
            if real_weights is not None:
                output_scale = 1 << int(frac_out)
                amplified = sum(
                    abs(Fraction.from_float(float(weight)))
                    * output_scale
                    * int(delta)
                    for weight, delta in zip(real_weights[neuron], delta_vec)
                )
                delta_input = self._ceil_fraction(amplified / scale_in)
            else:
                # 2*S_out*|W_real| <= 2*|W_int| + 1.
                amplified_twice = sum(
                    (2 * abs(int(weight)) + 1) * int(delta)
                    for weight, delta in zip(row, delta_vec)
                )
                delta_input = (
                    amplified_twice + (2 * scale_in) - 1
                ) // (2 * scale_in)
            budget_values.append(int(delta_weights_scalar + 1 + delta_input))

        max_int64 = int(np.iinfo(np.int64).max)
        if any(value > max_int64 for value in budget_values):
            raise OverflowError("Derived error budget exceeds int64 reporting range.")
        budget = np.asarray(budget_values, dtype=np.int64)
        assert np.all(budget >= 1)
        return budget

    @staticmethod
    def _floor_fraction(value: Fraction) -> int:
        return value.numerator // value.denominator

    @staticmethod
    def _ceil_fraction(value: Fraction) -> int:
        return -((-value.numerator) // value.denominator)

    def _inherited_error_budget_int(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
    ) -> np.ndarray:
        if cur_layer.layer_index == 1:
            return np.zeros(in_layer.layer_size, dtype=np.int64)
        inherited = getattr(in_layer, "error_budget_int", None)
        if inherited is None:
            if self.error_budget_mode == "derived":
                raise RuntimeError("Previous layer has no accepted derived error budget.")
            return np.zeros(in_layer.layer_size, dtype=np.int64)
        return np.asarray(inherited, dtype=np.int64)

    def _candidate_error_budget_int(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        weights_int: np.ndarray,
        frac_bit: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        output_scale = 1 << int(frac_bit)
        pre_lo_int, pre_hi_int = self._layer_preimage_bounds_int(cur_layer, output_scale)
        assumed_lo, assumed_hi, frac_in = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        budget = self._contract_tolerance_int(
            pre_lo_int,
            pre_hi_int,
            output_scale,
            cur_layer=cur_layer,
            weights_int=weights_int,
            assumed_lo_int=assumed_lo,
            assumed_hi_int=assumed_hi,
            frac_in=frac_in,
            delta_in_int=self._inherited_error_budget_int(cur_layer, in_layer),
        )
        return budget, assumed_lo, assumed_hi, frac_in

    def _real_affine_bounds_on_integer_box(
        self,
        cur_layer: LayerEncoding,
        assumed_lo_int: np.ndarray,
        assumed_hi_int: np.ndarray,
        *,
        frac_in: int,
        frac_out: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact outward real-affine bounds in output integer ULPs.

        The model parameters are stored IEEE floating-point values. Converting
        each value with ``Fraction.from_float`` makes this computation exact
        for those model values, avoiding an unsound float-rounding gap in the
        decision-margin guard.
        """

        weights = np.asarray(cur_layer.layer_paras[0])
        biases = np.asarray(cur_layer.layer_paras[1])
        low = np.asarray(assumed_lo_int, dtype=object).reshape(-1)
        high = np.asarray(assumed_hi_int, dtype=object).reshape(-1)
        if weights.ndim != 2 or weights.shape[1] != low.size or high.size != low.size:
            raise ValueError("Real affine bound dimensions do not match the assumption box.")

        input_scale = 1 << int(frac_in)
        output_scale = 1 << int(frac_out)
        lower_values: list[int] = []
        upper_values: list[int] = []
        for row, bias in zip(weights, biases):
            lower = Fraction.from_float(float(bias))
            upper = Fraction.from_float(float(bias))
            for weight, lo, hi in zip(row, low, high):
                coefficient = Fraction.from_float(float(weight))
                lo_real = Fraction(int(lo), input_scale)
                hi_real = Fraction(int(hi), input_scale)
                if coefficient >= 0:
                    lower += coefficient * lo_real
                    upper += coefficient * hi_real
                else:
                    lower += coefficient * hi_real
                    upper += coefficient * lo_real
            lower_values.append(self._floor_fraction(lower * output_scale))
            upper_values.append(self._ceil_fraction(upper * output_scale))

        return (
            np.asarray(lower_values, dtype=np.int64),
            np.asarray(upper_values, dtype=np.int64),
        )

    @staticmethod
    def _containment_margins_int(
        guaranteed_low: np.ndarray,
        guaranteed_high: np.ndarray,
        assumed_low: np.ndarray,
        assumed_high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        lower_margin = np.asarray(guaranteed_low, dtype=np.int64) - np.asarray(assumed_low, dtype=np.int64)
        upper_margin = np.asarray(assumed_high, dtype=np.int64) - np.asarray(guaranteed_high, dtype=np.int64)
        lower_ok = lower_margin >= 0
        upper_ok = upper_margin >= 0
        ok = bool(np.all(lower_ok & upper_ok))
        violation_indices = np.flatnonzero(~(lower_ok & upper_ok))
        return lower_margin, upper_margin, ok, violation_indices

    def _store_verified_activation_bounds(
        self,
        cur_layer: LayerEncoding,
        guaranteed_low_int: np.ndarray,
        guaranteed_high_int: np.ndarray,
        scale: int,
        *,
        source: str,
    ) -> None:
        cur_layer.verified_activation_lb = np.asarray(guaranteed_low_int, dtype=np.float64) / float(scale)
        cur_layer.verified_activation_ub = np.asarray(guaranteed_high_int, dtype=np.float64) / float(scale)
        cur_layer.verified_activation_source = source

    def _record_hidden_chaining_check(
        self,
        cur_layer: LayerEncoding,
        layer_index: int,
        all_bit: int,
        frac_bit: int,
        in_layer: LayerEncoding | None = None,
        weights_int: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Check that a hidden contract guarantee composes with the next input box.

        The ESBMC hidden harness proves a pre-activation guarantee. The next layer
        assumes the current layer's post-activation clipped box. In integer domain,
        ReLU(G_l +/- tol) must be contained in that assumed box.
        """

        scale = 1 << int(frac_bit)
        pre_lo_int, pre_hi_int = self._layer_preimage_bounds_int(cur_layer, scale)
        if self.error_budget_mode == "derived":
            if in_layer is None or weights_int is None:
                raise ValueError("Derived chaining requires the input layer and quantized weights.")
            tolerance_int, _, _, _ = self._candidate_error_budget_int(
                cur_layer,
                in_layer,
                weights_int,
                frac_bit,
            )
        else:
            tolerance_int = self._contract_tolerance_int(pre_lo_int, pre_hi_int, scale)

        if self.error_budget_mode == "derived":
            q_min = -(1 << (int(all_bit) - 1))
            q_max = (1 << (int(all_bit) - 1)) - 1
            guaranteed_low = np.maximum(np.clip(pre_lo_int - tolerance_int, q_min, q_max), 0)
            guaranteed_high = np.maximum(np.clip(pre_hi_int + tolerance_int, q_min, q_max), 0)
        else:
            guaranteed_low = np.maximum(pre_lo_int - tolerance_int, 0)
            guaranteed_high = np.maximum(pre_hi_int + tolerance_int, 0)

        legacy_assumed_low = np.floor(np.asarray(cur_layer.clipped_lb, dtype=np.float64) * scale).astype(np.int64)
        legacy_assumed_high = np.ceil(np.asarray(cur_layer.clipped_ub, dtype=np.float64) * scale).astype(np.int64)
        legacy_lower_margin, legacy_upper_margin, legacy_ok, legacy_violations = self._containment_margins_int(
            guaranteed_low,
            guaranteed_high,
            legacy_assumed_low,
            legacy_assumed_high,
        )

        effective_propagation = self.error_budget_mode == "derived" or self.propagate_contract_tolerance
        if effective_propagation:
            assumed_low = guaranteed_low
            assumed_high = guaranteed_high
            assumption_source = "verified_contract"
        else:
            assumed_low = legacy_assumed_low
            assumed_high = legacy_assumed_high
            assumption_source = "deeppoly_clipped"

        lower_margin, upper_margin, ok, violation_indices = self._containment_margins_int(
            guaranteed_low,
            guaranteed_high,
            assumed_low,
            assumed_high,
        )
        if ok and effective_propagation:
            self._store_verified_activation_bounds(
                cur_layer,
                guaranteed_low,
                guaranteed_high,
                scale,
                source=assumption_source,
            )
        if self.error_budget_mode == "derived":
            cur_layer.error_budget_int = np.asarray(tolerance_int, dtype=np.int64)

        record: dict[str, Any] = {
            "layer_index": int(layer_index),
            "network_layer_index": int(cur_layer.layer_index),
            "Q": int(all_bit),
            "I": int(max(all_bit - frac_bit - 1, 0)),
            "F": int(frac_bit),
            "scale_factor": int(scale),
            "contract_tolerance_enabled": bool(
                self.error_budget_mode == "derived"
                or self.error_budget_mode == "heuristic"
                or self.unsound_contract_tolerance
            ),
            "contract_tolerance_propagated": bool(
                self.error_budget_mode == "derived"
                or self.propagate_contract_tolerance
            ),
            "error_budget_mode": self.error_budget_mode,
            "error_budget_int": [int(value) for value in tolerance_int],
            "chaining_enforced": bool(self.enforce_contract_chaining),
            "soundness": (
                "degraded"
                if not self.enforce_contract_chaining
                else "derived_budget"
                if self.error_budget_mode == "derived"
                else "degraded"
                if self.error_budget_mode == "heuristic"
                else "strict"
            ),
            "assumption_source": assumption_source,
            "legacy_box_chaining_ok": legacy_ok,
            "legacy_min_lower_margin_int": int(np.min(legacy_lower_margin)) if legacy_lower_margin.size else 0,
            "legacy_min_upper_margin_int": int(np.min(legacy_upper_margin)) if legacy_upper_margin.size else 0,
            "legacy_violating_neuron_count": int(legacy_violations.size),
            "abs_tol_int": int(
                (scale // 1000)
                if self.error_budget_mode == "heuristic"
                or self.unsound_contract_tolerance
                else 0
            ),
            "rel_tol_num": int(
                1
                if self.error_budget_mode == "heuristic"
                or self.unsound_contract_tolerance
                else 0
            ),
            "rel_tol_den": 100,
            "max_tolerance_int": int(np.max(tolerance_int)) if tolerance_int.size else 0,
            "min_lower_margin_int": int(np.min(lower_margin)) if lower_margin.size else 0,
            "min_upper_margin_int": int(np.min(upper_margin)) if upper_margin.size else 0,
            "chaining_ok": ok,
            "ok": ok,
            "status": "VERIFIED" if ok else "FAILED",
            "violating_neurons": [int(index) for index in violation_indices[:20]],
            "violating_neuron_count": int(violation_indices.size),
        }
        self.chaining_records.append(record)
        return record

    def _record_output_margin_check(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        weights_int: np.ndarray,
        layer_index: int,
        all_bit: int,
        frac_bit: int,
    ) -> dict[str, Any]:
        """Check that the final real-semantics margin exceeds the derived budget."""

        budget, assumed_low, assumed_high, frac_in = self._candidate_error_budget_int(
            cur_layer,
            in_layer,
            weights_int,
            frac_bit,
        )
        guarantee_low, guarantee_high = self._real_affine_bounds_on_integer_box(
            cur_layer,
            assumed_low,
            assumed_high,
            frac_in=frac_in,
            frac_out=frac_bit,
        )
        q_min = -(1 << (int(all_bit) - 1))
        q_max = (1 << (int(all_bit) - 1)) - 1
        target = int(
            self.property_spec.target_label
            if self.property_spec.target_label is not None
            else self.targetCls
        )
        margins: list[dict[str, Any]] = []
        margin_ok = True
        for other in range(cur_layer.layer_size):
            if other == target:
                continue
            nominal_margin = int(guarantee_low[target]) - int(guarantee_high[other])
            required_budget = int(budget[target]) + int(budget[other])
            raw_survives = nominal_margin > required_budget
            deployed_target_low = min(
                max(int(guarantee_low[target]) - int(budget[target]), q_min),
                q_max,
            )
            deployed_other_high = min(
                max(int(guarantee_high[other]) + int(budget[other]), q_min),
                q_max,
            )
            survives = raw_survives and deployed_target_low > deployed_other_high
            margin_ok = margin_ok and survives
            margins.append(
                {
                    "other_class": int(other),
                    "nominal_margin_int": int(nominal_margin),
                    "required_budget_int": int(required_budget),
                    "residual_margin_int": int(nominal_margin - required_budget),
                    "raw_margin_ok": bool(raw_survives),
                    "deployed_target_low_int": int(deployed_target_low),
                    "deployed_other_high_int": int(deployed_other_high),
                    "ok": bool(survives),
                }
            )

        record = {
            "layer_index": int(layer_index),
            "network_layer_index": int(cur_layer.layer_index),
            "Q": int(all_bit),
            "I": int(max(all_bit - frac_bit - 1, 0)),
            "F": int(frac_bit),
            "input_fractional_bits": int(frac_in),
            "target_class": target,
            "error_budget_int": [int(value) for value in budget],
            "guarantee_source": "exact_real_affine_over_propagated_assumption_box",
            "guarantee_low_int": [int(value) for value in guarantee_low],
            "guarantee_high_int": [int(value) for value in guarantee_high],
            "margin_ok": bool(margin_ok),
            "status": "VERIFIED" if margin_ok else "MARGIN_TOO_SMALL",
            "class_margins": margins,
        }
        self.output_margin_records.append(record)
        if not margin_ok:
            self.synthesis_final_status = "MARGIN_TOO_SMALL"
        return record

    def _candidate_replay_violates(
        self,
        *,
        inputs_int: np.ndarray | list[int],
        input_fractional_bits: int,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
    ) -> bool:
        assumed_low, assumed_high, frac_in = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        inputs = self._rescale_integer_vector(
            inputs_int,
            from_fractional_bits=int(input_fractional_bits),
            to_fractional_bits=int(frac_in),
        )
        if inputs.size != assumed_low.size:
            return False
        if np.any(inputs < assumed_low) or np.any(inputs > assumed_high):
            return False

        outputs = replay_on_python(
            inputs,
            SimpleNamespace(
                weights_int=np.asarray(qu_w_int, dtype=np.int64),
                biases_int=np.asarray(qu_b_int, dtype=np.int64),
            ),
            LayerReplayFormat(
                input_fractional_bits=int(frac_in),
                total_bits=int(all_bit),
                apply_relu=False,
            ),
        )
        is_output_layer = cur_layer.layer_index == len(self.dense_layers) + 1
        if is_output_layer:
            if self.property_spec.valid_labels:
                valid = tuple(int(value) for value in self.property_spec.valid_labels)
                max_valid = max(int(outputs[index]) for index in valid)
                max_invalid = max(
                    int(outputs[index])
                    for index in range(outputs.size)
                    if index not in valid
                )
                return max_valid <= max_invalid
            target = int(
                self.property_spec.target_label
                if self.property_spec.target_label is not None
                else self.targetCls
            )
            return any(
                int(outputs[index]) >= int(outputs[target])
                for index in range(outputs.size)
                if index != target
            )

        scale = 1 << int(frac_bit)
        pre_low, pre_high = self._layer_preimage_bounds_int(cur_layer, scale)
        tolerance, _, _, _ = self._candidate_error_budget_int(
            cur_layer,
            in_layer,
            qu_w_int,
            frac_bit,
        )
        return bool(
            np.any(outputs < (pre_low - tolerance))
            or np.any(outputs > (pre_high + tolerance))
        )

    @staticmethod
    def _rescale_integer_vector(
        values: np.ndarray | list[int],
        *,
        from_fractional_bits: int,
        to_fractional_bits: int,
    ) -> np.ndarray:
        """Re-encode one concrete real vector at a candidate input scale."""

        source = np.asarray(values, dtype=np.int64).reshape(-1)
        shift = int(to_fractional_bits) - int(from_fractional_bits)
        if shift >= 0:
            return np.asarray(
                [int(value) << shift for value in source],
                dtype=np.int64,
            )

        denominator = 1 << (-shift)
        rescaled: list[int] = []
        for value in source:
            magnitude = abs(int(value))
            rounded = (magnitude + denominator // 2) // denominator
            rescaled.append(-rounded if int(value) < 0 else rounded)
        return np.asarray(rescaled, dtype=np.int64)

    def _record_failed_counterexample(
        self,
        *,
        result: ESBMCResult,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        layer_index: int,
    ) -> dict[str, Any] | None:
        if result.status != "FAILED" or result.counterexample_inputs is None:
            return None
        _, _, frac_in = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        try:
            confirmed = self._candidate_replay_violates(
                inputs_int=result.counterexample_inputs,
                input_fractional_bits=frac_in,
                cur_layer=cur_layer,
                in_layer=in_layer,
                qu_w_int=qu_w_int,
                qu_b_int=qu_b_int,
                frac_bit=frac_bit,
                all_bit=all_bit,
            )
        except Exception as exc:  # noqa: BLE001 - replay is diagnostic.
            LOGGER.warning(
                "Could not replay ESBMC counterexample for layer %s: %s",
                layer_index,
                exc,
            )
            confirmed = False

        record = {
            "layer_index": int(layer_index),
            "Q": int(all_bit),
            "I": int(max(all_bit - frac_bit - 1, 0)),
            "F": int(frac_bit),
            "input_fractional_bits": int(frac_in),
            "inputs_int": [int(value) for value in result.counterexample_inputs],
            "counterexample_neuron": result.counterexample_neuron,
            "counterexample_preclamp": result.counterexample_preclamp,
            "replay_confirmed": bool(confirmed),
        }
        self.counterexample_records.append(record)
        if confirmed:
            self.cex_pool.setdefault(int(layer_index), []).append(record)
        return record

    def _candidate_filtered_by_counterexample(
        self,
        *,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        layer_index: int,
    ) -> dict[str, Any] | None:
        if self.cex_feedback not in {"filter", "filter+jump"}:
            return None
        for counterexample in self.cex_pool.get(int(layer_index), []):
            if self._candidate_replay_violates(
                inputs_int=counterexample["inputs_int"],
                input_fractional_bits=int(counterexample["input_fractional_bits"]),
                cur_layer=cur_layer,
                in_layer=in_layer,
                qu_w_int=qu_w_int,
                qu_b_int=qu_b_int,
                frac_bit=frac_bit,
                all_bit=all_bit,
            ):
                self.cex_filtered_counts[int(layer_index)] = (
                    self.cex_filtered_counts.get(int(layer_index), 0) + 1
                )
                record = {
                    "layer_index": int(layer_index),
                    "Q": int(all_bit),
                    "I": int(max(all_bit - frac_bit - 1, 0)),
                    "F": int(frac_bit),
                    "status": "CEX_FILTERED",
                    "property_type": (
                        "output"
                        if cur_layer.layer_index == len(self.dense_layers) + 1
                        else "preimage"
                    ),
                    "mode": "cex_filter",
                    "reason": "replayed_counterexample_still_violates_candidate",
                    "source_counterexample": dict(counterexample),
                }
                self.esbmc_call_records.append(record)
                return record
        return None

    def _record_invalid_assumption_box(
        self,
        *,
        layer_index: int,
        all_bit: int,
        frac_bit: int,
        cardinality: str,
    ) -> ESBMCResult:
        record = {
            "layer_index": int(layer_index),
            "Q": int(all_bit),
            "I": int(max(all_bit - frac_bit - 1, 0)),
            "F": int(frac_bit),
            "status": "VACUOUS",
            "sentinel_status": "SKIPPED",
            "assumption_box_cardinality": cardinality,
            "reason": "invalid_integer_assumption_box",
        }
        self.vacuity_records.append(record)
        self.synthesis_final_status = "VACUOUS"
        return ESBMCResult(
            status="VACUOUS",
            command=(),
            stdout="",
            stderr="invalid integer assumption box",
            return_code=1,
            timeout_seconds=int(self.config.esbmc.timeout_seconds),
            memlimit=str(self.config.esbmc.memlimit),
            resource_control={
                "status": "VACUOUS",
                "assumption_box_cardinality": cardinality,
                "reason": "invalid_integer_assumption_box",
            },
        )

    def _run_esbmc_file(
        self,
        harness: Path,
        *,
        extract_counterexample: bool = False,
    ) -> ESBMCResult:
        """Preserve the legacy runner call shape unless extraction is needed."""

        if extract_counterexample:
            return self.esbmc_runner.run_file(
                harness,
                extract_counterexample=True,
            )
        return self.esbmc_runner.run_file(harness)

    def _run_vacuity_sentinel(
        self,
        *,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        layer_index: int,
        all_bit: int,
        frac_bit: int,
    ) -> dict[str, Any]:
        assumed_lo, assumed_hi, _ = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        cardinality, valid = self._assumption_box_cardinality(assumed_lo, assumed_hi)
        if not valid:
            self._record_invalid_assumption_box(
                layer_index=layer_index,
                all_bit=all_bit,
                frac_bit=frac_bit,
                cardinality=cardinality,
            )
            return self.vacuity_records[-1]

        if not self.vacuity_check:
            record = {
                "layer_index": int(layer_index),
                "Q": int(all_bit),
                "I": int(max(all_bit - frac_bit - 1, 0)),
                "F": int(frac_bit),
                "status": "SKIPPED",
                "sentinel_status": "SKIPPED",
                "assumption_box_cardinality": cardinality,
            }
            self.vacuity_records.append(record)
            return record

        source = render_assumption_sentinel_program(
            input_size=in_layer.layer_size,
            input_bounds_low_c_int=self.numpy_to_c_int_array(assumed_lo),
            input_bounds_high_c_int=self.numpy_to_c_int_array(assumed_hi),
        )
        sentinel_dir = self.output_dir / "layers" / "vacuity"
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        harness = sentinel_dir / f"layer_{layer_index}_Q{all_bit}_F{frac_bit}_sentinel.c"
        harness.write_text(source, encoding="utf-8")
        result = self.esbmc_runner.run_file(harness)
        self._stats["esbmc_calls"] += 1.0

        status = (
            "NONVACUOUS"
            if result.status == "FAILED"
            else "VACUOUS" if result.status == "VERIFIED" else result.status
        )
        call_status = (
            "SENTINEL_EXPECTED_FAILURE"
            if status == "NONVACUOUS"
            else status
        )
        call_record = self._esbmc_call_record(
            result=result,
            layer_index=layer_index,
            block_index=None,
            start_neuron=None,
            end_neuron=None,
            all_bit=all_bit,
            frac_bit=frac_bit,
            harness=harness,
            property_type="vacuity_sentinel",
            mode="sentinel",
            input_dim=in_layer.layer_size,
            output_neurons=0,
            status=call_status,
        )
        call_record["assumption_box_cardinality"] = cardinality
        self.esbmc_call_records.append(call_record)
        record = {
            "layer_index": int(layer_index),
            "Q": int(all_bit),
            "I": int(max(all_bit - frac_bit - 1, 0)),
            "F": int(frac_bit),
            "status": status,
            "sentinel_status": result.status,
            "assumption_box_cardinality": cardinality,
            "harness": str(harness),
            "elapsed_seconds": float(result.elapsed_seconds),
            "resource_control": result.resource_control,
        }
        self.vacuity_records.append(record)
        if status not in {"NONVACUOUS", "SKIPPED"}:
            self.synthesis_final_status = status
        return record

    def _annotate_assumption_cardinality(
        self,
        *,
        layer_index: int,
        all_bit: int,
        frac_bit: int,
        cardinality: str,
    ) -> None:
        for record in reversed(self.esbmc_call_records):
            if (
                int(record.get("layer_index", -1)) == int(layer_index)
                and int(record.get("Q", -1)) == int(all_bit)
                and int(record.get("F", -1)) == int(frac_bit)
                and record.get("property_type") in {"preimage", "output"}
            ):
                record["assumption_box_cardinality"] = cardinality
            elif record.get("assumption_box_cardinality") is not None:
                break

    def verify_layer_with_esbmc(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        layer_index: int,
    ) -> ESBMCResult:
        assumed_lo, assumed_hi, _ = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        cardinality, valid_box = self._assumption_box_cardinality(assumed_lo, assumed_hi)
        if not valid_box:
            return self._record_invalid_assumption_box(
                layer_index=layer_index,
                all_bit=all_bit,
                frac_bit=frac_bit,
                cardinality=cardinality,
            )

        if self._uses_hidden_block_verification(cur_layer):
            result = self.verify_hidden_layer_blocks_with_esbmc(
                cur_layer=cur_layer,
                in_layer=in_layer,
                qu_w_int=qu_w_int,
                qu_b_int=qu_b_int,
                frac_bit=frac_bit,
                all_bit=all_bit,
                layer_index=layer_index,
            )
            self._annotate_assumption_cardinality(
                layer_index=layer_index,
                all_bit=all_bit,
                frac_bit=frac_bit,
                cardinality=cardinality,
            )
            return result

        c_source = self.generate_esbmc_verification_code(
            cur_layer=cur_layer,
            in_layer=in_layer,
            qu_w_int=qu_w_int,
            qu_b_int=qu_b_int,
            frac_bit=frac_bit,
            all_bit=all_bit,
            layer_index=layer_index,
        )
        layers_dir = self.output_dir / "layers"
        layers_dir.mkdir(parents=True, exist_ok=True)
        archived_file = layers_dir / f"layer_{layer_index}_Q{all_bit}_F{frac_bit}.c"
        archived_file.write_text(c_source, encoding="utf-8")

        result = self._run_esbmc_file(
            archived_file,
            extract_counterexample=self.cex_feedback != "off",
        )
        self._stats["esbmc_calls"] += 1.0
        record = self._esbmc_call_record(
            result=result,
            layer_index=layer_index,
            block_index=None,
            start_neuron=None,
            end_neuron=None,
            all_bit=all_bit,
            frac_bit=frac_bit,
            harness=archived_file,
            property_type="output" if cur_layer.layer_index == len(self.dense_layers) + 1 else "preimage",
            mode="full_layer",
            input_dim=in_layer.layer_size,
            output_neurons=cur_layer.layer_size,
        )
        self.esbmc_call_records.append(record)
        record["assumption_box_cardinality"] = cardinality
        LOGGER.info("ESBMC layer=%s bits(Q=%s,F=%s) status=%s", cur_layer.layer_index, all_bit, frac_bit, result.status)
        return result

    def verify_hidden_layer_blocks_with_esbmc(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        layer_index: int,
    ) -> ESBMCResult:
        layers_dir = self.output_dir / "layers" / "blocks"
        layers_dir.mkdir(parents=True, exist_ok=True)

        if self.esbmc_jobs != 1:
            LOGGER.warning(
                "esbmc_jobs=%s requested; block-wise ESBMC verification is currently run sequentially.",
                self.esbmc_jobs,
            )

        block_ranges = self._hidden_block_ranges(cur_layer.layer_size)
        records: list[dict[str, Any]] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        elapsed_total = 0.0
        aggregate_return_code = 0
        first_failure: dict[str, Any] | None = None
        first_failure_result: ESBMCResult | None = None

        for block_index, (start_neuron, end_neuron) in enumerate(block_ranges):
            c_source = self.generate_esbmc_hidden_block_verification_code(
                cur_layer=cur_layer,
                in_layer=in_layer,
                qu_w_int=qu_w_int,
                qu_b_int=qu_b_int,
                frac_bit=frac_bit,
                all_bit=all_bit,
                start_neuron=start_neuron,
                end_neuron=end_neuron,
            )
            harness_name = (
                f"layer_{layer_index}_block_{block_index}_"
                f"n{start_neuron}_{end_neuron}_Q{all_bit}_F{frac_bit}.c"
            )
            archived_file = layers_dir / harness_name
            archived_file.write_text(c_source, encoding="utf-8")

            block_result = self._run_esbmc_file(
                archived_file,
                extract_counterexample=self.cex_feedback != "off",
            )

            self._stats["esbmc_calls"] += 1.0
            self._stats["esbmc_block_calls"] += 1.0
            elapsed_total += float(block_result.elapsed_seconds)
            aggregate_return_code = max(aggregate_return_code, int(block_result.return_code))
            stdout_parts.append(block_result.stdout)
            stderr_parts.append(block_result.stderr)

            record_status = self._record_block_status(block_result.status)
            record = self._esbmc_call_record(
                result=block_result,
                layer_index=layer_index,
                block_index=block_index,
                start_neuron=start_neuron,
                end_neuron=end_neuron,
                all_bit=all_bit,
                frac_bit=frac_bit,
                harness=archived_file,
                property_type="preimage",
                mode="blockwise",
                input_dim=in_layer.layer_size,
                output_neurons=end_neuron - start_neuron,
                status=record_status,
            )
            records.append(record)
            self.esbmc_call_records.append(record)
            self.esbmc_block_records.append(record)

            LOGGER.info(
                "ESBMC block layer=%s block=%s neurons=[%s,%s) bits(Q=%s,F=%s) status=%s",
                cur_layer.layer_index,
                block_index,
                start_neuron,
                end_neuron,
                all_bit,
                frac_bit,
                record_status,
            )

            if record_status != "VERIFIED":
                skipped_count = len(block_ranges) - block_index - 1
                record["reason"] = "candidate_rejected_by_block"
                record["skipped_remaining_blocks"] = int(
                    skipped_count if self._should_fail_fast_blocks() else 0
                )
                if first_failure is None:
                    first_failure = dict(record)
                    first_failure_result = block_result
                if self.blockwise_first_failed_block is None:
                    self.blockwise_first_failed_block = dict(record)

                if self._should_fail_fast_blocks():
                    self.blockwise_skipped_blocks_due_to_fail_fast += skipped_count
                    for skipped_offset, (skip_start, skip_end) in enumerate(
                        block_ranges[block_index + 1 :],
                        start=block_index + 1,
                    ):
                        skipped_record: dict[str, Any] = {
                            "layer_index": int(layer_index),
                            "block_index": int(skipped_offset),
                            "start_neuron": int(skip_start),
                            "end_neuron": int(skip_end),
                            "Q": int(all_bit),
                            "I": int(max(all_bit - frac_bit - 1, 0)),
                            "F": int(frac_bit),
                            "status": "SKIPPED",
                            "time": 0.0,
                            "elapsed_seconds": 0.0,
                            "return_code": 0,
                            "timeout": f"{int(self.config.esbmc.timeout_seconds)}s",
                            "memlimit": str(self.config.esbmc.memlimit),
                            "command": [],
                            "stdout_log_path": "",
                            "stderr_log_path": "",
                            "resource_control": {
                                "timeout": f"{int(self.config.esbmc.timeout_seconds)}s",
                                "memlimit": str(self.config.esbmc.memlimit),
                                "status": "SKIPPED",
                                "stdout_log_path": "",
                                "stderr_log_path": "",
                            },
                            "harness": None,
                            "property_type": "preimage",
                            "mode": "blockwise",
                            "input_dim": int(in_layer.layer_size),
                            "neurons_per_query": int(skip_end - skip_start),
                            "estimated_macs": int(in_layer.layer_size) * int(skip_end - skip_start),
                            "reason": "skipped_due_to_fail_fast",
                            "skipped_due_to_fail_fast": True,
                        }
                        records.append(skipped_record)
                        self.esbmc_call_records.append(skipped_record)
                        self.esbmc_block_records.append(skipped_record)
                    break

        if all(record["status"] in {"VERIFIED", "SKIPPED"} for record in records) and not first_failure:
            aggregate_status = "VERIFIED"
        else:
            aggregate_status = (
                str(first_failure.get("status"))
                if first_failure is not None
                else self._candidate_rejection_status(records)
            )
        aggregate_resource_control = {
            "timeout": f"{int(self.config.esbmc.timeout_seconds)}s",
            "memlimit": str(self.config.esbmc.memlimit),
            "elapsed_seconds": float(elapsed_total),
            "return_code": int(0 if aggregate_status == "VERIFIED" else aggregate_return_code or 1),
            "status": aggregate_status,
            "stdout_log_path": "",
            "stderr_log_path": "",
            "fail_fast": bool(self._should_fail_fast_blocks()),
            "run_all_blocks_on_failure": bool(self.blockwise_run_all_blocks_on_failure),
            "first_failed_block": first_failure,
        }

        return ESBMCResult(
            status=aggregate_status,
            command=(),
            stdout="\n".join(stdout_parts),
            stderr="\n".join(stderr_parts),
            return_code=0 if aggregate_status == "VERIFIED" else aggregate_return_code or 1,
            elapsed_seconds=elapsed_total,
            timeout_seconds=int(self.config.esbmc.timeout_seconds),
            memlimit=str(self.config.esbmc.memlimit),
            resource_control=aggregate_resource_control,
            blocks=tuple(records),
            counterexample_inputs=(
                first_failure_result.counterexample_inputs
                if first_failure_result is not None
                else None
            ),
            counterexample_neuron=(
                first_failure_result.counterexample_neuron
                if first_failure_result is not None
                else None
            ),
            counterexample_preclamp=(
                first_failure_result.counterexample_preclamp
                if first_failure_result is not None
                else None
            ),
        )

    def verify_layer_no_saturation_with_esbmc(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        layer_index: int,
    ) -> ESBMCResult:
        if self._uses_hidden_block_verification(cur_layer):
            return self.verify_layer_no_saturation_blocks_with_esbmc(
                cur_layer=cur_layer,
                in_layer=in_layer,
                qu_w_int=qu_w_int,
                qu_b_int=qu_b_int,
                frac_bit=frac_bit,
                all_bit=all_bit,
                layer_index=layer_index,
            )

        c_source = self.generate_esbmc_no_saturation_code(
            cur_layer=cur_layer,
            in_layer=in_layer,
            qu_w_int=qu_w_int,
            qu_b_int=qu_b_int,
            frac_bit=frac_bit,
            all_bit=all_bit,
            layer_index=layer_index,
        )
        layers_dir = self.output_dir / "layers"
        layers_dir.mkdir(parents=True, exist_ok=True)
        archived_file = layers_dir / f"layer_{layer_index}_Q{all_bit}_F{frac_bit}_no_saturation.c"
        archived_file.write_text(c_source, encoding="utf-8")

        result = self.esbmc_runner.run_file(archived_file)
        self._stats["esbmc_calls"] += 1.0
        record = self._esbmc_call_record(
            result=result,
            layer_index=layer_index,
            block_index=None,
            start_neuron=None,
            end_neuron=None,
            all_bit=all_bit,
            frac_bit=frac_bit,
            harness=archived_file,
            property_type="no_saturation",
            mode="full_layer",
            input_dim=in_layer.layer_size,
            output_neurons=cur_layer.layer_size,
        )
        self.esbmc_call_records.append(record)
        LOGGER.info(
            "ESBMC no-saturation layer=%s bits(Q=%s,F=%s) status=%s",
            cur_layer.layer_index,
            all_bit,
            frac_bit,
            result.status,
        )
        return result

    def verify_layer_no_saturation_blocks_with_esbmc(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        layer_index: int,
    ) -> ESBMCResult:
        layers_dir = self.output_dir / "layers" / "blocks"
        layers_dir.mkdir(parents=True, exist_ok=True)

        block_ranges = self._hidden_block_ranges(cur_layer.layer_size)
        records: list[dict[str, Any]] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        elapsed_total = 0.0
        aggregate_return_code = 0
        first_non_verified: dict[str, Any] | None = None

        for block_index, (start_neuron, end_neuron) in enumerate(block_ranges):
            c_source = self.generate_esbmc_no_saturation_block_code(
                cur_layer=cur_layer,
                in_layer=in_layer,
                qu_w_int=qu_w_int,
                qu_b_int=qu_b_int,
                frac_bit=frac_bit,
                all_bit=all_bit,
                start_neuron=start_neuron,
                end_neuron=end_neuron,
            )
            harness_name = (
                f"layer_{layer_index}_no_sat_block_{block_index}_"
                f"n{start_neuron}_{end_neuron}_Q{all_bit}_F{frac_bit}.c"
            )
            archived_file = layers_dir / harness_name
            archived_file.write_text(c_source, encoding="utf-8")

            block_result = self.esbmc_runner.run_file(archived_file)
            self._stats["esbmc_calls"] += 1.0
            self._stats["esbmc_block_calls"] += 1.0
            elapsed_total += float(block_result.elapsed_seconds)
            aggregate_return_code = max(aggregate_return_code, int(block_result.return_code))
            stdout_parts.append(block_result.stdout)
            stderr_parts.append(block_result.stderr)

            record_status = self._record_block_status(block_result.status)
            record = self._esbmc_call_record(
                result=block_result,
                layer_index=layer_index,
                block_index=block_index,
                start_neuron=start_neuron,
                end_neuron=end_neuron,
                all_bit=all_bit,
                frac_bit=frac_bit,
                harness=archived_file,
                property_type="no_saturation",
                mode="blockwise",
                input_dim=in_layer.layer_size,
                output_neurons=end_neuron - start_neuron,
                status=record_status,
            )
            records.append(record)
            self.esbmc_call_records.append(record)
            self.esbmc_no_saturation_block_records.append(record)

            LOGGER.info(
                "ESBMC no-saturation block layer=%s block=%s neurons=[%s,%s) bits(Q=%s,F=%s) status=%s",
                cur_layer.layer_index,
                block_index,
                start_neuron,
                end_neuron,
                all_bit,
                frac_bit,
                record_status,
            )

            if record_status != "VERIFIED":
                record["reason"] = "no_saturation_block_not_verified"
                if first_non_verified is None:
                    first_non_verified = dict(record)
                if self._should_stop_no_saturation_blocks(record_status):
                    for skipped_offset, (skip_start, skip_end) in enumerate(
                        block_ranges[block_index + 1 :],
                        start=block_index + 1,
                    ):
                        skipped_record: dict[str, Any] = {
                            "layer_index": int(layer_index),
                            "block_index": int(skipped_offset),
                            "start_neuron": int(skip_start),
                            "end_neuron": int(skip_end),
                            "Q": int(all_bit),
                            "I": int(max(all_bit - frac_bit - 1, 0)),
                            "F": int(frac_bit),
                            "status": "SKIPPED",
                            "time": 0.0,
                            "elapsed_seconds": 0.0,
                            "return_code": 0,
                            "timeout": f"{int(self.config.esbmc.timeout_seconds)}s",
                            "memlimit": str(self.config.esbmc.memlimit),
                            "command": [],
                            "stdout_log_path": "",
                            "stderr_log_path": "",
                            "resource_control": {
                                "timeout": f"{int(self.config.esbmc.timeout_seconds)}s",
                                "memlimit": str(self.config.esbmc.memlimit),
                                "status": "SKIPPED",
                                "stdout_log_path": "",
                                "stderr_log_path": "",
                            },
                            "harness": None,
                            "property_type": "no_saturation",
                            "mode": "blockwise",
                            "input_dim": int(in_layer.layer_size),
                            "neurons_per_query": int(skip_end - skip_start),
                            "estimated_macs": int(in_layer.layer_size) * int(skip_end - skip_start),
                            "reason": "skipped_after_no_saturation_block_status",
                            "skipped_due_to_no_saturation_policy": True,
                        }
                        records.append(skipped_record)
                        self.esbmc_call_records.append(skipped_record)
                        self.esbmc_no_saturation_block_records.append(skipped_record)
                    break

        aggregate_status = self._aggregate_no_saturation_status(records)
        aggregate_resource_control = {
            "timeout": f"{int(self.config.esbmc.timeout_seconds)}s",
            "memlimit": str(self.config.esbmc.memlimit),
            "elapsed_seconds": float(elapsed_total),
            "return_code": int(0 if aggregate_status == "VERIFIED" else aggregate_return_code or 1),
            "status": aggregate_status,
            "stdout_log_path": "",
            "stderr_log_path": "",
            "first_non_verified_block": first_non_verified,
            "continue_on_unknown": bool(self.no_saturation_continue_on_unknown),
        }

        return ESBMCResult(
            status=aggregate_status,
            command=(),
            stdout="\n".join(stdout_parts),
            stderr="\n".join(stderr_parts),
            return_code=0 if aggregate_status == "VERIFIED" else aggregate_return_code or 1,
            elapsed_seconds=elapsed_total,
            timeout_seconds=int(self.config.esbmc.timeout_seconds),
            memlimit=str(self.config.esbmc.memlimit),
            resource_control=aggregate_resource_control,
            blocks=tuple(records),
        )

    def generate_esbmc_verification_code(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        layer_index: int,
    ) -> str:
        del layer_index
        scale = 1 << int(frac_bit)
        weights_c_int = self.numpy_to_c_int_array(qu_w_int)
        biases_c_int = self.numpy_to_c_int_array(qu_b_int)

        pre_lo_int, pre_hi_int = self._layer_preimage_bounds_int(cur_layer, scale)
        tolerance_int, input_lo_int, input_hi_int, frac_in = self._candidate_error_budget_int(
            cur_layer,
            in_layer,
            qu_w_int,
            frac_bit,
        )
        input_scale = 1 << int(frac_in)

        is_output_layer = cur_layer.layer_index == len(self.dense_layers) + 1
        if is_output_layer:
            if self.property_spec.valid_labels:
                return render_output_valid_set_program(
                    output_size=cur_layer.layer_size,
                    input_size=in_layer.layer_size,
                    weights_c_int=weights_c_int,
                    biases_c_int=biases_c_int,
                    input_bounds_low_c_int=self.numpy_to_c_int_array(input_lo_int),
                    input_bounds_high_c_int=self.numpy_to_c_int_array(input_hi_int),
                    valid_classes=tuple(self.property_spec.valid_labels),
                    scale_factor=scale,
                    total_bits=all_bit,
                    input_scale_factor=input_scale,
                )
            return render_output_target_program(
                output_size=cur_layer.layer_size,
                input_size=in_layer.layer_size,
                weights_c_int=weights_c_int,
                biases_c_int=biases_c_int,
                input_bounds_low_c_int=self.numpy_to_c_int_array(input_lo_int),
                input_bounds_high_c_int=self.numpy_to_c_int_array(input_hi_int),
                target_label=int(self.property_spec.target_label if self.property_spec.target_label is not None else self.targetCls),
                scale_factor=scale,
                total_bits=all_bit,
                input_scale_factor=input_scale,
            )

        return render_hidden_affine_bounds_program(
            output_size=cur_layer.layer_size,
            input_size=in_layer.layer_size,
            weights_c_int=weights_c_int,
            biases_c_int=biases_c_int,
            preimage_low_c_int=self.numpy_to_c_int_array(pre_lo_int),
            preimage_high_c_int=self.numpy_to_c_int_array(pre_hi_int),
            input_bounds_low_c_int=self.numpy_to_c_int_array(input_lo_int),
            input_bounds_high_c_int=self.numpy_to_c_int_array(input_hi_int),
            scale_factor=scale,
            total_bits=all_bit,
            unsound_contract_tolerance=self.error_budget_mode == "heuristic",
            input_scale_factor=input_scale,
            contract_tolerance_c_int=(
                self.numpy_to_c_int_array(tolerance_int)
                if self.error_budget_mode == "derived"
                else None
            ),
        )

    def generate_esbmc_hidden_block_verification_code(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        start_neuron: int,
        end_neuron: int,
    ) -> str:
        scale = 1 << int(frac_bit)
        pre_lo_int, pre_hi_int = self._layer_preimage_bounds_int(cur_layer, scale)
        tolerance_int, input_lo_int, input_hi_int, frac_in = self._candidate_error_budget_int(
            cur_layer,
            in_layer,
            qu_w_int,
            frac_bit,
        )
        input_scale = 1 << int(frac_in)

        return render_hidden_affine_bounds_block_program(
            block_size=int(end_neuron - start_neuron),
            input_size=in_layer.layer_size,
            weights_c_int=self.numpy_to_c_int_array(qu_w_int[start_neuron:end_neuron]),
            biases_c_int=self.numpy_to_c_int_array(qu_b_int[start_neuron:end_neuron]),
            preimage_low_c_int=self.numpy_to_c_int_array(pre_lo_int[start_neuron:end_neuron]),
            preimage_high_c_int=self.numpy_to_c_int_array(pre_hi_int[start_neuron:end_neuron]),
            input_bounds_low_c_int=self.numpy_to_c_int_array(input_lo_int),
            input_bounds_high_c_int=self.numpy_to_c_int_array(input_hi_int),
            scale_factor=scale,
            total_bits=all_bit,
            unsound_contract_tolerance=self.error_budget_mode == "heuristic",
            input_scale_factor=input_scale,
            contract_tolerance_c_int=(
                self.numpy_to_c_int_array(tolerance_int[start_neuron:end_neuron])
                if self.error_budget_mode == "derived"
                else None
            ),
        )

    def generate_esbmc_no_saturation_code(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        layer_index: int,
    ) -> str:
        del layer_index
        scale = 1 << int(frac_bit)
        weights_c_int = self.numpy_to_c_int_array(qu_w_int)
        biases_c_int = self.numpy_to_c_int_array(qu_b_int)
        input_lo_int, input_hi_int, frac_in = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        input_scale = 1 << int(frac_in)

        return render_no_saturation_program(
            output_size=cur_layer.layer_size,
            input_size=in_layer.layer_size,
            weights_c_int=weights_c_int,
            biases_c_int=biases_c_int,
            input_bounds_low_c_int=self.numpy_to_c_int_array(input_lo_int),
            input_bounds_high_c_int=self.numpy_to_c_int_array(input_hi_int),
            scale_factor=scale,
            total_bits=all_bit,
            integer_bits=max(int(all_bit) - int(frac_bit) - 1, 0),
            fractional_bits=frac_bit,
            input_scale_factor=input_scale,
        )

    def generate_esbmc_no_saturation_block_code(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        start_neuron: int,
        end_neuron: int,
    ) -> str:
        scale = 1 << int(frac_bit)
        input_lo_int, input_hi_int, frac_in = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        input_scale = 1 << int(frac_in)

        return render_no_saturation_block_program(
            block_size=int(end_neuron - start_neuron),
            input_size=in_layer.layer_size,
            weights_c_int=self.numpy_to_c_int_array(qu_w_int[start_neuron:end_neuron]),
            biases_c_int=self.numpy_to_c_int_array(qu_b_int[start_neuron:end_neuron]),
            input_bounds_low_c_int=self.numpy_to_c_int_array(input_lo_int),
            input_bounds_high_c_int=self.numpy_to_c_int_array(input_hi_int),
            scale_factor=scale,
            total_bits=all_bit,
            integer_bits=max(int(all_bit) - int(frac_bit) - 1, 0),
            fractional_bits=frac_bit,
            input_scale_factor=input_scale,
        )

    def numpy_to_c_int_array(self, np_array: np.ndarray) -> str:
        if np_array.ndim == 1:
            return "{" + ", ".join(str(int(x)) for x in np_array) + "}"
        rows = []
        for row in np_array:
            rows.append("{" + ", ".join(str(int(x)) for x in row) + "}")
        return "{" + ", ".join(rows) + "}"

    def blockwise_verification_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.esbmc_block_records]
        verified_blocks = sum(1 for record in records if record.get("status") == "VERIFIED")
        timeout_blocks = sum(1 for record in records if record.get("status") == "TIMEOUT")
        failed_blocks = sum(1 for record in records if record.get("status") == "FAILED")
        memout_blocks = sum(1 for record in records if record.get("status") == "MEMOUT")
        unknown_blocks = sum(1 for record in records if record.get("status") == "UNKNOWN")
        skipped_blocks = sum(1 for record in records if record.get("status") == "SKIPPED")
        largest_neurons = max((int(record.get("neurons_per_query", 0) or 0) for record in records), default=0)
        largest_input_dim = max((int(record.get("input_dim", 0) or 0) for record in records), default=0)
        largest_estimated_macs = max((int(record.get("estimated_macs", 0) or 0) for record in records), default=0)

        layers: list[dict[str, Any]] = []
        for layer_index in sorted({int(record["layer_index"]) for record in records}):
            layers.append(
                {
                    "layer_index": int(layer_index),
                    "blocks": [
                        record
                        for record in records
                        if int(record.get("layer_index", -1)) == layer_index
                    ],
                }
            )

        return {
            "enabled": bool(
                self.verify_mode == "esbmc"
                and self.harness_scope == "layer"
                and self.esbmc_layer_block_size > 0
            ),
            "mode": (
                "blockwise_hidden_layers"
                if self.harness_scope == "layer"
                and self.esbmc_layer_block_size > 0
                else "not_applicable_network_scope"
                if self.harness_scope == "network"
                else "monolithic_per_layer"
            ),
            "block_size": int(self.esbmc_layer_block_size),
            "policy": "shared_layer_qif",
            "fail_fast": bool(self.blockwise_fail_fast),
            "effective_fail_fast": bool(self._should_fail_fast_blocks()),
            "run_all_blocks_on_failure": bool(self.blockwise_run_all_blocks_on_failure),
            "esbmc_jobs": int(self.esbmc_jobs),
            "total_blocks": int(len(records)),
            "verified_blocks": int(verified_blocks),
            "failed_blocks": int(failed_blocks),
            "timeout_blocks": int(timeout_blocks),
            "memout_blocks": int(memout_blocks),
            "unknown_blocks": int(unknown_blocks),
            "skipped_blocks": int(skipped_blocks),
            "skipped_blocks_due_to_fail_fast": int(self.blockwise_skipped_blocks_due_to_fail_fast),
            "largest_neurons_per_query": int(largest_neurons),
            "largest_input_dim_per_query": int(largest_input_dim),
            "largest_estimated_macs_per_query": int(largest_estimated_macs),
            "first_failed_block": self.blockwise_first_failed_block,
            "layers": layers,
        }

    def contract_tolerance_summary(self) -> dict[str, Any]:
        soundness = self.soundness_label()
        return {
            "mode": (
                "not_applicable_network_scope"
                if self.harness_scope == "network"
                else "derived"
                if self.error_budget_mode == "derived"
                else "zero"
                if self.error_budget_mode == "zero"
                else "heuristic"
            ),
            "abs_tol_num": int(1 if self.error_budget_mode == "heuristic" else 0),
            "abs_tol_den": 1000,
            "rel_tol_num": int(1 if self.error_budget_mode == "heuristic" else 0),
            "rel_tol_den": 100,
            "propagated": bool(
                self.harness_scope == "layer"
                and (
                    self.error_budget_mode == "derived"
                    or self.propagate_contract_tolerance
                )
            ),
            "error_budget_mode": self.error_budget_mode,
            "soundness": soundness,
        }

    def chaining_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.chaining_records]
        failed = [record for record in records if not bool(record.get("chaining_ok", False))]
        soundness = self.soundness_label()
        return {
            "enabled": bool(
                self.verify_mode == "esbmc" and self.harness_scope == "layer"
            ),
            "enforced": bool(self.enforce_contract_chaining),
            "contract_tolerance_propagated": bool(
                self.error_budget_mode == "derived"
                or self.propagate_contract_tolerance
            ),
            "policy": "activation_of_hidden_contract_subset_downstream_assumption",
            "soundness": soundness,
            "unsound_contract_tolerance": bool(self.unsound_contract_tolerance),
            "error_budget_mode": self.error_budget_mode,
            "all_ok": len(failed) == 0,
            "failed_count": int(len(failed)),
            "layers": records,
        }

    def soundness_label(self) -> str:
        if self.harness_scope == "network":
            return "end_to_end"
        if not self.enforce_contract_chaining:
            return "degraded"
        if (
            "error_budget_mode" not in self.__dict__
            and self.unsound_contract_tolerance
        ):
            return (
                "tolerance_propagated"
                if self.propagate_contract_tolerance
                else "degraded"
            )
        if self.error_budget_mode == "derived":
            if self.output_margin_records and bool(
                self.output_margin_records[-1].get("margin_ok", False)
            ):
                return "derived_budget"
            return "derived_budget_incomplete"
        if self.error_budget_mode == "heuristic":
            return "degraded"
        return "strict"

    def output_margin_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.output_margin_records]
        final_ok = bool(records and records[-1].get("margin_ok", False))
        return {
            "enabled": (
                self.harness_scope == "layer"
                and self.error_budget_mode == "derived"
            ),
            "all_ok": final_ok,
            "status": (
                "VERIFIED"
                if final_ok
                else "MARGIN_TOO_SMALL"
                if records and records[-1].get("status") == "MARGIN_TOO_SMALL"
                else "SKIPPED"
            ),
            "checks": records,
        }

    def vacuity_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.vacuity_records]
        latest_by_layer: dict[int, dict[str, Any]] = {}
        for record in records:
            latest_by_layer[int(record.get("layer_index", -1))] = record
        final_records = list(latest_by_layer.values())
        statuses = {str(record.get("status", "UNKNOWN")) for record in final_records}
        if "VACUOUS" in statuses:
            status = "VACUOUS"
        elif "TIMEOUT" in statuses:
            status = "TIMEOUT"
        elif "MEMOUT" in statuses:
            status = "MEMOUT"
        elif "UNKNOWN" in statuses:
            status = "UNKNOWN"
        elif records and statuses <= {"NONVACUOUS", "SKIPPED"}:
            status = "PASSED" if "NONVACUOUS" in statuses else "SKIPPED"
        else:
            status = "SKIPPED"

        sentinel_statuses = {
            str(record.get("sentinel_status", "SKIPPED"))
            for record in final_records
        }
        if sentinel_statuses == {"FAILED"}:
            sentinel_status = "FAILED"
        elif sentinel_statuses == {"SKIPPED"} or not records:
            sentinel_status = "SKIPPED"
        elif "VERIFIED" in sentinel_statuses:
            sentinel_status = "VERIFIED"
        else:
            sentinel_status = "MIXED"
        return {
            "enabled": bool(self.vacuity_check),
            "status": status,
            "sentinel_status": sentinel_status,
            "layers": records,
            "final_layers": final_records,
        }

    def counterexample_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.counterexample_records]
        confirmed = sum(
            1 for record in records if bool(record.get("replay_confirmed", False))
        )
        return {
            "feedback_mode": self.cex_feedback,
            "records": records,
            "counterexamples_total": int(len(records)),
            "counterexamples_confirmed": int(confirmed),
            "counterexample_confirmation_rate": (
                float(confirmed / len(records)) if records else None
            ),
            "layers": [
                {
                    "layer_index": int(layer_index),
                    "pool_size": int(len(self.cex_pool.get(layer_index, []))),
                    "esbmc_calls_saved_by_cex_filter": int(
                        self.cex_filtered_counts.get(layer_index, 0)
                    ),
                    "bit_jumps": [
                        dict(jump)
                        for jump in self.cex_bit_jumps.get(layer_index, [])
                    ],
                }
                for layer_index in sorted(
                    set(self.cex_pool)
                    | set(self.cex_filtered_counts)
                    | set(self.cex_bit_jumps)
                )
            ],
            "esbmc_calls_saved_by_cex_filter": int(
                sum(self.cex_filtered_counts.values())
            ),
        }

    def end_to_end_summary(self) -> dict[str, Any]:
        return dict(self.end_to_end_record)

    def esbmc_call_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.esbmc_call_records]
        statuses = [
            "VERIFIED",
            "FAILED",
            "TIMEOUT",
            "MEMOUT",
            "UNKNOWN",
            "SKIPPED",
            "SENTINEL_EXPECTED_FAILURE",
            "VACUOUS",
        ]
        counts = {
            status.lower(): int(sum(1 for record in records if record.get("status") == status))
            for status in statuses
        }
        executed_records = [
            record
            for record in records
            if record.get("status") in statuses
            and record.get("status") != "SKIPPED"
        ]
        query_times = [
            float(record.get("elapsed_seconds", record.get("time", 0.0)) or 0.0)
            for record in executed_records
        ]
        query_peak_memory_bytes = [
            int(record["peak_memory_bytes"])
            for record in executed_records
            if record.get("peak_memory_bytes") is not None
        ]
        total_calls = int(sum(counts.values()))
        total_non_skipped = int(total_calls - counts["skipped"])
        return {
            "records": records,
            "verified_count": counts["verified"],
            "failed_count": counts["failed"],
            "timeout_count": counts["timeout"],
            "memout_count": counts["memout"],
            "unknown_count": counts["unknown"],
            "skipped_count": counts["skipped"],
            "vacuity_sentinel_count": counts["sentinel_expected_failure"],
            "vacuous_count": counts["vacuous"],
            "total_count": total_calls,
            "executed_count": total_non_skipped,
            "timeout_rate": float(counts["timeout"] / total_calls) if total_calls else 0.0,
            "memout_rate": float(counts["memout"] / total_calls) if total_calls else 0.0,
            "unknown_rate": float(counts["unknown"] / total_calls) if total_calls else 0.0,
            "total_time_seconds": float(sum(query_times)),
            "max_query_time_seconds": float(max(query_times, default=0.0)),
            "mean_query_time_seconds": float(sum(query_times) / len(query_times)) if query_times else 0.0,
            "memory_measurement": (
                "linux_procfs_process_tree_rss"
                if query_peak_memory_bytes
                else "unavailable"
            ),
            "memory_queries_measured": int(len(query_peak_memory_bytes)),
            "max_query_peak_memory_bytes": int(max(query_peak_memory_bytes, default=0)),
            "max_query_peak_memory_mib": float(
                max(query_peak_memory_bytes, default=0) / (1024 * 1024)
            ),
            "mean_query_peak_memory_bytes": float(
                sum(query_peak_memory_bytes) / len(query_peak_memory_bytes)
            ) if query_peak_memory_bytes else 0.0,
            "mean_query_peak_memory_mib": float(
                (sum(query_peak_memory_bytes) / len(query_peak_memory_bytes)) / (1024 * 1024)
            ) if query_peak_memory_bytes else 0.0,
            "largest_neurons_per_query": int(
                max((int(record.get("neurons_per_query", 0) or 0) for record in records), default=0)
            ),
            "largest_input_dim_per_query": int(
                max((int(record.get("input_dim", 0) or 0) for record in records), default=0)
            ),
            "largest_estimated_macs_per_query": int(
                max((int(record.get("estimated_macs", 0) or 0) for record in records), default=0)
            ),
        }

    def stats_snapshot(self) -> dict[str, float]:
        return {key: float(value) for key, value in self._stats.items()}

    def no_saturation_block_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.esbmc_no_saturation_block_records]

        return {
            "no_saturation_blocks": records,
            "no_saturation_blocks_total": int(len(records)),
            "no_saturation_blocks_verified": int(
                sum(1 for record in records if record.get("status") == "VERIFIED")
            ),
            "no_saturation_blocks_failed": int(
                sum(1 for record in records if record.get("status") == "FAILED")
            ),
            "no_saturation_blocks_timeout": int(
                sum(1 for record in records if record.get("status") == "TIMEOUT")
            ),
            "no_saturation_blocks_memout": int(
                sum(1 for record in records if record.get("status") == "MEMOUT")
            ),
            "no_saturation_blocks_unknown": int(
                sum(1 for record in records if record.get("status") == "UNKNOWN")
            ),
            "no_saturation_blocks_skipped": int(
                sum(1 for record in records if record.get("status") == "SKIPPED")
            ),
        }

    def update_quantized_weights_affine(
        self,
        in_layer: LayerEncoding,
        out_layer: LayerEncoding,
        num_bit: int,
        frac_bit_weights: int,
        frac_bit_bias: int,
        in_layer_index: int,
    ) -> None:
        min_fp_weight, max_fp_weight = int_get_min_max(num_bit, frac_bit_weights)
        del min_fp_weight, max_fp_weight
        min_fp_bias, max_fp_bias = int_get_min_max(num_bit, frac_bit_bias)
        del min_fp_bias, max_fp_bias, in_layer
        for out_index in range(out_layer.layer_size):
            weight_row = out_layer.layer_paras[0][out_index]
            bias = out_layer.layer_paras[1][out_index]
            weight_row_int = quantize_int(np.asarray(weight_row), num_bit, frac_bit_weights)
            weight_row_fp = np.asarray(weight_row_int, dtype=np.float64) / (2**frac_bit_weights)
            bias_fp = float(quantize_int(bias, num_bit, frac_bit_bias) / (2**frac_bit_bias))

            neuron = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index]
            neuron.weight = weight_row_fp
            neuron.bias = bias_fp
            neuron.algebra_lower = np.append(weight_row_fp, [bias_fp])
            neuron.algebra_upper = np.append(weight_row_fp, [bias_fp])

    def write_result(self, qu_frac_list: list[int], file_name: str | Path) -> None:
        path = Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        real_qu_list: list[int] = []
        frac_qu_list: list[int] = []
        int_qu_list: list[int] = []
        for i, _ in enumerate(self.dense_layers):
            exported_int_bits = _export_integer_bits(int(self.dense_layers[i].int_bit))
            real_qu_list.append(qu_frac_list[i] + exported_int_bits + 1)
            frac_qu_list.append(qu_frac_list[i])
            int_qu_list.append(exported_int_bits)

        exported_output_int_bits = _export_integer_bits(int(self.output_layer.int_bit))
        real_qu_list.append(qu_frac_list[-1] + exported_output_int_bits + 1)
        frac_qu_list.append(qu_frac_list[-1])
        int_qu_list.append(exported_output_int_bits)

        text = {
            "Solving Result": True,
            "all_quantization_bits": real_qu_list,
            "fractional_bits": frac_qu_list,
            "integer_bits": int_qu_list,
            "stats": self._stats,
        }
        path.write_text(json.dumps(text, indent=2), encoding="utf-8")

    def forward_quantization(self) -> tuple[bool, Any, Any, Any]:
        if self.gp_model is None:
            raise RuntimeError("--verify-mode milp requires a MILP solver model; use --verify-mode esbmc with cached preimage.")
        qu_list: list[int] = []
        qu_frac_list: list[int] = []
        qu_int_list: list[int] = []

        nonInputLayers = self.dense_layers.copy()
        nonInputLayers.append(self.output_layer)
        in_layer_index = -1

        for cur_layer in nonInputLayers:
            in_layer_index += 1
            in_layer = self.input_layer if cur_layer.layer_index == 1 else self.dense_layers[cur_layer.layer_index - 2]
            w = cur_layer.layer_paras[0]
            b = cur_layer.layer_paras[1]
            ifFound = False

            for rela_bit in range(self.bit_ub - self.bit_lb + 1):
                pre_mul_qu_lb_deepPoly = []
                pre_mul_qu_ub_deepPoly = []
                if ifFound:
                    break

                model_cstr_ll: list[Any] = []
                prop_cstr_ll: list[Any] = []
                var_ll: list[Any] = []

                frac_bit = rela_bit + self.bit_lb
                int_bit = int(cur_layer.int_bit)
                all_bit = frac_bit + int_bit
                qu_w = quantize_int(w, all_bit, frac_bit) / (2**frac_bit)
                qu_b = quantize_int(b, all_bit, frac_bit) / (2**frac_bit)

                target_lb = 0
                other_ubs = []
                sumOfK = 0
                numOfK = 0

                for out_index in range(cur_layer.layer_size):
                    qu_weights = qu_w[out_index]
                    qu_bias = qu_b[out_index]

                    lower_bound = np.append(qu_weights, qu_bias)
                    upper_bound = np.append(qu_weights, qu_bias)
                    cur_neuron_concrete_algebra_lower = None
                    cur_neuron_concrete_algebra_upper = None

                    if in_layer_index == 0:
                        cur_neuron_concrete_algebra_lower = deepcopy(lower_bound)
                        cur_neuron_concrete_algebra_upper = deepcopy(upper_bound)

                    for kk in range(2 * (in_layer_index + 1) - 1)[::-1]:
                        tmp_lower = np.zeros(len(self.deepPolyNets_DNN.layers[kk].neurons[0].algebra_lower))
                        tmp_upper = np.zeros(len(self.deepPolyNets_DNN.layers[kk].neurons[0].algebra_lower))

                        for pp in range(self.deepPolyNets_DNN.layers[kk].size):
                            if lower_bound[pp] >= 0:
                                tmp_lower += np.float32(lower_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[pp].algebra_lower)
                            else:
                                tmp_lower += np.float32(lower_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[pp].algebra_upper)

                            if upper_bound[pp] >= 0:
                                tmp_upper += np.float32(upper_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[pp].algebra_upper)
                            else:
                                tmp_upper += np.float32(upper_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[pp].algebra_lower)

                        tmp_lower[-1] += lower_bound[-1]
                        tmp_upper[-1] += upper_bound[-1]
                        lower_bound = deepcopy(tmp_lower)
                        upper_bound = deepcopy(tmp_upper)
                        if kk == 1:
                            cur_neuron_concrete_algebra_lower = deepcopy(lower_bound)
                            cur_neuron_concrete_algebra_upper = deepcopy(upper_bound)

                    cur_neuron_concrete_lower = lower_bound[0]
                    cur_neuron_concrete_upper = upper_bound[0]

                    pre_mul_qu_lb_deepPoly.append(cur_neuron_concrete_lower)
                    pre_mul_qu_ub_deepPoly.append(cur_neuron_concrete_upper)

                    quantized_lb_expression = self._linear_combination(
                        cur_neuron_concrete_algebra_lower[:-1],
                        self.input_gp_vars,
                        cur_neuron_concrete_algebra_lower[-1],
                    )
                    quantized_ub_expression = self._linear_combination(
                        cur_neuron_concrete_algebra_upper[:-1],
                        self.input_gp_vars,
                        cur_neuron_concrete_algebra_upper[-1],
                    )

                    if cur_layer.layer_index == (len(self.dense_layers) + 1):
                        if out_index == self.targetCls:
                            target_lb = quantized_lb_expression
                        else:
                            other_ubs.append(quantized_ub_expression)
                    else:
                        k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_lb)
                        if cur_layer.relaxed_ub[out_index] > 0:
                            prop_cstr_ll.append(
                                self.gp_model.addConstr(
                                    quantized_lb_expression <= cur_layer.relaxed_lb_expression[out_index] - 1000 * (k_i_lb - 1) - self.tole
                                )
                            )
                            prop_cstr_ll.append(
                                self.gp_model.addConstr(
                                    quantized_lb_expression >= cur_layer.relaxed_lb_expression[out_index] - 1000 * k_i_lb + self.tole
                                )
                            )
                            sumOfK = sumOfK + k_i_lb
                            numOfK += 1

                        k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_ub)
                        prop_cstr_ll.append(
                            self.gp_model.addConstr(
                                quantized_ub_expression >= cur_layer.relaxed_ub_expression[out_index] + 1000 * (k_i_ub - 1) + self.tole
                            )
                        )
                        prop_cstr_ll.append(
                            self.gp_model.addConstr(
                                quantized_ub_expression <= cur_layer.relaxed_ub_expression[out_index] + 1000 * k_i_ub - self.tole
                            )
                        )
                        numOfK += 1
                        sumOfK = sumOfK + k_i_ub

                if other_ubs:
                    for other_single_ub in other_ubs:
                        k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_ub)
                        prop_cstr_ll.append(
                            self.gp_model.addConstr(other_single_ub >= target_lb + 1000 * (k_i_ub - 1) + self.tole)
                        )
                        prop_cstr_ll.append(
                            self.gp_model.addConstr(other_single_ub <= target_lb + 1000 * k_i_ub - self.tole)
                        )
                        sumOfK = sumOfK + k_i_ub
                        numOfK += 1

                if not other_ubs and self.ifRelax == 1:
                    prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= int(numOfK * 0.25) + 1))
                else:
                    prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

                self.gp_model.update()
                self.gp_model.setParam("DualReductions", 0)
                self.gp_model.optimize()

                if self.gp_model.status == GRB.INFEASIBLE:
                    cur_layer.frac_bit = frac_bit
                    qu_frac_list.append(frac_bit)
                    qu_int_list.append(_export_integer_bits(int_bit))
                    qu_list.append(all_bit)
                    ifFound = True
                    self.gp_model.remove(model_cstr_ll)
                    self.gp_model.remove(prop_cstr_ll)
                    self.gp_model.remove(var_ll)
                    self.gp_model.update()
                    self.update_quantized_weights_affine(in_layer, cur_layer, all_bit, frac_bit, frac_bit, in_layer_index)

                    if cur_layer.layer_index < (len(self.dense_layers) + 1):
                        for out_index in range(cur_layer.layer_size):
                            lb_new = pre_mul_qu_lb_deepPoly[out_index]
                            ub_new = pre_mul_qu_ub_deepPoly[out_index]
                            cur_neuron = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1)].neurons[out_index]
                            if lb_new >= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_lower[out_index] = 1
                                cur_neuron.algebra_upper[out_index] = 1
                            elif ub_new <= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                            elif lb_new + ub_new <= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                k_new = ub_new / (ub_new - lb_new)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper[out_index] = k_new
                                cur_neuron.algebra_upper[-1] = -k_new * lb_new
                            else:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_lower[out_index] = 1
                                k_new = ub_new / (ub_new - lb_new)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper[out_index] = k_new
                                cur_neuron.algebra_upper[-1] = -k_new * lb_new
                    else:
                        self.output_layer.qu_lb = pre_mul_qu_lb_deepPoly
                        self.output_layer.qu_ub = pre_mul_qu_ub_deepPoly
            if not ifFound:
                return False, None, None, None

        return True, qu_list, qu_frac_list, qu_int_list


QuadapterRobustnessSynthesizer = GPEncoding
