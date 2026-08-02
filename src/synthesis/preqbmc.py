from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
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
from utils.fixed_point import (
    clamp_to_signed_range,
    int_get_min_max,
    quantize_int,
    round_divide_half_away_from_zero,
)
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
    render_prefix_direction_cut_validation_program,
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
    margin_cuts: bool | None = None
    e2e_fallback: bool | None = None
    cegar_max_rounds: int = 3

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
            margin_cuts=(
                None
                if getattr(args, "margin_cuts", None) is None
                else str(getattr(args, "margin_cuts")).lower() == "on"
            ),
            e2e_fallback=(
                None
                if getattr(args, "e2e_fallback", None) is None
                else str(getattr(args, "e2e_fallback")).lower() == "on"
            ),
            cegar_max_rounds=max(0, int(getattr(args, "cegar_max_rounds", 3))),
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
        self.preimage_source = "deeppoly_forward_FALLBACK"
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
        self.margin_cuts = (
            self.error_budget_mode == "derived"
            if self.config.margin_cuts is None
            else bool(self.config.margin_cuts)
        )
        self.e2e_fallback = (
            self.error_budget_mode == "derived"
            if self.config.e2e_fallback is None
            else bool(self.config.e2e_fallback)
        )
        self.e2e_fallback_attempted = False
        self.cegar_max_rounds = max(0, int(self.config.cegar_max_rounds))
        self.composition_path = (
            "network_e2e" if self.harness_scope == "network" else "layer_contracts"
        )
        self.margin_cut_records: list[dict[str, Any]] = []
        self.hidden_contract_cut_records: list[dict[str, Any]] = []
        self._hidden_contract_cut_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._relational_cut_validation_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.preimage_attempt_records: list[dict[str, Any]] = []
        self.preimage_deflation_records: list[dict[str, Any]] = []
        self._last_contract_target_status = "NOT_RUN"
        self.esbmc_call_records: list[dict[str, Any]] = []
        self.esbmc_block_records: list[dict[str, Any]] = []
        self.esbmc_no_saturation_block_records: list[dict[str, Any]] = []
        self.chaining_records: list[dict[str, Any]] = []
        self.output_margin_records: list[dict[str, Any]] = []
        self.vacuity_records: list[dict[str, Any]] = []
        self.source_region_record: dict[str, Any] = {
            "method": "deeppoly",
            "status": "NOT_RUN",
            "eligible_for_transfer": False,
            "quantized_pipeline_started": False,
            "esbmc_attempted": False,
            "target_class": int(original_prediction),
            "input_epsilon": float(self.eps),
        }
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
        source_check_start = time.monotonic()
        self.assert_input_box(lb, ub)
        self.symbolic_propagate()

        out_bounds_lb = self.output_layer.lb
        other_max = -1000.0
        for index, value in enumerate(self.output_layer.ub):
            if index == self.targetCls:
                continue
            other_max = max(other_max, value)

        target_lower = float(out_bounds_lb[self.targetCls])
        certified_margin_lower = float(target_lower - other_max)
        source_verified = target_lower >= other_max
        self.source_region_record = {
            "method": "deeppoly",
            "status": "VERIFIED" if source_verified else "INCONCLUSIVE",
            "eligible_for_transfer": bool(source_verified),
            "quantized_pipeline_started": bool(source_verified),
            "esbmc_attempted": False,
            "target_class": int(self.targetCls),
            "input_epsilon": float(self.eps),
            "target_lower_bound": target_lower,
            "maximum_competitor_upper_bound": float(other_max),
            "certified_margin_lower_bound": certified_margin_lower,
            "elapsed_seconds": float(time.monotonic() - source_check_start),
            "interpretation": (
                "The original floating-point robustness property is certified over the input region."
                if source_verified
                else "DeepPoly did not establish the original floating-point robustness property; this is not a counterexample."
            ),
        }
        self._stats["source_property_time"] = float(
            self.source_region_record["elapsed_seconds"]
        )
        if not source_verified:
            self.synthesis_final_status = "SOURCE_PROPERTY_INCONCLUSIVE"
            self._stats["total_time"] = float(time.monotonic() - source_check_start)
            return SynthesisResult(
                success=False,
                total_bits=[],
                fractional_bits=[],
                integer_bits=[],
                stats={key: float(value) for key, value in self._stats.items()},
                final_status=self.synthesis_final_status,
            )

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
            missing_preimages = [
                layer
                for layer in self.dense_layers
                if str(getattr(layer, "preimage_source", ""))
                not in {
                    "milp_preimage",
                    "milp_preimage_no_violation_to_cap",
                    "quantized_milp_preimage",
                }
            ]
            if (
                self.error_budget_mode == "derived"
                and self.harness_scope == "layer"
                and missing_preimages
            ):
                self.synthesis_final_status = "PREIMAGE_UNAVAILABLE"
                LOGGER.error(
                    "Derived layer contracts require property preimages; unavailable for hidden layer(s): %s",
                    ", ".join(str(layer.layer_index) for layer in missing_preimages),
                )
                backward_end_time = time.time()
                self._stats["backward_time"] = backward_end_time - backward_start_time
                self._stats["forward_time"] = 0.0
                self._stats["total_time"] = self._stats["backward_time"]
                return SynthesisResult(
                    success=False,
                    total_bits=[],
                    fractional_bits=[],
                    integer_bits=[],
                    stats={key: float(value) for key, value in self._stats.items()},
                    final_status=self.synthesis_final_status,
                )
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
        self.source_region_record["esbmc_attempted"] = bool(
            self.esbmc_call_records
        )

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
            layer.preimage_source = "milp_preimage"

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
            milp_preimage_available = str(in_layer.preimage_source) in {
                "milp_preimage",
                "milp_preimage_no_violation_to_cap",
                "quantized_milp_preimage",
            }
            if self.preimg_mode == "abstr" or (
                self.preimg_mode == "comp" and not milp_preimage_available
            ):
                LOGGER.warning(
                    "Layer %s is using DeepPoly/abstract preimage fallback instead of a MILP preimage.",
                    in_layer.layer_index,
                )
                scale_value = self.underPreImageAbstr(in_layer_index, in_layer, cur_layer)
                if scale_value > 0:
                    in_layer.preimage_source = "deeppoly_forward_FALLBACK"
            property_preimage_available = str(in_layer.preimage_source) in {
                "milp_preimage",
                "milp_preimage_no_violation_to_cap",
                "quantized_milp_preimage",
            }
            if not property_preimage_available:
                LOGGER.warning(
                    "Layer %s has no successful preimage solve; using forward DeepPoly bounds.",
                    in_layer.layer_index,
                )
                in_layer.relaxed_lb = np.asarray(in_layer.lb, dtype=np.float32).copy()
                in_layer.relaxed_ub = np.asarray(in_layer.ub, dtype=np.float32).copy()
                in_layer.preimage_source = "deeppoly_forward_FALLBACK"
            self.scaleValueSet[in_layer.layer_index - 1] = scale_value
            if self.error_budget_mode == "derived" and not property_preimage_available:
                # A predecessor preimage cannot be derived through a missing
                # downstream property preimage. Stop instead of constructing
                # constraints from placeholder relaxed expressions.
                for predecessor in self.dense_layers[:in_layer_index]:
                    predecessor.relaxed_lb = np.asarray(
                        predecessor.lb, dtype=np.float32
                    ).copy()
                    predecessor.relaxed_ub = np.asarray(
                        predecessor.ub, dtype=np.float32
                    ).copy()
                    predecessor.preimage_source = "downstream_preimage_unavailable"
                    self.scaleValueSet[predecessor.layer_index - 1] = -10000.0
                break
            cur_layer = in_layer

    def _store_milp_preimage_solution(
        self,
        in_layer_index: int,
        in_layer: LayerEncoding,
        *,
        source: str,
    ) -> None:
        for in_index in range(in_layer.layer_size):
            alpha = self.gp_model.value(in_layer.alpha[in_index])
            beta = self.gp_model.value(in_layer.beta[in_index])
            in_layer.relaxed_ub[in_index] = in_layer.ub[in_index] + beta
            in_layer.relaxed_lb[in_index] = in_layer.lb[in_index] - alpha

            neuron = self.deepPolyNets_DNN.layers[
                2 * (in_layer_index + 1) - 1
            ].neurons[in_index]
            in_lb_algebra = neuron.concrete_algebra_lower
            in_ub_algebra = neuron.concrete_algebra_upper
            in_layer.relaxed_lb_expression[in_index] = self._linear_combination(
                in_lb_algebra[:-1],
                self.input_gp_vars,
                in_lb_algebra[-1] - alpha,
            )
            in_layer.relaxed_ub_expression[in_index] = self._linear_combination(
                in_ub_algebra[:-1],
                self.input_gp_vars,
                in_ub_algebra[-1] + beta,
            )
            if in_layer.relaxed_ub[in_index] <= 0:
                in_layer.relaxed_ub_expression[in_index] = 0

        in_layer.preimage_source = source

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
        initial_status = self.gp_model.status
        solver_status = str(initial_status)
        accepted_source: str | None = None
        cap_status: str | None = None
        cap_elapsed = 0.0
        if initial_status == GRB.OPTIMAL:
            candidate_scale = float(self.gp_model.value(relaxScale))
            # An existential violation at scale zero means the current box is
            # not a valid Cartesian property preimage.
            if candidate_scale > 0.0:
                scaleValue = candidate_scale
                accepted_source = "milp_preimage"
                self._store_milp_preimage_solution(
                    in_layer_index,
                    in_layer,
                    source=accepted_source,
                )
        elif initial_status == GRB.INFEASIBLE:
            # The primary query asks for any property violation over the whole
            # bounded expansion family. If it is proven infeasible, maximize
            # the expansion after removing only the violation constraints.
            # A feasible cap solution plus the prior infeasibility proof makes
            # that whole capped box a valid property preimage.
            self.gp_model.remove(prop_cstr_ll)
            prop_cstr_ll = []
            self.gp_model.update()
            self.gp_model.setObjective(relaxScale, GRB.MAXIMIZE)
            cap_started = time.time()
            self.gp_model.optimize()
            cap_finished = time.time()
            cap_elapsed = cap_finished - cap_started
            self._stats["solving_time"] += cap_elapsed
            cap_status = str(self.gp_model.status)
            if self.gp_model.status == GRB.OPTIMAL:
                scaleValue = float(self.gp_model.value(relaxScale))
                accepted_source = "milp_preimage_no_violation_to_cap"
                self._store_milp_preimage_solution(
                    in_layer_index,
                    in_layer,
                    source=accepted_source,
                )

        self.preimage_attempt_records.append(
            {
                "network_layer_index": int(in_layer.layer_index),
                "source": accepted_source or "milp_preimage",
                "solver_status": solver_status,
                "cap_solver_status": cap_status,
                "scale_value": float(scaleValue),
                "accepted": accepted_source is not None,
                "proof_kind": (
                    "minimum_violating_expansion_boundary"
                    if accepted_source == "milp_preimage"
                    else "no_violation_within_bounded_expansion_family"
                    if accepted_source == "milp_preimage_no_violation_to_cap"
                    else "none"
                ),
                "elapsed_seconds": float(
                    opt_finish_time - opt_start_time + cap_elapsed
                ),
            }
        )
        # Candidate-local constraints must not leak into the next backward
        # layer when the solver is infeasible, unbounded, or otherwise fails.
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
                margin_record: dict[str, Any] | None = None
                if self.error_budget_mode == "derived" and is_output_layer:
                    margin_record = self._record_output_margin_check(
                        cur_layer=cur_layer,
                        in_layer=in_layer,
                        weights_int=qu_w_int,
                        layer_index=layer_index,
                        all_bit=all_bit,
                        frac_bit=frac_bit,
                    )

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

                if margin_record is not None and margin_record["analytic_margin_ok"]:
                    esbmc_result = self._analytic_output_margin_result()
                else:
                    esbmc_result = self.verify_layer_with_esbmc(
                        cur_layer=cur_layer,
                        in_layer=in_layer,
                        qu_w_int=qu_w_int,
                        qu_b_int=qu_b_int,
                        frac_bit=frac_bit,
                        all_bit=all_bit,
                        layer_index=layer_index,
                    )
                    if margin_record is not None:
                        candidate_q = list(selected_q)
                        candidate_f = list(selected_f)
                        candidate_i = list(selected_i)
                        candidate_q[layer_index] = int(all_bit)
                        candidate_f[layer_index] = int(frac_bit)
                        candidate_i[layer_index] = _export_integer_bits(int_bit)
                        esbmc_result = self._resolve_output_margin_result(
                            margin_record,
                            esbmc_result,
                            total_bits=candidate_q,
                            fractional_bits=candidate_f,
                            integer_bits=candidate_i,
                        )
                if esbmc_result.status != "VERIFIED":
                    terminal_statuses.append(
                        str(margin_record.get("status", "MARGIN_INCONCLUSIVE"))
                        if margin_record is not None
                        else str(esbmc_result.status)
                    )
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

        if "PREIMAGE_UNAVAILABLE" in terminal_statuses:
            self.synthesis_final_status = "PREIMAGE_UNAVAILABLE"
        elif terminal_statuses and set(terminal_statuses) <= {"MARGIN_INCONCLUSIVE"}:
            self.synthesis_final_status = "MARGIN_INCONCLUSIVE"
        elif "MARGIN_REFUTED" in terminal_statuses:
            self.synthesis_final_status = "MARGIN_REFUTED"
        elif terminal_statuses and set(terminal_statuses) <= {"PREIMAGE_DEFLATION_EMPTY"}:
            self.synthesis_final_status = "PREIMAGE_DEFLATION_EMPTY"
        elif "TIMEOUT" in terminal_statuses:
            self.synthesis_final_status = "TIMEOUT"
        elif "MEMOUT" in terminal_statuses:
            self.synthesis_final_status = "MEMOUT"
        elif "UNKNOWN" in terminal_statuses or "ERROR" in terminal_statuses:
            self.synthesis_final_status = "UNKNOWN"
        elif "LAYER_INCONCLUSIVE" in terminal_statuses:
            self.synthesis_final_status = "LAYER_INCONCLUSIVE"
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
            margin_record: dict[str, Any] | None = None
            if self.error_budget_mode == "derived" and is_output_layer:
                margin_record = self._record_output_margin_check(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    weights_int=np.asarray(qu_w_int),
                    layer_index=layer_index,
                    all_bit=q_bits,
                    frac_bit=f_bits,
                )

            if margin_record is not None and margin_record["analytic_margin_ok"]:
                contract_result = self._analytic_output_margin_result()
            else:
                contract_result = self.verify_layer_with_esbmc(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    qu_w_int=np.asarray(qu_w_int),
                    qu_b_int=np.asarray(qu_b_int),
                    frac_bit=f_bits,
                    all_bit=q_bits,
                    layer_index=layer_index,
                )
                if margin_record is not None:
                    contract_result = self._resolve_output_margin_result(
                        margin_record,
                        contract_result,
                        total_bits=[int(value) for value in total_bits],
                        fractional_bits=[int(value) for value in fractional_bits],
                        integer_bits=[int(value) for value in integer_bits],
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
                if margin_record is not None:
                    margin_status = str(margin_record.get("status", "MARGIN_INCONCLUSIVE"))
                    record["status"] = margin_status
                    record["contract_status"] = margin_status
                    record["failure_type"] = (
                        "derived_output_margin_refuted"
                        if margin_status == "MARGIN_REFUTED"
                        else "derived_output_margin_inconclusive"
                    )
                    record["output_margin"] = margin_record
                else:
                    record["status"] = contract_result.status
                    if contract_result.status == "PREIMAGE_DEFLATION_EMPTY":
                        record["failure_type"] = "preimage_deflation_empty"
                    elif contract_result.status == "PREIMAGE_UNAVAILABLE":
                        record["failure_type"] = "property_preimage_unavailable"
                record["final_status"] = (
                    "FAILED"
                    if margin_record is None and contract_result.status == "FAILED"
                    else "MARGIN_INCONCLUSIVE"
                    if margin_record is not None and margin_record.get("status") != "MARGIN_REFUTED"
                    else "MARGIN_REFUTED"
                    if margin_record is not None
                    else "PREIMAGE_DEFLATION_EMPTY"
                    if contract_result.status == "PREIMAGE_DEFLATION_EMPTY"
                    else "PREIMAGE_UNAVAILABLE"
                    if contract_result.status == "PREIMAGE_UNAVAILABLE"
                    else "UNKNOWN"
                )
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
            "cpu_time_seconds": result.cpu_time_seconds,
            "average_cpu_utilization_percent": result.average_cpu_utilization_percent,
            "cpu_measurement": result.cpu_measurement,
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

    def _candidate_contract_target_bounds_int(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        weights_int: np.ndarray,
        frac_bit: int,
        *,
        record: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return the candidate hidden contract target and its emitted budget.

        For the final hidden layer in derived mode, ``P`` is the property preimage
        produced by the backward method. The hidden harness permits a deviation of
        ``delta`` around its target interval. Emitting ``P deflated by delta`` makes
        the proved guarantee ``(P deflated by delta) expanded by delta``, which is a
        subset of ``P`` in exact integer arithmetic. No tolerance is weakened. If a
        coordinate collapses, the candidate is rejected as
        ``PREIMAGE_DEFLATION_EMPTY`` before ESBMC. A forward DeepPoly box is
        never substituted for ``P``; that case is ``PREIMAGE_UNAVAILABLE``.
        """

        scale = 1 << int(frac_bit)
        pre_low, pre_high = self._layer_preimage_bounds_int(cur_layer, scale)
        budget, assumed_lo, assumed_hi, frac_in = self._candidate_error_budget_int(
            cur_layer,
            in_layer,
            weights_int,
            frac_bit,
        )
        budget_components = getattr(self, "_last_error_budget_components", None)
        if (
            not isinstance(budget_components, dict)
            or "total" not in budget_components
            or not np.array_equal(
                np.asarray(budget_components["total"], dtype=np.int64),
                np.asarray(budget, dtype=np.int64),
            )
        ):
            budget_components = {"total": np.asarray(budget, dtype=np.int64)}
        is_last_hidden = (
            cur_layer.layer_index == len(self.dense_layers)
            and cur_layer.layer_index < len(self.dense_layers) + 1
        )
        preimage_source = str(
            getattr(cur_layer, "preimage_source", "deeppoly_forward_FALLBACK")
        )
        property_preimage_available = preimage_source in {
            "milp_preimage",
            "milp_preimage_no_violation_to_cap",
            "quantized_milp_preimage",
        }
        if (
            self.error_budget_mode == "derived"
            and is_last_hidden
            and not property_preimage_available
        ):
            # Forward reachable bounds and property preimages are different
            # mathematical objects. A forward box has no property-interior
            # slack and must never be deflated into a derived contract target.
            target_low = np.asarray(pre_low, dtype=np.int64)
            target_high = np.asarray(pre_high, dtype=np.int64)
            valid = False
            status = "PREIMAGE_UNAVAILABLE"
        elif self.error_budget_mode == "derived" and is_last_hidden:
            target_low = np.asarray(pre_low, dtype=np.int64) + np.asarray(budget, dtype=np.int64)
            target_high = np.asarray(pre_high, dtype=np.int64) - np.asarray(budget, dtype=np.int64)
            valid = bool(np.all(target_low <= target_high))
            status = "DEFLATED" if valid else "PREIMAGE_DEFLATION_EMPTY"
        else:
            target_low = np.asarray(pre_low, dtype=np.int64)
            target_high = np.asarray(pre_high, dtype=np.int64)
            valid = True
            status = "NOT_REQUIRED"

        self._last_contract_target_status = status

        if record:
            collapsed = np.flatnonzero(target_low > target_high)
            widths = np.asarray(pre_high, dtype=np.int64) - np.asarray(pre_low, dtype=np.int64)
            symmetric_slack = np.maximum(widths // 2, 0)
            self.preimage_deflation_records.append(
                {
                    "layer_index": int(cur_layer.layer_index - 1),
                    "network_layer_index": int(cur_layer.layer_index),
                    "preimage_source": preimage_source,
                    "property_preimage_available": bool(property_preimage_available),
                    "Q": int(frac_bit + int(cur_layer.int_bit)),
                    "F": int(frac_bit),
                    "status": status,
                    "preimage_low_int": [int(value) for value in pre_low],
                    "preimage_high_int": [int(value) for value in pre_high],
                    "error_budget_int": [int(value) for value in budget],
                    "preimage_width_int": [int(value) for value in widths],
                    "preimage_symmetric_slack_int": [int(value) for value in symmetric_slack],
                    "budget_decomposition": {
                        name: [int(value) for value in values]
                        for name, values in budget_components.items()
                    },
                    "target_low_int": [int(value) for value in target_low],
                    "target_high_int": [int(value) for value in target_high],
                    "collapsed_neurons": [int(value) for value in collapsed],
                }
            )
        return target_low, target_high, np.asarray(budget, dtype=np.int64), valid

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

    def _derived_error_budget_components_int(
        self,
        cur_layer: LayerEncoding,
        weights_int: np.ndarray,
        assumed_lo_int: np.ndarray,
        assumed_hi_int: np.ndarray,
        frac_in: int,
        delta_in_int: np.ndarray | int,
        frac_out: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Decompose implementation/real-affine deviation in output ULPs.

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

        delta_input_values: list[int] = []
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
            delta_input_values.append(int(delta_input))

        max_int64 = int(np.iinfo(np.int64).max)
        budget_values = [
            int(delta_weights_scalar + 1 + delta_input)
            for delta_input in delta_input_values
        ]
        if any(value > max_int64 for value in budget_values):
            raise OverflowError("Derived error budget exceeds int64 reporting range.")
        return {
            "dw": np.full(
                weights.shape[0], int(delta_weights_scalar), dtype=np.int64
            ),
            "dr": np.ones(weights.shape[0], dtype=np.int64),
            "dp": np.asarray(delta_input_values, dtype=np.int64),
            "total": np.asarray(budget_values, dtype=np.int64),
        }

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
        """Return the conservative total derived budget in output ULPs."""

        components = self._derived_error_budget_components_int(
            cur_layer=cur_layer,
            weights_int=weights_int,
            assumed_lo_int=assumed_lo_int,
            assumed_hi_int=assumed_hi_int,
            frac_in=frac_in,
            delta_in_int=delta_in_int,
            frac_out=frac_out,
        )
        budget = components["total"]
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
        if self.error_budget_mode == "derived":
            self._last_error_budget_components = self._derived_error_budget_components_int(
                cur_layer=cur_layer,
                weights_int=weights_int,
                assumed_lo_int=assumed_lo,
                assumed_hi_int=assumed_hi,
                frac_in=frac_in,
                delta_in_int=self._inherited_error_budget_int(cur_layer, in_layer),
                frac_out=int(frac_bit),
            )
        else:
            self._last_error_budget_components = {
                "total": np.asarray(budget, dtype=np.int64)
            }
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
        if self.error_budget_mode == "derived":
            if in_layer is None or weights_int is None:
                raise ValueError("Derived chaining requires the input layer and quantized weights.")
            pre_lo_int, pre_hi_int, tolerance_int, target_valid = (
                self._candidate_contract_target_bounds_int(
                    cur_layer,
                    in_layer,
                    weights_int,
                    frac_bit,
                )
            )
            if not target_valid:
                raise RuntimeError("Cannot chain an empty deflated preimage target.")
        else:
            pre_lo_int, pre_hi_int = self._layer_preimage_bounds_int(cur_layer, scale)
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
        """Run the sufficient analytical output-margin pre-pass.

        This check bounds each logit independently, so a pass is a sound proof but a
        failure is only an abstraction artifact candidate. In particular, extrema for
        the target and a competitor may come from different hidden vectors. Callers
        must therefore send analytical failures to the exact deployed output harness
        instead of rejecting the format.
        """

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
            "analytic_margin_ok": bool(margin_ok),
            "margin_ok": bool(margin_ok),
            "output_margin": "analytic_pass" if margin_ok else "analytic_fail_pending_exact_query",
            "composition_path": "layer_analytic" if margin_ok else None,
            "status": "VERIFIED" if margin_ok else "PENDING_EXACT_QUERY",
            "class_margins": margins,
        }
        self.output_margin_records.append(record)
        if margin_ok:
            self.composition_path = "layer_analytic"
        return record

    @staticmethod
    def _round_fraction_half_away_from_zero(value: Fraction) -> int:
        if value >= 0:
            return (2 * value.numerator + value.denominator) // (2 * value.denominator)
        positive = -value
        return -(
            (2 * positive.numerator + positive.denominator)
            // (2 * positive.denominator)
        )

    def _solve_margin_direction_milp(
        self,
        direction: np.ndarray,
        *,
        hidden_layer_count: int | None = None,
    ) -> tuple[float, float, float]:
        """Bound one direction over an exact real-network hidden prefix.

        The MILP variables encode the original real input box, affine hidden
        pre-activations, and exact ReLUs. DeepPoly pre-activation bounds provide the
        finite big-M constants. The returned endpoints use each backend's global
        objective bound and are rounded outward with ``nextafter``.
        """

        prefix_count = (
            len(self.dense_layers)
            if hidden_layer_count is None
            else int(hidden_layer_count)
        )
        if prefix_count < 1 or prefix_count > len(self.dense_layers):
            raise ValueError("Directional MILP requires a non-empty hidden prefix.")
        prefix = self.dense_layers[:prefix_count]
        low = np.asarray(self.x_low_real, dtype=np.float64).reshape(-1)
        high = np.asarray(self.x_high_real, dtype=np.float64).reshape(-1)
        if direction.shape != (prefix[-1].layer_size,):
            raise ValueError("Output direction does not match the last hidden layer.")
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("Margin-cut MILP requires finite input and DeepPoly bounds.")

        model = build_model(
            self.solver,
            "output_margin_direction",
            threads=max(1, int(getattr(self.config, "gurobi_threads", 4))),
            output_flag=0,
        )
        previous_values = [
            model.add_var(lb=float(lo), ub=float(hi), name=f"margin_x_{index}")
            for index, (lo, hi) in enumerate(zip(low, high))
        ]
        previous_bounds = [(float(lo), float(hi)) for lo, hi in zip(low, high)]
        for layer_offset, hidden in enumerate(prefix):
            weights = np.asarray(hidden.layer_paras[0], dtype=np.float64)
            biases = np.asarray(hidden.layer_paras[1], dtype=np.float64)
            pre_low = np.asarray(hidden.lb, dtype=np.float64).reshape(-1)
            pre_high = np.asarray(hidden.ub, dtype=np.float64).reshape(-1)
            if weights.shape != (hidden.layer_size, len(previous_values)):
                raise ValueError("Hidden weights are not row-per-neuron for directional MILP.")
            if not (np.all(np.isfinite(pre_low)) and np.all(np.isfinite(pre_high))):
                raise ValueError("Directional MILP requires finite DeepPoly bounds.")

            hidden_values: list[Any] = []
            hidden_bounds: list[tuple[float, float]] = []
            for neuron, (row, bias, lo, hi) in enumerate(
                zip(weights, biases, pre_low, pre_high)
            ):
                if float(lo) > float(hi):
                    raise ValueError("DeepPoly produced an invalid hidden interval.")
                prefix_name = f"margin_l{layer_offset}_n{neuron}"
                pre = model.add_var(lb=float(lo), ub=float(hi), name=f"{prefix_name}_pre")
                model.add_constr(
                    pre == self._linear_combination(row, previous_values, float(bias)),
                    name=f"{prefix_name}_affine",
                )
                relu_low = max(0.0, float(lo))
                relu_high = max(0.0, float(hi))
                value = model.add_var(lb=relu_low, ub=relu_high, name=f"{prefix_name}_relu")
                if float(hi) <= 0.0:
                    model.add_constr(value == 0.0, name=f"{prefix_name}_zero")
                elif float(lo) >= 0.0:
                    model.add_constr(value == pre, name=f"{prefix_name}_linear")
                else:
                    active = model.add_var(
                        lb=0.0,
                        ub=1.0,
                        vtype=GRB.BINARY,
                        name=f"{prefix_name}_active",
                    )
                    model.add_constr(value >= pre, name=f"{prefix_name}_lower_pre")
                    model.add_constr(value >= 0.0, name=f"{prefix_name}_lower_zero")
                    model.add_constr(
                        value <= pre - float(lo) * (1 - active),
                        name=f"{prefix_name}_upper_pre",
                    )
                    model.add_constr(
                        value <= float(hi) * active,
                        name=f"{prefix_name}_upper_zero",
                    )
                hidden_values.append(value)
                hidden_bounds.append((relu_low, relu_high))
            previous_values = hidden_values
            previous_bounds = hidden_bounds

        direction_low, direction_high = self._interval_linear_combination(
            direction,
            previous_bounds,
            0.0,
        )
        objective = model.add_var(
            lb=float(direction_low),
            ub=float(direction_high),
            name="margin_direction_value",
        )
        model.add_constr(
            objective == self._linear_combination(direction, previous_values, 0.0),
            name="margin_direction_definition",
        )
        started = time.monotonic()
        model.set_objective(objective, GRB.MINIMIZE)
        if model.optimize() != GRB.OPTIMAL:
            raise RuntimeError("Margin-cut MILP minimization did not reach OPTIMAL.")
        lower = math.nextafter(float(model.objective_bound()), -math.inf)
        model.set_objective(objective, GRB.MAXIMIZE)
        if model.optimize() != GRB.OPTIMAL:
            raise RuntimeError("Margin-cut MILP maximization did not reach OPTIMAL.")
        upper = math.nextafter(float(model.objective_bound()), math.inf)
        return lower, upper, float(time.monotonic() - started)

    def _deployed_prefix_cut_payload(
        self,
        hidden_layer_count: int,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], tuple[tuple[int, int], ...]]:
        """Quantize one already-selected hidden prefix for exact cut validation."""

        prefix = self.dense_layers[: int(hidden_layer_count)]
        if not prefix or len(prefix) != int(hidden_layer_count):
            raise ValueError("Cut validation requires a non-empty selected hidden prefix.")

        formats: list[tuple[int, int]] = []
        payloads: list[dict[str, object]] = []
        previous_fractional_bits: int | None = None
        for layer in prefix:
            if layer.frac_bit is None or layer.int_bit is None:
                raise RuntimeError(
                    "Cannot validate a relational cut before the prefix Q/I/F is selected."
                )
            fractional_bits = int(layer.frac_bit)
            total_bits = fractional_bits + int(layer.int_bit)
            formats.append((total_bits, fractional_bits))
            input_fractional_bits = (
                fractional_bits
                if previous_fractional_bits is None
                else previous_fractional_bits
            )
            weights_int = np.asarray(
                quantize_int(layer.layer_paras[0], total_bits, fractional_bits),
                dtype=np.int64,
            )
            biases_int = np.asarray(
                quantize_int(layer.layer_paras[1], total_bits, fractional_bits),
                dtype=np.int64,
            )
            payloads.append(
                {
                    "input_size": int(weights_int.shape[1]),
                    "output_size": int(weights_int.shape[0]),
                    "total_bits": total_bits,
                    "fractional_bits": fractional_bits,
                    "input_fractional_bits": input_fractional_bits,
                    "weights_c_int": self.numpy_to_c_int_array(weights_int),
                    "biases_c_int": self.numpy_to_c_int_array(biases_int),
                }
            )
            previous_fractional_bits = fractional_bits

        input_total_bits, input_fractional_bits = formats[0]
        input_scale = 1 << input_fractional_bits
        input_min = -(1 << (input_total_bits - 1))
        input_max = (1 << (input_total_bits - 1)) - 1
        input_low = np.maximum(
            np.floor(np.asarray(self.x_low_real, dtype=np.float64) * input_scale),
            input_min,
        ).astype(np.int64)
        input_high = np.minimum(
            np.ceil(np.asarray(self.x_high_real, dtype=np.float64) * input_scale),
            input_max,
        ).astype(np.int64)
        return input_low, input_high, payloads, tuple(formats)

    def _formally_validate_relational_cut(
        self,
        record: dict[str, Any],
        *,
        hidden_layer_count: int,
        layer_index: int,
        all_bit: int,
        frac_bit: int,
        cut_kind: str,
        identifier: str,
    ) -> bool:
        """Validate a MILP-proposed cut over the exact deployed prefix with ESBMC."""

        input_low, input_high, layers, prefix_formats = self._deployed_prefix_cut_payload(
            hidden_layer_count
        )
        direction = tuple(int(value) for value in record["direction_int"])
        cut_low = int(record["cut_low_int"])
        cut_high = int(record["cut_high_int"])
        cache_key = (prefix_formats, direction, cut_low, cut_high)
        validation_cache = getattr(self, "_relational_cut_validation_cache", None)
        if validation_cache is None:
            validation_cache = {}
            self._relational_cut_validation_cache = validation_cache
        cached = validation_cache.get(cache_key)
        if cached is not None:
            record.update(
                {
                    "formal_validation_status": str(cached["status"]),
                    "formal_validation_harness": str(cached["harness"]),
                    "formal_validation_cache_hit": True,
                    "soundness": str(cached["soundness"]),
                }
            )
            return str(cached["status"]) == "VERIFIED"

        source = render_prefix_direction_cut_validation_program(
            input_size=int(input_low.size),
            input_bounds_low_c_int=self.numpy_to_c_int_array(input_low),
            input_bounds_high_c_int=self.numpy_to_c_int_array(input_high),
            layers=layers,
            direction_c_int=self.numpy_to_c_int_array(
                np.asarray(direction, dtype=object)
            ),
            cut_low_int=cut_low,
            cut_high_int=cut_high,
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "formats": prefix_formats,
                    "direction": direction,
                    "low": cut_low,
                    "high": cut_high,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        cut_dir = self.output_dir / "layers" / "cuts"
        cut_dir.mkdir(parents=True, exist_ok=True)
        harness = cut_dir / (
            f"{cut_kind}_layer_{int(layer_index)}_{identifier}_{fingerprint}.c"
        )
        harness.write_text(source, encoding="utf-8")
        result = self._run_esbmc_file(harness, extract_counterexample=False)
        self._stats["esbmc_calls"] += 1.0
        call_record = self._esbmc_call_record(
            result=result,
            layer_index=layer_index,
            block_index=None,
            start_neuron=None,
            end_neuron=None,
            all_bit=all_bit,
            frac_bit=frac_bit,
            harness=harness,
            property_type="relational_cut_validation",
            mode="exact_deployed_prefix",
            input_dim=int(input_low.size),
            output_neurons=1,
        )
        call_record.update(
            {
                "cut_kind": cut_kind,
                "cut_identifier": identifier,
                "hidden_prefix_layers": int(hidden_layer_count),
                "prefix_formats": [
                    {"Q": int(q), "F": int(f)} for q, f in prefix_formats
                ],
                "direction_int": list(direction),
                "cut_low_int": cut_low,
                "cut_high_int": cut_high,
            }
        )
        self.esbmc_call_records.append(call_record)

        validation_status = str(result.status)
        soundness = (
            "esbmc_exact_deployed_prefix_validated"
            if validation_status == "VERIFIED"
            else "untrusted_milp_proposal_not_injected"
        )
        record.update(
            {
                "formal_validation_status": validation_status,
                "formal_validation_harness": str(harness),
                "formal_validation_resource_control": result.resource_control,
                "formal_validation_cache_hit": False,
                "soundness": soundness,
            }
        )
        validation_cache[cache_key] = {
            "status": validation_status,
            "harness": str(harness),
            "soundness": soundness,
        }
        return validation_status == "VERIFIED"

    def _margin_cut_bounds(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        frac_bit: int,
        all_bit: int,
    ) -> list[dict[str, Any]]:
        """Build candidate integer cuts for decisive output directions.

        For competitor ``k``, the real MILP bounds ``d_k*h_real`` where
        ``d_k = V_real[k]-V_real[target]``. The output harness instead contains the
        deployed hidden integer vector ``h_int`` and a quantized direction ``D_int``.
        At product scale ``S_d*S_h`` the inherited hidden error contributes
        ``S_d * sum_j |d_j|*delta_j``. Direction quantization contributes
        ``sum_j |D_j-S_d*d_j|*max(|h_j|)``. Both exact rational sums are rounded
        upward, while MILP endpoints are converted outward.  Because the numerical
        MILP backend is not proof-producing, each resulting bound is additionally
        checked over the exact deployed prefix by ESBMC before it is returned.
        """

        if not self.margin_cuts or self.error_budget_mode != "derived":
            return []
        if self.property_spec.valid_labels:
            self.margin_cut_records.append(
                {
                    "enabled": True,
                    "status": "SKIPPED_UNSUPPORTED_VALID_SET_PROPERTY",
                    "hidden_layers": int(len(self.dense_layers)),
                }
            )
            return []

        real_weights = np.asarray(cur_layer.layer_paras[0], dtype=np.float64)
        real_biases = np.asarray(cur_layer.layer_paras[1], dtype=np.float64)
        target = int(
            self.property_spec.target_label
            if self.property_spec.target_label is not None
            else self.targetCls
        )
        if real_weights.shape != (cur_layer.layer_size, in_layer.layer_size):
            raise ValueError("Output weights must use row-per-class orientation.")
        center_hidden = np.asarray(in_layer.realVal, dtype=np.float64).reshape(-1)
        if center_hidden.size != in_layer.layer_size:
            raise ValueError("Center hidden activation is unavailable for orientation check.")
        center_logits = real_weights @ center_hidden + real_biases
        if int(np.argmax(center_logits)) != target:
            raise ValueError("Output row orientation check disagrees with the target class.")

        assumed_low, assumed_high, frac_in = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        delta = np.asarray(in_layer.error_budget_int, dtype=object).reshape(-1)
        if delta.size != in_layer.layer_size:
            raise ValueError("Last hidden layer has no compatible derived error budget.")
        direction_scale = 1 << int(frac_bit)
        hidden_scale = 1 << int(frac_in)
        max_abs_hidden = [
            max(abs(int(lo)), abs(int(hi)))
            for lo, hi in zip(assumed_low, assumed_high)
        ]

        records: list[dict[str, Any]] = []
        for competitor in range(cur_layer.layer_size):
            if competitor == target:
                continue
            direction = real_weights[competitor] - real_weights[target]
            lower_real, upper_real, elapsed = self._solve_margin_direction_milp(direction)
            direction_fraction = [Fraction.from_float(float(value)) for value in direction]
            direction_int = np.asarray(
                [
                    self._round_fraction_half_away_from_zero(value * direction_scale)
                    for value in direction_fraction
                ],
                dtype=np.int64,
            )
            inherited_widening = sum(
                Fraction(direction_scale) * abs(value) * int(error)
                for value, error in zip(direction_fraction, delta)
            )
            coefficient_widening = sum(
                abs(Fraction(int(integer)) - value * direction_scale) * int(maximum)
                for integer, value, maximum in zip(
                    direction_int,
                    direction_fraction,
                    max_abs_hidden,
                )
            )
            widening = self._ceil_fraction(inherited_widening + coefficient_widening)
            product_scale = direction_scale * hidden_scale
            cut_low = self._floor_fraction(
                Fraction.from_float(float(lower_real)) * product_scale
            ) - widening
            cut_high = self._ceil_fraction(
                Fraction.from_float(float(upper_real)) * product_scale
            ) + widening
            record = {
                "enabled": True,
                "status": "PROPOSED",
                "proposal_status": "OPTIMAL",
                "target_class": int(target),
                "competitor_class": int(competitor),
                "direction_scale": int(direction_scale),
                "hidden_scale": int(hidden_scale),
                "product_scale": int(product_scale),
                "direction_int": [int(value) for value in direction_int],
                "milp_low_real": float(lower_real),
                "milp_high_real": float(upper_real),
                "inherited_widening_product_units": int(
                    self._ceil_fraction(inherited_widening)
                ),
                "coefficient_widening_product_units": int(
                    self._ceil_fraction(coefficient_widening)
                ),
                "total_widening_product_units": int(widening),
                "cut_low_int": int(cut_low),
                "cut_high_int": int(cut_high),
                "milp_wall_time_seconds": float(elapsed),
                "soundness": "pending_exact_deployed_prefix_validation",
            }
            validated = self._formally_validate_relational_cut(
                record,
                hidden_layer_count=int(in_layer.layer_index),
                layer_index=int(cur_layer.layer_index - 1),
                all_bit=int(all_bit),
                frac_bit=int(frac_bit),
                cut_kind="output_margin",
                identifier=f"competitor_{int(competitor)}",
            )
            record["status"] = "VERIFIED" if validated else "VALIDATION_REJECTED"
            self.margin_cut_records.append(record)
            if validated:
                records.append(record)
        return records

    def _hidden_contract_cut_bounds(
        self,
        *,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        weights_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        start_neuron: int,
        end_neuron: int,
        maximum_cuts: int,
        neuron_indices: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Propose and formally validate relational hidden-prefix invariants.

        Each direction is one exact quantized row of the layer under check.
        The MILP bounds that direction over the exact real-network prefix. The
        inherited per-coordinate implementation budget widens the result. ESBMC
        then checks the proposed integer endpoints over the exact deployed prefix;
        only validated cuts can reach a contract harness.
        """

        if (
            not self.margin_cuts
            or self.error_budget_mode != "derived"
            or in_layer.layer_index <= 0
            or maximum_cuts <= 0
        ):
            return []
        inherited = getattr(in_layer, "error_budget_int", None)
        if inherited is None:
            self.hidden_contract_cut_records.append(
                {
                    "layer_index": int(cur_layer.layer_index - 1),
                    "status": "SKIPPED_NO_INHERITED_BUDGET",
                }
            )
            return []

        _, _, frac_in = self._assumption_box_int(cur_layer, in_layer, frac_bit)
        scale_in = 1 << int(frac_in)
        delta = np.asarray(inherited, dtype=object).reshape(-1)
        rows = np.asarray(weights_int, dtype=object)
        records: list[dict[str, Any]] = []
        selected_neurons = (
            [
                int(neuron)
                for neuron in neuron_indices
                if int(start_neuron) <= int(neuron) < int(end_neuron)
            ][: int(maximum_cuts)]
            if neuron_indices is not None
            else list(
                range(
                    int(start_neuron),
                    min(int(end_neuron), int(start_neuron) + int(maximum_cuts)),
                )
            )
        )
        for neuron in selected_neurons:
            direction = np.asarray(rows[neuron], dtype=object).reshape(-1)
            if direction.size != in_layer.layer_size or delta.size != direction.size:
                raise ValueError("Hidden contract cut dimensions do not match the input layer.")
            cache_key = (
                int(cur_layer.layer_index),
                int(frac_in),
                tuple(
                    (
                        int(layer.frac_bit) + int(layer.int_bit),
                        int(layer.frac_bit),
                    )
                    for layer in self.dense_layers[: int(in_layer.layer_index)]
                ),
                tuple(int(value) for value in direction),
                tuple(int(value) for value in delta),
            )
            cached = self._hidden_contract_cut_cache.get(cache_key)
            if cached is not None:
                record = dict(cached)
                record["cache_hit"] = True
                records.append(record)
                self.hidden_contract_cut_records.append(record)
                continue

            lower_real, upper_real, elapsed = self._solve_margin_direction_milp(
                np.asarray([float(value) for value in direction], dtype=np.float64),
                hidden_layer_count=int(in_layer.layer_index),
            )
            widening = sum(
                abs(int(coefficient)) * int(error)
                for coefficient, error in zip(direction, delta)
            )
            cut_low = self._floor_fraction(
                Fraction.from_float(float(lower_real)) * scale_in
            ) - int(widening)
            cut_high = self._ceil_fraction(
                Fraction.from_float(float(upper_real)) * scale_in
            ) + int(widening)
            int64_info = np.iinfo(np.int64)
            if cut_low < int(int64_info.min) or cut_high > int(int64_info.max):
                self.hidden_contract_cut_records.append(
                    {
                        "layer_index": int(cur_layer.layer_index - 1),
                        "network_layer_index": int(cur_layer.layer_index),
                        "neuron_index": int(neuron),
                        "status": "SKIPPED_INT64_REPRESENTATION",
                        "cut_low_int": int(cut_low),
                        "cut_high_int": int(cut_high),
                    }
                )
                continue
            record = {
                "layer_index": int(cur_layer.layer_index - 1),
                "network_layer_index": int(cur_layer.layer_index),
                "neuron_index": int(neuron),
                "status": "PROPOSED",
                "proposal_status": "OPTIMAL",
                "direction_int": [int(value) for value in direction],
                "input_fractional_bits": int(frac_in),
                "milp_low_real": float(lower_real),
                "milp_high_real": float(upper_real),
                "inherited_widening_int": int(widening),
                "cut_low_int": int(cut_low),
                "cut_high_int": int(cut_high),
                "milp_wall_time_seconds": float(elapsed),
                "cache_hit": False,
                "soundness": "pending_exact_deployed_prefix_validation",
            }
            validated = self._formally_validate_relational_cut(
                record,
                hidden_layer_count=int(in_layer.layer_index),
                layer_index=int(cur_layer.layer_index - 1),
                all_bit=int(all_bit),
                frac_bit=int(frac_bit),
                cut_kind="hidden_contract",
                identifier=f"neuron_{int(neuron)}",
            )
            record["status"] = "VERIFIED" if validated else "VALIDATION_REJECTED"
            self.hidden_contract_cut_records.append(record)
            if validated:
                self._hidden_contract_cut_cache[cache_key] = dict(record)
                records.append(record)
        return records

    def _hidden_contract_violation_order(
        self,
        *,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        weights_int: np.ndarray,
        biases_int: np.ndarray,
        frac_bit: int,
        all_bit: int,
        start_neuron: int,
        end_neuron: int,
    ) -> list[int]:
        target_low, target_high, budget, valid = self._candidate_contract_target_bounds_int(
            cur_layer,
            in_layer,
            weights_int,
            frac_bit,
        )
        if not valid:
            return []
        assumed_low, assumed_high, frac_in = self._assumption_box_int(
            cur_layer,
            in_layer,
            frac_bit,
        )
        denominator = 1 << int(frac_in)
        scored: list[tuple[int, int]] = []
        for neuron in range(int(start_neuron), int(end_neuron)):
            lower_acc = 0
            upper_acc = 0
            for weight, low, high in zip(
                np.asarray(weights_int[neuron]).reshape(-1),
                assumed_low,
                assumed_high,
            ):
                coefficient = int(weight)
                lower_acc += coefficient * int(low if coefficient >= 0 else high)
                upper_acc += coefficient * int(high if coefficient >= 0 else low)
            lower = clamp_to_signed_range(
                round_divide_half_away_from_zero(lower_acc, denominator)
                + int(biases_int[neuron]),
                int(all_bit),
            )
            upper = clamp_to_signed_range(
                round_divide_half_away_from_zero(upper_acc, denominator)
                + int(biases_int[neuron]),
                int(all_bit),
            )
            accepted_low = int(target_low[neuron]) - int(budget[neuron])
            accepted_high = int(target_high[neuron]) + int(budget[neuron])
            violation = max(accepted_low - lower, upper - accepted_high, 0)
            scored.append((int(violation), int(neuron)))
        return [neuron for _, neuron in sorted(scored, key=lambda item: (-item[0], item[1]))]

    @staticmethod
    def _analytic_output_margin_result() -> ESBMCResult:
        """Represent the sound analytical fast path without an ESBMC process."""

        return ESBMCResult(
            status="VERIFIED",
            command=(),
            stdout="",
            stderr="",
            return_code=0,
            elapsed_seconds=0.0,
            resource_control={
                "status": "VERIFIED",
                "mode": "analytic_output_margin_fast_path",
                "elapsed_seconds": 0.0,
            },
        )

    def _record_exact_output_margin_result(
        self,
        record: dict[str, Any],
        result: ESBMCResult,
    ) -> None:
        """Finalize an analytical failure using the exact deployed output harness.

        The harness quantizes and clamps the output layer with the shared deployment
        kernel. Its nondeterministic input is constrained to the last hidden layer's
        verified activation box, which already contains the inherited derived budget;
        adding that budget again would be unsoundly pessimistic. A VERIFIED result is
        therefore a sound box-level certificate. A failing hidden-box point is not
        necessarily reachable from a common network input, so every non-VERIFIED
        solver outcome remains MARGIN_INCONCLUSIVE rather than a refutation.
        """

        record["solver_status"] = str(result.status)
        record["resource_control"] = result.resource_control
        record["exact_query_elapsed_seconds"] = float(result.elapsed_seconds)
        record["hidden_box_counterexample"] = (
            [int(value) for value in result.counterexample_inputs]
            if result.counterexample_inputs is not None
            else None
        )
        if result.status == "VERIFIED":
            record["margin_ok"] = True
            record["output_margin"] = "exact_harness_pass"
            record["status"] = "VERIFIED"
            record["composition_path"] = "layer_exact_output"
            self.composition_path = "layer_exact_output"
            return

        record["margin_ok"] = False
        record["output_margin"] = "MARGIN_INCONCLUSIVE"
        record["status"] = "MARGIN_INCONCLUSIVE"
        self.synthesis_final_status = "MARGIN_INCONCLUSIVE"

    @staticmethod
    def _terminal_margin_result(status: str) -> ESBMCResult:
        return ESBMCResult(
            status=status,
            command=(),
            stdout="",
            stderr="",
            return_code=0 if status == "VERIFIED" else 1,
            elapsed_seconds=0.0,
            resource_control={"status": status, "mode": "margin_resolution"},
        )

    def _resolve_output_margin_result(
        self,
        record: dict[str, Any],
        result: ESBMCResult,
        *,
        total_bits: list[int],
        fractional_bits: list[int],
        integer_bits: list[int],
    ) -> ESBMCResult:
        """Resolve the output box query, with at most one sound E2E fallback.

        The E2E harness executes the same deployed integer program from the original
        input box. Therefore VERIFIED establishes the transfer claim directly, without
        relying on the inconclusive Cartesian hidden abstraction. FAILED is considered
        a refutation only when both Python and compiled-C replay confirm the witness.
        """

        self._record_exact_output_margin_result(record, result)
        if result.status == "VERIFIED" or not self.e2e_fallback:
            return result
        if self.e2e_fallback_attempted:
            record["e2e_fallback"] = {
                "attempted": False,
                "reason": "single fallback attempt already consumed",
            }
            return result

        self.e2e_fallback_attempted = True
        verified = self._verify_network_end_to_end(
            total_bits,
            fractional_bits,
            integer_bits,
        )
        fallback_status = str(self.end_to_end_record.get("status", "UNKNOWN"))
        record["e2e_fallback"] = {
            "attempted": True,
            "status": fallback_status,
            "harness": self.end_to_end_record.get("harness"),
            "resource_control": self.end_to_end_record.get("resource_control"),
        }
        if verified:
            record["margin_ok"] = True
            record["output_margin"] = "e2e_fallback_pass"
            record["status"] = "VERIFIED"
            record["composition_path"] = "e2e_fallback"
            self.composition_path = "e2e_fallback"
            self.synthesis_final_status = "VERIFIED"
            return self._terminal_margin_result("VERIFIED")

        replay = self.end_to_end_record.get("counterexample_replay") or {}
        replay_confirmed = bool(replay.get("python_replay_confirmed")) and bool(
            replay.get("so_replay_confirmed")
        )
        if fallback_status == "FAILED" and replay_confirmed:
            record["margin_ok"] = False
            record["output_margin"] = "MARGIN_REFUTED"
            record["status"] = "MARGIN_REFUTED"
            record["reachable_counterexample"] = dict(replay)
            self.synthesis_final_status = "MARGIN_REFUTED"
            return self._terminal_margin_result("MARGIN_REFUTED")

        record["margin_ok"] = False
        record["output_margin"] = "MARGIN_INCONCLUSIVE"
        record["status"] = "MARGIN_INCONCLUSIVE"
        self.synthesis_final_status = "MARGIN_INCONCLUSIVE"
        return result

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

        is_last_hidden = (
            cur_layer.layer_index == len(self.dense_layers)
            and cur_layer.layer_index < len(self.dense_layers) + 1
        )
        if self.error_budget_mode == "derived" and is_last_hidden:
            _, _, _, target_valid = self._candidate_contract_target_bounds_int(
                cur_layer,
                in_layer,
                qu_w_int,
                frac_bit,
                record=True,
            )
            if not target_valid:
                target_status = str(
                    getattr(
                        self,
                        "_last_contract_target_status",
                        "PREIMAGE_DEFLATION_EMPTY",
                    )
                )
                self.synthesis_final_status = target_status
                reason = (
                    "derived contracts require a MILP property preimage; "
                    "forward DeepPoly bounds cannot be deflated"
                    if target_status == "PREIMAGE_UNAVAILABLE"
                    else "derived budget collapses the property preimage"
                )
                return ESBMCResult(
                    status=target_status,
                    command=(),
                    stdout="",
                    stderr="",
                    return_code=1,
                    elapsed_seconds=0.0,
                    timeout_seconds=int(self.config.esbmc.timeout_seconds),
                    memlimit=str(self.config.esbmc.memlimit),
                    resource_control={
                        "status": target_status,
                        "elapsed_seconds": 0.0,
                        "reason": reason,
                        "preimage_source": str(
                            getattr(
                                cur_layer,
                                "preimage_source",
                                "deeppoly_forward_FALLBACK",
                            )
                        ),
                    },
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
        if (
            result.status == "FAILED"
            and cur_layer.layer_index <= len(self.dense_layers)
            and getattr(self, "cegar_max_rounds", 0) > 0
            and self.margin_cuts
            and self.error_budget_mode == "derived"
            and int(getattr(in_layer, "layer_index", 0)) > 0
        ):
            violation_order = self._hidden_contract_violation_order(
                cur_layer=cur_layer,
                in_layer=in_layer,
                weights_int=qu_w_int,
                biases_int=qu_b_int,
                frac_bit=frac_bit,
                all_bit=all_bit,
                start_neuron=0,
                end_neuron=cur_layer.layer_size,
            )
            accumulated_cuts: list[dict[str, Any]] = []
            for round_index, neuron in enumerate(
                violation_order[: int(getattr(self, "cegar_max_rounds", 0))], start=1
            ):
                new_cuts = self._hidden_contract_cut_bounds(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    weights_int=qu_w_int,
                    frac_bit=frac_bit,
                    all_bit=all_bit,
                    start_neuron=0,
                    end_neuron=cur_layer.layer_size,
                    maximum_cuts=1,
                    neuron_indices=[neuron],
                )
                if not new_cuts:
                    break
                accumulated_cuts.extend(new_cuts)
                refined_source = self.generate_esbmc_verification_code(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    qu_w_int=qu_w_int,
                    qu_b_int=qu_b_int,
                    frac_bit=frac_bit,
                    all_bit=all_bit,
                    layer_index=layer_index,
                    contract_cuts=accumulated_cuts,
                )
                refined_file = layers_dir / (
                    f"layer_{layer_index}_Q{all_bit}_F{frac_bit}_cegar_{round_index}.c"
                )
                refined_file.write_text(refined_source, encoding="utf-8")
                result = self._run_esbmc_file(
                    refined_file,
                    extract_counterexample=True,
                )
                self._stats["esbmc_calls"] += 1.0
                refined_record = self._esbmc_call_record(
                    result=result,
                    layer_index=layer_index,
                    block_index=None,
                    start_neuron=None,
                    end_neuron=None,
                    all_bit=all_bit,
                    frac_bit=frac_bit,
                    harness=refined_file,
                    property_type="preimage",
                    mode="full_layer_relational_cegar",
                    input_dim=in_layer.layer_size,
                    output_neurons=cur_layer.layer_size,
                )
                refined_record["cegar_round"] = int(round_index)
                refined_record["contract_cuts"] = [
                    dict(cut) for cut in accumulated_cuts
                ]
                refined_record["assumption_box_cardinality"] = cardinality
                self.esbmc_call_records.append(refined_record)
                if result.status != "FAILED":
                    break
        if (
            result.status == "FAILED"
            and cur_layer.layer_index <= len(self.dense_layers)
            and int(getattr(in_layer, "layer_index", 0)) > 0
            and self.error_budget_mode == "derived"
        ):
            # A hidden-box endpoint is not necessarily generated by one common
            # network input. Without a separately proved reachable witness it is
            # an abstraction limitation, not a refutation of the candidate format.
            self.esbmc_call_records[-1]["raw_status"] = "FAILED"
            self.esbmc_call_records[-1]["status"] = "LAYER_INCONCLUSIVE"
            result = replace(result, status="LAYER_INCONCLUSIVE")
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
            query_results: list[tuple[Path, ESBMCResult, int, list[dict[str, Any]]]] = [
                (archived_file, block_result, 0, [])
            ]
            accumulated_cuts: list[dict[str, Any]] = []
            if (
                block_result.status == "FAILED"
                and getattr(self, "cegar_max_rounds", 0) > 0
                and self.margin_cuts
                and self.error_budget_mode == "derived"
                and int(getattr(in_layer, "layer_index", 0)) > 0
            ):
                violation_order = self._hidden_contract_violation_order(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    weights_int=qu_w_int,
                    biases_int=qu_b_int,
                    frac_bit=frac_bit,
                    all_bit=all_bit,
                    start_neuron=start_neuron,
                    end_neuron=end_neuron,
                )
                for round_index, neuron in enumerate(
                    violation_order[: int(getattr(self, "cegar_max_rounds", 0))],
                    start=1,
                ):
                    new_cuts = self._hidden_contract_cut_bounds(
                        cur_layer=cur_layer,
                        in_layer=in_layer,
                        weights_int=qu_w_int,
                        frac_bit=frac_bit,
                        all_bit=all_bit,
                        start_neuron=start_neuron,
                        end_neuron=end_neuron,
                        maximum_cuts=1,
                        neuron_indices=[neuron],
                    )
                    if not new_cuts:
                        break
                    accumulated_cuts.extend(new_cuts)
                    refined_source = self.generate_esbmc_hidden_block_verification_code(
                        cur_layer=cur_layer,
                        in_layer=in_layer,
                        qu_w_int=qu_w_int,
                        qu_b_int=qu_b_int,
                        frac_bit=frac_bit,
                        all_bit=all_bit,
                        start_neuron=start_neuron,
                        end_neuron=end_neuron,
                        contract_cuts=accumulated_cuts,
                    )
                    refined_file = layers_dir / harness_name.replace(
                        ".c", f"_cegar_{round_index}.c"
                    )
                    refined_file.write_text(refined_source, encoding="utf-8")
                    block_result = self._run_esbmc_file(
                        refined_file,
                        extract_counterexample=True,
                    )
                    query_results.append(
                        (refined_file, block_result, round_index, list(accumulated_cuts))
                    )
                    if block_result.status != "FAILED":
                        break

            query_records: list[dict[str, Any]] = []
            for query_file, query_result, cegar_round, query_cuts in query_results:
                self._stats["esbmc_calls"] += 1.0
                self._stats["esbmc_block_calls"] += 1.0
                elapsed_total += float(query_result.elapsed_seconds)
                aggregate_return_code = max(
                    aggregate_return_code, int(query_result.return_code)
                )
                stdout_parts.append(query_result.stdout)
                stderr_parts.append(query_result.stderr)
                query_status = self._record_block_status(query_result.status)
                query_record = self._esbmc_call_record(
                    result=query_result,
                    layer_index=layer_index,
                    block_index=block_index,
                    start_neuron=start_neuron,
                    end_neuron=end_neuron,
                    all_bit=all_bit,
                    frac_bit=frac_bit,
                    harness=query_file,
                    property_type="preimage",
                    mode="blockwise" if cegar_round == 0 else "blockwise_relational_cegar",
                    input_dim=in_layer.layer_size,
                    output_neurons=end_neuron - start_neuron,
                    status=query_status,
                )
                query_record["cegar_round"] = int(cegar_round)
                query_record["contract_cuts"] = [dict(cut) for cut in query_cuts]
                query_record["intermediate_query"] = query_file != query_results[-1][0]
                self.esbmc_call_records.append(query_record)
                query_records.append(query_record)

            record = query_records[-1]
            record_status = str(record["status"])
            if (
                record_status == "FAILED"
                and int(getattr(in_layer, "layer_index", 0)) > 0
                and self.error_budget_mode == "derived"
            ):
                # The remaining witness is a point in a relational
                # over-approximation, not a confirmed deployed-network input.
                record_status = "LAYER_INCONCLUSIVE"
                record["raw_status"] = "FAILED"
                record["status"] = record_status
            record["cegar"] = {
                "attempted": len(query_results) > 1,
                "rounds": len(query_results) - 1,
                "cuts": [dict(cut) for cut in accumulated_cuts],
                "initial_status": str(query_results[0][1].status),
                "final_status": record_status,
            }
            records.append(record)
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
        contract_cuts: list[dict[str, Any]] | None = None,
    ) -> str:
        del layer_index
        self._require_validated_hidden_row_cuts(qu_w_int, contract_cuts)
        scale = 1 << int(frac_bit)
        weights_c_int = self.numpy_to_c_int_array(qu_w_int)
        biases_c_int = self.numpy_to_c_int_array(qu_b_int)

        tolerance_int, input_lo_int, input_hi_int, frac_in = self._candidate_error_budget_int(
            cur_layer,
            in_layer,
            qu_w_int,
            frac_bit,
        )
        if self.error_budget_mode == "derived" and cur_layer.layer_index <= len(self.dense_layers):
            pre_lo_int, pre_hi_int, tolerance_int, target_valid = (
                self._candidate_contract_target_bounds_int(
                    cur_layer,
                    in_layer,
                    qu_w_int,
                    frac_bit,
                )
            )
            if not target_valid:
                raise RuntimeError("Cannot generate a harness for an empty deflated preimage.")
        else:
            pre_lo_int, pre_hi_int = self._layer_preimage_bounds_int(cur_layer, scale)
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
            margin_cuts = self._margin_cut_bounds(
                cur_layer,
                in_layer,
                frac_bit,
                all_bit,
            )
            if self.output_margin_records:
                self.output_margin_records[-1]["margin_cuts"] = [
                    dict(cut) for cut in margin_cuts
                ]
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
                margin_cut_directions_c_int=(
                    self.numpy_to_c_int_array(
                        np.asarray(
                            [cut["direction_int"] for cut in margin_cuts],
                            dtype=object,
                        )
                    )
                    if margin_cuts
                    else None
                ),
                margin_cut_low_c_int=(
                    self.numpy_to_c_int_array(
                        np.asarray(
                            [cut["cut_low_int"] for cut in margin_cuts],
                            dtype=object,
                        )
                    )
                    if margin_cuts
                    else None
                ),
                margin_cut_high_c_int=(
                    self.numpy_to_c_int_array(
                        np.asarray(
                            [cut["cut_high_int"] for cut in margin_cuts],
                            dtype=object,
                        )
                    )
                    if margin_cuts
                    else None
                ),
                margin_cut_scale=(
                    int(margin_cuts[0]["direction_scale"])
                    if margin_cuts
                    else None
                ),
                margin_cut_count=len(margin_cuts),
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
            contract_cut_directions_c_int=(
                self.numpy_to_c_int_array(
                    np.asarray([cut["direction_int"] for cut in contract_cuts], dtype=object)
                )
                if contract_cuts
                else None
            ),
            contract_cut_low_c_int=(
                self.numpy_to_c_int_array(
                    np.asarray([cut["cut_low_int"] for cut in contract_cuts], dtype=object)
                )
                if contract_cuts
                else None
            ),
            contract_cut_high_c_int=(
                self.numpy_to_c_int_array(
                    np.asarray([cut["cut_high_int"] for cut in contract_cuts], dtype=object)
                )
                if contract_cuts
                else None
            ),
            contract_cut_output_indices_c_int=(
                self.numpy_to_c_int_array(
                    np.asarray([cut["neuron_index"] for cut in contract_cuts], dtype=object)
                )
                if contract_cuts
                else None
            ),
            contract_cut_count=len(contract_cuts or []),
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
        contract_cuts: list[dict[str, Any]] | None = None,
    ) -> str:
        self._require_validated_hidden_row_cuts(qu_w_int, contract_cuts)
        scale = 1 << int(frac_bit)
        tolerance_int, input_lo_int, input_hi_int, frac_in = self._candidate_error_budget_int(
            cur_layer,
            in_layer,
            qu_w_int,
            frac_bit,
        )
        if self.error_budget_mode == "derived":
            pre_lo_int, pre_hi_int, tolerance_int, target_valid = (
                self._candidate_contract_target_bounds_int(
                    cur_layer,
                    in_layer,
                    qu_w_int,
                    frac_bit,
                )
            )
            if not target_valid:
                raise RuntimeError("Cannot generate a block for an empty deflated preimage.")
        else:
            pre_lo_int, pre_hi_int = self._layer_preimage_bounds_int(cur_layer, scale)
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
            contract_cut_directions_c_int=(
                self.numpy_to_c_int_array(
                    np.asarray([cut["direction_int"] for cut in contract_cuts], dtype=object)
                )
                if contract_cuts
                else None
            ),
            contract_cut_low_c_int=(
                self.numpy_to_c_int_array(
                    np.asarray([cut["cut_low_int"] for cut in contract_cuts], dtype=object)
                )
                if contract_cuts
                else None
            ),
            contract_cut_high_c_int=(
                self.numpy_to_c_int_array(
                    np.asarray([cut["cut_high_int"] for cut in contract_cuts], dtype=object)
                )
                if contract_cuts
                else None
            ),
            contract_cut_output_indices_c_int=(
                self.numpy_to_c_int_array(
                    np.asarray(
                        [
                            int(cut["neuron_index"]) - int(start_neuron)
                            for cut in contract_cuts
                        ],
                        dtype=object,
                    )
                )
                if contract_cuts
                else None
            ),
            contract_cut_count=len(contract_cuts or []),
        )

    @staticmethod
    def _require_validated_hidden_row_cuts(
        weights_int: np.ndarray,
        contract_cuts: list[dict[str, Any]] | None,
    ) -> None:
        """Reject unproved or miswired cuts before rendering an assume harness."""

        rows = np.asarray(weights_int, dtype=object)
        for cut in contract_cuts or []:
            if str(cut.get("formal_validation_status")) != "VERIFIED":
                raise ValueError("A hidden relational cut must be formally validated.")
            neuron = int(cut["neuron_index"])
            if neuron < 0 or neuron >= rows.shape[0]:
                raise ValueError("Hidden relational cut neuron is outside the layer.")
            direction = tuple(int(value) for value in cut["direction_int"])
            expected = tuple(int(value) for value in rows[neuron].reshape(-1))
            if direction != expected:
                raise ValueError(
                    "Hidden relational cut direction must equal its quantized affine row."
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

    def source_region_summary(self) -> dict[str, Any]:
        record = dict(self.source_region_record)
        record["esbmc_attempted"] = bool(self.esbmc_call_records)
        return record

    def blockwise_verification_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.esbmc_block_records]
        verified_blocks = sum(1 for record in records if record.get("status") == "VERIFIED")
        timeout_blocks = sum(1 for record in records if record.get("status") == "TIMEOUT")
        failed_blocks = sum(1 for record in records if record.get("status") == "FAILED")
        memout_blocks = sum(1 for record in records if record.get("status") == "MEMOUT")
        unknown_blocks = sum(
            1
            for record in records
            if record.get("status") in {"UNKNOWN", "LAYER_INCONCLUSIVE"}
        )
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
        latest_status = str(records[-1].get("status", "SKIPPED")) if records else "SKIPPED"
        recovery: dict[str, Any] = {
            "analytic_worst_residual_margin_int": None,
            "step_a_alone_status": "NOT_RUN",
            "step_a_residual_recovery_int": None,
            "step_b_status": "NOT_RUN",
            "step_b_additional_recovery_int": None,
            "combined_recovery_lower_bound_int": None,
            "note": "A scalar recovery is reported only when the corresponding exact query proves a strict integer margin.",
        }
        if records:
            latest = records[-1]
            residuals = [
                int(item["residual_margin_int"])
                for item in latest.get("class_margins", [])
                if item.get("residual_margin_int") is not None
            ]
            worst = min(residuals) if residuals else None
            recovery["analytic_worst_residual_margin_int"] = worst
            cuts = latest.get("margin_cuts") or []
            solver_status = str(latest.get("solver_status", "NOT_RUN"))
            if cuts:
                recovery["step_a_alone_status"] = "NOT_RUN_WITHOUT_CUTS"
                recovery["step_b_status"] = solver_status
            else:
                recovery["step_a_alone_status"] = solver_status
                if solver_status == "VERIFIED" and worst is not None:
                    recovery["step_a_residual_recovery_int"] = 1 - worst
            if solver_status == "VERIFIED" and worst is not None:
                recovery["combined_recovery_lower_bound_int"] = 1 - worst
            fallback = latest.get("e2e_fallback") or {}
            if fallback:
                recovery["e2e_fallback_status"] = fallback.get("status", "UNKNOWN")
                if fallback.get("status") == "VERIFIED" and worst is not None:
                    recovery["reachable_recovery_lower_bound_int"] = 1 - worst
        return {
            "enabled": (
                self.harness_scope == "layer"
                and self.error_budget_mode == "derived"
            ),
            "all_ok": final_ok,
            "status": (
                "VERIFIED"
                if final_ok
                else "MARGIN_INCONCLUSIVE"
                if latest_status == "MARGIN_INCONCLUSIVE"
                else "PENDING_EXACT_QUERY"
                if latest_status == "PENDING_EXACT_QUERY"
                else "SKIPPED"
            ),
            "checks": records,
            "recovery_diagnostics": recovery,
        }

    def margin_cut_summary(self) -> dict[str, Any]:
        records = [dict(record) for record in self.margin_cut_records]
        validated = [
            record
            for record in records
            if record.get("formal_validation_status") == "VERIFIED"
        ]
        return {
            "enabled": bool(self.margin_cuts and self.error_budget_mode == "derived"),
            "status": (
                "VERIFIED"
                if validated
                else str(records[-1].get("status"))
                if records
                else "NOT_RUN"
            ),
            "cuts_total": int(len(records)),
            "cuts_formally_validated": int(len(validated)),
            "cuts": records,
        }

    def cegar_summary(self) -> dict[str, Any]:
        calls = [
            dict(record)
            for record in self.esbmc_call_records
            if record.get("mode")
            in {"blockwise_relational_cegar", "full_layer_relational_cegar"}
        ]
        statuses = [str(record.get("status", "UNKNOWN")) for record in calls]
        cut_validations = [
            dict(record)
            for record in self.esbmc_call_records
            if record.get("property_type") == "relational_cut_validation"
        ]
        return {
            "enabled": bool(
                self.margin_cuts
                and self.error_budget_mode == "derived"
                and int(getattr(self, "cegar_max_rounds", 0)) > 0
            ),
            "max_rounds": int(getattr(self, "cegar_max_rounds", 0)),
            "rounds_run": int(len(calls)),
            "status": (
                "VERIFIED"
                if "VERIFIED" in statuses
                else "MEMOUT"
                if "MEMOUT" in statuses
                else "TIMEOUT"
                if "TIMEOUT" in statuses
                else "LAYER_INCONCLUSIVE"
                if calls
                else "NOT_RUN"
            ),
            "cuts": [dict(record) for record in self.hidden_contract_cut_records],
            "queries": calls,
            "cut_validation_queries": cut_validations,
            "cuts_require_exact_prefix_validation": True,
            "witness_policy": "unconfirmed_relational_witnesses_are_inconclusive",
        }

    def preimage_provenance_summary(self) -> dict[str, Any]:
        layers = [
            {
                "layer_index": int(layer.layer_index - 1),
                "network_layer_index": int(layer.layer_index),
                "preimage_source": str(
                    getattr(layer, "preimage_source", "deeppoly_forward_FALLBACK")
                ),
            }
            for layer in self.dense_layers
        ]
        all_property_preimages_available = bool(
            layers
            and all(
                layer["preimage_source"]
                in {
                    "milp_preimage",
                    "milp_preimage_no_violation_to_cap",
                    "quantized_milp_preimage",
                }
                for layer in layers
            )
        )
        return {
            "layers": layers,
            "attempts": [
                dict(record)
                for record in getattr(self, "preimage_attempt_records", [])
            ],
            "deflation": [
                dict(record)
                for record in getattr(self, "preimage_deflation_records", [])
            ],
            "all_milp_preimage": bool(
                layers
                and all(
                    str(layer["preimage_source"]).startswith("milp_preimage")
                    for layer in layers
                )
            ),
            "all_property_preimages_available": all_property_preimages_available,
            "status": "AVAILABLE" if all_property_preimages_available else "UNAVAILABLE",
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
            "LAYER_INCONCLUSIVE",
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
        query_cpu_seconds = [
            float(record["cpu_time_seconds"])
            for record in executed_records
            if record.get("cpu_time_seconds") is not None
        ]
        query_cpu_utilization = [
            float(record["average_cpu_utilization_percent"])
            for record in executed_records
            if record.get("average_cpu_utilization_percent") is not None
        ]
        total_calls = int(sum(counts.values()))
        total_non_skipped = int(total_calls - counts["skipped"])
        unknown_count = int(counts["unknown"] + counts["layer_inconclusive"])
        verification_denominator = int(
            counts["verified"]
            + counts["failed"]
            + counts["timeout"]
            + counts["memout"]
            + unknown_count
            + counts["vacuous"]
        )
        return {
            "records": records,
            "verified_count": counts["verified"],
            "failed_count": counts["failed"],
            "timeout_count": counts["timeout"],
            "memout_count": counts["memout"],
            "unknown_count": unknown_count,
            "layer_inconclusive_count": counts["layer_inconclusive"],
            "skipped_count": counts["skipped"],
            "vacuity_sentinel_count": counts["sentinel_expected_failure"],
            "vacuous_count": counts["vacuous"],
            "total_count": total_calls,
            "executed_count": total_non_skipped,
            "timeout_rate": float(counts["timeout"] / total_calls) if total_calls else 0.0,
            "memout_rate": float(counts["memout"] / total_calls) if total_calls else 0.0,
            "unknown_rate": float(unknown_count / total_calls) if total_calls else 0.0,
            "verification_rate_percent": float(
                100.0 * counts["verified"] / verification_denominator
            ) if verification_denominator else 0.0,
            "verification_denominator": verification_denominator,
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
            "cpu_measurement": (
                "linux_procfs_process_tree_cpu_time"
                if query_cpu_seconds
                else "unavailable"
            ),
            "cpu_queries_measured": int(len(query_cpu_seconds)),
            "total_cpu_time_seconds": float(sum(query_cpu_seconds)),
            "average_cpu_utilization_percent": float(
                100.0 * sum(query_cpu_seconds) / sum(query_times)
            ) if query_cpu_seconds and sum(query_times) > 0 else 0.0,
            "mean_query_cpu_utilization_percent": float(
                sum(query_cpu_utilization) / len(query_cpu_utilization)
            ) if query_cpu_utilization else 0.0,
            "max_query_cpu_utilization_percent": float(
                max(query_cpu_utilization, default=0.0)
            ),
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
