from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from symbolic_pp.DeepPoly_quadapter import DP_DNN_network
from synthesis.preimage_cache import load_preimage_cache, save_preimage_cache
from utils.fixed_point import int_get_min_max, quantize_int
from utils.logging_utils import get_logger
from verification.c_templates import (
    render_hidden_affine_bounds_program,
    render_no_saturation_program,
    render_output_target_program,
    render_output_valid_set_program,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner, ESBMCResult
from verification.properties import ClassificationProperty

LOGGER = get_logger(__name__)

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - depends on host licensing/install.
    gp = None  # type: ignore[assignment]
    GRB = None  # type: ignore[assignment]


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise RuntimeError(
            "gurobipy is required for this pipeline mode. "
            "Use --no-gurobi with --verify-mode esbmc and a preimage cache, "
            "or install/configure Gurobi."
        )


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

    @classmethod
    def from_namespace(cls, args: Any) -> "QuadapterConfig":
        return cls(
            bit_lb=int(args.bit_lb),
            bit_ub=int(args.bit_ub),
            preimg_mode=str(args.preimg_mode),
            verify_mode=str(args.verify_mode),
            sample_id=int(args.sample_id),
            eps=float(args.eps),
            output_dir=Path(getattr(args, "output_dir", getattr(args, "outputPath", "output"))),
            if_relax=bool(int(getattr(args, "if_relax", getattr(args, "ifRelax", 0)))),
            no_gurobi=bool(getattr(args, "no_gurobi", False)),
            save_preimage_cache=bool(getattr(args, "save_preimage_cache", False)),
            preimage_cache_dir=(
                Path(getattr(args, "preimage_cache_dir"))
                if getattr(args, "preimage_cache_dir", None) is not None
                else None
            ),
            preimage_cache_key=getattr(args, "preimage_cache_key", None),
        )


@dataclass(frozen=True)
class SynthesisResult:
    """Stable result object for the robustness pipeline."""

    success: bool
    total_bits: list[int]
    fractional_bits: list[int]
    integer_bits: list[int]
    stats: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total_bits": self.total_bits,
            "fractional_bits": self.fractional_bits,
            "integer_bits": self.integer_bits,
            "stats": self.stats,
        }


class LayerEncoding:
    """Per-layer state used by the Gurobi/DeepPoly synthesis algorithm."""

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

        _require_gurobi()
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
        self.x_low_real = x_low_real
        self.x_high_real = x_high_real
        self.sample_id = self.config.sample_id
        self.eps = self.config.eps
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ifRelax = int(self.config.if_relax)
        self.scaleValueSet: list[float] = []

        if self.config.no_gurobi and self.verify_mode != "esbmc":
            raise ValueError("--no-gurobi requires --verify-mode esbmc because MILP forward verification uses Gurobi.")

        if self.config.no_gurobi:
            self.gp_model = None
        else:
            _require_gurobi()
            self.gp_model = gp.Model("gp_encoding")
            self.gp_model.Params.IntFeasTol = 1e-9
            self.gp_model.Params.FeasibilityTol = self.tole
            self.gp_model.setParam(GRB.Param.Threads, 30)
            self.gp_model.setParam(GRB.Param.OutputFlag, 0)

        self._stats = {
            "encoding_time": 0.0,
            "solving_time": 0.0,
            "backward_time": 0.0,
            "forward_time": 0.0,
            "total_time": 0.0,
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

    def verified_quant(self, lb: np.ndarray, ub: np.ndarray) -> tuple[bool, Any, Any, Any]:
        result = self.run(lb, ub)
        if not result.success:
            return False, None, None, None
        return True, result.total_bits, result.fractional_bits, result.integer_bits

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
        if self.config.no_gurobi:
            self.load_cached_preimage()
        else:
            self.backward_preimage_computation()
            if self.config.save_preimage_cache:
                self.save_cached_preimage()
        backward_end_time = time.time()

        if self.verify_mode == "esbmc":
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
        LOGGER.info("Saved Gurobi preimage cache to %s", cache_path)
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
        LOGGER.info("Loaded Gurobi preimage cache %s (%s)", self._preimage_cache_key(), metadata.get("format"))

    def backward_preimage_computation(self) -> None:
        if self.gp_model is None:
            raise RuntimeError("Cannot compute a preimage without Gurobi. Use load_cached_preimage() instead.")
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

            symbolic_lb_expression = np.dot(in_lb_algebra[:-1], self.input_gp_vars) + relaxed_lb_bias
            symbolic_ub_expression = np.dot(in_ub_algebra[:-1], self.input_gp_vars) + relaxed_ub_bias

            model_cstr_ll.append(self.gp_model.addConstr(in_layer.gp_vars_before[in_index] <= symbolic_ub_expression))
            model_cstr_ll.append(self.gp_model.addConstr(in_layer.gp_vars_before[in_index] >= symbolic_lb_expression))
            model_cstr_ll.append(
                self.gp_model.addGenConstrMax(in_layer.gp_vars_after[in_index], [in_layer.gp_vars_before[in_index], 0])
            )

        self.gp_model.update()

        for out_index in range(cur_layer.layer_size):
            accumulation = np.dot(w[out_index], in_layer.gp_vars_after) + b[out_index]
            model_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_before[out_index] == accumulation))

        enc_finish_time = time.time()
        self._stats["encoding_time"] += enc_finish_time - enc_start_time

        if cur_layer.layer_index == (len(self.dense_layers) + 1):
            other_vars = [
                cur_layer.gp_vars_before[i] for i in range(cur_layer.layer_size) if i != int(self.targetCls)
            ]
            other_maximal = self.gp_model.addVar(lb=-1000, vtype=GRB.CONTINUOUS)
            prop_cstr_ll.append(self.gp_model.addGenConstrMax(other_maximal, other_vars))
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
            scaleValue = float(relaxScale.X)
            for in_index in range(in_layer.layer_size):
                alpha = in_layer.alpha[in_index].X
                beta = in_layer.beta[in_index].X
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
                in_layer.relaxed_lb_expression[in_index] = np.dot(in_lb_algebra[:-1], self.input_gp_vars) + relaxed_lb_bias
                in_layer.relaxed_ub_expression[in_index] = np.dot(in_ub_algebra[:-1], self.input_gp_vars) + relaxed_ub_bias
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
                model_cstr_ll.append(
                    self.gp_model.addGenConstrMin(
                        in_layer.alpha_after[in_index],
                        [in_layer.alpha_before[in_index], in_layer.lb[in_index]],
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

        scaleValue = float(relaxScale.X)
        for in_index in range(in_layer.layer_size):
            alpha_after = in_layer.alpha_after[in_index].X
            beta_after = in_layer.beta_after[in_index].X

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
            in_layer.relaxed_lb_expression[in_index] = np.dot(in_lb_algebra[:-1], self.input_gp_vars) + relaxed_lb_bias
            in_layer.relaxed_ub_expression[in_index] = np.dot(in_ub_algebra[:-1], self.input_gp_vars) + relaxed_ub_bias
            if in_layer.ub[in_index] <= 0:
                in_layer.relaxed_ub_expression[in_index] = 0

        self.gp_model.remove(prop_cstr_ll)
        self.gp_model.remove(model_cstr_ll)
        self.gp_model.remove(relaxScale_LL)
        self.gp_model.update()
        return scaleValue

    def forward_quantization_with_esbmc(self) -> tuple[bool, Any, Any, Any]:
        qu_list: list[int] = []
        qu_frac_list: list[int] = []
        qu_int_list: list[int] = []

        non_input_layers = self.dense_layers.copy()
        non_input_layers.append(self.output_layer)
        in_layer_index = -1

        for cur_layer in non_input_layers:
            in_layer_index += 1
            in_layer = self.input_layer if cur_layer.layer_index == 1 else self.dense_layers[cur_layer.layer_index - 2]
            w = cur_layer.layer_paras[0]
            b = cur_layer.layer_paras[1]
            if_found = False

            for frac_bit in range(self.bit_lb, self.bit_ub + 1):
                if if_found:
                    break

                int_bit = int(cur_layer.int_bit)
                all_bit = frac_bit + int_bit
                qu_w_int = quantize_int(w, all_bit, frac_bit)
                qu_b_int = quantize_int(b, all_bit, frac_bit)

                esbmc_result = self.verify_layer_with_esbmc(
                    cur_layer=cur_layer,
                    in_layer=in_layer,
                    qu_w_int=np.asarray(qu_w_int),
                    qu_b_int=np.asarray(qu_b_int),
                    frac_bit=frac_bit,
                    all_bit=all_bit,
                    layer_index=in_layer_index,
                )

                if esbmc_result.status == "VERIFIED":
                    cur_layer.frac_bit = frac_bit
                    qu_frac_list.append(frac_bit)
                    qu_int_list.append(_export_integer_bits(int_bit))
                    qu_list.append(all_bit)
                    if_found = True
                    self.update_quantized_weights_affine(in_layer, cur_layer, all_bit, frac_bit, frac_bit, in_layer_index)

            if not if_found:
                return False, None, None, None

        return True, qu_list, qu_frac_list, qu_int_list

    def verify_exported_quantization_with_esbmc(
        self,
        total_bits: list[int],
        fractional_bits: list[int],
        integer_bits: list[int],
        *,
        formal_saturation_check: bool = False,
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
                        "no_saturation_status": "NOT_RUN",
                        "failure_type": "invalid_qif",
                    }
                )
                return False, records

            qu_w_int = quantize_int(cur_layer.layer_paras[0], q_bits, f_bits)
            qu_b_int = quantize_int(cur_layer.layer_paras[1], q_bits, f_bits)
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
                "no_saturation_status": "DISABLED" if not formal_saturation_check else "PENDING",
            }
            if contract_result.status != "VERIFIED":
                record["no_saturation_status"] = "NOT_RUN"
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
                record["no_saturation_status"] = no_saturation_result.status
                if no_saturation_result.status != "VERIFIED":
                    record["status"] = "FAILED"
                    record["failure_type"] = "formal_saturation_possible"
                    records.append(record)
                    return False, records

            record["status"] = "VERIFIED"
            records.append(record)
            self.update_quantized_weights_affine(in_layer, cur_layer, q_bits, f_bits, f_bits, layer_index)

        return True, records

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
        c_source = self.generate_esbmc_verification_code(
            cur_layer=cur_layer,
            in_layer=in_layer,
            qu_w_int=qu_w_int,
            qu_b_int=qu_b_int,
            frac_bit=frac_bit,
            layer_index=layer_index,
        )
        layers_dir = self.output_dir / "layers"
        layers_dir.mkdir(parents=True, exist_ok=True)
        archived_file = layers_dir / f"layer_{layer_index}_Q{all_bit}_F{frac_bit}.c"
        archived_file.write_text(c_source, encoding="utf-8")

        temp_file = Path(f"esbmc_verify_layer_{layer_index}_Q{all_bit}_F{frac_bit}.c")
        temp_file.write_text(c_source, encoding="utf-8")
        try:
            result = self.esbmc_runner.run_file(temp_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()
        LOGGER.info("ESBMC layer=%s bits(Q=%s,F=%s) status=%s", cur_layer.layer_index, all_bit, frac_bit, result.status)
        return result

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

        temp_file = Path(f"esbmc_verify_layer_{layer_index}_Q{all_bit}_F{frac_bit}_no_saturation.c")
        temp_file.write_text(c_source, encoding="utf-8")
        try:
            result = self.esbmc_runner.run_file(temp_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()
        LOGGER.info(
            "ESBMC no-saturation layer=%s bits(Q=%s,F=%s) status=%s",
            cur_layer.layer_index,
            all_bit,
            frac_bit,
            result.status,
        )
        return result

    def generate_esbmc_verification_code(
        self,
        cur_layer: LayerEncoding,
        in_layer: LayerEncoding,
        qu_w_int: np.ndarray,
        qu_b_int: np.ndarray,
        frac_bit: int,
        layer_index: int,
    ) -> str:
        del layer_index
        scale = 1 << int(frac_bit)
        weights_c_int = self.numpy_to_c_int_array(qu_w_int)
        biases_c_int = self.numpy_to_c_int_array(qu_b_int)

        pre_lo = np.array(cur_layer.relaxed_lb if cur_layer.relaxed_lb is not None else cur_layer.lb, dtype=np.float64)
        pre_hi = np.array(cur_layer.relaxed_ub if cur_layer.relaxed_ub is not None else cur_layer.ub, dtype=np.float64)
        pre_lo_int = np.floor(pre_lo * scale).astype(np.int64)
        pre_hi_int = np.ceil(pre_hi * scale).astype(np.int64)

        if cur_layer.layer_index == 1:
            x_lo = np.array(self.x_low_real, dtype=np.float64)
            x_hi = np.array(self.x_high_real, dtype=np.float64)
        else:
            x_lo = np.array(in_layer.clipped_lb, dtype=np.float64)
            x_hi = np.array(in_layer.clipped_ub, dtype=np.float64)
        input_lo_int = np.floor(x_lo * scale).astype(np.int64)
        input_hi_int = np.ceil(x_hi * scale).astype(np.int64)

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

        if cur_layer.layer_index == 1:
            x_lo = np.array(self.x_low_real, dtype=np.float64)
            x_hi = np.array(self.x_high_real, dtype=np.float64)
        else:
            x_lo = np.array(in_layer.clipped_lb, dtype=np.float64)
            x_hi = np.array(in_layer.clipped_ub, dtype=np.float64)
        input_lo_int = np.floor(x_lo * scale).astype(np.int64)
        input_hi_int = np.ceil(x_hi * scale).astype(np.int64)

        return render_no_saturation_program(
            output_size=cur_layer.layer_size,
            input_size=in_layer.layer_size,
            weights_c_int=weights_c_int,
            biases_c_int=biases_c_int,
            input_bounds_low_c_int=self.numpy_to_c_int_array(input_lo_int),
            input_bounds_high_c_int=self.numpy_to_c_int_array(input_hi_int),
            scale_factor=scale,
            total_bits=all_bit,
        )

    def numpy_to_c_int_array(self, np_array: np.ndarray) -> str:
        if np_array.ndim == 1:
            return "{" + ", ".join(str(int(x)) for x in np_array) + "}"
        rows = []
        for row in np_array:
            rows.append("{" + ", ".join(str(int(x)) for x in row) + "}")
        return "{" + ", ".join(rows) + "}"

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
            raise RuntimeError("--verify-mode milp requires Gurobi; use --verify-mode esbmc with --no-gurobi.")
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

                    quantized_lb_expression = np.dot(cur_neuron_concrete_algebra_lower[:-1], self.input_gp_vars) + cur_neuron_concrete_algebra_lower[-1]
                    quantized_ub_expression = np.dot(cur_neuron_concrete_algebra_upper[:-1], self.input_gp_vars) + cur_neuron_concrete_algebra_upper[-1]

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
