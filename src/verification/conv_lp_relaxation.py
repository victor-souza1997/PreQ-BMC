"""Sparse relational LP relaxation for convolution-native fixed-point networks.

This module is a bound synthesizer, not a proof checker.  It constructs a
continuous over-approximation of the deployed integer network.  Every rounded
affine result is enclosed by a half-unit error interval, and every clamp/ReLU
is replaced by the convex hull of its scalar graph over a sound interval.
Positive margin bounds are therefore candidates for a proof-carrying
certificate; they must not be reported as ESBMC results by themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import coo_matrix

from backends.conv_fixed_point import (
    ConvFixedPointNetwork,
    QuantizedConv2D,
    QuantizedDense,
)
from utils.fixed_point import round_divide_half_away_from_zero
from verification.conv_contracts import (
    IntegerInvariant,
    conv_output_terms,
    propagate_network,
)


@dataclass(frozen=True)
class VariableRange:
    start: int
    size: int

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self.start + index


@dataclass(frozen=True)
class RelationalLP:
    network: ConvFixedPointNetwork
    input_variables: VariableRange
    rounded_variables: tuple[VariableRange, ...]
    output_variables: tuple[VariableRange, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    constraint_matrix: object
    constraint_upper: np.ndarray

    @property
    def variable_count(self) -> int:
        return len(self.lower_bounds)

    @property
    def constraint_count(self) -> int:
        return len(self.constraint_upper)

    def solve_margin(
        self,
        target: int,
        competitor: int,
        *,
        time_limit_seconds: float | None = None,
    ) -> OptimizeResult:
        final = self.output_variables[-1]
        if target == competitor or not (0 <= target < final.size and 0 <= competitor < final.size):
            raise ValueError("Invalid target/competitor pair")
        objective = np.zeros(self.variable_count, dtype=np.float64)
        objective[final[target]] = 1.0
        objective[final[competitor]] = -1.0
        options = {"presolve": True}
        if time_limit_seconds is not None:
            options["time_limit"] = float(time_limit_seconds)
        return linprog(
            objective,
            A_ub=self.constraint_matrix,
            b_ub=self.constraint_upper,
            bounds=list(zip(self.lower_bounds, self.upper_bounds, strict=True)),
            method="highs",
            options=options,
        )


class _Rows:
    def __init__(self) -> None:
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.upper: list[float] = []

    def add(self, terms: Iterable[tuple[int, float]], upper: float) -> None:
        row = len(self.upper)
        for column, value in terms:
            if value:
                self.rows.append(row)
                self.columns.append(int(column))
                self.values.append(float(value))
        self.upper.append(float(upper))


def _rounded_affine_bounds(layer, invariant: IntegerInvariant, output_index: int):
    if isinstance(layer, QuantizedConv2D):
        terms = conv_output_terms(layer, output_index)
        channel = np.unravel_index(output_index, layer.geometry.output_shape)[-1]
        bias = int(layer.bias_int[channel])
    else:
        terms = tuple(enumerate(layer.weights_int[output_index]))
        bias = int(layer.bias_int[output_index])
    low = high = 0
    for index, weight in terms:
        products = (int(weight) * invariant.low[index], int(weight) * invariant.high[index])
        low += min(products)
        high += max(products)
    denominator = 1 << layer.input_fractional_bits
    return (
        round_divide_half_away_from_zero(low, denominator) + bias,
        round_divide_half_away_from_zero(high, denominator) + bias,
        terms,
        bias,
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _add_clip_hull(
    rows: _Rows,
    source: int,
    target: int,
    source_low: int,
    source_high: int,
    clip_low: int,
    clip_high: int,
) -> None:
    """Add the exact convex hull of ``target = clip(source)`` on an interval."""

    if source_low > source_high or clip_low >= clip_high:
        raise ValueError("Invalid clip interval")
    if source_low == source_high:
        value = _clip(source_low, clip_low, clip_high)
        rows.add(((target, 1.0),), value)
        rows.add(((target, -1.0),), -value)
        return
    if source_high <= clip_low:
        rows.add(((target, 1.0),), clip_low)
        rows.add(((target, -1.0),), -clip_low)
        return
    if source_low >= clip_high:
        rows.add(((target, 1.0),), clip_high)
        rows.add(((target, -1.0),), -clip_high)
        return
    if source_low >= clip_low and source_high <= clip_high:
        rows.add(((target, 1.0), (source, -1.0)), 0.0)
        rows.add(((target, -1.0), (source, 1.0)), 0.0)
        return
    if source_low < clip_low and source_high <= clip_high:
        slope = (source_high - clip_low) / (source_high - source_low)
        intercept = clip_low - slope * source_low
        rows.add(((target, -1.0),), -clip_low)
        rows.add(((source, 1.0), (target, -1.0)), 0.0)
        rows.add(((target, 1.0), (source, -slope)), intercept)
        return
    if source_low >= clip_low and source_high > clip_high:
        slope = (clip_high - source_low) / (source_high - source_low)
        intercept = source_low - slope * source_low
        rows.add(((target, 1.0),), clip_high)
        rows.add(((target, 1.0), (source, -1.0)), 0.0)
        rows.add(((target, -1.0), (source, slope)), -intercept)
        return

    lower_slope = (clip_high - clip_low) / (source_high - clip_low)
    lower_intercept = clip_low - lower_slope * clip_low
    upper_slope = (clip_high - clip_low) / (clip_high - source_low)
    upper_intercept = clip_low - upper_slope * source_low
    rows.add(((target, -1.0),), -clip_low)
    rows.add(((target, -1.0), (source, lower_slope)), -lower_intercept)
    rows.add(((target, 1.0),), clip_high)
    rows.add(((target, 1.0), (source, -upper_slope)), upper_intercept)


def build_relational_lp(
    network: ConvFixedPointNetwork,
    input_invariant: IntegerInvariant,
) -> RelationalLP:
    """Build a sparse continuous over-approximation of the integer network."""

    if len(input_invariant.low) != math.prod(network.input_shape):
        raise ValueError("Input invariant does not match network")
    certificates = propagate_network(network.layers, input_invariant)
    lower_bounds = [float(value) for value in input_invariant.low]
    upper_bounds = [float(value) for value in input_invariant.high]
    input_variables = VariableRange(0, len(lower_bounds))
    previous = input_variables
    rounded_variables: list[VariableRange] = []
    output_variables: list[VariableRange] = []
    rows = _Rows()

    for layer, certificate in zip(network.layers, certificates, strict=True):
        rounded_start = len(lower_bounds)
        rounded_bounds = [
            _rounded_affine_bounds(layer, certificate.input_invariant, output_index)
            for output_index in range(layer.output_size)
        ]
        for low, high, _, _ in rounded_bounds:
            lower_bounds.append(float(low))
            upper_bounds.append(float(high))
        rounded = VariableRange(rounded_start, layer.output_size)
        rounded_variables.append(rounded)

        output_start = len(lower_bounds)
        lower_bounds.extend(float(value) for value in certificate.output_invariant.low)
        upper_bounds.extend(float(value) for value in certificate.output_invariant.high)
        output = VariableRange(output_start, layer.output_size)
        output_variables.append(output)

        denominator = float(1 << layer.input_fractional_bits)
        for output_index, (raw_low, raw_high, terms, bias) in enumerate(rounded_bounds):
            affine = [(rounded[output_index], 1.0)]
            affine.extend((previous[index], -float(weight) / denominator)
                          for index, weight in terms)
            rows.add(affine, float(bias) + 0.5)
            rows.add(((column, -value) for column, value in affine), -float(bias) + 0.5)
            q_min = -(1 << (layer.spec.total_bits - 1))
            q_max = (1 << (layer.spec.total_bits - 1)) - 1
            clip_low = 0 if layer.apply_relu else q_min
            _add_clip_hull(
                rows,
                rounded[output_index],
                output[output_index],
                raw_low,
                raw_high,
                clip_low,
                q_max,
            )
        previous = output

    matrix = coo_matrix(
        (rows.values, (rows.rows, rows.columns)),
        shape=(len(rows.upper), len(lower_bounds)),
        dtype=np.float64,
    ).tocsr()
    return RelationalLP(
        network=network,
        input_variables=input_variables,
        rounded_variables=tuple(rounded_variables),
        output_variables=tuple(output_variables),
        lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
        upper_bounds=np.asarray(upper_bounds, dtype=np.float64),
        constraint_matrix=matrix,
        constraint_upper=np.asarray(rows.upper, dtype=np.float64),
    )
