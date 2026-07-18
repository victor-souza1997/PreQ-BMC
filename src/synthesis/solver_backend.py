from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any, Literal, Protocol, Sequence


SolverBackendName = Literal["cbc", "gurobi"]


class SolverStatus(str, Enum):
    """Backend-neutral MILP status values used by the robustness pipeline."""

    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    INF_OR_UNBD = "INF_OR_UNBD"
    TIME_LIMIT = "TIME_LIMIT"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class _ParamNames:
    Threads = "Threads"
    OutputFlag = "OutputFlag"


class BackendConstants:
    """Small Gurobi-compatible constant surface for legacy formulation code."""

    BINARY = "BINARY"
    CONTINUOUS = "CONTINUOUS"
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"
    MAXINT = 1_000_000_000.0
    OPTIMAL = SolverStatus.OPTIMAL
    FEASIBLE = SolverStatus.FEASIBLE
    INFEASIBLE = SolverStatus.INFEASIBLE
    UNBOUNDED = SolverStatus.UNBOUNDED
    INF_OR_UNBD = SolverStatus.INF_OR_UNBD
    TIME_LIMIT = SolverStatus.TIME_LIMIT
    UNKNOWN = SolverStatus.UNKNOWN
    Param = _ParamNames


@dataclass(frozen=True)
class GeneratedConstraint:
    constraints: tuple[Any, ...]
    variables: tuple[Any, ...] = ()


@dataclass(frozen=True)
class NoOpConstraint:
    reason: str = "constant-true"


class MilpModel(Protocol):
    name: str

    @property
    def status(self) -> SolverStatus: ...

    def add_var(self, *, lb: float | None = None, ub: float | None = None, vtype: str | None = None, name: str | None = None) -> Any: ...

    def add_constr(self, expr: Any, name: str | None = None) -> Any: ...

    def set_objective(self, expr: Any, sense: str) -> None: ...

    def optimize(self) -> SolverStatus: ...

    def value(self, var: Any) -> float: ...

    def remove(self, objects: Any) -> None: ...

    def update(self) -> None: ...

    def add_max_constr(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> Any: ...

    def add_min_constr(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> Any: ...


def _finite_pair(bounds: tuple[float, float], *, context: str) -> tuple[float, float]:
    lb, ub = float(bounds[0]), float(bounds[1])
    if not math.isfinite(lb) or not math.isfinite(ub):
        raise ValueError(f"CBC {context} requires finite operand bounds, got {bounds!r}.")
    if lb > ub:
        raise ValueError(f"CBC {context} received invalid bounds {bounds!r}.")
    return lb, ub


def _flatten_generated(objects: Any) -> tuple[list[Any], list[Any]]:
    if objects is None:
        return [], []
    if isinstance(objects, (list, tuple, set)):
        iterable = list(objects)
    else:
        iterable = [objects]
    constraints: list[Any] = []
    variables: list[Any] = []
    for item in iterable:
        if item is None or isinstance(item, NoOpConstraint):
            continue
        if isinstance(item, GeneratedConstraint):
            constraints.extend(item.constraints)
            variables.extend(item.variables)
        else:
            constraints.append(item)
    return constraints, variables


class GurobiBackend:
    """Gurobi adapter that preserves the existing native expression semantics."""

    def __init__(self, name: str, *, threads: int | None = None, output_flag: int = 0) -> None:
        try:
            import gurobipy as gp
            from gurobipy import GRB as native_grb
        except ImportError as exc:  # pragma: no cover - depends on host install/license.
            raise RuntimeError(
                "Gurobi backend requested, but gurobipy is not importable. "
                "Install `pip install -e '.[gurobi]'` and configure a Gurobi license."
            ) from exc
        self.name = name
        self._gp = gp
        self._grb = native_grb
        self._model = gp.Model(name)
        self._status = SolverStatus.UNKNOWN
        self.setParam(BackendConstants.Param.OutputFlag, output_flag)
        if threads is not None:
            self.setParam(BackendConstants.Param.Threads, int(threads))

    def add_var(
        self,
        *,
        lb: float | None = None,
        ub: float | None = None,
        vtype: str | None = None,
        name: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if lb is not None:
            kwargs["lb"] = float(lb)
        if ub is not None:
            kwargs["ub"] = float(ub)
        if name:
            kwargs["name"] = name
        if vtype == BackendConstants.BINARY:
            kwargs["vtype"] = self._grb.BINARY
        elif vtype == BackendConstants.CONTINUOUS or vtype is None:
            kwargs["vtype"] = self._grb.CONTINUOUS
        else:
            kwargs["vtype"] = vtype
        return self._model.addVar(**kwargs)

    def addVar(self, *args: Any, **kwargs: Any) -> Any:
        return self.add_var(*args, **kwargs)

    def add_constr(self, expr: Any, name: str | None = None) -> Any:
        if isinstance(expr, bool):
            if expr:
                return NoOpConstraint()
            raise ValueError("Attempted to add a constant-false Gurobi constraint.")
        if name is None:
            return self._model.addConstr(expr)
        return self._model.addConstr(expr, name=name)

    def addConstr(self, expr: Any, name: str | None = None) -> Any:
        return self.add_constr(expr, name=name)

    def add_max_constr(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> Any:
        del operand_bounds
        variables, constants = self._split_variable_and_constant_operands(operands)
        if not variables:
            return self.add_constr(target == max(constants), name=name)
        kwargs: dict[str, Any] = {"name": name or ""}
        if constants:
            kwargs["constant"] = max(constants)
        return self._model.addGenConstrMax(target, variables, **kwargs)

    def addGenConstrMax(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> Any:
        return self.add_max_constr(target, operands, operand_bounds=operand_bounds, name=name)

    def add_min_constr(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> Any:
        del operand_bounds
        variables, constants = self._split_variable_and_constant_operands(operands)
        if not variables:
            return self.add_constr(target == min(constants), name=name)
        kwargs: dict[str, Any] = {"name": name or ""}
        if constants:
            kwargs["constant"] = min(constants)
        return self._model.addGenConstrMin(target, variables, **kwargs)

    def addGenConstrMin(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> Any:
        return self.add_min_constr(target, operands, operand_bounds=operand_bounds, name=name)

    def set_objective(self, expr: Any, sense: str) -> None:
        native_sense = self._grb.MINIMIZE if sense == BackendConstants.MINIMIZE else self._grb.MAXIMIZE
        self._model.setObjective(expr, native_sense)

    def setObjective(self, expr: Any, sense: str) -> None:
        self.set_objective(expr, sense)

    def setParam(self, name: Any, value: Any) -> None:
        self._model.setParam(str(name), value)

    def optimize(self) -> SolverStatus:
        self._model.optimize()
        self._status = self._map_status(self._model.status)
        return self._status

    @property
    def status(self) -> SolverStatus:
        return self._map_status(self._model.status)

    def value(self, var: Any) -> float:
        return float(var.X)

    def remove(self, objects: Any) -> None:
        constraints, variables = _flatten_generated(objects)
        if constraints:
            self._model.remove(constraints)
        if variables:
            self._model.remove(variables)

    def update(self) -> None:
        self._model.update()

    @staticmethod
    def _split_variable_and_constant_operands(operands: Sequence[Any]) -> tuple[list[Any], list[float]]:
        variables: list[Any] = []
        constants: list[float] = []
        for operand in operands:
            if isinstance(operand, Real):
                constants.append(float(operand))
            else:
                variables.append(operand)
        return variables, constants

    def _map_status(self, status: Any) -> SolverStatus:
        if status == self._grb.OPTIMAL:
            return SolverStatus.OPTIMAL
        if status == self._grb.INFEASIBLE:
            return SolverStatus.INFEASIBLE
        if status == self._grb.UNBOUNDED:
            return SolverStatus.UNBOUNDED
        if status == self._grb.INF_OR_UNBD:
            return SolverStatus.INF_OR_UNBD
        if status == self._grb.TIME_LIMIT:
            return SolverStatus.TIME_LIMIT
        return SolverStatus.UNKNOWN


class CbcBackend:
    """python-mip/CBC adapter with finite-bound max/min encodings."""

    def __init__(self, name: str, *, threads: int | None = None, output_flag: int = 0) -> None:
        try:
            import mip
            from mip import CBC, Model
        except ImportError as exc:  # pragma: no cover - exercised when optional dep is absent.
            raise RuntimeError(
                "CBC backend requested, but python-mip is not importable. "
                "Install it with `pip install -e '.[cbc]'`."
            ) from exc
        self.name = name
        self._mip = mip
        self._model = Model(name=name, solver_name=CBC)
        self._bounds: dict[int, tuple[float, float]] = {}
        self._status = SolverStatus.UNKNOWN
        self._generated_constraint_counter = 0
        self.setParam(BackendConstants.Param.OutputFlag, output_flag)
        if threads is not None:
            self.setParam(BackendConstants.Param.Threads, int(threads))

    def add_var(
        self,
        *,
        lb: float | None = None,
        ub: float | None = None,
        vtype: str | None = None,
        name: str | None = None,
    ) -> Any:
        var_type = self._mip.BINARY if vtype == BackendConstants.BINARY else self._mip.CONTINUOUS
        kwargs: dict[str, Any] = {"var_type": var_type}
        if lb is not None:
            kwargs["lb"] = float(lb)
        if ub is not None:
            kwargs["ub"] = float(ub)
        if name:
            kwargs["name"] = name
        var = self._model.add_var(**kwargs)
        self._bounds[id(var)] = (float(var.lb), float(var.ub))
        return var

    def addVar(self, *args: Any, **kwargs: Any) -> Any:
        return self.add_var(*args, **kwargs)

    def add_constr(self, expr: Any, name: str | None = None) -> Any:
        if isinstance(expr, bool):
            if expr:
                return NoOpConstraint()
            raise ValueError("Attempted to add a constant-false CBC constraint.")
        if name is None:
            return self._model.add_constr(expr)
        return self._model.add_constr(expr, name=name)

    def addConstr(self, expr: Any, name: str | None = None) -> Any:
        return self.add_constr(expr, name=name)

    def add_max_constr(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> GeneratedConstraint:
        del name
        constraint_id = self._next_generated_constraint_id()
        ops = list(operands)
        bounds = self._resolve_operand_bounds(ops, operand_bounds, context="max")
        max_ub = max(ub for _, ub in bounds)
        constraints: list[Any] = []
        selectors: list[Any] = []
        constraints.append(self.add_constr(target <= max_ub))
        for index, (operand, (operand_lb, _operand_ub)) in enumerate(zip(ops, bounds)):
            selector = self.add_var(vtype=BackendConstants.BINARY, name=f"max_sel_{constraint_id}_{index}")
            selectors.append(selector)
            big_m = max(0.0, float(max_ub - operand_lb))
            constraints.append(self.add_constr(target >= operand))
            constraints.append(self.add_constr(target <= operand + big_m * (1 - selector)))
        constraints.append(self.add_constr(self._mip.xsum(selectors) == 1))
        return GeneratedConstraint(tuple(constraints), tuple(selectors))

    def addGenConstrMax(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> GeneratedConstraint:
        return self.add_max_constr(target, operands, operand_bounds=operand_bounds, name=name)

    def add_min_constr(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> GeneratedConstraint:
        del name
        constraint_id = self._next_generated_constraint_id()
        ops = list(operands)
        bounds = self._resolve_operand_bounds(ops, operand_bounds, context="min")
        min_lb = min(lb for lb, _ in bounds)
        constraints: list[Any] = []
        selectors: list[Any] = []
        constraints.append(self.add_constr(target >= min_lb))
        for index, (operand, (_operand_lb, operand_ub)) in enumerate(zip(ops, bounds)):
            selector = self.add_var(vtype=BackendConstants.BINARY, name=f"min_sel_{constraint_id}_{index}")
            selectors.append(selector)
            big_m = max(0.0, float(operand_ub - min_lb))
            constraints.append(self.add_constr(target <= operand))
            constraints.append(self.add_constr(target >= operand - big_m * (1 - selector)))
        constraints.append(self.add_constr(self._mip.xsum(selectors) == 1))
        return GeneratedConstraint(tuple(constraints), tuple(selectors))

    def addGenConstrMin(
        self,
        target: Any,
        operands: Sequence[Any],
        *,
        operand_bounds: Sequence[tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> GeneratedConstraint:
        return self.add_min_constr(target, operands, operand_bounds=operand_bounds, name=name)

    def set_objective(self, expr: Any, sense: str) -> None:
        if sense == BackendConstants.MINIMIZE:
            self._model.objective = self._mip.minimize(expr)
        else:
            self._model.objective = self._mip.maximize(expr)

    def setObjective(self, expr: Any, sense: str) -> None:
        self.set_objective(expr, sense)

    def setParam(self, name: Any, value: Any) -> None:
        key = str(name)
        if key == BackendConstants.Param.Threads:
            self._model.threads = max(1, int(value))
        elif key == BackendConstants.Param.OutputFlag:
            self._model.verbose = int(value)
        elif key in {"TimeLimit", "max_seconds"}:
            self._model.max_seconds = float(value)
        else:
            # Gurobi-only tolerances and reductions are ignored by CBC unless
            # python-mip exposes a direct equivalent.
            return

    def optimize(self) -> SolverStatus:
        raw_status = self._model.optimize()
        self._status = self._map_status(raw_status if raw_status is not None else self._model.status)
        return self._status

    @property
    def status(self) -> SolverStatus:
        return self._map_status(self._model.status)

    def value(self, var: Any) -> float:
        return float(var.x)

    def remove(self, objects: Any) -> None:
        raw_constraints, generated_variables = _flatten_generated(objects)
        constraints: list[Any] = []
        variables: list[Any] = [item for item in generated_variables if not isinstance(item, NoOpConstraint)]
        for item in raw_constraints:
            if isinstance(item, NoOpConstraint):
                continue
            if id(item) in self._bounds:
                variables.append(item)
            else:
                constraints.append(item)
        if constraints:
            self._model.remove(constraints)
        if variables:
            self._model.remove(variables)
            for var in variables:
                self._bounds.pop(id(var), None)

    def update(self) -> None:
        return None

    def _resolve_operand_bounds(
        self,
        operands: Sequence[Any],
        operand_bounds: Sequence[tuple[float, float]] | None,
        *,
        context: str,
    ) -> list[tuple[float, float]]:
        if operand_bounds is not None:
            if len(operand_bounds) != len(operands):
                raise ValueError(f"CBC {context} constraint got {len(operands)} operands and {len(operand_bounds)} bounds.")
            return [_finite_pair(bounds, context=context) for bounds in operand_bounds]

        resolved: list[tuple[float, float]] = []
        for operand in operands:
            if isinstance(operand, Real):
                value = float(operand)
                resolved.append(_finite_pair((value, value), context=context))
                continue
            bounds = self._bounds.get(id(operand))
            if bounds is None:
                raise ValueError(
                    f"CBC {context} constraint requires finite bounds for every nonconstant operand. "
                    "Pass operand_bounds from the formulation context."
                )
            resolved.append(_finite_pair(bounds, context=context))
        return resolved

    def _next_generated_constraint_id(self) -> int:
        self._generated_constraint_counter += 1
        return self._generated_constraint_counter

    def _map_status(self, status: Any) -> SolverStatus:
        opt_status = self._mip.OptimizationStatus
        if status == opt_status.OPTIMAL:
            return SolverStatus.OPTIMAL
        if status == opt_status.FEASIBLE:
            return SolverStatus.FEASIBLE
        if status == opt_status.INFEASIBLE:
            return SolverStatus.INFEASIBLE
        if status == opt_status.UNBOUNDED:
            return SolverStatus.UNBOUNDED
        if status == opt_status.NO_SOLUTION_FOUND:
            return SolverStatus.TIME_LIMIT
        if status == opt_status.ERROR:
            return SolverStatus.ERROR
        return SolverStatus.UNKNOWN


def build_model(
    backend: SolverBackendName,
    name: str,
    *,
    threads: int | None = None,
    output_flag: int = 0,
    **_kwargs: Any,
) -> MilpModel:
    if backend == "cbc":
        return CbcBackend(name, threads=threads, output_flag=output_flag)
    if backend == "gurobi":
        return GurobiBackend(name, threads=threads, output_flag=output_flag)
    raise ValueError(f"Unsupported MILP solver backend: {backend!r}. Expected 'cbc' or 'gurobi'.")
