from __future__ import annotations

import importlib.util
import unittest

from synthesis.preqbmc import GPEncoding
from synthesis.solver_backend import BackendConstants as GRB
from synthesis.solver_backend import SolverStatus, build_model


def _has_mip() -> bool:
    return importlib.util.find_spec("mip") is not None


@unittest.skipUnless(_has_mip(), "python-mip is not installed")
class CbcBackendTest(unittest.TestCase):
    def test_continuous_objective(self) -> None:
        model = build_model("cbc", "continuous_objective")
        x = model.add_var(lb=0, ub=10, vtype=GRB.CONTINUOUS)
        model.set_objective(2 * x + 1, GRB.MAXIMIZE)

        self.assertEqual(model.optimize(), SolverStatus.OPTIMAL)
        self.assertAlmostEqual(model.value(x), 10.0, places=6)

    def test_binary_disjunction(self) -> None:
        model = build_model("cbc", "binary_disjunction")
        x = model.add_var(lb=0, ub=10, vtype=GRB.CONTINUOUS)
        y = model.add_var(vtype=GRB.BINARY)
        model.add_constr(x <= 3 + 7 * y)
        model.add_constr(x >= 5)
        model.set_objective(y, GRB.MINIMIZE)

        self.assertEqual(model.optimize(), SolverStatus.OPTIMAL)
        self.assertAlmostEqual(model.value(y), 1.0, places=6)

    def test_max_min_encodings(self) -> None:
        model = build_model("cbc", "max_min")
        x = model.add_var(lb=-2, ub=4, vtype=GRB.CONTINUOUS)
        y = model.add_var(lb=1, ub=3, vtype=GRB.CONTINUOUS)
        z_max = model.add_var(lb=-2, ub=4, vtype=GRB.CONTINUOUS)
        z_min = model.add_var(lb=-2, ub=4, vtype=GRB.CONTINUOUS)
        model.add_constr(x == 2)
        model.add_constr(y == 3)
        model.add_max_constr(z_max, [x, y], operand_bounds=[(-2, 4), (1, 3)])
        model.add_min_constr(z_min, [x, y], operand_bounds=[(-2, 4), (1, 3)])
        model.set_objective(z_max + z_min, GRB.MINIMIZE)

        self.assertEqual(model.optimize(), SolverStatus.OPTIMAL)
        self.assertAlmostEqual(model.value(z_max), 3.0, places=6)
        self.assertAlmostEqual(model.value(z_min), 2.0, places=6)

    def test_constraint_removal_and_resolve(self) -> None:
        model = build_model("cbc", "removal")
        x = model.add_var(lb=0, ub=10, vtype=GRB.CONTINUOUS)
        constraint = model.add_constr(x <= 2)
        model.set_objective(x, GRB.MAXIMIZE)

        self.assertEqual(model.optimize(), SolverStatus.OPTIMAL)
        self.assertAlmostEqual(model.value(x), 2.0, places=6)

        model.remove([constraint])
        model.update()
        self.assertEqual(model.optimize(), SolverStatus.OPTIMAL)
        self.assertAlmostEqual(model.value(x), 10.0, places=6)

    def test_linear_combination_accepts_expression_constant(self) -> None:
        model = build_model("cbc", "linear_expression_constant")
        alpha = model.add_var(lb=0, ub=10, vtype=GRB.CONTINUOUS)
        x = model.add_var(lb=0, ub=5, vtype=GRB.CONTINUOUS)
        y = model.add_var(lb=-10, ub=10, vtype=GRB.CONTINUOUS)
        expression_constant = 1.0 - alpha
        expression = GPEncoding._linear_combination([2.0], [x], expression_constant)
        model.add_constr(alpha == 3)
        model.add_constr(x == 4)
        model.add_constr(y == expression)
        model.set_objective(y, GRB.MINIMIZE)

        self.assertEqual(model.optimize(), SolverStatus.OPTIMAL)
        self.assertAlmostEqual(model.value(y), 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
