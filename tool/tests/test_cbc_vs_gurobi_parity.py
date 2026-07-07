from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from synthesis.preqbmc import GPEncoding
from synthesis.solver_backend import BackendConstants as GRB
from synthesis.solver_backend import SolverStatus, build_model


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _can_build_gurobi() -> bool:
    if not _module_available("gurobipy"):
        return False
    try:
        model = build_model("gurobi", "license_probe")
        x = model.add_var(lb=0, ub=1, vtype=GRB.CONTINUOUS)
        model.set_objective(x, GRB.MAXIMIZE)
        return model.optimize() == SolverStatus.OPTIMAL
    except Exception:
        return False


@unittest.skipUnless(_module_available("mip"), "python-mip is not installed")
class SolverParityTest(unittest.TestCase):
    def _solve_reference_problem(self, solver: str) -> tuple[float, float, float]:
        model = build_model(solver, f"{solver}_parity")
        x = model.add_var(lb=0, ub=10, vtype=GRB.CONTINUOUS)
        y = model.add_var(vtype=GRB.BINARY)
        z = model.add_var(lb=0, ub=10, vtype=GRB.CONTINUOUS)
        model.add_constr(x <= 3 + 7 * y)
        model.add_constr(x >= 5)
        model.add_max_constr(z, [x, 6.0], operand_bounds=[(0, 10), (6, 6)])
        model.set_objective(z - y, GRB.MINIMIZE)
        self.assertEqual(model.optimize(), SolverStatus.OPTIMAL)
        return model.value(x), model.value(y), model.value(z)

    def test_cbc_solves_reference_problem(self) -> None:
        x, y, z = self._solve_reference_problem("cbc")
        self.assertAlmostEqual(x, 5.0, places=6)
        self.assertAlmostEqual(y, 1.0, places=6)
        self.assertAlmostEqual(z, 6.0, places=6)

    @unittest.skipUnless(_can_build_gurobi(), "gurobipy is unavailable or unlicensed")
    def test_cbc_and_gurobi_objective_parity(self) -> None:
        cbc_solution = self._solve_reference_problem("cbc")
        gurobi_solution = self._solve_reference_problem("gurobi")
        for cbc_value, gurobi_value in zip(cbc_solution, gurobi_solution):
            self.assertAlmostEqual(cbc_value, gurobi_value, places=6)

    @unittest.skipUnless(
        os.environ.get("PREQBMC_RUN_SOLVER_PARITY") == "1",
        "set PREQBMC_RUN_SOLVER_PARITY=1 to run end-to-end CBC/Gurobi pipeline parity",
    )
    @unittest.skipUnless(shutil.which("esbmc"), "esbmc binary is not installed")
    @unittest.skipUnless(_module_available("tensorflow"), "tensorflow is not installed")
    @unittest.skipUnless(_module_available("h5py"), "h5py is not installed")
    @unittest.skipUnless(_module_available("sklearn"), "scikit-learn is not installed")
    @unittest.skipUnless(_can_build_gurobi(), "gurobipy is unavailable or unlicensed")
    def test_optional_iris_pipeline_parity_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "tool" / "scripts" / "compare_solver_backends.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "solver_parity"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--case",
                    "iris",
                    "--quick",
                    "--output-root",
                    str(output_root),
                ],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=7200,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}",
            )
            report = json.loads((output_root / "solver_parity_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["matches"])


class GurobiExpressionConstantRegressionTest(unittest.TestCase):
    @unittest.skipUnless(_can_build_gurobi(), "gurobipy is unavailable or unlicensed")
    def test_linear_combination_accepts_gurobi_expression_constant(self) -> None:
        model = build_model("gurobi", "gurobi_linear_expression_constant")
        alpha = model.add_var(lb=0, ub=10, vtype=GRB.CONTINUOUS)
        x = model.add_var(lb=0, ub=5, vtype=GRB.CONTINUOUS)
        y = model.add_var(lb=-10, ub=10, vtype=GRB.CONTINUOUS)
        expression = GPEncoding._linear_combination([2.0], [x], 1.0 - alpha)
        model.add_constr(alpha == 3)
        model.add_constr(x == 4)
        model.add_constr(y == expression)
        model.set_objective(y, GRB.MINIMIZE)

        self.assertEqual(model.optimize(), SolverStatus.OPTIMAL)
        self.assertAlmostEqual(model.value(y), 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
