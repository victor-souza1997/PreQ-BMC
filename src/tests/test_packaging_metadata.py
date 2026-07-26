from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class PackagingMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.pyproject = tomllib.loads((cls.repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    def test_default_requirements_install_local_paper_extra(self) -> None:
        lines = [
            line.strip()
            for line in (self.repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(lines, ["-e .[paper]"])

    def test_paper_extra_is_license_free_and_excludes_conversion_tooling(self) -> None:
        extras = self.pyproject["project"]["optional-dependencies"]
        paper = set(extras["paper"])

        self.assertEqual(
            paper,
            {"tensorflow", "h5py", "scikit-learn", "matplotlib", "mip"},
        )
        self.assertTrue({"gurobipy", "torch", "onnx", "pandas"}.isdisjoint(paper))

    def test_console_script_targets_repository_cli(self) -> None:
        scripts = self.pyproject["project"]["scripts"]
        self.assertEqual(scripts["preqbmc"], "cli:main")


if __name__ == "__main__":
    unittest.main()
