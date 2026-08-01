from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_article_experiments


class ArticleExperimentSelectionTest(unittest.TestCase):
    def test_only_filter_preserves_existing_success_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run_dir = output_root / "existing_run"
            run_dir.mkdir()
            status_path = run_dir / "run_status.json"
            status_path.write_text(
                json.dumps({"status": "success", "final_status": "VERIFIED"}),
                encoding="utf-8",
            )
            config = {
                "metadata": {"output_root": str(output_root)},
                "runs": [
                    {
                        "name": "existing_run",
                        "dataset": "iris",
                        "arch": "1blk_10",
                        "sample_id": 0,
                        "eps": 0.01,
                    }
                ],
            }
            config_path = output_root / "experiments.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with mock.patch.object(
                run_article_experiments,
                "_has_existing_output",
                return_value=True,
            ):
                run_article_experiments.main(
                    [
                        "--config",
                        str(config_path),
                        "--only",
                        "different_run",
                    ]
                )

            self.assertEqual(
                json.loads(status_path.read_text(encoding="utf-8")),
                {"status": "success", "final_status": "VERIFIED"},
            )


if __name__ == "__main__":
    unittest.main()
