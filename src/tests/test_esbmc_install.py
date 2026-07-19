from __future__ import annotations

from pathlib import Path
import os
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

from verification import esbmc_install


class ESBMCInstallTest(unittest.TestCase):
    def test_resolve_prefers_repo_local_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = esbmc_install.local_esbmc_path(root)
            local.parent.mkdir(parents=True)
            local.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            local.chmod(local.stat().st_mode | stat.S_IXUSR)

            self.assertEqual(esbmc_install.resolve_esbmc_executable(root=root), str(local))

    def test_env_override_wins(self) -> None:
        with mock.patch.dict(os.environ, {esbmc_install.ENV_ESBMC_EXECUTABLE: "/tmp/custom-esbmc"}):
            self.assertEqual(esbmc_install.resolve_esbmc_executable(root=Path("/missing")), "/tmp/custom-esbmc")

    def test_install_extracts_release_and_creates_local_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_zip = root / "fixture.zip"
            with zipfile.ZipFile(fixture_zip, "w") as archive:
                archive.writestr("esbmc/bin/esbmc", "#!/bin/sh\necho 'ESBMC fake 1.0'\n")

            release = {
                "tag_name": "v-test",
                "assets": [
                    {
                        "name": "esbmc-linux.zip",
                        "browser_download_url": "https://example.invalid/esbmc-linux.zip",
                    }
                ],
            }

            def copy_fixture(_url: str, destination: Path) -> None:
                destination.write_bytes(fixture_zip.read_bytes())

            with mock.patch.object(esbmc_install, "_read_json_url", return_value=release), mock.patch.object(
                esbmc_install,
                "_download",
                side_effect=copy_fixture,
            ):
                result = esbmc_install.install_esbmc(root=root, asset_pattern="*linux*.zip")

            local = esbmc_install.local_esbmc_path(root)
            self.assertTrue(result["installed"])
            self.assertTrue(local.exists())
            self.assertEqual(esbmc_install.resolve_esbmc_executable(root=root), str(local))
            self.assertIn("ESBMC fake 1.0", result["version_output"])


if __name__ == "__main__":
    unittest.main()
