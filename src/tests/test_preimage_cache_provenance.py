"""Regression tests for preimage-cache provenance and model identity.

A cached preimage is only as trustworthy as the record of where it came from.
These tests pin the two properties that make the cache safe to reload:

* provenance survives the round trip and is never invented on load, and
* a cache is refused if it was solved for a different model.

Both matter because a forward DeepPoly box and a property preimage have the
same shape and dtype, so nothing downstream can tell them apart once the label
is lost.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from synthesis.preimage_cache import load_preimage_cache, save_preimage_cache
from synthesis.preqbmc import GPEncoding
from verification.esbmc import ESBMCRunner


def _layers(source: str) -> list[dict[str, object]]:
    return [
        {
            "layer_index": 1,
            "layer_size": 2,
            "relaxed_lb": np.asarray([0.0, -1.0]),
            "relaxed_ub": np.asarray([1.0, 2.0]),
            "preimage_source": source,
        }
    ]


def _encoder(cache_root: Path, key: str, *, expected_sha: str | None) -> GPEncoding:
    encoder = object.__new__(GPEncoding)
    encoder.dense_layers = [SimpleNamespace(layer_index=1, layer_size=2)]
    encoder.config = SimpleNamespace(
        preimage_cache_dir=cache_root,
        preimage_cache_key=key,
        preimage_cache_metadata=(
            {"weights_sha256": expected_sha} if expected_sha else {}
        ),
    )
    return encoder


class PreimageCacheProvenanceTest(unittest.TestCase):
    def _write(self, root: Path, key: str, source: str, sha: str) -> None:
        save_preimage_cache(
            cache_root=root,
            cache_key=key,
            layers=_layers(source),
            scale_values=np.asarray([1.0]),
            metadata={"weights_sha256": sha},
        )

    def test_provenance_is_persisted_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "k", "milp_preimage", "abc")
            enc = _encoder(root, "k", expected_sha="abc")
            enc.load_cached_preimage()
            self.assertEqual(enc.dense_layers[0].preimage_source, "milp_preimage")

    def test_forward_fallback_is_not_laundered_into_a_milp_preimage(self) -> None:
        """The bug this test exists for: a fallback box reloaded as a preimage."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "k", "deeppoly_forward_FALLBACK", "abc")
            enc = _encoder(root, "k", expected_sha="abc")
            enc.load_cached_preimage()
            self.assertEqual(
                enc.dense_layers[0].preimage_source, "deeppoly_forward_FALLBACK"
            )

    def test_cache_without_recorded_provenance_is_not_assumed_to_be_milp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "k", "milp_preimage", "abc")
            meta_path = root / "k" / "metadata.json"
            payload = json.loads(meta_path.read_text())
            for entry in payload["layers"]:
                entry.pop("preimage_source")
            meta_path.write_text(json.dumps(payload))
            enc = _encoder(root, "k", expected_sha="abc")
            enc.load_cached_preimage()
            self.assertEqual(
                enc.dense_layers[0].preimage_source, "cache_provenance_unavailable"
            )

    def test_cache_for_a_different_model_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "k", "milp_preimage", "cache-digest")
            enc = _encoder(root, "k", expected_sha="different-model-digest")
            with self.assertRaises(ValueError) as ctx:
                enc.load_cached_preimage()
            self.assertIn("different model", str(ctx.exception))

    def test_cache_missing_a_weight_digest_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_preimage_cache(
                cache_root=root,
                cache_key="k",
                layers=_layers("milp_preimage"),
                scale_values=np.asarray([1.0]),
                metadata={},
            )
            enc = _encoder(root, "k", expected_sha="abc")
            with self.assertRaises(ValueError):
                enc.load_cached_preimage()

    def test_round_trip_arrays_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "k", "milp_preimage", "abc")
            _, arrays = load_preimage_cache(cache_root=root, cache_key="k")
            np.testing.assert_allclose(arrays["relaxed_lb_0"], [0.0, -1.0])
            np.testing.assert_allclose(arrays["relaxed_ub_0"], [1.0, 2.0])


class UnwindInferenceTest(unittest.TestCase):
    """Cut and prefix loops must not exceed the inferred unwind bound.

    An insufficient unwind is not unsound (ESBMC reports VERIFICATION FAILED),
    but it manufactures spurious refutations, so the bound has to cover every
    loop the templates actually emit.
    """

    def _unwind(self, source: str) -> int:
        return ESBMCRunner.infer_unwind(object.__new__(ESBMCRunner), source)

    def test_contract_cut_count_is_covered(self) -> None:
        src = "#define LAYER_SIZE 4\n#define CONTRACT_CUT_COUNT 40\n"
        self.assertGreater(self._unwind(src), 40)

    def test_margin_cut_count_is_covered(self) -> None:
        src = "#define INPUT_SIZE 4\n#define MARGIN_CUT_COUNT 25\n"
        self.assertGreater(self._unwind(src), 25)

    def test_prefix_output_size_is_covered(self) -> None:
        src = "#define INPUT_SIZE 4\n#define PREFIX_OUTPUT_SIZE 32\n"
        self.assertGreater(self._unwind(src), 32)

    def test_layer_width_still_dominates_when_it_is_largest(self) -> None:
        src = "#define INPUT_SIZE 784\n#define MARGIN_CUT_COUNT 3\n"
        self.assertEqual(self._unwind(src), 785)


if __name__ == "__main__":
    unittest.main()
