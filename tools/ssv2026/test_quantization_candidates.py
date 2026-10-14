"""Small standalone regressions: no dataset, training, or ESBMC required."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("quantization_candidates.py")
spec = importlib.util.spec_from_file_location("ssv_quantization_candidates", SCRIPT)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class CandidateTests(unittest.TestCase):
    def model(self):
        np = q.np
        geometry = q.ConvGeometry((2, 2, 1), (1, 1, 1, 2), padding="SAME")
        return q.RestrictedSequentialCNN(
            (geometry,), (np.array([[[[1.0, -0.5]]]], dtype=np.float32),),
            (np.array([-0.25, 0.5], dtype=np.float32),),
            (np.arange(16, dtype=np.float32).reshape(8, 2) / 16 - 0.5,),
            (np.array([0.25, -0.5], dtype=np.float32),), max_affine_entries=1000,
        )

    def test_compact_c_matches_bigint_at_both_widths(self):
        np = q.np
        for bits, frac in ((8, 4), (16, 8)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory(prefix="ssv_qnn_") as tmp:
                folder = Path(tmp)
                network, encoder = q.make_network(self.model(), bits, frac, "nearest_center")
                wide = q.generate_conv_qnn_source(network, encoder_source=encoder.render_c())
                compact = q.compact_source(wide, network, bits)
                self.assertIn(f"static const int{bits}_t LAYER_0_KERNEL", compact)
                self.assertIn(f"int{bits}_t buffer_0", compact)
                self.assertIn("(__int128)w * (__int128)x", compact)
                path = folder / "compact.c"
                path.write_text(compact)
                q.compile_library(path, folder / "compact.so")
                c = q.CNetwork(folder / "compact.so", network)
                # Every input byte, plus signed input extremes (legal forward ABI).
                for byte in range(256):
                    image = np.full((3, 5, 1), byte, dtype=np.uint8)
                    encoded = encoder.encode(image)
                    np.testing.assert_array_equal(c.encode(image), encoded)
                    np.testing.assert_array_equal(c.forward(encoded), q.forward_conv_fixed_point_single(network, encoded))
                    np.testing.assert_array_equal(c.forward(encoded), q.numpy_reference(network, encoded))
                for value in (-(1 << (bits - 1)), (1 << (bits - 1)) - 1):
                    encoded = np.full(4, value, dtype=np.int64)
                    np.testing.assert_array_equal(c.forward(encoded), q.forward_conv_fixed_point_single(network, encoded))

    def test_changed_generator_fails_closed(self):
        network, encoder = q.make_network(self.model(), 8, 4, "nearest_center")
        wide = q.generate_conv_qnn_source(network, encoder_source=encoder.render_c())
        with self.assertRaises(ValueError):
            q.compact_source(wide.replace("buffer_0[i] = input[i];", "buffer_0[i] = 0;"), network, 8)

    def test_out_of_range_parameters_rejected(self):
        network, encoder = q.make_network(self.model(), 16, 8, "nearest_center")
        wide = q.generate_conv_qnn_source(network, encoder_source=encoder.render_c())
        with self.assertRaises(ValueError):
            q.compact_source(wide, network, 8)

    def test_format_constraints_and_negative_control(self):
        with self.assertRaises(ValueError):
            q.make_network(self.model(), 8, 8, "nearest_center")
        lemma = q.storage_lemma(8, 4)
        self.assertEqual(lemma.count('(__int128)s == v, "clamped storage cast identity"'), 1)

    def test_memory_guard_never_launches(self):
        self.assertEqual(q.classify_result("VERIFICATION SUCCESSFUL", 1, 0), "UNKNOWN")
        self.assertEqual(q.classify_result("VERIFICATION SUCCESSFUL", 0, 0), "VERIFIED")
        with tempfile.TemporaryDirectory(prefix="ssv_guard_") as tmp:
            path = Path(tmp) / "lemma.c"
            path.write_text(q.storage_lemma(8, 4))
            with patch.object(q, "available_gib", return_value=7.5), patch.object(
                q.subprocess, "Popen", side_effect=AssertionError("must not launch")
            ):
                result = q.run_obligation(path, 30, 8)
            self.assertEqual(result["status"], "NOT_RUN_LOW_MEMORY")

    def test_failed_quality_gate_precedes_solver_and_output_creation(self):
        with tempfile.TemporaryDirectory(prefix="ssv_gate_") as tmp:
            out = Path(tmp) / "must_not_exist"
            args = SimpleNamespace(artifact=Path(tmp), complete=True,
                                   validation_report=Path(tmp) / "report.json", output=out)
            with patch.object(q, "load_candidate", return_value=(Path(tmp), {}, None, None, {}, [{}])), patch.object(
                q, "read_json", return_value={"candidate_quality_eligible": False}
            ):
                with self.assertRaises(ValueError):
                    q.verify(args)
            self.assertFalse(out.exists())

    def test_direct_harness_compiles_and_uses_actual_compact_source(self):
        np = q.np
        with tempfile.TemporaryDirectory(prefix="ssv_direct_") as tmp:
            folder = Path(tmp)
            network, encoder = q.make_network(self.model(), 8, 4, "nearest_center")
            wide = q.generate_conv_qnn_source(network, encoder_source=encoder.render_c())
            (folder / "qnn_compact.c").write_text(q.compact_source(wide, network, 8))
            direct = q.direct_harness(folder, network, np.zeros((2, 2, 1), np.uint8),
                                      np.ones((2, 2, 1), np.uint8), 0)
            path = folder / "direct.c"
            path.write_text(direct)
            self.assertIn(str(folder / "qnn_compact.c"), direct)
            self.assertIn('"strict class margin"', direct)
            q.subprocess.run(["gcc", "-std=c11", "-fsyntax-only", str(path)], check=True,
                             capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
