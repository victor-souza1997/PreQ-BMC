import unittest

from verification.conv_cegar import CEGARConfig, DemandDrivenMarginCEGAR, SparseRelationalCut


class DemandDrivenCEGARTest(unittest.TestCase):
    def test_only_failed_competitor_is_refined_and_verified_cut_is_assumed(self):
        calls = {"propose": [], "validate": [], "retry": []}
        cut = SparseRelationalCut(2, (1, 7), (3, -2), 8, -10, 12)

        def propose(competitor, round_index, accepted):
            calls["propose"].append((competitor, round_index, len(accepted)))
            return [cut]

        def validate(candidate):
            calls["validate"].append(candidate)
            return {"status": "VERIFIED", "checker": "ESBMC"}

        def retry(competitor, accepted):
            calls["retry"].append((competitor, list(accepted)))
            return {"status": "VERIFIED", "checker": "ESBMC"}

        result = DemandDrivenMarginCEGAR(
            propose=propose, validate_with_esbmc=validate,
            retry_margin_with_esbmc=retry, config=CEGARConfig(max_rounds=2),
        ).refine({0: {"status": "VERIFIED"}, 1: {"status": "FAILED"}})
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(calls["propose"], [(1, 0, 0)])
        self.assertEqual(calls["retry"][0][1], [cut])
        self.assertTrue(result["records"][0]["assumed_by_retry"])

    def test_unverified_cut_can_never_reach_retry(self):
        cut = SparseRelationalCut(0, (0,), (1,), 1, 0, 1)
        retries = []
        result = DemandDrivenMarginCEGAR(
            propose=lambda *_: [cut],
            validate_with_esbmc=lambda _: {"status": "TIMEOUT", "checker": "ESBMC"},
            retry_margin_with_esbmc=lambda *args: retries.append(args),
        ).refine({4: {"status": "FAILED"}})
        self.assertEqual(result["status"], "ABSTRACTION_INCONCLUSIVE")
        self.assertEqual(retries, [])
        self.assertFalse(result["records"][0]["assumed_by_retry"])

    def test_empty_or_malformed_cut_is_rejected(self):
        with self.assertRaises(ValueError):
            SparseRelationalCut(0, (), (), 1, 0, 0)
        with self.assertRaises(ValueError):
            SparseRelationalCut(0, (1,), (1,), 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
