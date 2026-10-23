import unittest

import numpy as np

from paiec.metrics import BUDGETS, alc_weights, brier, brier_alc, ece, summarise


class AlcWeightTests(unittest.TestCase):
    def test_budgets_reproduce_the_published_weights(self):
        np.testing.assert_allclose(
            alc_weights(BUDGETS), [0.1, 0.2, 0.2, 0.2, 0.2, 0.1], atol=1e-12
        )

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(float(alc_weights(BUDGETS).sum()), 1.0)

    def test_flat_curve_equals_its_brier(self):
        flat = {budget: 0.25 for budget in BUDGETS}
        self.assertAlmostEqual(brier_alc(flat), 0.25)

    def test_endpoints_carry_half_weight(self):
        # A curve that is 0 everywhere except the last budget: only the last
        # weight (0.1) should see it, even if the violation is large.
        curve = {budget: 0.0 for budget in BUDGETS}
        curve[31] = 1.0
        self.assertAlmostEqual(brier_alc(curve), 0.1)


class BrierTests(unittest.TestCase):
    def test_perfect_predictions_score_zero(self):
        self.assertEqual(brier([0, 1, 1], [0, 1, 1]), 0.0)

    def test_constant_half_scores_quarter(self):
        self.assertAlmostEqual(brier([0, 1, 0, 1], [0.5] * 4), 0.25)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            brier([0, 1], [0.5])

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            brier([], [])


class EceTests(unittest.TestCase):
    def test_calibrated_bins_score_zero(self):
        # One bin at 0.5 holding half successes is perfectly calibrated.
        self.assertAlmostEqual(ece([0, 1, 0, 1], [0.5] * 4), 0.0)

    def test_overconfidence_is_penalised(self):
        self.assertAlmostEqual(ece([0, 0, 0, 0], [0.9] * 4), 0.9)

    def test_bins_are_weighted_by_count(self):
        labels = [0, 0, 0, 1]
        probs = [0.05, 0.05, 0.05, 1.0]
        # Three items sit in the lowest bin (predicted 0.05, observed 0) and one
        # in the top bin (predicted 1.0, observed 1), so only the first counts.
        self.assertAlmostEqual(ece(labels, probs), 0.75 * 0.05)


class SummaryTests(unittest.TestCase):
    def test_summary_contains_curve_and_alc(self):
        result = summarise({0: 0.25, 1: 0.25})
        self.assertIn("brier_alc", result)
        self.assertIn("brier_0", result)
        self.assertAlmostEqual(result["brier_alc"], 0.25)


if __name__ == "__main__":
    unittest.main()
