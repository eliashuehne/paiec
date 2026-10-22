import math
import unittest

import numpy as np
import pandas as pd

from paiec.compare import (
    RunScores, alc_from_curves, bootstrap_interval, by_benchmark, compare_to,
    curves_from_pairs, pair_curves, paired_t_test, regularized_incomplete_beta,
    summary_table,
)
from paiec.metrics import alc_weights, brier_alc


def item_rows(benchmark, subject, probabilities, outcomes, budget=0):
    """Prediction rows with one response per item, as the harness logs them."""
    return pd.DataFrame({
        "benchmark": benchmark,
        "subject_id": subject,
        "item_id": [f"i{n}" for n in range(len(probabilities))],
        "budget": budget,
        "probability": probabilities,
        "n_responses": 1,
        "n_outcomes_one": outcomes,
    })


class PairCurveTests(unittest.TestCase):
    def test_single_response_items_give_mean_squared_error(self):
        items = item_rows("toy", "s1", [0.2, 0.9], [0, 1])
        curves = pair_curves(items)
        self.assertEqual(list(curves), [("toy", "s1")])
        expected = ((0.2 - 0) ** 2 + (0.9 - 1) ** 2) / 2
        self.assertAlmostEqual(curves[("toy", "s1")][0], expected)

    def test_multiple_responses_are_weighted_by_response_count(self):
        # One item with 3 responses (2 positive) and one with 1 response (0).
        items = pd.DataFrame({
            "benchmark": ["toy", "toy"], "subject_id": ["s1", "s1"],
            "item_id": ["a", "b"], "budget": [0, 0], "probability": [0.5, 0.5],
            "n_responses": [3, 1], "n_outcomes_one": [2, 0],
        })
        expected = (2 * (0.5 - 1) ** 2 + 1 * (0.5) ** 2 + 1 * (0.5) ** 2) / 4
        self.assertAlmostEqual(pair_curves(items)[("toy", "s1")][0], expected)

    def test_curve_matches_the_metric_helpers(self):
        items = pd.DataFrame({
            "benchmark": ["toy"] * 12, "subject_id": ["s1"] * 12,
            "item_id": [f"i{n % 2}" for n in range(12)],
            "budget": [0, 1, 3, 7, 15, 31] * 2,
            "probability": [0.3] * 6 + [0.7] * 6,
            "n_responses": [1] * 12, "n_outcomes_one": [0] * 6 + [1] * 6,
        })
        curve = pair_curves(items)[("toy", "s1")]
        self.assertEqual(sorted(curve), [0, 1, 3, 7, 15, 31])
        # i0 predicts 0.3 and is never correct; i1 predicts 0.7 and is always correct.
        manual = {budget: ((0.3 - 0) ** 2 + (0.7 - 1) ** 2) / 2 for budget in curve}
        self.assertAlmostEqual(alc_from_curves({("toy", "s1"): curve})[("toy", "s1")],
                               brier_alc(manual))

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no prediction rows"):
            pair_curves(pd.DataFrame(columns=["benchmark", "subject_id", "item_id", "budget",
                                              "probability", "n_responses", "n_outcomes_one"]))

    def test_missing_columns_are_named(self):
        with self.assertRaisesRegex(ValueError, "probability"):
            pair_curves(pd.DataFrame({"benchmark": ["toy"]}))

    def test_zero_response_rows_are_rejected(self):
        items = item_rows("toy", "s1", [0.5], [0])
        items.loc[0, "n_responses"] = 0
        with self.assertRaisesRegex(ValueError, "no responses"):
            pair_curves(items)

    def test_scores_table_and_predictions_agree(self):
        items = pd.DataFrame({
            "benchmark": ["toy"] * 6, "subject_id": ["s1"] * 6,
            "item_id": [f"i{n % 2}" for n in range(6)], "budget": [0, 1, 3, 7, 15, 31],
            "probability": [0.4] * 6, "n_responses": [1] * 6, "n_outcomes_one": [0, 1, 0, 1, 0, 1],
        })
        scores = pd.DataFrame([
            {"benchmark": "toy", "subject_id": "s1", "budget": budget, "brier": value}
            for budget, value in pair_curves(items)[("toy", "s1")].items()
        ])
        left = alc_from_curves(pair_curves(items))
        right = alc_from_curves(curves_from_pairs(scores))
        self.assertAlmostEqual(float(left.iloc[0]), float(right.iloc[0]), places=12)


class StatisticsTests(unittest.TestCase):
    def test_incomplete_beta_matches_known_values(self):
        self.assertAlmostEqual(regularized_incomplete_beta(1, 1, 0.25), 0.25)
        # I_{0.5}(0.5, 0.5) = (2/pi) * arcsin(sqrt(0.5)) = 0.5 exactly, by symmetry.
        self.assertAlmostEqual(regularized_incomplete_beta(0.5, 0.5, 0.5), 0.5, places=10)
        self.assertAlmostEqual(regularized_incomplete_beta(2, 2, 0.0), 0.0)
        self.assertAlmostEqual(regularized_incomplete_beta(2, 2, 1.0), 1.0)

    def test_incomplete_beta_matches_scipy_when_available(self):
        try:
            from scipy.special import betainc
        except ImportError:
            self.skipTest("scipy is not installed")
        for a, b, x in [(1.5, 2.5, 0.3), (5, 0.5, 0.9), (10, 10, 0.05), (0.75, 3, 0.6)]:
            self.assertAlmostEqual(regularized_incomplete_beta(a, b, x), float(betainc(a, b, x)),
                                   places=10)

    def test_paired_t_test_matches_hand_computation(self):
        # Differences 0, 1, 2 have mean 1 and sample sd 1, so t = 1 / (1/sqrt(3)) = sqrt(3).
        # For t=sqrt(3), df=2: p = I_{2/5}(1, 1/2) = 1 - sqrt(3/5).
        differences = [0.0, 1.0, 2.0]
        statistic, p = paired_t_test(differences)
        self.assertAlmostEqual(statistic, math.sqrt(3), places=10)
        self.assertAlmostEqual(p, 1.0 - math.sqrt(0.6), places=10)

    def test_paired_t_test_matches_scipy_when_available(self):
        try:
            from scipy import stats
        except ImportError:
            self.skipTest("scipy is not installed")
        rng = np.random.default_rng(7)
        differences = rng.normal(0.4, 1.0, size=12)
        statistic, p = paired_t_test(differences)
        reference = stats.ttest_1samp(differences, 0.0)
        self.assertAlmostEqual(statistic, float(reference.statistic), places=10)
        self.assertAlmostEqual(p, float(reference.pvalue), places=10)

    def test_paired_t_test_needs_two_pairs(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            paired_t_test([0.1])

    def test_flat_zero_difference_is_not_significant(self):
        self.assertEqual(paired_t_test([0.0, 0.0, 0.0]), (0.0, 1.0))

    def test_bootstrap_interval_is_seeded_and_contains_the_mean(self):
        differences = [0.1, 0.2, -0.05, 0.3, 0.15]
        first = bootstrap_interval(differences, seed=3, samples=2000)
        second = bootstrap_interval(differences, seed=3, samples=2000)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], np.mean(differences))
        self.assertGreaterEqual(first[1], np.mean(differences))


def synthetic_run(label, offsets, *, benchmark="toy", seed=0):
    """A run whose pair scores are a fixed value plus per-pair offsets."""
    curves = pd.Series({
        (f"{benchmark}", f"s{index}"): value for index, value in enumerate(offsets)
    }, name="brier_alc")
    return RunScores(label, "test", curves, None, {"run_id": label})


class CompareTests(unittest.TestCase):
    def test_delta_sign_means_worse_is_positive(self):
        reference = synthetic_run("a", [0.30, 0.40, 0.35])
        worse = synthetic_run("b", [0.35, 0.45, 0.40])
        better = synthetic_run("c", [0.25, 0.35, 0.30])
        self.assertAlmostEqual(compare_to(reference, worse).mean_delta, 0.05)
        self.assertAlmostEqual(compare_to(reference, better).mean_delta, -0.05)

    def test_wins_and_losses_count_pairs(self):
        reference = synthetic_run("a", [0.30, 0.40, 0.35])
        mixed = synthetic_run("b", [0.31, 0.39, 0.35])
        result = compare_to(reference, mixed)
        self.assertEqual((result.wins, result.losses, result.ties), (1, 1, 1))

    def test_mismatched_pairs_are_refused_without_allow_partial(self):
        reference = synthetic_run("a", [0.30, 0.40])
        other = synthetic_run("b", [0.30, 0.40], benchmark="other")
        with self.assertRaisesRegex(ValueError, "do not score the same items"):
            compare_to(reference, other)

    def test_partial_comparison_uses_shared_pairs(self):
        reference = synthetic_run("a", [0.30, 0.40])
        curves = reference.curves.copy()
        curves[("toy", "s9")] = 0.10
        other = RunScores("b", "test", curves, None, {})
        result = compare_to(reference, other, strict=False)
        self.assertEqual(result.n_pairs, 2)
        self.assertTrue(result.partial)

    def test_item_level_runs_restrict_to_shared_items(self):
        # Two pairs over the six budgets; the challenger scored only items a and b.
        budgets = [0, 1, 3, 7, 15, 31]
        reference_items = pd.DataFrame([
            {"benchmark": "toy", "subject_id": subject, "item_id": item, "budget": budget,
             "probability": 0.5, "n_responses": 1, "n_outcomes_one": label}
            for subject in ("s1", "s2")
            for item, label in [("a", 0), ("b", 1), ("c", 0), ("d", 1)]
            for budget in budgets
        ])
        challenger_items = reference_items[reference_items.item_id.isin(["a", "b"])].assign(
            probability=0.25)
        reference = RunScores("a", "test", alc_from_curves(pair_curves(reference_items)),
                              reference_items, {})
        challenger = RunScores("b", "test", alc_from_curves(pair_curves(challenger_items)),
                               challenger_items, {})
        with self.assertRaisesRegex(ValueError, "do not score the same items"):
            compare_to(reference, challenger)
        result = compare_to(reference, challenger, strict=False)
        self.assertEqual(result.n_pairs, 2)
        self.assertTrue(result.partial)
        # Only items a (label 0) and b (label 1) count, so the shared curve is flat:
        # ((0.25-0)^2 + (0.25-1)^2)/2 versus the reference's 0.25 on the same items.
        challenger_shared = ((0.25 - 0) ** 2 + (0.25 - 1) ** 2) / 2
        self.assertAlmostEqual(result.mean_delta, challenger_shared - 0.25)

    def test_summary_and_benchmark_tables(self):
        reference = synthetic_run("a", [0.30, 0.40])
        table = summary_table([reference])
        self.assertEqual(list(table.columns), ["run", "pairs", "brier_alc", "sd", "source"])
        self.assertAlmostEqual(float(table.loc[0, "brier_alc"]), 0.35)
        grouped = by_benchmark({"a": reference})
        self.assertAlmostEqual(float(grouped.loc["toy", "a"]), 0.35)


if __name__ == "__main__":
    unittest.main()
