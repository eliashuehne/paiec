import unittest

import numpy as np
import pandas as pd

from paiec.evaluate import evaluate, score_pair
from paiec.metrics import BUDGETS
from paiec.pairs import Pair
from paiec.predictors import Constant
from paiec.schema import SUBJECT_FIELDS, ITEM_FIELDS

SUBJECT = {field: "" for field in SUBJECT_FIELDS}
SUBJECT["normalized_name"] = "toy-system"


def make_pair(n=100, seed=0, subject_id="s0") -> Pair:
    rng = np.random.default_rng(seed)
    item_ids = [f"item{k:03d}" for k in range(n)]
    labels = rng.integers(0, 2, size=n).astype(int)
    items = pd.DataFrame(
        {
            "content": [str(int(y)) for y in labels],
            "item_features": [""] * n,
            "interactors": [""] * n,
        },
        index=item_ids,
    )
    responses = pd.DataFrame({"item_id": item_ids, "label": labels})
    return Pair(
        benchmark="toy",
        subject_id=subject_id,
        subject=dict(SUBJECT),
        items=items,
        responses=responses,
        item_ids=tuple(item_ids),
    )


class ScorePairTests(unittest.TestCase):
    def test_predictor_boundary_has_anonymous_ids_and_budget_limited_evidence(self):
        pair = make_pair()
        pair.subject["display_name"] = "SECRET_DISPLAY"
        pair.subject["normalized_name"] = ""
        seen_budgets = set()

        def recorder(subject, item, labeled):
            self.assertEqual(tuple(subject), SUBJECT_FIELDS)
            self.assertEqual(tuple(item), ITEM_FIELDS)
            self.assertEqual(subject["normalized_name"], "")
            self.assertRegex(item["benchmark_id"], r"^benchmark_[0-9a-f]{16}$")
            self.assertNotEqual(item["benchmark_id"], pair.benchmark)
            for (observed_subject, observed_item), label in labeled:
                self.assertEqual(observed_subject, subject)
                self.assertEqual(tuple(observed_item), ITEM_FIELDS)
                self.assertEqual(observed_item["benchmark_id"], item["benchmark_id"])
                self.assertIn(label, (0, 1))
            seen_budgets.add(len(labeled))
            return .5

        score_pair(pair, recorder, evaluation_items=8, acquisition_items=40)
        self.assertEqual(seen_budgets, set(BUDGETS))

    def test_constant_half_scores_a_flat_quarter(self):
        rows = score_pair(make_pair(), Constant(0.5), evaluation_items=50)
        self.assertEqual([row["budget"] for row in rows], list(BUDGETS))
        for row in rows:
            self.assertAlmostEqual(row["brier"], 0.25)

    def test_oracle_scores_zero(self):
        # Item content encodes its own label, so a perfect predictor is trivial.
        def oracle(subject, item, labeled):
            return float(item["item_content"])

        rows = score_pair(make_pair(), oracle, evaluation_items=50)
        for row in rows:
            self.assertAlmostEqual(row["brier"], 0.0)

    def test_labels_grow_with_the_budget(self):
        seen = []

        def recorder(subject, item, labeled):
            seen.append(len(labeled))
            return 0.5

        score_pair(make_pair(), recorder, evaluation_items=50)
        # One call per evaluation item per budget.
        unique = sorted(set(seen))
        self.assertEqual(unique, list(BUDGETS))

    def test_out_of_range_probability_raises(self):
        def broken(subject, item, labeled):
            return 1.5

        with self.assertRaises(ValueError):
            score_pair(make_pair(), broken, evaluation_items=20)

    def test_batch_predictions_match_scalar_scoring(self):
        class BatchOracle:
            def predict_many(self, subject, items, labeled):
                return [float(item["item_content"]) for item in items]

        result = score_pair(make_pair(), BatchOracle(), evaluation_items=20)
        self.assertTrue(all(row["brier"] == 0 for row in result))

    def test_batch_cannot_silently_drop_an_item(self):
        class BrokenBatch:
            def predict_many(self, subject, items, labeled):
                return [0.5]

        with self.assertRaisesRegex(ValueError, "wrong number"):
            score_pair(make_pair(), BrokenBatch(), evaluation_items=20)

    def test_caps_change_counts_not_structure(self):
        rows = score_pair(
            make_pair(), Constant(0.5), acquisition_items=40, evaluation_items=30
        )
        self.assertTrue(all(row["n_items"] == 30 for row in rows))

    def test_logged_predictions_reconstruct_scores_without_acquisition_items(self):
        records = []
        rows = score_pair(make_pair(), Constant(.2), evaluation_items=20,
                          prediction_records=records)
        captured = pd.DataFrame(records)
        self.assertEqual(len(captured), 20 * len(BUDGETS))
        for row in rows:
            values = captured[captured.budget.eq(row["budget"])]
            n, hits, p = values.n_responses, values.n_outcomes_one, values.probability
            reconstructed = (hits * (1 - p) ** 2 + (n - hits) * p ** 2).sum() / n.sum()
            self.assertAlmostEqual(reconstructed, row["brier"])


class EvaluateTests(unittest.TestCase):
    def test_constant_suite_has_brier_alc_quarter(self):
        pairs = [make_pair(seed=s, subject_id=f"s{s}") for s in range(3)]
        result = evaluate(pairs, Constant(0.5), evaluation_items=40)
        self.assertAlmostEqual(result.brier_alc, 0.25)
        self.assertEqual(len(result.pair_alc), 3)
        self.assertTrue(np.allclose(result.per_budget.to_numpy(), 0.25))

    def test_oracle_curve_is_zero(self):
        def oracle(subject, item, labeled):
            return float(item["item_content"])

        result = evaluate([make_pair()], oracle, evaluation_items=40)
        self.assertAlmostEqual(result.brier_alc, 0.0)

    def test_by_benchmark_returns_one_row(self):
        result = evaluate([make_pair()], Constant(0.5), evaluation_items=40)
        self.assertEqual(list(result.by_benchmark().index), ["toy"])


if __name__ == "__main__":
    unittest.main()
