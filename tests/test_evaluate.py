import unittest

import numpy as np
import pandas as pd

from paiec.evaluate import evaluate, score_pair
from paiec.metrics import BUDGETS
from paiec.pairs import Pair
from paiec.schema import SUBJECT_FIELDS

SUBJECT = {field: "" for field in SUBJECT_FIELDS}
SUBJECT["normalized_name"] = "toy-system"


def constant_predictor(value):
    return lambda subject, item, labeled: value


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
    def test_constant_half_scores_a_flat_quarter(self):
        rows = score_pair(make_pair(), constant_predictor(0.5), evaluation_items=50)
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

    def test_caps_change_counts_not_structure(self):
        rows = score_pair(
            make_pair(), constant_predictor(0.5), acquisition_items=40, evaluation_items=30
        )
        self.assertTrue(all(row["n_items"] == 30 for row in rows))


class EvaluateTests(unittest.TestCase):
    def test_constant_suite_has_brier_alc_quarter(self):
        pairs = [make_pair(seed=s, subject_id=f"s{s}") for s in range(3)]
        result = evaluate(pairs, constant_predictor(0.5), evaluation_items=40)
        self.assertAlmostEqual(result.brier_alc, 0.25)
        self.assertEqual(len(result.pair_alc), 3)
        self.assertTrue(np.allclose(result.per_budget.to_numpy(), 0.25))

    def test_oracle_curve_is_zero(self):
        def oracle(subject, item, labeled):
            return float(item["item_content"])

        result = evaluate([make_pair()], oracle, evaluation_items=40)
        self.assertAlmostEqual(result.brier_alc, 0.0)

    def test_by_benchmark_returns_one_row(self):
        result = evaluate([make_pair()], constant_predictor(0.5), evaluation_items=40)
        self.assertEqual(list(result.by_benchmark().index), ["toy"])


if __name__ == "__main__":
    unittest.main()
