import unittest

import numpy as np
import pandas as pd

from paiec.evaluate import evaluate, make_pools, score_pair
from paiec.metrics import BUDGETS
from paiec.pairs import Pair
from paiec.schema import SUBJECT_FIELDS
from paiec.split import make_split

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


def named_pair(name, benchmark="toy", seed=0) -> Pair:
    """A pair whose visible subject record is unique, unlike ``make_pair``."""
    pair = make_pair(seed=seed, subject_id=name)
    subject = dict(SUBJECT, normalized_name=name)
    items = pair.items.assign(content=pair.items.index)  # content = item id, for tracing
    return Pair(benchmark, name, subject, items, pair.responses, pair.item_ids)


def recorder(log):
    def predict(subject, item, labeled):
        log.append((subject["normalized_name"], item["benchmark_id"], list(labeled)))
        return 0.5

    return predict


class PoolingTests(unittest.TestCase):
    def test_pool_of_one_matches_own_labels(self):
        pairs = [named_pair(f"s{k}", seed=k) for k in range(3)]

        def own_rate(subject, item, labeled):
            own = [y for (s, _), y in labeled if s == subject]
            return (sum(own) + 1) / (len(own) + 2)

        default = evaluate(pairs, own_rate, evaluation_items=30)
        pooled = evaluate(pairs, own_rate, evaluation_items=30, pool_size=3)
        np.testing.assert_allclose(default.pair_alc, pooled.pair_alc)

    def test_each_pair_sees_its_pool_at_the_same_budget(self):
        pairs = [named_pair(f"s{k}", seed=k) for k in range(4)]
        log = []
        evaluate(pairs, recorder(log), evaluation_items=10, pool_size=2)
        pools = {frozenset(pair.subject_id for pair in pool) for pool in make_pools(pairs, 2)}
        for target, _, labeled in log:
            subjects = {s["normalized_name"] for (s, _), _ in labeled}
            budget = len(labeled) // 2
            self.assertIn(len(labeled), {2 * b for b in BUDGETS})
            if budget:
                self.assertIn(frozenset(subjects), pools)
                self.assertIn(target, subjects)
                for name in subjects:
                    self.assertEqual(
                        sum(s["normalized_name"] == name for (s, _), _ in labeled), budget
                    )

    def test_own_evaluation_items_are_never_revealed_as_own_labels(self):
        pairs = [named_pair(f"s{k}", seed=k) for k in range(3)]
        log = []
        evaluate(pairs, recorder(log), evaluation_items=20, pool_size=None)
        evaluation = {
            pair.subject_id: set(
                make_split(pair.item_ids, benchmark="toy", subject_id=pair.subject_id).evaluation
            )
            for pair in pairs
        }
        others_cover_own_evaluation = False
        for target, _, labeled in log:
            own = [item["item_content"] for (s, item), _ in labeled
                   if s["normalized_name"] == target]
            others = {item["item_content"] for (s, item), _ in labeled
                      if s["normalized_name"] != target}
            self.assertEqual(len(own), len(labeled) // 3)
            self.assertFalse(set(own) & evaluation[target])
            others_cover_own_evaluation |= bool(others & evaluation[target])
        # Other systems may reveal the target's evaluation items: that is allowed.
        self.assertTrue(others_cover_own_evaluation)

    def test_pools_never_mix_benchmarks_and_cover_every_pair(self):
        pairs = [named_pair(f"a{k}", "alpha", seed=k) for k in range(5)]
        pairs += [named_pair(f"b{k}", "beta", seed=k) for k in range(3)]
        pools = make_pools(pairs, 2, seed=0)
        self.assertEqual(sorted(len(pool) for pool in pools), [1, 1, 2, 2, 2])
        self.assertTrue(all(len({pair.benchmark for pair in pool}) == 1 for pool in pools))
        self.assertEqual(sorted(p.subject_id for pool in pools for p in pool),
                         sorted(p.subject_id for p in pairs))
        self.assertEqual([len(pool) for pool in make_pools(pairs, None)], [5, 3])

    def test_invalid_pool_size_raises(self):
        with self.assertRaises(ValueError):
            evaluate([named_pair("s0")], constant_predictor(0.5), pool_size=0)


if __name__ == "__main__":
    unittest.main()
