import unittest

import numpy as np
import pandas as pd

from paiec.lift import LiftResult
from paiec.lift_budget import (
    BUDGET_BASELINES,
    attach_reveal_order,
    budget_frame,
    combine_budgets,
    empirical_mean,
    reveal_order,
    run_budgets,
    shrunk_offset,
    shrunk_rate,
)
from paiec.lift_sample import SEALED
from paiec.metrics import alc_weights
from paiec.split import make_split

FOLDS = [str(k) for k in range(5)] * 5 + [SEALED] * 5


def budget_sample(seed=0, systems=6, items=40):
    """Benchmarks differ a lot in level; systems differ within each, by a visible ``skill``."""
    rng = np.random.default_rng(seed)
    parts = []
    for number, fold in enumerate(FOLDS):
        level = rng.normal(scale=2)
        for system in range(systems):
            skill = 0.8 * rng.integers(-1, 2)  # a shared level, like a model generation
            p = 1 / (1 + np.exp(-(level + skill)))
            parts.append(pd.DataFrame({
                "benchmark": f"b{number:02d}",
                "subject_id": f"b{number:02d}_s{system}",
                "item_id": [f"b{number:02d}_i{k}" for k in range(items)],
                "trial": 1,
                "label": (rng.random(items) < p).astype(int),
                "fold": fold,
                "skill": skill,
            }))
    return pd.concat(parts, ignore_index=True)


def tiny():
    """Two systems on one benchmark, four items each; s1 answers i3 twice."""
    return pd.DataFrame({
        "benchmark": "x",
        "subject_id": ["s0"] * 4 + ["s1"] * 5,
        "item_id": ["i0", "i1", "i2", "i3", "i0", "i1", "i2", "i3", "i3"],
        "trial": [1, 1, 1, 1, 1, 1, 1, 1, 2],
        "label": [1, 0, 1, 1, 0, 0, 1, 1, 0],
        "fold": "0",
    })


def scored_keys(frame):
    return set(map(tuple, frame[["subject_id", "item_id"]].to_numpy()))


class RevealOrderTests(unittest.TestCase):
    def test_matches_the_official_split(self):
        order = reveal_order(tiny()).query("subject_id == 's1'")
        split = make_split(["i0", "i1", "i2", "i3"], benchmark="x", subject_id="s1")
        reveal = order[order["pool"] == "reveal"].sort_values("order")["item_id"].tolist()
        self.assertEqual(tuple(reveal), split.acquisition)
        self.assertEqual(set(order.loc[order["pool"] == "evaluate", "item_id"]), set(split.evaluation))

    def test_single_item_pair_is_only_evaluated(self):
        frame = tiny().iloc[:1]
        self.assertEqual(reveal_order(frame)["pool"].tolist(), ["evaluate"])

    def test_sealed_rows_get_no_pool(self):
        frame = pd.concat([tiny(), tiny().assign(benchmark="y", fold=SEALED)], ignore_index=True)
        keyed = attach_reveal_order(frame)
        self.assertTrue(keyed.loc[keyed["fold"] == SEALED, "pool"].isna().all())
        self.assertTrue(keyed.index.equals(frame.index))


class BudgetFrameTests(unittest.TestCase):
    def setUp(self):
        self.keyed = attach_reveal_order(budget_sample())

    def test_scored_rows_are_the_evaluation_pool_at_every_budget(self):
        evaluate = self.keyed[self.keyed["pool"] == "evaluate"]
        reveal = scored_keys(self.keyed[self.keyed["pool"] == "reveal"])
        for budget in (0, 1, 7, 31):
            frame = budget_frame(self.keyed, budget)
            self.assertTrue(frame.index.equals(evaluate.index))
            self.assertFalse(scored_keys(frame) & reveal)

    def test_pair_count_is_budget_capped_by_the_pool(self):
        pool = self.keyed[self.keyed["pool"] == "reveal"].groupby("subject_id").size()
        for budget in (0, 3, 31):
            frame = budget_frame(self.keyed, budget)
            expected = frame["subject_id"].map(pool).clip(upper=budget)
            np.testing.assert_array_equal(frame["pair_revealed_n"], expected)

    def test_budget_zero_reveals_nothing(self):
        frame = budget_frame(self.keyed, 0)
        self.assertTrue((frame[["bench_revealed_n", "pair_revealed_n"]] == 0).all().all())
        self.assertTrue(frame[["bench_revealed_rate", "pair_revealed_rate"]].isna().all().all())

    def test_scored_labels_do_not_change_revealed_columns(self):
        flipped = self.keyed.copy()
        evaluate = flipped["pool"] == "evaluate"
        flipped.loc[evaluate, "label"] = 1 - flipped.loc[evaluate, "label"]
        columns = ["bench_revealed_rate", "bench_revealed_n", "pair_revealed_rate", "pair_revealed_n"]
        for budget in (1, 7):
            pd.testing.assert_frame_equal(budget_frame(self.keyed, budget)[columns],
                                          budget_frame(flipped, budget)[columns])

    def test_known_values_and_one_label_per_revealed_item(self):
        keyed = attach_reveal_order(tiny())
        frame = budget_frame(keyed, 2)
        revealed = keyed[keyed["pool"] == "reveal"].sort_values("trial").drop_duplicates(
            ["subject_id", "item_id"])
        self.assertEqual(frame["bench_revealed_n"].iloc[0], 4)  # 2 items per system, i3 once
        self.assertEqual(len(revealed), 4)
        self.assertAlmostEqual(frame["bench_revealed_rate"].iloc[0], revealed["label"].mean())
        # s1 answered i3 twice: revealed once (trial 1) and never scored, or both trials scored.
        s1_i3 = keyed[(keyed["subject_id"] == "s1") & (keyed["item_id"] == "i3")]
        scored = frame[(frame["subject_id"] == "s1") & (frame["item_id"] == "i3")]
        if s1_i3["pool"].iloc[0] == "reveal":
            self.assertTrue(scored.empty)
            self.assertEqual(revealed.loc[revealed["subject_id"] == "s1", "trial"].max(), 1)
        else:
            self.assertEqual(sorted(scored["trial"]), [1, 2])

    def test_negative_budget_raises(self):
        with self.assertRaises(ValueError):
            budget_frame(self.keyed, -1)


class RuleTests(unittest.TestCase):
    TRAIN = pd.DataFrame({"label": [1, 1, 1, 0]})

    def rows(self, bench_rate, bench_n, pair_rate, pair_n):
        return pd.DataFrame({"bench_revealed_rate": [bench_rate], "bench_revealed_n": [bench_n],
                             "pair_revealed_rate": [pair_rate], "pair_revealed_n": [pair_n]})

    def test_budget_zero_gives_half_and_the_training_rate(self):
        rows = self.rows(np.nan, 0, np.nan, 0)
        self.assertEqual(empirical_mean(self.TRAIN, rows, ())[0], 0.5)
        self.assertAlmostEqual(shrunk_rate(self.TRAIN, rows, ())[0], 0.75)

    def test_shrunk_rate_known_value(self):
        rows = self.rows(0.2, 20, 1.0, 5)
        bench = (4 + 20 * 0.75) / 40
        self.assertAlmostEqual(shrunk_rate(self.TRAIN, rows, ())[0], (5 + 5 * bench) / 10)


class RunBudgetTests(unittest.TestCase):
    BUDGETS = (0, 1, 7)

    @classmethod
    def setUpClass(cls):
        cls.sample = budget_sample()
        cls.runs = {
            name: run_budgets(cls.sample, cls.BUDGETS, **vars(BUDGET_BASELINES[name]))
            for name in ("A", "E", "K", "KM")
        }

    def test_scored_rows_are_identical_across_budgets_and_baselines(self):
        reference = self.runs["A"][0].benchmarks[["fold", "benchmark", "rows"]]
        for runs in self.runs.values():
            for result in runs.values():
                self.assertTrue(result.benchmarks[["fold", "benchmark", "rows"]].equals(reference))

    def test_k_equals_a_at_budget_zero(self):
        np.testing.assert_allclose(self.runs["K"][0].folds["brier"], self.runs["A"][0].folds["brier"])

    def test_revealed_labels_beat_the_training_rate(self):
        for name in ("K", "KM"):
            for budget in (1, 7):
                better = self.runs[name][budget].folds["brier"] < self.runs["A"][budget].folds["brier"]
                self.assertTrue(better.all(), f"{name} at budget {budget}")

    def test_km_without_features_is_k(self):
        for budget in self.BUDGETS:
            np.testing.assert_allclose(self.runs["KM"][budget].folds["brier"], self.runs["K"][budget].folds["brier"])

    def test_within_benchmark_signal_improves_km(self):
        # A shared system level (like a model generation) helps most at low
        # budgets; with more labels the pair's own revealed rate covers it.
        with_skill = run_budgets(self.sample, (1,), columns=["skill"], predictor="shrunk_offset")
        gain = self.runs["KM"][1].folds["brier"] - with_skill[1].folds["brier"]
        self.assertGreater(gain.mean(), 0.002)
        self.assertGreaterEqual((gain > 0).sum(), 4)

    def test_sealed_labels_do_not_change_the_scores(self):
        flipped = self.sample.copy()
        sealed = flipped["fold"] == SEALED
        flipped.loc[sealed, "label"] = 1 - flipped.loc[sealed, "label"]
        km_skill = {"columns": ["skill"], "predictor": "shrunk_offset"}
        pd.testing.assert_frame_equal(
            run_budgets(flipped, (3,), **km_skill)[3].folds, run_budgets(self.sample, (3,), **km_skill)[3].folds
        )


class MonotoneTests(unittest.TestCase):
    def test_increasing_column_never_lowers_the_prediction(self):
        # The true relation is decreasing; the constraint must keep predictions non-decreasing.
        frame = budget_frame(attach_reveal_order(budget_sample()), 3)
        frame = frame.assign(feature=-frame["skill"])
        train, test = frame[frame["fold"] != "0"], frame[frame["fold"] == "0"]
        grid = test.iloc[[0] * 5].assign(feature=np.linspace(-1, 1, 5))
        p = shrunk_offset(train, grid, ["feature"], increasing=["feature"])
        self.assertTrue(np.all(np.diff(p) >= -1e-12))


class CombineBudgetTests(unittest.TestCase):
    def result(self, brier):
        return LiftResult(
            (), (),
            pd.DataFrame({"fold": ["0"], "benchmarks": [1], "rows": [10], "brier": [brier],
                          "log_loss": [brier], "brier_macro": [brier]}),
            pd.DataFrame({"fold": ["0"], "benchmark": ["b"], "rows": [10], "brier": [brier], "log_loss": [brier]}),
        )

    def test_official_weights(self):
        budgets = (0, 1, 3, 7, 15, 31)
        values = [0.30, 0.25, 0.22, 0.20, 0.19, 0.18]
        combined = combine_budgets({b: self.result(v) for b, v in zip(budgets, values)})
        self.assertAlmostEqual(combined.folds["brier"].iloc[0], float(np.dot(alc_weights(budgets), values)))
        np.testing.assert_allclose(alc_weights(budgets), [0.1, 0.2, 0.2, 0.2, 0.2, 0.1])

    def test_single_budget_is_returned_as_is(self):
        result = self.result(0.2)
        self.assertIs(combine_budgets({3: result}), result)

    def test_different_rows_raise(self):
        other = self.result(0.2)
        other.folds.loc[0, "rows"] = 11
        with self.assertRaises(ValueError):
            combine_budgets({0: self.result(0.2), 1: other})


if __name__ == "__main__":
    unittest.main()
