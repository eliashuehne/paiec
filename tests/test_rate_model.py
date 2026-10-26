import unittest

import numpy as np

from paiec.competition_gate import assess
from paiec.competition_scorer import PairEpisode
from paiec.rate_model import (ITEM_FEATURES, AbilityOnlyPredictor, HierarchicalPredictor, ability_key,
                              assign_folds, difficulty_cv, empirical_logit, fit_difficulty, fit_two_way,
                              item_feature_frame, logit, predict_difficulty, sigmoid, thin_label)


def cells_from(rates, *, responses=100, direction="higher_is_better"):
    cells = []
    for index, ((name, benchmark), rate) in enumerate(rates.items()):
        cells.append({"cell_index": index, "cell_key": f"{benchmark}|{name}", "benchmark": benchmark,
                      "direction": direction, "n_responses": responses, "n_items": responses,
                      "label_sum": int(round(rate * responses)), "rate": rate,
                      "subject": {"normalized_name": name, "provider": "", "release_date": "",
                                  "access_date": "", "harness": "", "harness_version": "",
                                  "reasoning_effort": "", "subject_features_extra": ""}})
    return cells


def synthetic_two_way():
    ability = {"Model A": .8, "Model B": -.8}
    difficulty = {"easy": 1.0, "hard": -1.0}
    rates = {(name, benchmark): float(sigmoid(a + difficulty[benchmark]))
             for name, a in ability.items() for benchmark in difficulty}
    return ability, difficulty, rates


class TwoWayTests(unittest.TestCase):
    def test_ability_and_difficulty_are_recovered_in_order(self):
        _, _, rates = synthetic_two_way()
        fit = fit_two_way(cells_from(rates), name_shrinkage=0.0, benchmark_shrinkage=0.0)
        self.assertGreater(fit.ability[ability_key({"normalized_name": "Model A"})],
                           fit.ability[ability_key({"normalized_name": "Model B"})])
        self.assertGreater(fit.difficulty["easy"], fit.difficulty["hard"])
        self.assertLess(fit.residual_sd, .01)
        self.assertGreater(fit.as_report()["variance_share_explained"], .999)

    def test_extreme_cells_stay_finite(self):
        rates = {("Model A", "none"): 0.0, ("Model B", "none"): 1.0,
                 ("Model A", "all"): 1.0, ("Model B", "all"): 0.0}
        fit = fit_two_way(cells_from(rates, responses=80), name_shrinkage=0.0,
                          benchmark_shrinkage=0.0)
        for value in list(fit.ability.values()) + list(fit.difficulty.values()):
            self.assertLess(abs(value), 7.0)
        self.assertGreater(empirical_logit(0, 80), -7.0)
        self.assertGreater(empirical_logit(80, 80), 0.0)
        self.assertLess(fit.residual_sd, 7.0)

    def test_other_directions_are_excluded_from_the_ability_fit(self):
        _, _, rates = synthetic_two_way()
        cells = cells_from(rates)
        flipped = [dict(cell, direction="lower_is_better", rate=1 - cell["rate"]) for cell in cells]
        with self.assertRaises(ValueError):
            fit_two_way(flipped)
        mixed = cells + flipped
        fit = fit_two_way(mixed)
        self.assertEqual(fit.cells, 4)

    def test_canonical_names_collapse_benchmark_prefixes(self):
        self.assertEqual(ability_key({"normalized_name": "CMMLU / gemma3_27b_it"}),
                         ability_key({"normalized_name": "gemma3_27b_it"}))
        self.assertNotEqual(ability_key({"normalized_name": "GPT-4"}),
                            ability_key({"normalized_name": "GPT-4o"}))
        self.assertEqual(ability_key({"normalized_name": ""}), "")

    def test_shrinkage_pulls_small_groups_toward_the_mean(self):
        _, _, rates = synthetic_two_way()
        cells = cells_from(rates)
        weak = fit_two_way(cells, name_shrinkage=0.0, benchmark_shrinkage=0.0)
        strong = fit_two_way(cells, name_shrinkage=200.0, benchmark_shrinkage=200.0)
        key = ability_key({"normalized_name": "Model A"})
        self.assertLess(abs(strong.ability[key]), abs(weak.ability[key]))
        self.assertLess(abs(strong.difficulty["easy"]), abs(weak.difficulty["easy"]))

    def test_folds_are_deterministic_and_cover_every_benchmark(self):
        cells = cells_from({("A", f"b{index}"): .5 for index in range(9)})
        first = assign_folds(cells, 3, 11)
        self.assertEqual(first, assign_folds(cells, 3, 11))
        self.assertEqual(set(first.values()), {0, 1, 2})
        with self.assertRaises(ValueError):
            assign_folds(cells, 1, 11)


class DifficultyTests(unittest.TestCase):
    def test_regression_learns_the_driving_feature(self):
        cells, features, difficulties = [], {}, {}
        for index in range(6):
            name = f"b{index}"
            cell_key = f"{name}|A"
            vector = np.zeros(len(ITEM_FEATURES))
            vector[0] = index / 5.0
            features[cell_key] = vector
            difficulties[name] = -1.0 + index * 0.4
            cells.append({"cell_key": cell_key, "benchmark": name, "rate": .5, "n_responses": 100,
                          "direction": "higher_is_better",
                          "subject": {"normalized_name": "A"}})
        state = fit_difficulty(cells, difficulties, features, alpha=1e-6)
        self.assertGreater(state["coefficients"][0], .5)
        self.assertAlmostEqual(predict_difficulty(state, features["b0|A"]), difficulties["b0"], places=2)
        cross = difficulty_cv(cells, difficulties, features, alpha=1e-6, seed=1)
        self.assertLess(cross["mse"], cross["baseline_mse"])

    def test_extrapolation_is_bounded(self):
        state = {"alpha": 3.0, "mean": np.zeros(len(ITEM_FEATURES)),
                 "scale": np.ones(len(ITEM_FEATURES)), "coefficients": np.full(len(ITEM_FEATURES), 4.0),
                 "intercept": 0.0, "rows": 10}
        self.assertLessEqual(abs(predict_difficulty(state, np.full(len(ITEM_FEATURES), 1e6))), 4.0)
        self.assertEqual(predict_difficulty({"intercept": 9.0}, None), 4.0)

    def test_near_constant_features_keep_unit_scale(self):
        cells, features, difficulties = [], {}, {}
        for index in range(4):
            cell_key = f"b{index}|A"
            vector = np.zeros(len(ITEM_FEATURES))
            vector[0] = index / 3.0
            features[cell_key] = vector
            difficulties[f"b{index}"] = float(index) - 1.5
            cells.append({"cell_key": cell_key, "benchmark": f"b{index}", "rate": .5,
                          "label_sum": 50, "n_responses": 100, "direction": "higher_is_better",
                          "subject": {"normalized_name": "A"}})
        state = fit_difficulty(cells, difficulties, features, alpha=1.0)
        self.assertTrue(all(value >= 1e-6 for value in state["scale"]))
        self.assertLessEqual(abs(predict_difficulty(state, np.full(len(ITEM_FEATURES), 1e9))), 4.0)

    def test_missing_features_fall_back_to_the_intercept(self):
        state = fit_difficulty([], {}, {}, alpha=3.0)
        self.assertEqual(predict_difficulty(state, None), 0.0)
        self.assertEqual(len(state["coefficients"]), len(ITEM_FEATURES))


class StubStore:
    """Each item differs in the first task feature."""

    def matrix(self, benchmark, item_ids):
        matrix = np.tile(np.linspace(0, 1, len(ITEM_FEATURES)), (len(item_ids), 1))
        matrix[:, 0] = np.linspace(-1, 1, len(item_ids))
        return matrix


class StubModel:
    """Two-column design: logit prior plus the first task feature."""

    def predict_proba(self, design):
        probability = sigmoid(design[:, 0] + 2 * design[:, 1])
        return np.column_stack([1 - probability, probability])


def episode(subject="Model A"):
    return PairEpisode(cell_index=0, cell_key="easy|Model A", benchmark="easy", benchmark_id="benchmark_e",
                       subject_key=ability_key({"normalized_name": subject}),
                       acquisition=("a1", "a2", "a3"), evaluation=("e1", "e2"),
                       n_responses={"a1": 1, "a2": 1, "a3": 1, "e1": 1, "e2": 1},
                       label_sum={"a1": 1, "a2": 0, "a3": 1, "e1": 1, "e2": 0})


class PredictorTests(unittest.TestCase):
    def setUp(self):
        _, _, rates = synthetic_two_way()
        self.fit = fit_two_way(cells_from(rates))

    def test_budget_zero_is_the_prior_and_ignores_difficulty_when_asked(self):
        with_difficulty = HierarchicalPredictor(self.fit, {"intercept": 0.0}, {"easy|Model A": .5})
        without = AbilityOnlyPredictor(self.fit)
        self.assertGreater(with_difficulty.pair_prior(episode()), without.pair_prior(episode()))
        self.assertTrue(np.allclose(without.base(episode()), without.pair_prior(episode())))

    def test_an_unknown_name_falls_back_to_the_benchmark_term(self):
        predictor = HierarchicalPredictor(self.fit, {"intercept": 0.0}, {"easy|Model A": .5})
        known = predictor.pair_prior(episode())
        unknown = predictor.pair_prior(episode("Unseen Model"))
        self.assertNotAlmostEqual(known, unknown)

    def test_evidence_preserves_within_pair_order(self):
        predictor = HierarchicalPredictor(self.fit, {"intercept": 0.0}, {"easy|Model A": .5},
                                          store=StubStore(), item_model=StubModel())
        base = predictor.base(episode())
        self.assertGreater(np.ptp(base), .01)
        shifted = predictor.update(base, [1, 1, 1], [1])
        self.assertGreater(shifted.mean(), base.mean())
        self.assertTrue(np.all(np.diff(shifted) > 0))
        self.assertTrue(np.allclose(np.sort(shifted), shifted))
        self.assertTrue(np.all((shifted > 0) & (shifted < 1)))

    def test_no_evidence_reproduces_the_base(self):
        predictor = HierarchicalPredictor(self.fit, {"intercept": 0.0}, {"easy|Model A": .5},
                                          store=StubStore(), item_model=StubModel())
        base = predictor.base(episode())
        self.assertTrue(np.array_equal(predictor.update(base, [], []), base))

    def test_benchmark_weight_scales_the_shared_evidence(self):
        strong = HierarchicalPredictor(self.fit, {"intercept": 0.0}, {"easy|Model A": 0.0},
                                       prior_strength=5.0, benchmark_weight=1.0)
        mild = HierarchicalPredictor(self.fit, {"intercept": 0.0}, {"easy|Model A": 0.0},
                                     prior_strength=5.0, benchmark_weight=0.1)
        base = mild.base(episode())
        self.assertLess(strong.update(base, [], [0, 0, 0, 0]).mean(),
                        mild.update(base, [], [0, 0, 0, 0]).mean())
        self.assertLess(strong.update(base, [], [0, 0, 0, 0]).mean(), base.mean())

    def test_thinning_keeps_extreme_items_exact(self):
        self.assertEqual(thin_label("c", "i", 3, 3, 1), 1)
        self.assertEqual(thin_label("c", "i", 3, 0, 1), 0)
        values = [thin_label("c", f"i{index}", 4, 1, 7) for index in range(200)]
        self.assertAlmostEqual(sum(values) / 200, .25, delta=.08)


class GateTests(unittest.TestCase):
    def summary(self, model, acquired, constant, fixed=.22):
        curve = {str(budget): value for budget, value in
                 zip((0, 1, 3, 7, 15, 31), (constant, constant, constant, constant, constant, constant))}
        return {"arms": {"all": {
            "hierarchical_rate": {"brier_alc_mean": model, "pairs": 400,
                                  "curve_mean": {**curve, "0": model}},
            "acquired_label_mean": {"brier_alc_mean": acquired, "curve_mean": curve},
            "fixed_beta_k5": {"brier_alc_mean": fixed, "curve_mean": curve},
            "pooled_plus_own": {"brier_alc_mean": fixed + .005, "curve_mean": curve},
            "constant_0.5": {"brier_alc_mean": constant, "curve_mean": curve}}},
            "folds": [{"regime": "all", "arms": {
                "hierarchical_rate": {"brier_alc": model},
                "acquired_label_mean": {"brier_alc": acquired}}} for _ in range(4)]}

    def test_a_model_that_loses_to_the_label_mean_is_not_promoted(self):
        result = assess(self.summary(.19, .15, .25))
        self.assertEqual(result["verdict"], "no_gain_over_revealed_label_mean")
        self.assertFalse(result["release_candidate"])
        self.assertIn("alc_beats_revealed_label_mean", result["failed"])

    def test_a_model_that_beats_the_controls_clears(self):
        result = assess(self.summary(.12, .15, .25))
        self.assertEqual(result["verdict"], "clears_matched_adaptive_control")
        self.assertFalse(result["submission_ready"])

    def test_a_model_that_loses_to_a_fixed_control_is_rejected(self):
        result = assess(self.summary(.21, .25, .25, fixed=.20))
        self.assertEqual(result["verdict"], "no_gain_over_fixed_controls")
        self.assertIn("alc_beats_strongest_fixed_control", result["failed"])
        self.assertTrue(any(check["check"] == "alc_beats_strongest_fixed_control"
                            for check in result["checks"]))

    def test_too_few_folds_is_inconclusive(self):
        summary = self.summary(.12, .15, .25)
        summary["folds"] = summary["folds"][:1]
        result = assess(summary)
        self.assertEqual(result["verdict"], "inconclusive")

    def test_a_run_without_the_reference_arm_fails_closed(self):
        with self.assertRaises(ValueError):
            assess({"arms": {"all": {}}, "folds": []})


class PairEpisodeRateTests(unittest.TestCase):
    def test_pair_rate_uses_every_recorded_response(self):
        pair = PairEpisode(cell_index=0, cell_key="c", benchmark="b", benchmark_id="benchmark_b",
                           subject_key="s", acquisition=("a",), evaluation=("e1", "e2"),
                           n_responses={"a": 2, "e1": 1, "e2": 3}, label_sum={"a": 1, "e1": 1, "e2": 1})
        self.assertAlmostEqual(pair.rate, .5)
        self.assertEqual(pair.n_evaluation_responses, 4)


class FoldPersistenceTests(unittest.TestCase):
    def test_persisted_fold_rows_restore_and_summarize(self):
        import json

        from scripts import run_rate_model
        from paiec.metrics import BUDGETS

        def fold_row(fold, base):
            return {"fold": fold, "regime": "all", "held_benchmarks": ["b"], "pairs": 100,
                    "arms": {name: {"brier_alc": base + offset / 100, "auroc": .5,
                                    "pairs": 100,
                                    "curve": {str(budget): base + offset / 100
                                              for budget in BUDGETS}}
                             for offset, name in enumerate(run_rate_model.DEVICES)}}

        rows = [run_rate_model.restore_row(json.loads(json.dumps(fold_row(fold, .2 + fold / 100))))
                for fold in range(3)]
        arms = run_rate_model.summarize_arms(rows)
        self.assertEqual(len(rows[0]["arms"]["constant_0.5"]["curve"]), len(BUDGETS))
        self.assertAlmostEqual(arms["acquired_label_mean"]["difference_from_acquired_mean"], 0.)
        expected = float(np.mean([row["arms"]["constant_0.5"]["brier_alc"] for row in rows]))
        self.assertAlmostEqual(arms["constant_0.5"]["brier_alc_mean"], expected)
        self.assertEqual(arms["hierarchical_rate"]["pairs"], 300)
        self.assertEqual(set(arms["hierarchical_rate"]["curve_mean"]), {str(b) for b in BUDGETS})
        self.assertAlmostEqual(arms["hierarchical_rate"]["curve_mean"]["0"],
                               float(np.mean([row["arms"]["hierarchical_rate"]["brier_alc"]
                                              for row in rows])))


if __name__ == "__main__":
    unittest.main()
