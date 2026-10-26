import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from paiec.budget_evaluation import curve_scores, full_protocol_episodes
from paiec.catboost_gate import assess_catboost_base
from paiec.catboost_training import (CatBoostClassifier, checked_configs, log_loss, scope_weights,
                                     select_configuration)
from paiec.feature_preparation import write_jsonl
from paiec.metrics import BUDGETS

CONFIGS = [{"depth": 2, "learning_rate": .1, "l2_leaf_reg": 3.0},
           {"depth": 3, "learning_rate": .05, "l2_leaf_reg": 10.0}]


def synthetic_frame(families=("a", "b", "c", "d"), rows_per_family=24):
    rng = np.random.default_rng(11)
    records = []
    for family in families:
        sign = 1.0 if family != "d" else -1.0
        for index in range(rows_per_family):
            value = float(rng.normal())
            records.append({"family": family, "value": value,
                            "shape": family if index % 2 else "shared",
                            "scope": family + ("_0" if index % 2 else "_1"),
                            "group": family + str(index),
                            "label": float(sign * value > 0)})
    return pd.DataFrame(records)


class CatboostHelperTests(unittest.TestCase):
    def test_configurations_and_scope_weights_are_declared_and_balanced(self):
        self.assertEqual(checked_configs(CONFIGS), CONFIGS)
        for invalid in [[], [{"depth": 1, "learning_rate": .1, "l2_leaf_reg": 3.}],
                        [{"depth": 3, "learning_rate": 0, "l2_leaf_reg": 3.}],
                        [{"depth": 3, "learning_rate": .1, "l2_leaf_reg": -1.}],
                        [{"depth": 3, "learning_rate": .1}], CONFIGS + CONFIGS[:1]]:
            with self.assertRaises(ValueError):
                checked_configs(invalid)
        weights = scope_weights(["a", "a", "a", "b"])
        self.assertAlmostEqual(float(weights.mean()), 1.)
        self.assertEqual([round(value, 6) for value in weights], [.666667, .666667, .666667, 2.])

    def test_log_loss_matches_known_values(self):
        self.assertAlmostEqual(log_loss([0, 1], [.5, .5]), np.log(2), places=9)
        self.assertLess(log_loss([0, 1], [.1, .9]), log_loss([0, 1], [.5, .5]))
        with self.assertRaises(ValueError):
            log_loss([0, 1], [.5])


class SharedBudgetScorerTests(unittest.TestCase):
    def fixture(self, root, episodes):
        (root / "evidence_plan_v1").mkdir(parents=True)
        write_jsonl(root / "evidence_plan_v1/evidence_episode_links.jsonl", episodes)

    def test_small_scopes_are_excluded_and_isolation_violations_fail(self):
        rows = [{"row_key": "e" + str(i)} for i in range(80)] + [{"row_key": "s" + str(i)} for i in range(40)]
        scopes = np.asarray(["eligible"] * 80 + ["small"] * 40)
        families = np.asarray(["f"] * 120)
        episodes = [{"scope_key": "eligible", "acquisition_task_groups": [f"q{i}" for i in range(40)],
                     "query_task_groups": [f"q{i}" for i in range(40, 80)],
                     "support_order_row_keys": [f"e{i}" for i in range(31)],
                     "query_row_keys": [f"e{i}" for i in range(40, 80)]},
                    {"scope_key": "small", "acquisition_task_groups": [f"r{i}" for i in range(20)],
                     "query_task_groups": [f"r{i}" for i in range(20, 40)],
                     "support_order_row_keys": [f"s{i}" for i in range(5)],
                     "query_row_keys": [f"s{i}" for i in range(20, 40)]}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "prepared"
            self.fixture(root, episodes)
            kept, excluded = full_protocol_episodes(root, rows, scopes, families)
            self.assertEqual([episode["scope_key"] for episode in kept], ["eligible"])
            self.assertEqual(excluded[0]["scope_key"], "small")
            self.assertEqual(len(kept[0]["support"]), 31)
            self.assertEqual(kept[0]["validation_family"], "f")
            episodes[0] = dict(episodes[0], query_row_keys=["e0"] + episodes[0]["query_row_keys"][1:])
            write_jsonl(root / "evidence_plan_v1/evidence_episode_links.jsonl", episodes)
            with self.assertRaisesRegex(ValueError, "isolation"):
                full_protocol_episodes(root, rows, scopes, families)
            (root / "evidence_plan_v1/evidence_episode_links.jsonl").unlink()
            write_jsonl(root / "evidence_plan_v1/evidence_episode_links.jsonl", [episodes[1]])
            with self.assertRaisesRegex(ValueError, "complete protocol"):
                full_protocol_episodes(root, rows, scopes, families)

    def test_budgets_zero_reproduce_base_and_adaptation_preserves_order(self):
        labels = np.asarray([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
        episode = {"scope_key": "s", "validation_family": "f", "support": list(range(4)), "query": [4, 5, 6, 7]}
        base = np.asarray([.4, .6, .3, .7, .5, .5, .5, .5])
        scored, curves, alc = curve_scores([episode], labels, {"catboost": base}, prior_strength=5.,
                                          offset_regularization=1.)
        self.assertEqual(len(scored), len(BUDGETS) * 6)
        self.assertAlmostEqual(alc["constant"], .25)
        self.assertAlmostEqual(curves["catboost/none"][0], float(np.mean((base[4:] - labels[4:]) ** 2)))
        self.assertAlmostEqual(curves["catboost/logit_shift"][0], curves["catboost/none"][0])
        expected = (5 * base[4:] + float(labels[:3].sum())) / (5 + 3)
        np.testing.assert_allclose([row["brier"] for row in scored if row["method"] == "catboost/beta_mean"
                                    and row["budget"] == 3],
                                   [float(np.mean((expected - labels[4:]) ** 2))], rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            curve_scores([episode], labels, {"bad": np.asarray([1.5] * 8)})


class CatboostGateTests(unittest.TestCase):
    def report(self, cold, alc, **changes):
        value = {"source_policy": "provided_dataset_only", "training_families": 7,
                 "original_development_outcomes_used": False, "monitor_outcomes_used": False,
                 "sealed_roles_opened": False, "visible_scopes": 170,
                 "full_budget_eligible_scopes": 170, "excluded_small_scopes": 0,
                 "zero_budget_brier_all_held_rows_equal_scope": {"constant": .25, "catboost": cold},
                 "brier_alc_on_common_eligible_scopes": {"constant": .25, "beta_k5": .21,
                                                         "catboost/none": alc, "catboost/beta_mean": alc}}
        value.update(changes)
        return value

    def test_base_gate_and_isolation_requirements(self):
        passing = assess_catboost_base(self.report(.24, .2))
        self.assertTrue(passing["base_predictor_out_of_family_cold_start_beats_constant"])
        self.assertTrue(passing["any_adaptation_candidate_passed_initial_checks"])
        failing = assess_catboost_base(self.report(.31, .24))
        self.assertFalse(failing["base_predictor_out_of_family_cold_start_beats_constant"])
        self.assertFalse(failing["any_adaptation_candidate_passed_initial_checks"])
        self.assertEqual(len(failing["candidates"]["catboost/none"]["reasons"]), 2)
        with self.assertRaises(ValueError):
            assess_catboost_base(self.report(.24, .2, sealed_roles_opened=True))
        with self.assertRaises(ValueError):
            assess_catboost_base(self.report(.24, .2, visible_scopes=170, full_budget_eligible_scopes=5,
                                             excluded_small_scopes=5, zero_budget_brier_all_held_rows_equal_scope={
                                                 "constant": .2, "catboost": .19}))
        with self.assertRaises(ValueError):
            assess_catboost_base(self.report(.24, .2, brier_alc_on_common_eligible_scopes={"catboost/none": .2}))

    def test_ablations_are_evidence_not_release_candidates(self):
        report = self.report(.24, .2)
        report["zero_budget_brier_all_held_rows_equal_scope"]["catboost_no_task_structure"] = .24
        report["brier_alc_on_common_eligible_scopes"]["catboost_no_task_structure/logit_shift"] = .2
        assessment = assess_catboost_base(report)
        ablation = assessment["candidates"]["catboost_no_task_structure/logit_shift"]
        self.assertTrue(ablation["predictive_checks_passed"])
        self.assertFalse(ablation["initial_release_checks_passed"])
        self.assertEqual(ablation["reasons"], ["ablation_variant_not_a_release_candidate"])
        self.assertTrue(assess_catboost_base(report)["candidates"]["catboost/none"]["initial_release_checks_passed"])


@unittest.skipIf(CatBoostClassifier is None, "CatBoost is not installed in this environment")
class CatboostSelectionTests(unittest.TestCase):
    def test_selection_never_reads_the_outer_held_family(self):
        full = synthetic_frame()
        frame = full[["value", "shape"]]
        labels = full.label.to_numpy(dtype=float)
        families = full.family.to_numpy()
        groups = full.group.to_numpy()
        scopes = full.scope.to_numpy()
        outer_train = families != "d"
        kwargs = dict(cat_features=["shape"], seed=3, threads=1, max_iterations=12,
                      early_stopping_rounds=4, min_iterations=5)
        first = select_configuration(frame, labels, scopes, families, groups, outer_train, CONFIGS, **kwargs)
        flipped = labels.copy()
        flipped[families == "d"] = 1 - flipped[families == "d"]
        second = select_configuration(frame, flipped, scopes, families, groups, outer_train, CONFIGS, **kwargs)
        self.assertEqual(first["selected_config"], second["selected_config"])
        self.assertEqual(first["iterations"], second["iterations"])
        self.assertEqual([row["mean_inner_logloss"] for row in first["candidates"]],
                         [row["mean_inner_logloss"] for row in second["candidates"]])
        self.assertEqual(len(first["candidates"]), len(CONFIGS))
        self.assertTrue(all(row["best_iteration"] >= 1 for row in first["candidates"][0]["inner_folds"]))
        with self.assertRaisesRegex(ValueError, "two training families"):
            select_configuration(frame, labels, scopes, families, groups, families == "a", CONFIGS, **kwargs)


if __name__ == "__main__":
    unittest.main()
