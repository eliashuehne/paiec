"""Tests for the scrambled-profile permutation distribution."""
import json
from pathlib import Path
import tempfile
import unittest

from paiec.catboost_permutation import (METRIC_DIRECTION, PRIMARY_METRIC, PermutationConfig,
                                        compare_to_null, collect_metrics, paired_scope_ordering,
                                        permuted_recipe, scope_ordering, scored_row_signature,
                                        verified_replicate)
from paiec.catboost_unseen_benchmark import UnseenBenchmarkConfig

RECIPE = {"version": "catboost_base_v4", "prepared": "runs/competition_pipeline/protocol_panel_v2",
          "feature_groups": ["task_structure", "system_profile"],
          "profile_catalog": "research/system_profiles/collection_v2",
          "profile_policy": "configs/head_finetuning/enrichment_catalog_v2.json",
          "threads": 8, "seed": 7}


def panel(cold=0.2543, auroc=0.5364, shift=0.1589):
    return {"zero_budget_brier_all_held_rows_equal_scope": {"catboost": cold},
            "cold_start_metrics": {"catboost": {"mean_within_scope_auroc": auroc}},
            "brier_alc_on_common_eligible_scopes": {"catboost/none": cold, "catboost/beta_mean": 0.1615,
                                                    "catboost/logit_shift": shift}}


def unseen(primary=0.2510, pooled=0.2490, auroc=0.5530, folds=(0.51, 0.52, 0.53)):
    return {"primary_equal_scope_brier": primary,
            "pooled_row_metrics": {"equal_scope_brier": pooled, "mean_within_scope_auroc": auroc},
            "folds": [{"external_metrics": {"equal_scope_brier": primary,
                                            "mean_within_scope_auroc": value}} for value in folds]}


def metrics(**overrides):
    values = {metric: 0.5 for metric in METRIC_DIRECTION}
    values.update(overrides)
    return values


class RecipeTests(unittest.TestCase):
    def test_declares_the_seeded_control(self):
        control = permuted_recipe(RECIPE, 3)
        self.assertEqual(control["version"], "catboost_base_v5")
        self.assertEqual(control["profile_permutation_seed"], 3)
        self.assertNotIn("profile_permutation_seed", RECIPE)
        self.assertEqual(control["feature_groups"], RECIPE["feature_groups"])

    def test_rejects_a_recipe_it_could_not_have_measured(self):
        for broken in [dict(RECIPE, version="catboost_base_v5"),
                       dict(RECIPE, profile_permutation_seed=1),
                       {key: value for key, value in RECIPE.items() if key != "profile_policy"},
                       dict(RECIPE, feature_groups=["task_structure"]),
                       dict(RECIPE, version="catboost_base_v3")]:
            with self.subTest(broken=broken.get("version")):
                with self.assertRaises(ValueError):
                    permuted_recipe(broken, 1)

    def test_seed_must_be_an_integer(self):
        for seed in (True, "1", 1.0, None):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    permuted_recipe(RECIPE, seed)


class ReplicateTests(unittest.TestCase):
    def unseen(self):
        return UnseenBenchmarkConfig.model_validate({
            "panel_run": "runs/x", "data_root": "payload/data", "seed": 1,
            "min_pair_items": 20, "rows_per_system": 100, "benchmark_cap": 6000,
            "max_panel_group_overlap": 0.0})

    def test_a_replicate_from_another_permutation_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = Path(folder) / "panel"
            panel.mkdir()
            (panel / "resolved_config.json").write_text(
                json.dumps(permuted_recipe(RECIPE, 9)))
            with self.assertRaises(ValueError):
                verified_replicate(Path(folder), RECIPE, self.unseen(), 2, folder)

    def test_a_replicate_from_another_recipe_version_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            panel = Path(folder) / "panel"
            panel.mkdir()
            (panel / "resolved_config.json").write_text(json.dumps(RECIPE))
            with self.assertRaises(ValueError):
                verified_replicate(Path(folder), RECIPE, self.unseen(), 1, folder)


class MetricTests(unittest.TestCase):
    def test_collect_metrics_reads_each_declared_value(self):
        values = collect_metrics(panel(), unseen())
        self.assertEqual(set(values), set(METRIC_DIRECTION))
        self.assertAlmostEqual(values["panel_cold_equal_scope_brier"], 0.2543)
        self.assertAlmostEqual(values[PRIMARY_METRIC], 0.52)
        self.assertAlmostEqual(values["unseen_fold_mean_brier"], 0.2510)

    def test_fold_average_replaces_cross_fold_averaging(self):
        # Pooled AUROC is inflated by averaging fold models; the per-fold mean is not.
        values = collect_metrics(panel(), unseen(auroc=0.5530, folds=(0.50, 0.50, 0.50)))
        self.assertAlmostEqual(values["unseen_pooled_auroc"], 0.5530)
        self.assertAlmostEqual(values[PRIMARY_METRIC], 0.50)

    def test_no_eligible_fold_is_an_error(self):
        with self.assertRaises(ValueError):
            collect_metrics(panel(), unseen(folds=()))

    def test_row_signature_is_independent_of_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "row_predictions.jsonl"
            path.write_text(json.dumps({"benchmark": "b", "subject_id": "s", "item_id": "i",
                                        "label": 1, "prediction": 0.4}) + "\n")
            first = scored_row_signature(folder)
            path.write_text(json.dumps({"benchmark": "b", "subject_id": "s", "item_id": "i",
                                        "label": 1, "prediction": 0.9}) + "\n")
            self.assertEqual(first, scored_row_signature(folder))


class ScopeOrderingTests(unittest.TestCase):
    def write(self, folder, rows):
        path = Path(folder) / "row_predictions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return folder

    def test_per_scope_auroc_and_singletons(self):
        rows = []
        for label, score in [(0, .1), (1, .9), (0, .2), (1, .8)]:
            rows.append({"scope_key": "perfect", "label": label, "prediction": score})
        for label, score in [(0, .9), (1, .1)]:
            rows.append({"scope_key": "inverted", "label": label, "prediction": score})
        rows.append({"scope_key": "single_class", "label": 1, "prediction": .5})
        rows.append({"scope_key": "single_class", "label": 1, "prediction": .4})
        with tempfile.TemporaryDirectory() as folder:
            values = scope_ordering(self.write(folder, rows))
        self.assertEqual(set(values), {"perfect", "inverted"})
        self.assertAlmostEqual(values["perfect"], 1.0)
        self.assertAlmostEqual(values["inverted"], 0.0)

    def arm(self, **scopes):
        return {"scope_auroc": dict(scopes)}

    def test_paired_deltas_and_counts(self):
        observed = self.arm(a=.70, b=.55, c=.60)
        baseline = self.arm(a=.50, b=.50, c=.50)
        nulls = {1: self.arm(a=.75, b=.55, c=.45), 2: self.arm(a=.50, b=.50, c=.55)}
        report = paired_scope_ordering(observed, baseline, nulls, resamples=200)
        self.assertEqual(report["scopes"], 3)
        self.assertEqual(len(report["rows"]), 3)
        # 0.075, 0.025 and 0.10 against the mean permuted scope score.
        self.assertAlmostEqual(report["mean_delta_vs_null_mean"], 0.2 / 3)
        self.assertEqual(report["scopes_above_null_mean"], 3)
        self.assertEqual(report["scopes_above_every_permutation"], 1)
        self.assertEqual(report["scopes_below_every_permutation"], 0)
        low, high = report["mean_delta_ci95"]
        self.assertGreaterEqual(low, 0.025)
        self.assertLessEqual(high, 0.10)
        self.assertLessEqual(low, report["mean_delta_vs_null_mean"])
        self.assertGreaterEqual(high, report["mean_delta_vs_null_mean"])

    def test_requires_a_shared_scope(self):
        with self.assertRaises(ValueError):
            paired_scope_ordering(self.arm(a=.6), self.arm(a=.5), {1: self.arm(b=.5)})


class NullComparisonTests(unittest.TestCase):
    def build(self, observed_value, null_values, *, alpha=0.05):
        observed = metrics(**{PRIMARY_METRIC: observed_value})
        baseline = metrics(**{PRIMARY_METRIC: 0.5097})
        nulls = {seed: metrics(**{PRIMARY_METRIC: value})
                 for seed, value in enumerate(null_values, start=1)}
        return compare_to_null(observed, baseline, nulls, alpha=alpha)

    def test_p_value_counts_draws_at_least_as_good(self):
        report = self.build(0.52, [0.50, 0.51, 0.53])
        primary = report["primary"]
        # One of three permutations reached the observed arm; the +1 correction gives 2/4.
        self.assertAlmostEqual(primary["p_value"], 0.5)
        self.assertAlmostEqual(primary["p_value_floor"], 0.25)

    def test_residual_separates_path_effect_from_observed_gain(self):
        report = self.build(0.52, [0.50, 0.50, 0.50])
        row = report["comparison"][PRIMARY_METRIC]
        self.assertAlmostEqual(row["observed_gain_over_baseline"], 0.0103)
        self.assertAlmostEqual(row["path_only_gain_null_mean"], -0.0097)
        self.assertAlmostEqual(row["residual_gain"], 0.0200)

    def test_lower_is_better_uses_the_opposite_tail(self):
        metric = "unseen_primary_equal_scope_brier"
        observed, baseline = metrics(**{metric: 0.25}), metrics(**{metric: 0.26})
        nulls = {1: metrics(**{metric: 0.24}), 2: metrics(**{metric: 0.27})}
        row = compare_to_null(observed, baseline, nulls, alpha=0.05)["comparison"][metric]
        self.assertEqual(row["null_seeds_at_least_as_good"], [1])
        self.assertAlmostEqual(row["observed_gain_over_baseline"], 0.01)
        self.assertAlmostEqual(row["path_only_gain_null_mean"], 0.005)
        self.assertAlmostEqual(row["residual_gain"], 0.005)

    def test_verdicts_cover_beyond_within_and_worse(self):
        self.assertEqual(self.build(0.60, [0.50, 0.51], alpha=0.5)["primary"]["verdict"],
                         "residual_ordering_gain_beyond_permuted_control")
        self.assertEqual(self.build(0.52, [0.50, 0.51], alpha=0.05)["primary"]["verdict"],
                         "residual_ordering_gain_within_permuted_control_range")
        self.assertEqual(self.build(0.40, [0.50, 0.51])["primary"]["verdict"],
                         "observed_arm_no_better_than_permuted_control")

    def test_single_permutation_cannot_reach_a_conventional_threshold(self):
        report = self.build(0.60, [0.50], alpha=0.05)
        self.assertAlmostEqual(report["primary"]["p_value_floor"], 0.5)
        self.assertGreater(report["primary"]["p_value"], 0.05)


class ConfigTests(unittest.TestCase):
    def config(self, **overrides):
        values = {"recipe": "configs/pipelines/catboost_base_v4.json",
                  "unseen_config": "configs/pipelines/catboost_unseen_benchmark_v2.json",
                  "permutation_seeds": [2, 3],
                  "reused_replicates": [{"seed": 1, "panel_run": "runs/a", "unseen_run": "runs/b"}],
                  "observed_panel_run": "runs/c", "observed_unseen_run": "runs/d",
                  "baseline_panel_run": "runs/e", "baseline_unseen_run": "runs/f"}
        values.update(overrides)
        return PermutationConfig.model_validate(values)

    def test_accepts_a_coherent_plan(self):
        self.assertEqual(self.config().permutation_seeds, [2, 3])

    def test_rejects_overlapping_or_degenerate_plans(self):
        for broken in [{"permutation_seeds": []},
                       {"permutation_seeds": [2, 2]},
                       {"permutation_seeds": [1, 2]},
                       {"alpha": 0.0},
                       {"alpha": 1.0}]:
            with self.subTest(broken=broken):
                with self.assertRaises(ValueError):
                    self.config(**broken)

    def test_reused_seed_must_be_unique(self):
        with self.assertRaises(ValueError):
            self.config(reused_replicates=[{"seed": 1, "panel_run": "runs/a", "unseen_run": "runs/b"},
                                           {"seed": 1, "panel_run": "runs/h", "unseen_run": "runs/i"}])

    def test_unknown_keys_are_refused(self):
        with self.assertRaises(ValueError):
            self.config(permutation_seed=[1, 2])


if __name__ == "__main__":
    unittest.main()
