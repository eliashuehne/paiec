import unittest

import numpy as np
import pandas as pd

from paiec.competition_scorer import (BUDGETS, ConstantPredictor, EvidencePredictor, benchmark_presence,
                                      constant_baseline, draw_evidence, evidence_for, fixed_beta,
                                      item_brier, label_list, oracle_curve, own_label_mean, pair_auroc,
                                      revealed_units, score_draw, select_draw, split_pair)
from paiec.metrics import alc_weights


def build(cells_spec):
    """Cells and their item labels from (benchmark, name, successes, items) tuples."""
    cells, labels = [], {}
    for index, (benchmark, name, successes, n_items, direction) in enumerate(cells_spec):
        items = [f"i{index}_{position}" for position in range(n_items)]
        counts = pd.DataFrame({"item_id": items,
                               "n_responses": [1] * n_items,
                               "label_sum": [1 if position < successes else 0
                                             for position in range(n_items)]})
        labels[index] = counts
        cells.append({"cell_index": index, "cell_key": f"cell{index}", "benchmark": benchmark,
                      "benchmark_id": f"benchmark_{benchmark}", "subject_id": f"s{index}",
                      "subject": {"normalized_name": name, "provider": "", "release_date": "",
                                  "access_date": "", "harness": "", "harness_version": "",
                                  "reasoning_effort": "", "subject_features_extra": ""},
                      "direction": direction, "n_items": n_items,
                      "rate": successes / n_items, "n_responses": n_items, "label_sum": successes})
    return cells, labels


SPEC = [("alpha", "Strong", 8, 10, "higher_is_better"),
        ("alpha", "Weak", 2, 10, "higher_is_better"),
        ("beta", "Strong", 5, 10, "higher_is_better"),
        ("beta", "Other", 5, 10, "higher_is_better")]


class SplitTests(unittest.TestCase):
    def test_split_is_half_and_half_and_fixed(self):
        cells, labels = build(SPEC)
        first = split_pair(cells[0], labels[0], seed=7)
        second = split_pair(cells[0], labels[0], seed=7)
        self.assertEqual(first.acquisition, second.acquisition)
        self.assertEqual(first.evaluation, second.evaluation)
        self.assertEqual(len(first.acquisition), 5)
        self.assertEqual(len(first.evaluation), 5)
        self.assertFalse(set(first.acquisition) & set(first.evaluation))
        self.assertEqual(set(first.acquisition) | set(first.evaluation), set(labels[0]["item_id"]))

    def test_revealed_prefix_is_nested_and_respects_response_counts(self):
        cells, labels = build(SPEC)
        labels[0].loc[0, "n_responses"] = 3
        labels[0].loc[0, "label_sum"] = 2
        pair = split_pair(cells[0], labels[0], seed=7)
        seen = [set(revealed_units(pair, budget)) for budget in BUDGETS]
        for earlier, later in zip(seen, seen[1:]):
            self.assertTrue(earlier <= later)
        self.assertTrue(all(not unit for unit in [revealed_units(pair, 0)]))
        counts = sum(pair.n_responses[item] for item in revealed_units(pair, 1))
        self.assertLessEqual(counts, 1)

    def test_labels_expose_every_recorded_response(self):
        cells, labels = build(SPEC)
        pair = split_pair(cells[0], labels[0], seed=7)
        item = pair.acquisition[0]
        labels[0].loc[labels[0]["item_id"] == item, "n_responses"] = 4
        labels[0].loc[labels[0]["item_id"] == item, "label_sum"] = 3
        refreshed = split_pair(cells[0], labels[0], seed=7)
        self.assertEqual(sorted(label_list(refreshed, item)), [0, 1, 1, 1])


class BrierTests(unittest.TestCase):
    def test_response_weighted_brier_is_exact_with_trials(self):
        cells, labels = build(SPEC)
        labels[0].loc[0, "n_responses"] = 3
        labels[0].loc[0, "label_sum"] = 2
        pair = split_pair(cells[0], labels[0], seed=7)
        values = np.full(len(pair.evaluation), .5)
        self.assertAlmostEqual(item_brier(pair, pair.evaluation, values), .25)

    def test_constant_arm_reproduces_the_published_quarter(self):
        cells, labels = build(SPEC)
        draw = select_draw(cells, labels, seed=3, max_pairs=10, min_items=2)
        scored = score_draw(draw, constant_baseline(), with_auroc=False)
        self.assertAlmostEqual(scored["brier_alc"], .25)
        for budget in BUDGETS:
            self.assertAlmostEqual(scored["curve"][budget], .25)

    def test_weights_match_the_published_curve(self):
        self.assertEqual([round(value, 6) for value in alc_weights(BUDGETS)],
                         [.1, .2, .2, .2, .2, .1])

    def test_auroc_separates_ordered_and_tied_predictions(self):
        cells, labels = build(SPEC)
        for position in range(10):
            labels[0].loc[position, "label_sum"] = 1 if position < 5 else 0
        pair = split_pair(cells[0], labels[0], seed=7)
        targets = [pair.label_sum[item] for item in pair.evaluation]
        perfect = np.array(targets, dtype=float)
        inverted = 1 - perfect
        self.assertAlmostEqual(pair_auroc(pair, pair.evaluation, perfect), 1.0)
        self.assertAlmostEqual(pair_auroc(pair, pair.evaluation, inverted), 0.0)
        self.assertAlmostEqual(pair_auroc(pair, pair.evaluation, np.full(len(targets), .5)), .5)

    def test_auroc_counts_every_recorded_response(self):
        cells, labels = build(SPEC)
        for position in range(10):
            labels[0].loc[position, "label_sum"] = 5 if position < 5 else 0
            labels[0].loc[position, "n_responses"] = 5 if position < 5 else 5
        labels[0].loc[0, "label_sum"] = 3
        pair = split_pair(cells[0], labels[0], seed=7)
        targets = [pair.label_sum[item] / pair.n_responses[item] for item in pair.evaluation]
        self.assertAlmostEqual(pair_auroc(pair, pair.evaluation, np.full(len(targets), .5)), .5)
        # The mixed item's own responses tie with each other, so 1.0 is not reachable.
        self.assertGreater(pair_auroc(pair, pair.evaluation, np.array(targets, dtype=float)), .97)


class EvidenceTests(unittest.TestCase):
    def test_other_benchmarks_never_leak(self):
        cells, labels = build(SPEC + [("gamma", "Solo", 5, 10, "higher_is_better")])
        draw = select_draw(cells, labels, seed=5, max_pairs=10, min_items=2)
        solo = [pair for pair in draw.pairs if pair.benchmark == "gamma"][0]
        own, other = evidence_for(draw, solo, 7)
        self.assertTrue(own)
        self.assertEqual(other, [])
        alpha = [pair for pair in draw.pairs
                 if pair.benchmark == "alpha" and pair.subject_key == "strong"][0]
        sibling = [pair for pair in draw.pairs
                   if pair.benchmark == "alpha" and pair.cell_key != alpha.cell_key][0]
        _, shared = evidence_for(draw, alpha, 7)
        self.assertEqual(shared, [value for item in revealed_units(sibling, 7)
                                  for value in label_list(sibling, item)])

    def test_benchmark_evidence_excludes_the_pair_itself(self):
        cells, labels = build(SPEC)
        draw = select_draw(cells, labels, seed=5, max_pairs=10, min_items=2)
        pair = [value for value in draw.pairs if value.benchmark == "beta"][0]
        own, other = evidence_for(draw, pair, 7)
        self.assertEqual(len(other), sum(len(label_list(sibling, item))
                                         for sibling in draw.pairs
                                         if sibling.benchmark == "beta" and sibling.cell_key != pair.cell_key
                                         for item in revealed_units(sibling, 7)))

    def test_evaluation_outcomes_never_enter_evidence(self):
        cells, labels = build(SPEC)
        draw = select_draw(cells, labels, seed=5, max_pairs=10, min_items=2)
        cache = draw_evidence(draw)
        for budget, per_pair in cache.items():
            for pair in draw.pairs:
                own, other = per_pair[pair.cell_key]
                revealed_items = set(revealed_units(pair, budget))
                self.assertFalse(revealed_items & set(pair.evaluation))
                expected = sum(pair.n_responses[item] for item in revealed_items)
                self.assertEqual(len(own), expected)
        expected = [evidence_for(draw, pair, 7) for pair in draw.pairs]
        self.assertEqual([cache[7][pair.cell_key] for pair in draw.pairs], expected)

    def test_flat_controls_report_the_posterior_rate(self):
        cells, labels = build(SPEC)
        draw = select_draw(cells, labels, seed=5, max_pairs=10, min_items=2)
        pair = [value for value in draw.pairs if value.benchmark == "alpha"][0]
        base = np.linspace(.2, .8, len(pair.evaluation))
        predictor = EvidencePredictor("beta", prior_strength=2.0, prior="base")
        shifted = predictor.update(base, [1, 1, 1], [])
        expected = (2.0 * base.mean() + 3) / 5.0
        self.assertTrue(np.allclose(shifted, expected))
        self.assertGreater(shifted.mean(), base.mean())
        self.assertEqual(len(set(np.round(shifted, 12))), 1)

    def test_zero_prior_means_and_cross_benchmark_pooling(self):
        mean = own_label_mean()
        base = np.full(4, .5)
        self.assertAlmostEqual(float(mean.update(base, [1, 0, 1], [])[0]), 2 / 3)
        pooled = EvidencePredictor("pooled", prior_strength=0.0, own_weight=0.0, other_weight=1.0,
                                   prior="half")
        self.assertAlmostEqual(float(pooled.update(base, [], [1, 1, 0, 0])[0]), .5)
        self.assertGreater(float(pooled.update(base, [], [1, 1])[0]), .9999)
        self.assertAlmostEqual(float(pooled.update(base, [], [])[0]), .5)
        self.assertAlmostEqual(float(fixed_beta(5.).update(base, [1], [])[0]), 3.5 / 6)
        # Evidence this arm weights at zero must leave the base untouched.
        self.assertTrue(np.array_equal(pooled.update(base, [1], []), base))
        self.assertTrue(np.array_equal(mean.update(base, [], [1, 0]), base))

    def test_oracle_bounds_the_curve(self):
        cells, labels = build(SPEC)
        draw = select_draw(cells, labels, seed=5, max_pairs=10, min_items=2)
        oracle = oracle_curve(draw)
        scored = score_draw(draw, ConstantPredictor("half", .5), with_auroc=False)
        self.assertLess(oracle["brier_alc"], scored["brier_alc"])


class DrawTests(unittest.TestCase):
    def test_regimes_follow_name_overlap(self):
        cells, labels = build(SPEC)
        presence = benchmark_presence(cells)
        self.assertEqual(presence["strong"], {"alpha", "beta"})
        shared = select_draw(cells, labels, seed=1, regime="shared_subject", min_items=2)
        cold = select_draw(cells, labels, seed=1, regime="cold_subject", min_items=2)
        self.assertEqual({pair.benchmark for pair in shared.pairs}, {"alpha", "beta"})
        self.assertEqual({pair.subject_key for pair in shared.pairs}, {"strong"})
        self.assertEqual({pair.subject_key for pair in cold.pairs}, {"other", "weak"})
        with self.assertRaises(ValueError):
            select_draw(cells, labels, seed=1, regime="unknown")

    def test_direction_and_benchmark_filters(self):
        cells, labels = build(SPEC + [("gamma", "Strong", 5, 10, "lower_is_better")])
        draw = select_draw(cells, labels, seed=1, directions=["higher_is_better"], min_items=2)
        self.assertNotIn("gamma", {pair.benchmark for pair in draw.pairs})
        draw = select_draw(cells, labels, seed=1, benchmarks=["beta"], min_items=2)
        self.assertEqual({pair.benchmark for pair in draw.pairs}, {"beta"})
        self.assertEqual(draw.pairs[0].subject_key, "other")

    def test_pairs_are_capped_and_seeded(self):
        cells, labels = build(SPEC)
        first = select_draw(cells, labels, seed=2, max_pairs=2, min_items=2)
        second = select_draw(cells, labels, seed=2, max_pairs=2, min_items=2)
        self.assertEqual([pair.cell_key for pair in first.pairs], [pair.cell_key for pair in second.pairs])
        self.assertEqual(len(first.pairs), 2)


if __name__ == "__main__":
    unittest.main()
