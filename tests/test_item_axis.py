import unittest

import numpy as np
import pandas as pd

from paiec.competition_scorer import Draw, split_pair
from paiec.item_axis import (item_frames, item_observations, oracle_ceilings, rasch_fit,
                             split_half_reliability, summarise, transfer_auroc,
                             variance_components)
from paiec.rate_model import sigmoid


def frame_from(rates, *, responses=40, prefix="c"):
    """A frame in the shape ``item_frames`` produces, from a rate table."""
    cells = sorted({cell for cell, _ in rates})
    items = sorted({item for _, item in rates})
    index = {item: position for position, item in enumerate(items)}
    rows, cols, n, s = [], [], [], []
    for (cell, item), rate in rates.items():
        rows.append(cells.index(cell))
        cols.append(index[item])
        n.append(responses)
        s.append(int(round(rate * responses)))
    return {"cells": [{"cell_index": position, "cell_key": name} for position, name in enumerate(cells)],
            "items": index, "item_ids": items, "rows": np.asarray(rows), "cols": np.asarray(cols),
            "n": np.asarray(n, dtype=float), "s": np.asarray(s, dtype=float)}


def shared_rates(subjects=8, items=12, seed=3):
    """Ability and difficulty both real: logit p = theta - b."""
    rng = np.random.default_rng(seed)
    ability = dict(zip([f"s{index}" for index in range(subjects)], rng.normal(0, 1, subjects)))
    difficulty = dict(zip([f"i{index}" for index in range(items)], rng.normal(0, 1, items)))
    return {(cell, item): float(sigmoid(ability[cell] - difficulty[item]))
            for cell in ability for item in difficulty}


def interaction_rates(subjects=8, items=12, seed=3):
    """Only subject-item interaction: nothing shared about an item."""
    rng = np.random.default_rng(seed)
    return {(f"s{cell}", f"i{item}"): float(rng.uniform(.05, .95))
            for cell in range(subjects) for item in range(items)}


def cell(index, cell_key, direction="higher_is_better", n_items=80):
    return {"cell_index": index, "cell_key": cell_key, "benchmark": "b", "direction": direction,
            "n_items": n_items, "label_sum": 1, "n_responses": 2,
            "subject": {"normalized_name": cell_key, "provider": "", "release_date": "",
                        "access_date": "", "harness": "", "harness_version": "",
                        "reasoning_effort": "", "subject_features_extra": ""}}


class FrameTests(unittest.TestCase):
    def test_only_the_judged_direction_and_large_enough_cells_enter(self):
        table = {"item_id": ["a", "b"], "n_responses": [1, 1], "label_sum": [1, 0]}
        cells = [cell(0, "keep"), cell(1, "flip", direction="lower_is_better"),
                 cell(2, "small", n_items=79)]
        frames = item_frames(cells, {0: table, 1: table, 2: table}, min_items=80)
        self.assertEqual(list(frames), ["b"])
        self.assertEqual(frames["b"]["item_ids"], ["a", "b"])
        self.assertEqual(len(frames["b"]["cells"]), 1)

    def test_repeated_trials_collapse_to_one_weighted_observation(self):
        frame = {"cells": [{"cell_index": 0, "cell_key": "c"}], "items": {"a": 0}, "item_ids": ["a"],
                 "rows": np.array([0, 0]), "cols": np.array([0, 0]),
                 "n": np.array([2.0, 3.0]), "s": np.array([1.0, 0.0])}
        rows, cols, counts, sums = item_observations(frame)
        self.assertEqual((rows.tolist(), cols.tolist()), ([0], [0]))
        self.assertEqual((counts.tolist(), sums.tolist()), ([5.0], [1.0]))


class RaschTests(unittest.TestCase):
    def test_ability_and_difficulty_recover_their_order(self):
        rates = shared_rates()
        frame = frame_from(rates)
        fit = rasch_fit(frame)
        for strong, weak in (("s0", "s1"), ("s2", "s3")):
            if fit["theta"][int(strong[1:])] < fit["theta"][int(weak[1:])]:
                self.fail("ability order inverted")
        order = np.argsort(-fit["difficulty"])
        hardest, easiest = order[0], order[-1]
        self.assertGreater(rates[("s0", frame["item_ids"][easiest])],
                           rates[("s0", frame["item_ids"][hardest])])

    def test_difficulty_is_centred_and_penalties_are_checked(self):
        fit = rasch_fit(frame_from(shared_rates()))
        self.assertAlmostEqual(float(fit["difficulty"].mean()), 0.0, places=9)
        with self.assertRaises(ValueError):
            rasch_fit(frame_from(shared_rates()), item_penalty=-1)

    def test_item_difficulty_transfers_to_held_out_systems_when_shared(self):
        result = transfer_auroc(frame_from(shared_rates()), seed=0)
        self.assertIsNotNone(result["auc"])
        self.assertGreater(result["auc"], .7)

    def test_transfer_is_near_chance_when_difficulty_is_model_specific(self):
        result = transfer_auroc(frame_from(interaction_rates()), seed=0)
        self.assertIsNotNone(result["auc"])
        self.assertTrue(.4 < result["auc"] < .6, result["auc"])

    def test_reliability_separates_shared_from_model_specific(self):
        shared = split_half_reliability(frame_from(shared_rates()), seed=0)
        specific = split_half_reliability(frame_from(interaction_rates()), seed=0)
        self.assertGreater(shared["pearson"], .6)
        self.assertLess(specific["pearson"], .45)

    def test_variance_components_report_positive_spread(self):
        components = variance_components(frame_from(shared_rates()))
        self.assertGreater(components["sd_ability"], 0)
        self.assertGreater(components["sd_difficulty"], 0)
        self.assertTrue(0 <= components["residual_brier"] <= .25)

    def test_summarise_pools_the_per_benchmark_rows(self):
        report = summarise({"b": frame_from(shared_rates())})
        self.assertEqual(report["benchmarks"], 1)
        self.assertGreater(report["transfer"]["pooled_auc"], .7)
        self.assertEqual(report["transfer"]["benchmarks"], 1)


def pair(cell_index, cell_key, rates, *, responses=20):
    """A real episode: the production split gives each subject its own halves."""
    items = sorted(rates)
    table = pd.DataFrame({"item_id": items,
                          "n_responses": [responses] * len(items),
                          "label_sum": [int(round(rates[item] * responses)) for item in items]})
    cell = {"cell_index": cell_index, "cell_key": cell_key, "benchmark": "b",
            "benchmark_id": "benchmark_b", "subject": {"normalized_name": cell_key}}
    return split_pair(cell, table, seed=0)


PROFILE = {"i0": .95, "i1": .9, "i2": .7, "i3": .3, "i4": .1, "i5": .05}


class CeilingTests(unittest.TestCase):
    def draw(self):
        """Two subjects with the same item profile: hard items are hard for both."""
        return Draw(seed=0, regime="all",
                    pairs=[pair(0, "a", dict(PROFILE)), pair(1, "b", dict(PROFILE))])

    def test_ceilings_are_ordered_and_coverage_grows_with_budget(self):
        report = oracle_ceilings(self.draw())
        budget_zero = report["rows"][0]
        budget_last = report["rows"][-1]
        self.assertEqual(budget_zero["budget"], 0)
        self.assertEqual(budget_zero["covered_item_share"], 0.0)
        self.assertGreater(budget_last["covered_item_share"], 0.0)
        self.assertGreaterEqual(budget_last["covered_item_share"], budget_zero["covered_item_share"])
        # Knowing each item beats knowing only the pair's rate, which is what the
        # free evidence rule already does.
        self.assertLess(budget_last["own_item_oracle"], .25)
        self.assertLess(report["own_item_alc"], report["sibling_facility_alc"] + 1e-9)
        self.assertLess(report["sibling_facility_alc"], .25)


if __name__ == "__main__":
    unittest.main()
