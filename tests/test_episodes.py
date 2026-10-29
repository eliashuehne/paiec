import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.episodes import pair_episodes, tag_episodes, training_episodes
from paiec.metrics import BUDGETS
from paiec.split import make_split


def write_benchmark(root, name, subjects, items=100, seed=0):
    rng = np.random.default_rng(seed)
    target = Path(root) / name / "formatted_tables"
    target.mkdir(parents=True)
    item_ids = [f"{name}-i{k:03d}" for k in range(items)]
    pd.DataFrame({"subject_id": subjects}).to_parquet(target / "subjects.parquet")
    pd.DataFrame({
        "item_id": item_ids,
        "item_features": [f"item={item_id}" for item_id in item_ids],
    }).to_parquet(target / "items.parquet")
    rows = [
        {"subject_id": s, "item_id": i, "response": float(rng.integers(0, 2))}
        for s in subjects
        for i in item_ids
    ]
    rows.append({"subject_id": subjects[0], "item_id": item_ids[0], "response": np.nan})
    pd.DataFrame(rows).to_parquet(target / "responses.parquet")
    return pd.DataFrame(rows).dropna()


class PairEpisodeTests(unittest.TestCase):
    def test_reveals_follow_the_harness_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = write_benchmark(tmp, "toy", ["s0", "s1"])
            episodes = pair_episodes(tmp, "toy")
            self.assertEqual(list(episodes["subject_id"]), ["s0", "s1"])

            group = rows[rows["subject_id"] == "s0"]
            labels = group.drop_duplicates("item_id").set_index("item_id")["response"]
            split = make_split(tuple(labels.index), benchmark="toy", subject_id="s0")
            row = episodes.iloc[0]
            for b in BUDGETS:
                self.assertEqual(row[f"k{b}"], labels.loc[list(split.acquisition[:b])].sum())
            expected_eval = group[group["item_id"].isin(split.evaluation)]["response"].mean()
            self.assertAlmostEqual(row["eval_mean"], expected_eval)

    def test_pairs_below_eighty_items_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_benchmark(tmp, "small", ["s0"], items=79)
            self.assertTrue(pair_episodes(tmp, "small").empty)

    def test_large_benchmarks_are_capped_in_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_benchmark(tmp, "big", [f"s{k}" for k in range(5)])
            write_benchmark(tmp, "one", ["t0"], seed=1)
            frame = training_episodes(tmp, ["big", "one"], pairs_per_benchmark=2)
            totals = frame.groupby("benchmark")["weight"].sum()
            self.assertAlmostEqual(totals["big"], 2.0)
            self.assertAlmostEqual(totals["one"], 1.0)


class TagEpisodeTests(unittest.TestCase):
    def test_reveals_first_trial_and_scores_all_trials_of_capped_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = write_benchmark(tmp, "toy", ["s0"])
            first_labels = rows.set_index("item_id")["response"]
            split = make_split(
                tuple(first_labels.index), benchmark="toy", subject_id="s0"
            )

            # A second trial disagrees with the first on every item. Reveals must
            # keep the original labels; evaluation must count both trials.
            second_trials = rows.copy()
            second_trials["response"] = 1.0 - second_trials["response"]
            invalid_rows = pd.DataFrame(
                [
                    ("unknown", split.evaluation[0], 0.0),
                    ("s0", "unknown", 1.0),
                    ("s0", split.evaluation[0], 0.5),
                ],
                columns=["subject_id", "item_id", "response"],
            )
            directory = Path(tmp) / "toy" / "formatted_tables"
            pd.concat([rows, second_trials, invalid_rows]).to_parquet(
                directory / "responses.parquet"
            )

            (episode,) = tag_episodes(tmp, "toy", evaluation_items=7)
            self.assertEqual(episode["weight"], 1.0)
            revealed = list(split.acquisition[: max(BUDGETS)])
            self.assertEqual(
                episode["revealed_tags"], [{"item": item} for item in revealed]
            )
            np.testing.assert_array_equal(
                episode["revealed_labels"], first_labels.loc[revealed]
            )
            evaluation = sorted(split.evaluation[:7])
            self.assertEqual(
                episode["eval_tags"], [{"item": item} for item in evaluation]
            )
            np.testing.assert_array_equal(episode["eval_rows"], np.full(7, 2.0))
            np.testing.assert_array_equal(episode["eval_correct"], np.ones(7))

            summary = pair_episodes(tmp, "toy").iloc[0]
            self.assertEqual(summary["accuracy"], 0.5)
            self.assertEqual(summary["eval_mean"], 0.5)
            for budget in BUDGETS:
                self.assertEqual(
                    summary[f"k{budget}"], first_labels.loc[revealed[:budget]].sum()
                )

    def test_sampling_is_repeatable_and_respects_pair_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_benchmark(tmp, "toy", [f"s{number}" for number in range(6)])
            sampled = tag_episodes(tmp, "toy", seed=12, max_pairs=2)
            tag_episodes.cache_clear()
            repeated = tag_episodes(tmp, "toy", seed=12, max_pairs=2)
            all_subjects = tag_episodes(tmp, "toy", seed=12, max_pairs=6)

            self.assertEqual(len(sampled), 2)
            self.assertEqual(len(all_subjects), 6)
            for original, replay in zip(sampled, repeated):
                self.assertEqual(original["revealed_tags"], replay["revealed_tags"])
                np.testing.assert_array_equal(
                    original["revealed_labels"], replay["revealed_labels"]
                )
                self.assertIn(
                    original["revealed_tags"],
                    [entry["revealed_tags"] for entry in all_subjects],
                )

    def test_repeated_trials_do_not_make_a_small_pair_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = write_benchmark(tmp, "small", ["s0"], items=79)
            pd.concat([rows, rows, rows]).to_parquet(
                Path(tmp) / "small" / "formatted_tables" / "responses.parquet"
            )
            self.assertEqual(tag_episodes(tmp, "small"), ())


if __name__ == "__main__":
    unittest.main()
