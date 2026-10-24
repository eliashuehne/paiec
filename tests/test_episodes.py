import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.episodes import pair_episodes, training_episodes
from paiec.metrics import BUDGETS
from paiec.split import make_split


def write_benchmark(root, name, subjects, items=100, seed=0):
    rng = np.random.default_rng(seed)
    target = Path(root) / name / "formatted_tables"
    target.mkdir(parents=True)
    item_ids = [f"{name}-i{k:03d}" for k in range(items)]
    pd.DataFrame({"subject_id": subjects}).to_parquet(target / "subjects.parquet")
    pd.DataFrame({"item_id": item_ids}).to_parquet(target / "items.parquet")
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


if __name__ == "__main__":
    unittest.main()
