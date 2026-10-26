import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.episodes import tag_episodes
from paiec.gbdt_prior import GBDTPrior
from paiec.prior_offset import episode_logits, episode_vectors
from paiec.schema import item_record, load_benchmark, subject_record

SUBJECT_COLUMNS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                   "harness_version", "reasoning_effort", "subject_features_extra", "display_name"]
NAMES = ["Alpha 7B", "Beta 7B", "Gamma 70B", "Delta 70B"]


def write_crossover_benchmark(root, name, seed):
    """Large models solve math and fail law; small models the opposite."""
    rng = np.random.default_rng(seed)
    target = Path(root) / name / "formatted_tables"
    target.mkdir(parents=True)
    ids = [f"{name}-{k}" for k in range(100)]
    topics = ["math" if k % 2 else "law" for k in range(100)]
    pd.DataFrame({"item_id": ids, "content": [f"{t} question number {k}" for k, t in enumerate(topics)],
                  "item_features": [""] * 100}).to_parquet(target / "items.parquet")
    subjects = pd.DataFrame({"subject_id": [f"{name}-s{k}" for k in range(len(NAMES))]})
    for column in SUBJECT_COLUMNS:
        subjects[column] = ""
    subjects["normalized_name"] = NAMES
    subjects.to_parquet(target / "subjects.parquet")
    rows = []
    for subject_id, model in zip(subjects["subject_id"], NAMES):
        big = "70B" in model
        for item_id, topic in zip(ids, topics):
            p = 0.9 if (topic == "math") == big else 0.1
            rows.append({"subject_id": subject_id, "item_id": item_id, "trial": 1, "interactors": "",
                         "response": float(rng.random() < p)})
    pd.DataFrame(rows).to_parquet(target / "responses.parquet")


def subject(name):
    record = {column: "" for column in SUBJECT_COLUMNS[:-1]}
    record["normalized_name"] = name
    return record


def item(topic):
    return {"item_content": f"{topic} question number 9999", "item_features": "", "interactors": "",
            "benchmark_id": "new"}


class GBDTPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "data"
        cls.names = [f"bench{k}" for k in range(8)]
        for k, name in enumerate(cls.names):
            write_crossover_benchmark(cls.data, name, seed=k)
        cls.prior = GBDTPrior().fit(cls.data, cls.names)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_learns_a_subject_item_crossover(self):
        big, small = subject("Omega 70B"), subject("Omega 7B")
        self.assertGreater(self.prior(big, item("math"), []), self.prior(big, item("law"), []))
        self.assertGreater(self.prior(small, item("law"), []), self.prior(small, item("math"), []))

    def test_batch_scores_match_live_scores(self):
        episodes = tag_episodes(str(self.data), "bench0")
        cached = episode_logits(self.prior, episode_vectors(self.data, "bench0"), episodes)
        tables = load_benchmark(self.data, "bench0")
        for episode, (revealed, _) in zip(episodes, cached):
            record = subject_record(tables.subjects.loc[episode["subject_id"]])
            for item_id, logit in zip(episode["revealed_ids"], revealed):
                stored = tables.items.loc[item_id]
                live = item_record({"content": stored["content"], "item_features": stored["item_features"],
                                    "interactors": ""}, "bench0")
                self.assertAlmostEqual(self.prior.logit(record, live), logit, places=8)


if __name__ == "__main__":
    unittest.main()
