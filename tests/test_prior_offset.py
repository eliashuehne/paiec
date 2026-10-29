import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.episodes import tag_episodes
from paiec.metrics import BUDGETS, alc_weights
from paiec.offset import TagOffset
from paiec.prior import LogisticPrior
from paiec.prior_offset import (PriorOffset, episode_logits, episode_vectors, prior_offset_alc,
                                prior_states)
from paiec.schema import item_record, load_benchmark, subject_record
from paiec.update import sigmoid

SUBJECT = {"normalized_name": "model-a"}
TASKS = ["a", "b", "c"]


class StubPrior:
    """Prior logit written into the item text, so tests control it exactly."""

    def logit(self, subject, item):
        return float(item["item_content"])


def item(logit, task):
    return {"item_content": repr(float(logit)), "item_features": f"task={task}", "benchmark_id": "bench"}


def synthetic_episodes(pairs, informative, seed=0, level=0.0, tau=1.0):
    """Episodes whose prior logits either drive the labels or are pure noise."""
    rng = np.random.default_rng(seed)
    episodes, logits = [], []
    for _ in range(pairs):
        shift = rng.normal(0, tau)
        def draw(n):
            prior = rng.normal(0, 1.5, size=n)
            truth = level + shift + (prior if informative else 0.0)
            tasks = rng.choice(TASKS, size=n)
            return prior, [{"task": t} for t in tasks], (rng.random(n) < sigmoid(truth)).astype(float)
        r_logit, r_tags, r_labels = draw(max(BUDGETS))
        e_logit, e_tags, e_labels = draw(80)
        episodes.append({"weight": 1.0, "revealed_tags": r_tags, "revealed_labels": r_labels,
                         "eval_tags": e_tags, "eval_rows": np.ones(80), "eval_correct": e_labels})
        logits.append((r_logit, e_logit))
    return episodes, logits


class PriorOffsetTests(unittest.TestCase):
    def test_fast_path_matches_per_call_predictions(self):
        (episode,), ((r_logit, e_logit),) = synthetic_episodes(1, informative=True, seed=4)
        model = PriorOffset(level=0.1, shift=0.2, scale=0.8, center=0.3, tau=1.1, sigma=0.6, prior=StubPrior())
        fast = prior_offset_alc([prior_states(episode, r_logit, e_logit)], [1.0], 0.1, 0.2, 0.8, 0.3, 1.1, 0.6)
        alc = 0.0
        for weight, budget in zip(alc_weights(BUDGETS), BUDGETS):
            labeled = [[[SUBJECT, item(l, t["task"])], y] for l, t, y in
                       zip(r_logit[:budget], episode["revealed_tags"][:budget], episode["revealed_labels"][:budget])]
            p = np.array([model(SUBJECT, item(l, t["task"]), labeled)
                          for l, t in zip(e_logit, episode["eval_tags"])])
            alc += weight * np.mean((p - episode["eval_correct"]) ** 2)
        self.assertAlmostEqual(fast, alc, places=10)

    def test_zero_scale_is_the_tag_offset(self):
        model = PriorOffset(level=0.1, shift=0.2, scale=0.0, tau=1.3, sigma=0.75, prior=StubPrior())
        tag = TagOffset(level=0.3, tau=1.3, sigma=0.75)
        labeled = [[[SUBJECT, item(2.0, t)], y] for t, y in zip(TASKS * 3, [1, 0, 1, 1, 1, 0, 0, 1, 1])]
        for task in TASKS + ["z"]:
            self.assertAlmostEqual(model(SUBJECT, item(-1.0, task), labeled),
                                   tag(SUBJECT, item(-1.0, task), labeled), places=10)

    def test_fit_trusts_an_informative_prior(self):
        episodes, logits = synthetic_episodes(150, informative=True)
        states = [prior_states(e, r, v) for e, (r, v) in zip(episodes, logits)]
        model = PriorOffset().fit_states(states, [1.0] * len(states), level=0.0, tau=1.0, center=0.0)
        self.assertGreaterEqual(model.scale, 0.75)

    def test_fit_ignores_a_useless_prior(self):
        episodes, logits = synthetic_episodes(150, informative=False)
        states = [prior_states(e, r, v) for e, (r, v) in zip(episodes, logits)]
        model = PriorOffset().fit_states(states, [1.0] * len(states), level=0.0, tau=1.0, center=0.0)
        self.assertLessEqual(model.scale, 0.25)


SUBJECT_COLUMNS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                   "harness_version", "reasoning_effort", "subject_features_extra", "display_name"]


def write_benchmark(root, name="toy", n_subjects=2, n_items=90):
    rng = np.random.default_rng(1)
    target = Path(root) / name / "formatted_tables"
    target.mkdir(parents=True)
    ids = [f"{name}-{k}" for k in range(n_items)]
    pd.DataFrame({"item_id": ids, "content": [f"question {k} about {'math' if k % 2 else 'law'}" for k in range(n_items)],
                  "item_features": [f"task={'x' if k % 3 else 'y'}" for k in range(n_items)]}).to_parquet(target / "items.parquet")
    subjects = pd.DataFrame({"subject_id": [f"s{k}" for k in range(n_subjects)]})
    for column in SUBJECT_COLUMNS:
        subjects[column] = ""
    subjects["normalized_name"] = ["Model 7B", "Model 70B"][:n_subjects]
    subjects.to_parquet(target / "subjects.parquet")
    rows = [{"subject_id": s, "item_id": i, "response": float(rng.integers(0, 2)), "trial": 1,
             "interactors": "tool" if s == "s1" else ""} for s in subjects["subject_id"] for i in ids]
    pd.DataFrame(rows).to_parquet(target / "responses.parquet")


class EpisodeVectorTests(unittest.TestCase):
    def test_cached_vectors_give_the_same_logits_as_live_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            write_benchmark(data)
            prior = LogisticPrior()
            prior.coef = np.random.default_rng(3).normal(size=prior.coef.size)
            prior.intercept = 0.2
            episodes = tag_episodes(str(data), "toy")
            cached = episode_logits(prior, episode_vectors(data, "toy"), episodes)
            reloaded = episode_logits(prior, episode_vectors(data, "toy"), episodes)
            tables = load_benchmark(data, "toy")
            first = tables.responses.drop_duplicates(["subject_id", "item_id"]).set_index(["subject_id", "item_id"])
            for episode, (revealed, evaluated) in zip(episodes, cached):
                subject = subject_record(tables.subjects.loc[episode["subject_id"]])
                for ids, logits in ((episode["revealed_ids"], revealed), (episode["eval_ids"], evaluated)):
                    for item_id, logit in zip(ids, logits):
                        stored = tables.items.loc[item_id]
                        live = item_record({"content": stored["content"], "item_features": stored["item_features"],
                                            "interactors": first.loc[(episode["subject_id"], item_id), "interactors"]},
                                           "toy")
                        self.assertAlmostEqual(prior.logit(subject, live), logit, places=10)
            for (a, b), (c, d) in zip(cached, reloaded):
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(b, d)


if __name__ == "__main__":
    unittest.main()
