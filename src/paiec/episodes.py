"""Build training records that replay the evaluator's item splits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .groups import parse_tags
from .metrics import BUDGETS
from .schema import benchmark_dir, response_table_path
from .split import Split, make_split, stable_seed


EPISODE_COLUMNS = ("benchmark", "subject_id", "accuracy", "eval_mean") + tuple(
    f"k{budget}" for budget in BUDGETS
)


@dataclass
class _SubjectEpisode:
    """One subject's responses and the item split shared by training formats."""

    responses: pd.DataFrame
    first_labels: pd.Series
    split: Split

    @classmethod
    def build(
        cls,
        responses: pd.DataFrame,
        benchmark: str,
        subject_id: str,
        seed: int,
        min_items: int,
    ) -> _SubjectEpisode | None:
        first_labels = responses.drop_duplicates("item_id").set_index("item_id")[
            "response"
        ]
        if first_labels.size < min_items:
            return None

        split = make_split(
            tuple(first_labels.index),
            benchmark=benchmark,
            subject_id=str(subject_id),
            seed=seed,
        )
        return cls(responses, first_labels, split)

    @property
    def revealed_items(self) -> tuple[str, ...]:
        return self.split.acquisition[: max(BUDGETS)]

    @property
    def revealed_labels(self) -> np.ndarray:
        # Revealing an item exposes its first trial, even if more trials exist.
        return self.first_labels.loc[list(self.revealed_items)].to_numpy()

    def evaluation_responses(self, limit: int | None = None) -> pd.DataFrame:
        item_ids = self.split.evaluation[:limit]
        # Scoring includes every trial of each selected evaluation item.
        return self.responses[self.responses["item_id"].isin(item_ids)]


def load_valid_responses(data_root: str, benchmark: str) -> pd.DataFrame:
    """Load binary responses whose subjects and items are present locally."""
    directory = benchmark_dir(data_root, benchmark)
    subject_ids = pd.read_parquet(
        directory / "subjects.parquet", columns=["subject_id"]
    )["subject_id"]
    item_ids = pd.read_parquet(directory / "items.parquet", columns=["item_id"])[
        "item_id"
    ]
    responses = pd.read_parquet(
        response_table_path(directory),
        columns=["subject_id", "item_id", "response"],
    )
    return responses[
        responses["response"].isin([0.0, 1.0])
        & responses["item_id"].isin(item_ids)
        & responses["subject_id"].isin(subject_ids)
    ]


def episode_for_subject(
    responses: pd.DataFrame,
    benchmark: str,
    subject_id: str,
    seed: int,
    min_items: int,
) -> dict | None:
    """Summarize one scoreable subject/benchmark pair for model fitting."""
    subject = _SubjectEpisode.build(responses, benchmark, subject_id, seed, min_items)
    if subject is None:
        return None

    cumulative_correct = np.concatenate([[0.0], np.cumsum(subject.revealed_labels)])
    evaluation_responses = subject.evaluation_responses()

    episode = {
        "benchmark": benchmark,
        "subject_id": str(subject_id),
        "accuracy": float(responses["response"].mean()),
        "eval_mean": float(evaluation_responses["response"].mean()),
    }
    episode.update(
        {f"k{budget}": float(cumulative_correct[budget]) for budget in BUDGETS}
    )
    return episode


@lru_cache(maxsize=None)
def pair_episodes(
    data_root: str, benchmark: str, seed: int = 0, min_items: int = 80
) -> pd.DataFrame:
    """Return one training episode for every scoreable subject in a benchmark."""
    benchmark_name = benchmark.split("/")[0]
    responses = load_valid_responses(data_root, benchmark)
    episodes = []
    for subject_id, subject_responses in responses.groupby("subject_id", sort=True):
        episode = episode_for_subject(
            subject_responses, benchmark_name, subject_id, seed, min_items
        )
        if episode is not None:
            episodes.append(episode)
    return pd.DataFrame(episodes, columns=EPISODE_COLUMNS)


def training_episodes(
    data_root: str,
    benchmarks: list[str],
    seed: int = 0,
    pairs_per_benchmark: int = 20,
) -> pd.DataFrame:
    """Stack episodes of several benchmarks with a ``weight`` column.

    Each benchmark contributes at most ``pairs_per_benchmark`` total weight,
    matching run_eval's default cap, so a few benchmarks with thousands of
    subjects do not dominate what is learned.
    """
    frames = [pair_episodes(str(data_root), benchmark, seed) for benchmark in benchmarks]
    nonempty_frames = [frame for frame in frames if not frame.empty]
    if not nonempty_frames:
        return pd.DataFrame(columns=(*EPISODE_COLUMNS, "weight"))

    episodes = pd.concat(nonempty_frames, ignore_index=True)
    episodes_per_benchmark = episodes.groupby("benchmark")["benchmark"].transform("size")
    episodes["weight"] = np.minimum(1.0, pairs_per_benchmark / episodes_per_benchmark)
    return episodes


def _sample_subjects(
    responses: pd.DataFrame,
    benchmark: str,
    seed: int,
    max_pairs: int,
    min_items: int,
) -> list:
    """Sample scoreable subjects without letting repeated trials affect eligibility."""
    distinct_items = responses.drop_duplicates(["subject_id", "item_id"])
    item_counts = distinct_items.groupby("subject_id").size()
    eligible = sorted(item_counts[item_counts >= min_items].index)
    if not eligible:
        return []

    rng = np.random.default_rng(stable_seed(seed, benchmark, "tags"))
    chosen = rng.choice(eligible, size=min(max_pairs, len(eligible)), replace=False)
    return sorted(chosen)


def _tag_record(
    episode: _SubjectEpisode, subject_id: str, item_tags: pd.Series, evaluation_items: int
) -> dict:
    evaluation = episode.evaluation_responses(limit=evaluation_items)
    scores = evaluation.groupby("item_id")["response"].agg(["size", "sum"])
    return {
        "weight": 1.0,
        "subject_id": str(subject_id),
        "revealed_ids": list(episode.revealed_items),
        "eval_ids": list(scores.index),
        "revealed_tags": [item_tags[item_id] for item_id in episode.revealed_items],
        "revealed_labels": episode.revealed_labels,
        "eval_tags": [item_tags[item_id] for item_id in scores.index],
        "eval_rows": scores["size"].to_numpy(dtype=float),
        "eval_correct": scores["sum"].to_numpy(dtype=float),
    }


@lru_cache(maxsize=None)
def tag_episodes(
    data_root: str,
    benchmark: str,
    seed: int = 0,
    max_pairs: int = 20,
    evaluation_items: int = 200,
    min_items: int = 80,
) -> tuple[dict, ...]:
    """Sample subject episodes with tags for the revealed and evaluation items.

    Each episode includes the first 31 revealed labels. Evaluation records retain
    the trial count and correct count for each item, capped at ``evaluation_items``.
    """
    benchmark_name = benchmark.split("/")[0]
    directory = benchmark_dir(data_root, benchmark)
    items = pd.read_parquet(
        directory / "items.parquet", columns=["item_id", "item_features"]
    )
    items = items.drop_duplicates("item_id").set_index("item_id")
    item_tags = items["item_features"].map(parse_tags)
    responses = load_valid_responses(data_root, benchmark)
    subjects = _sample_subjects(responses, benchmark_name, seed, max_pairs, min_items)

    episodes = []
    for subject_id in subjects:
        subject_responses = responses[responses["subject_id"] == subject_id]
        episode = _SubjectEpisode.build(
            subject_responses, benchmark_name, subject_id, seed, min_items
        )
        if episode is not None:
            episodes.append(_tag_record(episode, subject_id, item_tags, evaluation_items))
    return tuple(episodes)


__all__ = [
    "EPISODE_COLUMNS",
    "episode_for_subject",
    "load_valid_responses",
    "pair_episodes",
    "tag_episodes",
    "training_episodes",
]
