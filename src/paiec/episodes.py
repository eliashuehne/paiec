"""Build training records that replay the evaluator's item splits."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .metrics import BUDGETS
from .schema import benchmark_dir, response_table_path
from .split import make_split


EPISODE_COLUMNS = ("benchmark", "subject_id", "accuracy", "eval_mean") + tuple(
    f"k{budget}" for budget in BUDGETS
)


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
    first_response = responses.drop_duplicates("item_id").set_index("item_id")["response"]
    if first_response.size < min_items:
        return None

    split = make_split(
        tuple(first_response.index),
        benchmark=benchmark,
        subject_id=str(subject_id),
        seed=seed,
    )
    revealed = first_response.loc[list(split.acquisition[: max(BUDGETS)])].to_numpy()
    cumulative_correct = np.concatenate([[0.0], np.cumsum(revealed)])
    evaluation_responses = responses[responses["item_id"].isin(split.evaluation)]

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
    episodes = [
        episode
        for subject_id, subject_responses in responses.groupby("subject_id", sort=True)
        if (
            episode := episode_for_subject(
                subject_responses,
                benchmark_name,
                subject_id,
                seed,
                min_items,
            )
        )
        is not None
    ]
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
    episodes["weight"] = np.minimum(
        1.0, pairs_per_benchmark / episodes_per_benchmark
    )
    return episodes


__all__ = [
    "EPISODE_COLUMNS",
    "episode_for_subject",
    "load_valid_responses",
    "pair_episodes",
    "training_episodes",
]
