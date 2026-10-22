"""Cross-benchmark subject statistics for the prior predictors.

The runner builds these counts only from public benchmarks outside the full
evaluation suite. It keys caches by the training selection and data manifest,
so a lookup never needs to identify an anonymous test benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import benchmark_dir, canonical_identity, clean

def _identity_series(subjects: pd.DataFrame) -> pd.Series:
    """Use only the published normalized name, without display-name enrichment."""
    normalized = subjects["normalized_name"].astype("string").fillna("").str.strip()
    return normalized.map(canonical_identity)


def build_subject_stats(
    data_root: str | Path,
    benchmarks: list[str],
) -> pd.DataFrame:
    """Count responses per (benchmark, subject identity).

    Reads only two columns per benchmark; traces are never touched.
    """
    frames = []
    for benchmark in benchmarks:
        directory = benchmark_dir(data_root, benchmark)
        subjects = pd.read_parquet(
            directory / "subjects.parquet",
            columns=["subject_id", "normalized_name"],
        )
        mapping = dict(zip(subjects["subject_id"], _identity_series(subjects)))
        responses = pd.read_parquet(
            directory / "response.parquet", columns=["subject_id", "response"]
        )
        responses = responses[responses["response"].isin([0.0, 1.0])]
        responses = responses.assign(
            identity=responses["subject_id"].map(mapping).fillna("")
        )
        grouped = (
            responses.groupby("identity")["response"]
            .agg(n="size", successes="sum")
            .reset_index()
        )
        grouped["benchmark"] = benchmark.split("/")[0]
        frames.append(grouped)
    stats = pd.concat(frames, ignore_index=True)
    stats["successes"] = stats["successes"].astype(int)
    stats["n"] = stats["n"].astype(int)
    return stats[["benchmark", "identity", "n", "successes"]]


def load_or_build_subject_stats(
    data_root: str | Path,
    benchmarks: list[str],
    cache: str | Path,
) -> pd.DataFrame:
    """Build the counts once and reuse them across runs."""
    path = Path(cache)
    if path.is_file():
        return pd.read_parquet(path)
    stats = build_subject_stats(data_root, benchmarks)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(path, index=False)
    return stats


__all__ = ["build_subject_stats", "load_or_build_subject_stats", "clean"]
