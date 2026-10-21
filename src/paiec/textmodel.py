"""A cheap text-only control: hashed word n-grams and ridge regression.

The point of this predictor is falsification. If a 2,000-dimensional ridge on
the item text cannot beat the subject prior, then whatever Laya adds has to be
worth the whole training and packaging effort.
"""

from __future__ import annotations

import re
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

from .schema import benchmark_dir

_TOKEN = re.compile(r"[a-z0-9]+")
_MAX_TOKENS = 400
_DIMS = 2048
_MIN, _MAX = 0.02, 0.98


def _hash(token: str, dims: int) -> tuple[int, float]:
    value = crc32(token.encode("utf-8"))
    return value % dims, 1.0 if value & 1 else -1.0


def features(text: str, dims: int = _DIMS) -> np.ndarray:
    """Signed hashed counts over word unigrams and bigrams."""
    tokens = _TOKEN.findall(str(text).lower())[:_MAX_TOKENS]
    vector = np.zeros(dims, dtype=np.float32)
    for token in tokens:
        index, sign = _hash(token, dims)
        vector[index] += sign
    for left, right in zip(tokens, tokens[1:]):
        index, sign = _hash(left + "_" + right, dims)
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class TextRidge:
    """Ridge on hashed item text, with a global intercept."""

    name = "text_ridge"

    def __init__(self, l2: float = 1.0):
        self.l2 = float(l2)
        self.dims = _DIMS
        self.weights: np.ndarray | None = None
        self.intercept = 0.5

    def fit(self, texts, labels) -> "TextRidge":
        matrix = np.stack([features(text, self.dims) for text in texts])
        y = np.asarray(labels, dtype=np.float32)
        self.intercept = float(y.mean())
        gram = matrix.T @ matrix
        gram += self.l2 * np.eye(self.dims, dtype=np.float32)
        self.weights = np.linalg.solve(gram, matrix.T @ (y - self.intercept))
        return self

    def __call__(self, subject: dict, item: dict, labeled: list) -> float:
        if self.weights is None:
            raise RuntimeError("TextRidge.predict called before fit")
        value = self.intercept + float(features(item.get("item_content", "")) @ self.weights)
        return min(max(value, _MIN), _MAX)


def build_text_training(
    data_root: str | Path,
    benchmarks: list[str],
    *,
    per_benchmark: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample item text and binary outcomes from the training benchmarks."""
    rng = np.random.default_rng(seed)
    frames = []
    for benchmark in benchmarks:
        directory = benchmark_dir(data_root, benchmark)
        items = pd.read_parquet(directory / "items.parquet", columns=["item_id", "content"])
        responses = pd.read_parquet(
            directory / "response.parquet", columns=["item_id", "response"]
        )
        responses = responses[responses["response"].isin([0.0, 1.0])]
        merged = responses.merge(items, on="item_id", how="inner")
        merged = merged.drop_duplicates("item_id")
        if merged.empty:
            continue
        take = min(per_benchmark, len(merged))
        chosen = merged.iloc[rng.choice(len(merged), size=take, replace=False)]
        frames.append(
            pd.DataFrame(
                {
                    "benchmark": benchmark.split("/")[0],
                    "text": chosen["content"].fillna("").astype(str),
                    "label": chosen["response"].astype(int),
                }
            )
        )
    if not frames:
        raise ValueError("no training text was collected")
    return pd.concat(frames, ignore_index=True)


def load_or_build_text_training(
    data_root: str | Path,
    benchmarks: list[str],
    cache: str | Path,
    *,
    per_benchmark: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Cache the sampled texts so repeated runs skip 53 parquet reads."""
    path = Path(cache)
    if path.is_file():
        return pd.read_parquet(path)
    frame = build_text_training(
        data_root, benchmarks, per_benchmark=per_benchmark, seed=seed
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


__all__ = ["TextRidge", "features", "build_text_training", "load_or_build_text_training"]
