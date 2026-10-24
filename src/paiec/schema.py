"""Competition-shaped records built from the public measurement tables.

The evaluator hands `predict` two dictionaries: eight subject fields and four
item fields, every value a plain string, missing values as "". The public
tables store the same information under different names and use NaN freely, so
all table access goes through the helpers here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

SUBJECT_FIELDS = (
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "harness_version",
    "reasoning_effort",
    "subject_features_extra",
)

ITEM_FIELDS = (
    "item_content",
    "item_features",
    "interactors",
    "benchmark_id",
)

_NULLS = {"", "nan", "none", "nat", "<na>", "null"}

_PREFIX = re.compile(r"^[^/]*\s+/\s+")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_NON_KEY = re.compile(r"[^a-z0-9.+_-]+")


def canonical_identity(name: str) -> str:
    """Reduce a system name to a comparable key.

    The public tables name the same system differently across benchmarks:
    ``CMMLU / gemma3_27b_it`` carries the benchmark as a prefix and
    ``UnifiedQA-T5-11B (RACE prompt)`` carries the prompt in parentheses.
    Without this reduction a cross-benchmark prior matches almost nothing.
    """
    text = clean(name).lower()
    text = _PREFIX.sub("", text)
    text = _PARENTHETICAL.sub(" ", text)
    text = _NON_KEY.sub(" ", text)
    return " ".join(text.split())


def clean(value) -> str:
    """Render a table cell as the non-null string the evaluator would supply."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _NULLS else text


def benchmark_dir(data_root: str | Path, benchmark: str) -> Path:
    """Resolve one benchmark's table directory.

    Manifest entries may already end in ``formatted_tables``; accept either
    spelling so callers can pass whatever the manifest recorded.
    """
    root = Path(data_root)
    name = benchmark.split("/")[0]
    for candidate in (root / name / "formatted_tables", root / name):
        if (candidate / "items.parquet").is_file():
            return candidate
    raise FileNotFoundError(f"no measurement tables under {root / name}")


def response_table_path(directory: Path) -> Path:
    """Find the response table under either supported upstream file name."""
    for filename in ("responses.parquet", "response.parquet"):
        path = directory / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"no response table in {directory}")


def training_benchmarks(data_root: str | Path) -> list[str]:
    """List benchmarks usable for training: binary, item-level and publicly released.

    ``release == "withheld"`` marks benchmarks the dataset card excludes from
    the active public release.
    """
    names = []
    for path in sorted(Path(data_root).glob("*/formatted_tables/benchmarks.parquet")):
        meta = pd.read_parquet(path).iloc[0]
        if (
            meta.get("response_type") == "binary"
            and meta.get("granularity") == "item"
            and meta.get("release") == "public"
            and (path.parent / "items.parquet").is_file()
        ):
            names.append(path.parent.parent.name)
    return names


@dataclass(frozen=True)
class BenchTables:
    """One benchmark's public tables, indexed for lookup by id."""

    benchmark: str
    subjects: pd.DataFrame
    items: pd.DataFrame
    responses: pd.DataFrame


def load_benchmark(data_root: str | Path, benchmark: str) -> BenchTables:
    """Load the three tables the harness needs, dropping non-binary responses.

    Traces are never read: they are 23 of the payload's 24 GB and no predictor
    here uses them.
    """
    directory = benchmark_dir(data_root, benchmark)
    name = benchmark.split("/")[0]

    subjects = pd.read_parquet(directory / "subjects.parquet")
    subjects = subjects.set_index("subject_id", drop=False)

    items = pd.read_parquet(
        directory / "items.parquet",
        columns=["item_id", "content", "item_features"],
    )
    items = items.drop_duplicates("item_id").set_index("item_id", drop=False)

    # Our downloader keeps the upstream plural name. The baseline renames it.
    responses = pd.read_parquet(
        response_table_path(directory),
        columns=["subject_id", "item_id", "response", "trial", "interactors"],
    )
    responses = responses.rename(columns={"response": "label"})
    responses = responses[responses["label"].isin([0.0, 1.0])].copy()
    responses["label"] = responses["label"].astype(int)
    # Keep only responses whose item text is present; a missing item cannot be
    # predicted and would silently turn into an empty-string prompt.
    responses = responses[responses["item_id"].isin(items.index)]

    return BenchTables(name, subjects, items, responses)


def subject_record(row) -> dict:
    """Build the eight-field subject dictionary for one subject row.

    ``normalized_name`` is the evaluator's canonical identity, but the public
    tables leave it empty for many subjects. Falling back to ``display_name``
    keeps the identity available locally without inventing a string.
    """
    record = {field: clean(row.get(field)) for field in SUBJECT_FIELDS}
    if not record["normalized_name"]:
        record["normalized_name"] = clean(row.get("display_name"))
    return record


def item_record(row, benchmark: str) -> dict:
    """Build the four-field item dictionary for one paired item row."""
    return {
        "item_content": clean(row.get("content")),
        "item_features": clean(row.get("item_features")),
        "interactors": clean(row.get("interactors")),
        "benchmark_id": benchmark,
    }
