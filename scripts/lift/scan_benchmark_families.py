"""List benchmark pairs that may belong in the same family group.

    python scripts/lift/scan_benchmark_families.py

Cheap signals pick candidate pairs: a shared source_url or paper_url, similar
names (digits and separators removed, one contained in the other, or a shared
prefix), or many shared subject_ids. Each candidate is then checked for
near-duplicate items. The output is evidence for a human decision in
configs/lift/benchmark_groups.json, not a grouping on its own.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paiec.folds import load_benchmark_groups  # noqa: E402
from paiec.schema import benchmark_dir, training_benchmarks  # noqa: E402

PIECE, STEP = 40, 10  # distinctive text pieces: 40 characters, every 10


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "payload" / "data")
    parser.add_argument("--groups", type=Path, default=ROOT / "configs" / "lift" / "benchmark_groups.json")
    parser.add_argument("--sample", type=int, default=3000, help="items sampled from the checked benchmark")
    parser.add_argument("--pool", type=int, default=30000, help="items sampled from the other benchmark")
    parser.add_argument("--min-shared-subjects", type=int, default=5)
    return parser.parse_args()


def url_key(url) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    return url.lower().strip().rstrip("/").replace("http://", "https://").replace("www.", "")


def similar_names(a: str, b: str) -> bool:
    bare_a, bare_b = re.sub(r"[\W\d_]+", "", a), re.sub(r"[\W\d_]+", "", b)
    contained = len(min(bare_a, bare_b, key=len)) >= 3 and (bare_a in bare_b or bare_b in bare_a)
    return contained or a.split("_")[0] == b.split("_")[0]


def text_pieces(texts: pd.Series) -> list[set[str]]:
    """Distinctive pieces per item; pieces repeated across many items are prompt templates."""
    pieces = []
    for text in texts:
        flat = re.sub(r"\W+", "", str(text).lower())
        pieces.append({flat[i:i + PIECE] for i in range(0, max(1, len(flat) - PIECE), STEP)})
    counts = collections.Counter(piece for item in pieces for piece in item)
    limit = max(3, 0.005 * len(pieces))
    return [{piece for piece in item if counts[piece] <= limit} for item in pieces]


def main():
    args = parse_arguments()
    names = training_benchmarks(args.data)
    tables = {name: benchmark_dir(args.data, name) for name in names}
    meta = {
        name: pd.read_parquet(tables[name] / "benchmarks.parquet").iloc[0] for name in names
    }
    subjects = {
        name: set(pd.read_parquet(tables[name] / "subjects.parquet", columns=["subject_id"])["subject_id"])
        for name in names
    }
    group_of = {
        member: number
        for number, group in enumerate(load_benchmark_groups(args.groups))
        for member in group
    }

    candidates = []
    for a, b in itertools.combinations(names, 2):
        reasons = [
            field for field in ("source_url", "paper_url")
            if url_key(meta[a][field]) and url_key(meta[a][field]) == url_key(meta[b][field])
        ]
        if similar_names(a, b):
            reasons.append("name")
        shared = len(subjects[a] & subjects[b])
        if shared >= args.min_shared_subjects:
            reasons.append(f"{shared} subjects")
        if reasons:
            candidates.append((a, b, reasons))
    print(f"{len(candidates)} candidate pairs among {len(names)} benchmarks; checking item text")

    cache = {}

    def pieces(name, n):
        if (name, n) not in cache:
            texts = pd.read_parquet(tables[name] / "items.parquet", columns=["content"])["content"].dropna()
            cache[name, n] = text_pieces(texts.sample(min(n, len(texts)), random_state=0))
        return cache[name, n]

    rows = []
    for a, b, reasons in candidates:
        # Check the smaller benchmark's items against a pool from the larger one.
        small, large = sorted((a, b), key=lambda name: meta[name]["n_items"])
        pool = set().union(*pieces(large, args.pool))
        checked = pieces(small, args.sample)
        overlap = sum(len(item & pool) >= 2 for item in checked) / max(1, len(checked))
        grouped = a in group_of and group_of.get(a) == group_of.get(b)
        rows.append((overlap, a, b, ", ".join(reasons), grouped))
    for overlap, a, b, reasons, grouped in sorted(rows, reverse=True):
        mark = "grouped" if grouped else ""
        print(f"{overlap:6.1%}  {a:28s} {b:28s} {reasons:40s} {mark}")


if __name__ == "__main__":
    main()
