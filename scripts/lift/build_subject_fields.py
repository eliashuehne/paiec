"""Write the predictor-visible subject fields for every training benchmark.

    python scripts/lift/build_subject_fields.py

Writes runs/features/subject_fields.parquet, keyed by subject_id.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paiec.schema import benchmark_dir, training_benchmarks  # noqa: E402
from paiec.subject_fields import COLUMNS, subject_fields  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "payload" / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "features" / "subject_fields.parquet")
    return parser.parse_args()


def main():
    args = parse_arguments()
    benchmarks = training_benchmarks(args.data)
    tables = (
        pd.read_parquet(benchmark_dir(args.data, name) / "subjects.parquet", columns=COLUMNS)
        for name in benchmarks
    )
    fields = subject_fields(tables)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields.to_parquet(args.out, index=False)
    named = fields["normalized_name"].notna()
    print(f"{len(fields)} subjects from {len(benchmarks)} benchmarks, "
          f"{named.mean():.0%} named ({fields.loc[named, 'normalized_name'].nunique()} distinct) -> {args.out}")


if __name__ == "__main__":
    main()
