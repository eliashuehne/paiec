"""Build the subject feature table from the visible subject fields.

    python scripts/lift/build_subject_fields.py      # once, if missing
    python scripts/subjects/build_subject_features.py

Writes runs/features/subjects.parquet (one row per subject_id) and
runs/features/subjects_manifest.json with the fill rate of every column, by
subjects and by lift-sample rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paiec.subject_features import COLUMNS, build_subject_features, load_families  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fields", type=Path, default=ROOT / "runs" / "features" / "subject_fields.parquet")
    parser.add_argument("--families", type=Path, default=ROOT / "configs" / "subjects" / "model_families.json")
    parser.add_argument("--sample", type=Path, default=ROOT / "runs" / "lift" / "sample.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "features" / "subjects.parquet")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fill_rates(table: pd.DataFrame, weights: pd.Series | None) -> dict:
    """Share of subjects, and of sample rows, with each column filled (flags: share set to 1)."""
    rates = {}
    for column in COLUMNS[1:]:
        filled = table[column].eq(1) if column.startswith("is_") else table[column].notna()
        rates[column] = {"subjects": round(float(filled.mean()), 4)}
        if weights is not None:
            rates[column]["sample_rows"] = round(float(weights[filled].sum() / weights.sum()), 4)
    return rates


def main():
    args = parse_arguments()
    if not args.fields.is_file():
        sys.exit(f"{args.fields} is missing; run scripts/lift/build_subject_fields.py first")
    table = build_subject_features(pd.read_parquet(args.fields), load_families(args.families))
    weights = None
    if args.sample.is_file():
        counts = pd.read_parquet(args.sample, columns=["subject_id"])["subject_id"].value_counts()
        weights = table["subject_id"].map(counts).fillna(0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    rates = fill_rates(table, weights)
    manifest = {
        "subjects": int(len(table)),
        "fields_sha256": sha256(args.fields),
        "families_sha256": sha256(args.families),
        "fill_rates": rates,
    }
    args.out.with_name("subjects_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(table)} subjects -> {args.out}")
    print(f"{'column':22s} subjects  sample rows")
    for column, rate in rates.items():
        print(f"{column:22s} {rate['subjects']:7.1%}  {rate.get('sample_rows', float('nan')):7.1%}")


if __name__ == "__main__":
    main()
