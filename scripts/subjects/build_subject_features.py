"""Build the subject feature table from the visible subject fields.

    python scripts/lift/build_subject_fields.py      # once, if missing
    python scripts/subjects/build_subject_features.py

Writes runs/features/subjects.parquet (one row per subject_id) and
runs/features/subjects_manifest.json with the fill rate of every column, by
subjects and by lift-sample rows. Missing release dates are estimated from
family, product line and version (src/paiec/subject_dates.py), using
references from other lift folds only; only sources that pass the back-test
are kept, and the back-test is recorded in the manifest.
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

from paiec.subject_dates import (  # noqa: E402
    add_version_columns,
    backtest,
    fill_release_dates,
    subject_folds,
    summarise_backtest,
)
from paiec.subject_features import COLUMNS, build_subject_features, load_families  # noqa: E402

DATE_COLUMNS = ["line", "version", "name_date_days", "release_days_filled", "release_year_filled",
                "release_lo", "release_hi", "release_source", "release_filled"]


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
    for column in [*COLUMNS[1:], *DATE_COLUMNS]:
        filled = table[column].eq(1) if column.startswith("is_") or column == "release_filled" else table[column].notna()
        rates[column] = {"subjects": round(float(filled.mean()), 4)}
        if weights is not None:
            rates[column]["sample_rows"] = round(float(weights[filled].sum() / weights.sum()), 4)
    return rates


def main():
    args = parse_arguments()
    if not args.fields.is_file():
        sys.exit(f"{args.fields} is missing; run scripts/lift/build_subject_fields.py first")
    if not args.sample.is_file():
        sys.exit(f"{args.sample} is missing; run scripts/lift/build_lift_sample.py first (dates use its folds)")
    families = load_families(args.families)
    table = add_version_columns(build_subject_features(pd.read_parquet(args.fields), families), families)
    sample = pd.read_parquet(args.sample, columns=["subject_id", "fold"])
    weights = table["subject_id"].map(sample["subject_id"].value_counts()).fillna(0)

    sampled_folds = subject_folds(sample)
    # Eligibility filtering can omit sealed-only subjects. Without membership
    # evidence, retain their fields in the export but never use them as references.
    folds_of = {s: sampled_folds.get(s, frozenset({"sealed"})) for s in table["subject_id"]}
    unsealed = table.loc[table["subject_id"].map(lambda s: "sealed" not in folds_of.get(s, frozenset()))]
    # These pooled reports are diagnostics only, never the method-selection gate.
    reports = {mode: summarise_backtest(backtest(unsealed, mode)) for mode in ("identity", "signature")}
    dates = fill_release_dates(table, folds_of, accepted=None)
    table = table.merge(dates, on="subject_id", how="left", validate="one_to_one")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    rates = fill_rates(table, weights)
    manifest = {
        "subjects": int(len(table)),
        "fields_sha256": sha256(args.fields),
        "families_sha256": sha256(args.families),
        "sample_sha256": sha256(args.sample),
        "date_source_selection": "backtest_per_permitted_reference_set",
        "date_backtest": {mode: report.round(4).to_dict("records") for mode, report in reports.items()},
        "release_sources": table["release_source"].value_counts().to_dict(),
        "fill_rates": rates,
    }
    args.out.with_name("subjects_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(table)} subjects -> {args.out}")
    for mode, report in reports.items():
        print(f"date back-test, leaving out each {mode}:")
        print(report.round(3).to_string(index=False))
    print("date sources selected separately from each permitted reference set")
    print(f"{'column':22s} subjects  sample rows")
    for column, rate in rates.items():
        print(f"{column:22s} {rate['subjects']:7.1%}  {rate.get('sample_rows', float('nan')):7.1%}")


if __name__ == "__main__":
    main()
