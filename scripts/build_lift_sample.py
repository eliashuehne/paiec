"""Build the fixed lift sample: capped responses per benchmark, with folds.

    python scripts/build_lift_sample.py

Writes runs/lift/sample.parquet and runs/lift/sample_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paiec.folds import grouped_benchmark_folds, load_benchmark_groups  # noqa: E402
from paiec.lift_sample import assign_folds, build_sample, fold_labels, summarise_sample  # noqa: E402
from paiec.schema import load_benchmark, training_benchmarks  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "payload" / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "lift")
    parser.add_argument("--groups", type=Path, default=ROOT / "configs" / "lift" / "benchmark_groups.json")
    parser.add_argument("--cap", type=int, default=20_000, help="responses per benchmark")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sealed-fraction", type=float, default=0.2)
    parser.add_argument("--force", action="store_true", help="overwrite an existing sample")
    return parser.parse_args()


def data_revision(data_root: Path) -> str | None:
    """Read the dataset revision our downloader records beside the data."""
    manifest = data_root / "manifest.json"
    return json.loads(manifest.read_text()).get("revision") if manifest.is_file() else None


def main():
    args = parse_arguments()
    sample_path = args.out / "sample.parquet"
    if sample_path.exists() and not args.force:
        sys.exit(f"{sample_path} exists; pass --force to rebuild it")

    benchmarks = training_benchmarks(args.data)
    started = time.time()

    def tables():
        for count, benchmark in enumerate(benchmarks, start=1):
            yield benchmark, load_benchmark(args.data, benchmark).responses
            if count % 20 == 0 or count == len(benchmarks):
                print(f"  {count}/{len(benchmarks)} ({time.time() - started:.0f}s)", flush=True)

    sample = build_sample(tables(), cap=args.cap, seed=args.seed)
    # Folds are drawn only over benchmarks that have labelled responses.
    sampled = sorted(sample["benchmark"].unique())
    no_labels = sorted(set(benchmarks) - set(sampled))
    sealed, folds = grouped_benchmark_folds(
        sampled,
        load_benchmark_groups(args.groups),
        k=args.folds,
        seed=args.seed,
        sealed_fraction=args.sealed_fraction,
    )
    fold_of = fold_labels(sealed, folds)
    sample = assign_folds(sample, fold_of)
    print(f"{len(sampled)} benchmarks with labels: {len(sealed)} sealed, {args.folds} folds; "
          f"skipped {no_labels}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False)
    manifest = {
        "data_revision": data_revision(args.data),
        "cap": args.cap,
        "seed": args.seed,
        "folds": args.folds,
        "sealed_fraction": args.sealed_fraction,
        "groups_sha256": hashlib.sha256(args.groups.read_bytes()).hexdigest(),
        "skipped_no_labels": no_labels,
        "sealed": sealed,
        "fold_of": fold_of,
        **summarise_sample(sample),
        "created_unix": int(time.time()),
    }
    (args.out / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{manifest['rows']} rows from {manifest['benchmarks']} benchmarks -> {sample_path}")


if __name__ == "__main__":
    main()
