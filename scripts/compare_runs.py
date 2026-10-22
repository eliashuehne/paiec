#!/usr/bin/env python
"""Compare completed MLflow evaluation runs, pair by pair, with uncertainty.

Brier is lower-is-better, so a positive delta means the run is worse than the
reference. Pairs are compared only where both runs scored the same items, which
is why the reference's item IDs are read from ``predictions.csv`` rather than
assumed from the suite name.

    # every sample_200 Laya run against the English 512 baseline
    .venv-laya-mlx/bin/python scripts/compare_runs.py --filter "params.suite = 'sample_200'"

    # explicit runs, first one is the reference
    .venv-laya-mlx/bin/python scripts/compare_runs.py --run e3f4ee7a --run 5bbe8fd4

    # save the tables
    .venv-laya-mlx/bin/python scripts/compare_runs.py --suite sample_200 --csv runs/compare
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from paiec.compare import (
    RunScores, alc_from_curves, by_benchmark, compare_to, curves_from_pairs,
    pair_curves, summary_table,
)
from paiec.tracking import local_tracking_uri

ROOT = Path(__file__).resolve().parents[1]
MISSING = float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", default=[],
                        help="MLflow run ID or unique prefix; repeat, first is the reference")
    parser.add_argument("--filter", default=None,
                        help="MLflow search filter, e.g. \"params.suite = 'sample_200'\"")
    parser.add_argument("--suite", default=None, help="shorthand for params.suite = '<name>'")
    parser.add_argument("--predictor", default=None, help="shorthand for params.predictor = '<name>'")
    parser.add_argument("--experiment", default="PAIEC")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--reference", default=None,
                        help="run ID/prefix to use as the baseline; default is the first selected")
    parser.add_argument("--allow-partial", action="store_true",
                        help="compare the shared subset instead of refusing mismatched runs")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--level", type=float, default=0.95, help="bootstrap interval level")
    parser.add_argument("--csv", default=None,
                        help="directory to write summary/comparison/per_benchmark tables")
    return parser.parse_args()


def build_filter(args: argparse.Namespace) -> str:
    terms = []
    if args.filter:
        terms.append(f"({args.filter})")
    if args.suite:
        terms.append(f"params.suite = '{args.suite}'")
    if args.predictor:
        terms.append(f"params.predictor = '{args.predictor}'")
    return " and ".join(terms)


def label_for(params: dict, run_id: str) -> str:
    """Short, readable run label from tracked parameters."""
    predictor = params.get("predictor", "unknown")
    if predictor != "laya":
        return predictor
    parts = [f"laya/{params.get('laya_model', '?')}/{params.get('laya_max_len', '?')}"]
    if params.get("laya_backend", "mlx") != "mlx":
        parts.append(params["laya_backend"])
    if params.get("laya_dtype", "auto") not in {"auto", "float16"}:
        parts.append(params["laya_dtype"])
    if params.get("laya_use_labels", "False") in {"True", "true"}:
        parts.append("labels")
    return " ".join(parts)


def unique_labels(runs: list[RunScores]) -> list[RunScores]:
    counts = pd.Series([run.label for run in runs]).value_counts()
    for run in runs:
        if counts[run.label] > 1:
            run.label = f"{run.label} [{run.metadata['run_id'][:8]}]"
    return runs


def load_run(client, run, directory: Path, allow_partial: bool) -> RunScores:
    run_id = run.info.run_id
    params = dict(run.data.params)
    label = label_for(params, run_id)
    metadata = {
        "run_id": run_id,
        "name": run.data.tags.get("mlflow.runName", run_id),
        "status": run.info.status,
        "params": params,
        "brier_alc_logged": run.data.metrics.get("brier_alc"),
    }
    artifacts = {entry.path for entry in client.list_artifacts(run_id)}
    if "predictions.csv" in artifacts:
        path = client.download_artifacts(run_id, "predictions.csv", str(directory))
        items = pd.read_csv(path)
        return RunScores(label, f"mlflow:{run_id[:8]}", alc_from_curves(pair_curves(items)),
                         items, metadata)
    if allow_partial:
        path = client.download_artifacts(run_id, "scores/per_pair_budget.csv", str(directory))
        scores = pd.read_csv(path)
        return RunScores(label, f"mlflow:{run_id[:8]} (pair-level only)",
                         alc_from_curves(curves_from_pairs(scores)), None, metadata)
    raise SystemExit(
        f"run {run_id[:8]} has no predictions.csv, so its items cannot be matched; "
        "pass --allow-partial to compare pair-level scores instead"
    )


def resolve(client, spec: str, experiment_id: str):
    """Resolve a full run ID or unique prefix within the experiment."""
    if len(spec) == 32:
        try:
            return client.get_run(spec)
        except Exception:
            pass
    matches = [run for run in client.search_runs([experiment_id])
               if run.info.run_id.startswith(spec)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"no run in this experiment matches {spec!r}")
    listing = ", ".join(f"{run.info.run_id[:8]} ({run.data.tags.get('mlflow.runName', '')})"
                        for run in matches[:8])
    raise SystemExit(f"{spec!r} matches {len(matches)} runs: {listing}")


def print_summary(frame: pd.DataFrame, reference: RunScores) -> None:
    print("runs (lower is better)")
    view = frame.copy()
    view["delta"] = view["brier_alc"] - reference.brier_alc
    for column in ("brier_alc", "sd", "delta"):
        view[column] = view[column].map(lambda value: f"{value:+.4f}" if column == "delta"
                                        else (f"{value:.4f}" if value == value else "n/a"))
    print(view.to_string(index=False))


def print_comparisons(results, reference: RunScores) -> None:
    print(f"\npaired vs {reference.label} (positive delta = worse; "
          f"n pairs, wins/losses for the challenger)")
    rows = []
    for result in results:
        rows.append({
            "run": result.label,
            "delta": f"{result.mean_delta:+.4f}",
            "median": f"{result.median_delta:+.4f}",
            "ci": f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]",
            "p": f"{result.p:.3f}",
            "w/l/t": f"{result.wins}/{result.losses}/{result.ties}",
            "pairs": result.n_pairs,
            "subset": "partial" if result.partial else "full",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\ndelta = challenger mean pair ALC minus reference; p is a two-sided paired "
          "t-test over pairs.\nWith 10 pairs only large, consistent differences reach "
          "significance; read the interval too.")


def main() -> int:
    args = parse_args()
    try:
        from mlflow import MlflowClient
    except ImportError:
        print("MLflow is not installed in this environment; use .venv-laya-mlx.", file=sys.stderr)
        return 2
    client = MlflowClient(tracking_uri=args.tracking_uri or local_tracking_uri(ROOT))
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(f"experiment {args.experiment!r} not found")

    selected = [resolve(client, spec, experiment.experiment_id) for spec in args.run]
    search = build_filter(args)
    if not selected and not search:
        raise SystemExit("pass --run, --suite, --filter, or --predictor to choose runs")
    if search:
        selected.extend(run for run in client.search_runs(
            [experiment.experiment_id], filter_string=search,
            order_by=["attributes.start_time ASC"]))
    finished = [run for run in selected if run.info.status == "FINISHED"]
    for run in selected:
        if run.info.status != "FINISHED":
            print(f"skipping {run.info.run_id[:8]}: status {run.info.status} "
                  "(no partial predictions are recoverable)")
    if not finished:
        raise SystemExit("no finished runs selected")

    with tempfile.TemporaryDirectory() as directory:
        runs = unique_labels([load_run(client, run, Path(directory), args.allow_partial)
                              for run in finished])
        ordered = list(runs)
        if args.reference:
            wanted = resolve(client, args.reference, experiment.experiment_id).info.run_id
            match = [run for run in ordered if run.metadata["run_id"] == wanted]
            if not match:
                raise SystemExit("the reference run was not among the selected runs")
            ordered.remove(match[0])
            ordered.insert(0, match[0])
        reference, others = ordered[0], ordered[1:]

        for run in runs:
            logged = run.metadata.get("brier_alc_logged")
            if logged is not None and abs(logged - run.brier_alc) > 1e-6:
                print(f"warning: {run.label} reconstructs {run.brier_alc:.4f} but logged "
                      f"{logged:.4f}; its predictions and scores disagree")

        print_summary(summary_table(runs), reference)
        if others:
            results = []
            for run in others:
                try:
                    results.append(compare_to(reference, run, seed=args.seed,
                                              strict=not args.allow_partial, level=args.level))
                except ValueError as exc:
                    raise SystemExit(f"{exc}") from None
            print_comparisons(results, reference)

        table = by_benchmark({run.label: run for run in runs})
        if len(table.columns) > 1:
            print("\nper-benchmark mean pair ALC")
            print(table.to_string(float_format=lambda value: f"{value:.4f}"))

        if args.csv:
            out = Path(args.csv)
            out.mkdir(parents=True, exist_ok=True)
            summary_table(runs).to_csv(out / "summary.csv", index=False)
            if others:
                pd.DataFrame([result.__dict__ for result in results]).to_csv(
                    out / "comparisons.csv", index=False)
            table.to_csv(out / "per_benchmark.csv")
            print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
