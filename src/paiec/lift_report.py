"""Compare two lift runs, flag benchmark fingerprints, and keep a record of every run.

A feature is judged by paired differences: candidate minus base in the same
fold, on the same benchmarks. Raw per-fold scores swing with each fold's base
rate (a fold full of hard benchmarks punishes every model alike), so their
spread says little; the spread of paired differences is what a gain must beat.

The verdict is a screening rule for the fast loop, not a final answer: confirm
winners with ``scripts/run_eval.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .lift import LiftResult

# Share of a column explained by which benchmark a row comes from, above which
# the model can use the column to recognise benchmarks.
FINGERPRINT_WARNING = 0.8
# Folds that must agree before the verdict calls a change helpful or harmful.
FOLDS_TO_AGREE = 4
# Smallest mean Brier change the verdict acts on. A pure noise column costs a
# steady +0.0003 (the model fits it a little), so smaller changes are noise.
MIN_DELTA = 0.001


@dataclass(frozen=True)
class Comparison:
    """Paired differences between a candidate run and its base (negative = better)."""

    folds: pd.DataFrame
    benchmarks: pd.DataFrame
    summary: dict


def compare(base: LiftResult, candidate: LiftResult) -> Comparison:
    """Pair two runs fold by fold and benchmark by benchmark."""
    on = ["fold", "benchmark"]
    if not base.benchmarks[on].equals(candidate.benchmarks[on]):
        raise ValueError("runs were scored on different folds or benchmarks")
    folds = base.folds[["fold", "benchmarks", "rows"]].copy()
    for metric in ("brier", "brier_macro", "log_loss"):
        folds[f"base_{metric}"] = base.folds[metric]
        folds[f"candidate_{metric}"] = candidate.folds[metric]
        folds[f"delta_{metric}"] = candidate.folds[metric] - base.folds[metric]
    benchmarks = base.benchmarks[[*on, "rows"]].assign(
        base_brier=base.benchmarks["brier"],
        candidate_brier=candidate.benchmarks["brier"],
        delta_brier=candidate.benchmarks["brier"] - base.benchmarks["brier"],
    )
    delta = folds["delta_brier"]
    summary = {
        "mean_delta_brier": float(delta.mean()),
        "std_delta_brier": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        "mean_delta_brier_macro": float(folds["delta_brier_macro"].mean()),
        "mean_delta_log_loss": float(folds["delta_log_loss"].mean()),
        "folds_better": int((delta < 0).sum()),
        "folds_worse": int((delta > 0).sum()),
        "folds": int(len(delta)),
        "benchmarks_better": int((benchmarks["delta_brier"] < 0).sum()),
        "benchmarks_worse": int((benchmarks["delta_brier"] > 0).sum()),
        "benchmarks": int(len(benchmarks)),
        "base_brier": float(base.folds["brier"].mean()),
        "candidate_brier": float(candidate.folds["brier"].mean()),
    }
    summary["verdict"] = verdict(summary)
    return Comparison(folds, benchmarks, summary)


def verdict(summary: dict) -> str:
    """HELPS or HURTS only when the mean moves by MIN_DELTA and FOLDS_TO_AGREE folds agree."""
    needed = min(FOLDS_TO_AGREE, summary["folds"])
    if summary["mean_delta_brier"] <= -MIN_DELTA and summary["folds_better"] >= needed:
        return "HELPS"
    if summary["mean_delta_brier"] >= MIN_DELTA and summary["folds_worse"] >= needed:
        return "HURTS"
    return "NO CLEAR EFFECT"


def fingerprint_scores(sample: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """How much of each column is explained by which benchmark a row comes from.

    Numbers: the share of variance between benchmark means (1.0 when the
    column is constant within each benchmark). Text: the share of rows whose
    value occurs in only one benchmark. Both use non-missing rows only. A high
    score means the model can recognise training benchmarks from the column
    and memorise their difficulty, which never transfers to a new benchmark.
    """
    open_rows = sample[sample["fold"] != "sealed"]
    rows = []
    for column in columns:
        present = open_rows[[column, "benchmark"]].dropna()
        values = present[column]
        if values.empty:
            score = float("nan")
        elif values.dtype.kind in "biuf":
            values = values.astype(float)
            total = float(((values - values.mean()) ** 2).sum())
            means = values.groupby(present["benchmark"]).transform("mean")
            score = float(((means - values.mean()) ** 2).sum() / total) if total > 0 else 1.0
        else:
            spread = present.groupby(column)["benchmark"].nunique()
            score = float(values.map(spread).eq(1).mean())
        rows.append({
            "column": column,
            "filled": float(open_rows[column].notna().mean()),
            "fingerprint": score,
            "warning": bool(score >= FINGERPRINT_WARNING),
        })
    return pd.DataFrame(rows, columns=["column", "filled", "fingerprint", "warning"])


def is_label_free(build: Callable[[pd.DataFrame], pd.DataFrame], frame: pd.DataFrame, seed: int = 0) -> bool:
    """True when ``build`` returns the same table after the labels are shuffled.

    A feature table must not carry label information: at evaluation nobody's
    labels on hidden items exist. Topic scripts should pass this check.
    """
    shuffled = frame.assign(label=np.random.default_rng(seed).permutation(frame["label"].to_numpy()))
    return build(frame).reset_index(drop=True).equals(build(shuffled).reset_index(drop=True))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def format_report(comparison: Comparison, fingerprints: pd.DataFrame, title: str, top: int = 5) -> str:
    """The text a person reads: verdict, paired fold table, benchmarks, warnings."""
    s = comparison.summary
    lines = [
        f"== {title}",
        f"verdict: {s['verdict']}   (screening rule: |mean| >= {MIN_DELTA} and >= {FOLDS_TO_AGREE} folds agree;"
        " confirm with scripts/run_eval.py)",
        f"brier  base {s['base_brier']:.4f}  candidate {s['candidate_brier']:.4f}  "
        f"delta {s['mean_delta_brier']:+.4f} ± {s['std_delta_brier']:.4f}  "
        f"(better in {s['folds_better']}/{s['folds']} folds)",
        f"macro delta {s['mean_delta_brier_macro']:+.4f}   log loss delta {s['mean_delta_log_loss']:+.4f}   "
        f"benchmarks better {s['benchmarks_better']} / worse {s['benchmarks_worse']} of {s['benchmarks']}",
        "",
        "fold  bench    rows   base    cand    delta   macro Δ  logloss Δ",
    ]
    for row in comparison.folds.itertuples():
        lines.append(
            f"{row.fold:>4}  {row.benchmarks:>5}  {row.rows:>7}  {row.base_brier:.4f}  "
            f"{row.candidate_brier:.4f}  {row.delta_brier:+.4f}  {row.delta_brier_macro:+.4f}  "
            f"{row.delta_log_loss:+.4f}"
        )
    ranked = comparison.benchmarks.sort_values("delta_brier")
    for label, part in (("most improved", ranked.head(top)), ("most worsened", ranked.tail(top).iloc[::-1])):
        lines += ["", f"{label}:"]
        lines += [
            f"  {row.benchmark:28s} fold {row.fold}  {row.base_brier:.4f} -> {row.candidate_brier:.4f}"
            f"  ({row.delta_brier:+.4f})"
            for row in part.itertuples()
        ]
    if len(fingerprints):
        lines += ["", "fingerprint check (share explained by benchmark; >= "
                  f"{FINGERPRINT_WARNING} can memorise benchmarks):"]
        lines += [
            f"  {row.column:28s} filled {row.filled:5.1%}  fingerprint {row.fingerprint:.2f}"
            + ("  WARNING" if row.warning else "")
            for row in fingerprints.itertuples()
        ]
    return "\n".join(lines)


def save_run(out: Path, comparison: Comparison, fingerprints: pd.DataFrame, report: str, record: dict) -> None:
    """Write one run's files and append its record to the shared history.

    Every run is kept, not only winners, so the number of attempts behind any
    reported gain stays visible.
    """
    out.mkdir(parents=True, exist_ok=False)
    comparison.folds.to_csv(out / "folds.csv", index=False)
    comparison.benchmarks.to_csv(out / "benchmarks.csv", index=False)
    fingerprints.to_csv(out / "fingerprints.csv", index=False)
    (out / "report.txt").write_text(report + "\n")
    full = {**record, "summary": comparison.summary, "run_dir": out.name}
    (out / "run.json").write_text(json.dumps(full, indent=2))
    with (out.parent / "history.jsonl").open("a") as history:
        history.write(json.dumps(full) + "\n")


__all__ = [
    "FINGERPRINT_WARNING",
    "FOLDS_TO_AGREE",
    "MIN_DELTA",
    "Comparison",
    "compare",
    "file_sha256",
    "fingerprint_scores",
    "format_report",
    "is_label_free",
    "save_run",
    "verdict",
]
