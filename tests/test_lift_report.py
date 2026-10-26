import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.lift import LiftResult
from paiec.lift_report import (
    compare,
    fingerprint_scores,
    format_report,
    is_label_free,
    save_run,
    verdict,
)


def result(fold_brier, benchmark_brier):
    """A LiftResult with one benchmark per fold, enough to test the pairing."""
    folds = [str(k) for k in range(len(fold_brier))]
    return LiftResult(
        columns=(),
        abilities=(),
        folds=pd.DataFrame({
            "fold": folds,
            "benchmarks": 1,
            "rows": 100,
            "brier": fold_brier,
            "log_loss": [2 * b for b in fold_brier],
            "brier_macro": benchmark_brier,
        }),
        benchmarks=pd.DataFrame({
            "fold": folds,
            "benchmark": [f"b{k}" for k in range(len(folds))],
            "rows": 100,
            "brier": benchmark_brier,
            "log_loss": [2 * b for b in benchmark_brier],
        }),
    )


class CompareTests(unittest.TestCase):
    def test_paired_deltas_ignore_fold_base_rates(self):
        # Folds differ a lot in level but the candidate is 0.01 better in each.
        base = result([0.20, 0.30, 0.25, 0.22, 0.28], [0.20, 0.30, 0.25, 0.22, 0.28])
        candidate = result([0.19, 0.29, 0.24, 0.21, 0.27], [0.19, 0.29, 0.24, 0.21, 0.27])
        summary = compare(base, candidate).summary
        self.assertAlmostEqual(summary["mean_delta_brier"], -0.01)
        self.assertAlmostEqual(summary["std_delta_brier"], 0.0)
        self.assertEqual((summary["folds_better"], summary["folds_worse"]), (5, 0))
        self.assertEqual(summary["verdict"], "HELPS")

    def test_benchmark_table_and_counts(self):
        base = result([0.25] * 5, [0.25] * 5)
        candidate = result([0.25] * 5, [0.24, 0.26, 0.25, 0.25, 0.25])
        comparison = compare(base, candidate)
        self.assertEqual(comparison.benchmarks["delta_brier"].round(4).tolist(), [-0.01, 0.01, 0, 0, 0])
        self.assertEqual(comparison.summary["benchmarks_better"], 1)
        self.assertEqual(comparison.summary["benchmarks_worse"], 1)

    def test_different_benchmarks_raise(self):
        base = result([0.25] * 5, [0.25] * 5)
        other = result([0.25] * 4, [0.25] * 4)
        with self.assertRaises(ValueError):
            compare(base, other)


class VerdictTests(unittest.TestCase):
    def summary(self, mean, better, worse):
        return {"mean_delta_brier": mean, "folds_better": better, "folds_worse": worse, "folds": 5}

    def test_noise_sized_changes_are_no_clear_effect(self):
        self.assertEqual(verdict(self.summary(0.0003, 0, 5)), "NO CLEAR EFFECT")
        self.assertEqual(verdict(self.summary(-0.0003, 5, 0)), "NO CLEAR EFFECT")

    def test_needs_mean_and_four_folds_to_agree(self):
        self.assertEqual(verdict(self.summary(-0.002, 4, 1)), "HELPS")
        self.assertEqual(verdict(self.summary(-0.002, 3, 2)), "NO CLEAR EFFECT")
        self.assertEqual(verdict(self.summary(0.002, 1, 4)), "HURTS")
        self.assertEqual(verdict(self.summary(0.002, 4, 1)), "NO CLEAR EFFECT")


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        benchmark = np.repeat([f"b{k}" for k in range(20)], 100)
        self.sample = pd.DataFrame({
            "benchmark": benchmark,
            "fold": "0",
            "per_benchmark": pd.Series(benchmark).str[1:].astype(float).to_numpy(),
            "noise": rng.normal(size=2000),
            "own_word": [f"w_{name}" for name in benchmark],
            "shared_word": rng.choice(["yes", "no"], size=2000),
        })

    def test_constant_within_benchmark_warns_and_noise_does_not(self):
        scores = fingerprint_scores(self.sample, ["per_benchmark", "noise", "own_word", "shared_word"])
        scores = scores.set_index("column")
        self.assertAlmostEqual(scores.loc["per_benchmark", "fingerprint"], 1.0)
        self.assertLess(scores.loc["noise", "fingerprint"], 0.05)
        self.assertEqual(scores.loc["own_word", "fingerprint"], 1.0)
        self.assertEqual(scores.loc["shared_word", "fingerprint"], 0.0)
        self.assertEqual(scores["warning"].tolist(), [True, False, True, False])

    def test_sealed_rows_are_ignored(self):
        sample = self.sample.assign(fold=np.where(self.sample["benchmark"] == "b0", "sealed", "0"))
        sample.loc[sample["fold"] == "sealed", "noise"] = 1e6
        self.assertLess(fingerprint_scores(sample, ["noise"])["fingerprint"].iloc[0], 0.05)


class LabelFreeTests(unittest.TestCase):
    FRAME = pd.DataFrame({"item_id": [f"i{k}" for k in range(50)], "text": ["abc"] * 50, "label": [0, 1] * 25})

    def test_feature_from_text_passes(self):
        self.assertTrue(is_label_free(lambda f: f[["item_id"]].assign(n=f["text"].str.len()), self.FRAME))

    def test_feature_from_labels_fails(self):
        self.assertFalse(is_label_free(lambda f: f[["item_id"]].assign(leak=f["label"]), self.FRAME))


class ReportAndSaveTests(unittest.TestCase):
    def setUp(self):
        self.comparison = compare(result([0.25] * 5, [0.25] * 5), result([0.24] * 5, [0.24] * 5))
        self.fingerprints = pd.DataFrame(
            [{"column": "n_chars", "filled": 1.0, "fingerprint": 0.9, "warning": True}]
        )

    def test_report_mentions_verdict_and_warning(self):
        report = format_report(self.comparison, self.fingerprints, "demo")
        self.assertIn("verdict: HELPS", report)
        self.assertIn("WARNING", report)

    def test_every_run_is_saved_and_appended_to_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("first", "second"):
                save_run(root / name, self.comparison, self.fingerprints, "report", {"name": name})
            for name in ("first", "second"):
                files = {path.name for path in (root / name).iterdir()}
                self.assertEqual(
                    files, {"folds.csv", "benchmarks.csv", "fingerprints.csv", "report.txt", "run.json"}
                )
            history = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()]
            self.assertEqual([entry["name"] for entry in history], ["first", "second"])
            self.assertEqual(history[0]["summary"]["verdict"], "HELPS")

    def test_existing_run_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            save_run(Path(directory) / "run", self.comparison, self.fingerprints, "report", {})
            with self.assertRaises(FileExistsError):
                save_run(Path(directory) / "run", self.comparison, self.fingerprints, "report", {})


if __name__ == "__main__":
    unittest.main()
