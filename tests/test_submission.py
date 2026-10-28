import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paiec.prior import LogisticPrior
from paiec.prior_offset import PriorOffset
from paiec.submission import assemble, build_zip, save_artifacts

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "harness_version", "reasoning_effort", "subject_features_extra"]


def subject(name):
    record = {field: "" for field in FIELDS}
    record.update(normalized_name=name, provider="Acme", release_date="2025-01-01")
    return record


def item(text, task, benchmark="bench"):
    return {"item_content": text, "item_features": f"task={task};lang=en", "interactors": "",
            "benchmark_id": benchmark}


def fitted_model():
    prior = LogisticPrior()
    prior.coef = np.random.default_rng(0).normal(scale=0.3, size=prior.coef.size)
    prior.intercept = -0.2
    return PriorOffset(level=0.0, shift=0.1, scale=0.7, center=0.05, tau=1.5, sigma=0.75, prior=prior)


def load_entry_point(directory):
    spec = importlib.util.spec_from_file_location("submission_model_under_test", directory / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.model = fitted_model()
        cls.directory = assemble(tmp / "submission", save_artifacts(cls.model, tmp / "prior_offset.npz"))
        cls.entry = load_entry_point(cls.directory)
        cls.zip_path = build_zip(cls.directory, tmp / "submission.zip")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("submission_model_under_test", None)
        cls.tmp.cleanup()

    def test_zipped_model_matches_the_evaluated_model(self):
        target = subject("Model 70B")
        other = subject("Other 7B")
        labeled = [[[target, item(f"question {k}", "a" if k % 2 else "b")], k % 3 != 0] for k in range(7)]
        labeled += [[[other, item("question 99", "a")], 1], [[target, item("elsewhere", "a", "other")], 0]]
        for budget in (0, 1, 3, 7):
            evidence = labeled[:budget] + labeled[7:]
            for text, task in [("new question", "a"), ("another one", "b"), ("unseen tag", "z")]:
                expected = self.model(target, item(text, task), evidence)
                actual = self.entry.predict([target, item(text, task)], evidence)
                self.assertAlmostEqual(actual, expected, places=10)

    def test_other_subjects_labels_do_not_change_predictions(self):
        target, other = subject("Model 70B"), subject("Other 7B")
        own = [[[target, item("q1", "a")], 1]]
        noisy = own + [[[other, item(f"q{k}", "a")], 0] for k in range(20)]
        self.assertEqual(self.entry.predict([target, item("x", "a")], own),
                         self.entry.predict([target, item("x", "a")], noisy))

    def test_official_checker_accepts_the_zip(self):
        result = subprocess.run([sys.executable, str(ROOT / "baseline" / "check_submission_zip.py"),
                                 str(self.zip_path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
