import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from paiec.evaluate import EvalResult
from paiec.tracking import ExperimentTracker


@unittest.skipUnless(importlib.util.find_spec("mlflow"), "optional tracking dependency is not installed")
class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        (cls.root / "pyproject.toml").write_text("# synthetic test project\n")
        cls.tracker = ExperimentTracker(cls.root, experiment="test")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_completed_run_saves_metrics_configuration_and_source(self):
        frame = pd.DataFrame({"benchmark": ["toy"] * 6, "subject_id": ["s"] * 6,
                              "budget": [0, 1, 3, 7, 15, 31], "brier": [.25] * 6,
                              "ece": [1., 0, 0, 0, 0, 0], "n_items": [10] * 6})
        index = pd.MultiIndex.from_tuples([("toy", "s")], names=["benchmark", "subject_id"])
        result = EvalResult(frame, frame.set_index("budget").brier,
                            pd.Series([.25], index=index), .25, 1 / 6)
        with self.tracker.run("completed", {"seed": 0}, {"predictor": "constant"}, {}) as tracked:
            tracked.result(result, 1.5, pd.DataFrame({"probability": [.5]}))
            run_id = tracked.run_id
        record = self.tracker.client.get_run(run_id)
        self.assertEqual(record.info.status, "FINISHED")
        self.assertEqual(record.data.params["predictor"], "constant")
        self.assertAlmostEqual(record.data.metrics["brier_alc"], .25)
        self.assertEqual(record.data.metrics["completed_eval_inputs"], 10)
        self.assertNotIn("brier_by_budget", record.data.metrics)
        self.assertAlmostEqual(record.data.metrics["ece_alc"], .1)
        self.assertAlmostEqual(record.data.metrics["ece_budget_mean"], 1 / 6)
        self.assertIn("source_snapshot_sha256", record.data.tags)
        path = self.tracker.client.download_artifacts(run_id, "config.json")
        self.assertEqual(json.loads(Path(path).read_text()), {"seed": 0})
        self.assertTrue(self.tracker.client.download_artifacts(run_id, "predictions.csv"))

    def test_failed_run_remains_visible_and_reraises_original_error(self):
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with self.tracker.run("failed", {}, {}, {}) as tracked:
                run_id = tracked.run_id
                raise RuntimeError("intentional")
        self.assertEqual(self.tracker.client.get_run(run_id).info.status, "FAILED")
        self.assertTrue(self.tracker.client.download_artifacts(run_id, "error.txt"))

    def test_disabled_tracking_does_not_create_a_run(self):
        tracker = ExperimentTracker(self.root, enabled=False)
        with tracker.run("disabled", {}, {}, {}) as tracked:
            self.assertIsNone(tracked)
