import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.run_eval import make_predictor


class TrainingBoundaryTests(unittest.TestCase):
    def test_priors_are_built_without_the_entire_evaluation_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "corpus_manifest.json"
            manifest.write_text(json.dumps({"revision": "a", "eligible_dirs": ["bbq/formatted_tables", "ai2d_test/formatted_tables", "train/formatted_tables"]}))
            stats = pd.DataFrame({"benchmark": ["train"], "identity": ["m"], "n": [10], "successes": [2]})
            with patch("scripts.run_eval.load_or_build_subject_stats", return_value=stats) as load:
                prior = make_predictor("subject_prior", root, {"benchmarks": ["bbq", "ai2d_test"]}, root, 5)
                self.assertEqual(load.call_args.args[1], ["train"])
                first_cache = load.call_args.args[2]
                self.assertAlmostEqual(prior({"normalized_name": "unknown"}, {"benchmark_id": "benchmark_123"}, []), 3 / 12)
                make_predictor("shrunk_mean", root, {"benchmarks": ["bbq"]}, root, 5)
                self.assertEqual(load.call_args.args[1], ["ai2d_test", "train"])
                self.assertNotEqual(first_cache, load.call_args.args[2])
                manifest.write_text(manifest.read_text().replace('"a"', '"b"'))
                make_predictor("subject_prior", root, {"benchmarks": ["bbq", "ai2d_test"]}, root, 5)
                self.assertNotEqual(first_cache, load.call_args.args[2])

    def test_text_cache_distinguishes_suite_and_dataset_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "corpus_manifest.json"
            manifest.write_text(json.dumps({"revision": "a", "eligible_dirs": ["bbq", "train"]}))
            training = pd.DataFrame({"text": ["q"], "label": [1]})
            with patch("scripts.run_eval.load_or_build_text_training", return_value=training) as load, patch("scripts.run_eval.TextRidge"):
                make_predictor("text_ridge", root, {"benchmarks": ["bbq"], "seed": 0}, root, 5)
                first_cache = load.call_args.args[2]
                self.assertEqual(load.call_args.args[1], ["train"])
                make_predictor("text_ridge", root, {"benchmarks": ["train"], "seed": 0}, root, 5)
                self.assertNotEqual(first_cache, load.call_args.args[2])
                manifest.write_text(manifest.read_text().replace('"a"', '"b"'))
                make_predictor("text_ridge", root, {"benchmarks": ["bbq"], "seed": 0}, root, 5)
                self.assertNotEqual(first_cache, load.call_args.args[2])
