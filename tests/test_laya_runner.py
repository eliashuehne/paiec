from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from paiec.laya_predictor import LayaPredictor, PROMPT_VERSION
from paiec.pairs import Pair
from scripts import run_eval


class Router:
    def predict_batch(self, requests, batch_size=None):
        return [{"answers": {"correct": {"noul": .5}}} for _ in requests]


class LayaRunnerTests(unittest.TestCase):
    def test_cli_exports_complete_versioned_results_with_shared_question(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "corpus_manifest.json").write_text(json.dumps({"revision": "test", "eligible_dirs": ["toy"]}))
            config = root / "suites.json"
            config.write_text(json.dumps({"smoke": {"benchmarks": ["toy"], "pairs_per_benchmark": 1,
                                                    "evaluation_items": 8, "acquisition_items": 40}}))
            ids = tuple(f"i{i}" for i in range(100))
            pair = Pair("toy", "s0", {"normalized_name": "m"},
                        pd.DataFrame({"content": ids, "item_features": "", "interactors": ""}, index=ids),
                        pd.DataFrame({"item_id": ids, "label": [i % 2 for i in range(100)]}), ids)
            predictor = LayaPredictor(router=Router(), device="cpu", revision="test")
            argv = ["run_eval.py", "--data-root", str(data), "--config", str(config),
                    "--suite", "smoke", "--predictor", "laya", "--no-tracking"]
            with patch.object(run_eval, "ROOT", root), patch.object(run_eval, "build_pairs", return_value=[pair]), \
                    patch.object(run_eval, "predictor_instance", return_value=predictor), \
                    patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(run_eval.main(), 0)
            output = root / "runs/eval_smoke" / f"laya_mlx_auto_codabench_v1_english_512_zeroshot_{PROMPT_VERSION}"
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["predictors"]["laya"]["prompt_version"], PROMPT_VERSION)
            question = json.loads((output / "question_definition.json").read_text())
            self.assertEqual(set(question["correct"]), {"type", "instructions"})
            self.assertTrue((output / "summary.csv").is_file())
            self.assertEqual(len((output / "competition_input_examples.jsonl").read_text().splitlines()), 4)
