import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from paiec.development_evidence import original_development_pairs
from paiec.evaluate import evaluate
from paiec.schema import load_benchmark
from paiec.pairs import iter_pairs
from paiec.split import make_split
from paiec.head_training import file_hash


class SelectiveDevelopmentTests(unittest.TestCase):
    def test_selective_outcomes_reproduce_splits_trials_and_revealed_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "payload/data/demo"
            folder.mkdir(parents=True)
            ids = [f"item-{i}" for i in range(320)]
            pd.DataFrame(dict(subject_id=["original", "supplemental"], normalized_name=["model-a", "model-b"])).to_parquet(folder / "subjects.parquet")
            pd.DataFrame(dict(item_id=ids, content=ids, item_features=[""]*320)).to_parquet(folder / "items.parquet")
            records = [dict(subject_id=subject, item_id=item, response=i%2, trial=0, interactors="")
                       for subject in ["original", "supplemental"] for i,item in enumerate(ids)]
            # Preserve repeated-response weighting and first recorded label.
            records += [dict(subject_id="original", item_id=ids[0], response=1, trial=1, interactors="")]
            pd.DataFrame(records).to_parquet(folder / "response.parquet")
            original = root / "runs/frozen"
            original.mkdir(parents=True)
            source_manifest = original / "source_file_hashes.json"
            source_manifest.write_text(json.dumps({str(p.relative_to(root / "payload/data")):file_hash(p)
                                                  for p in folder.glob("*.parquet")}))
            (root / "split.json").write_text(json.dumps(dict(roles={"development":["demo"]},
                artifact_directory="runs/frozen", artifact_fingerprints={"source_file_hashes.json":file_hash(source_manifest)})))
            split = make_split(ids, benchmark="demo", subject_id="original", seed=17,
                               acquisition_items=40, evaluation_items=120)
            development = pd.DataFrame(dict(benchmark=["demo"]*120, subject_id=["original"]*120, item_id=split.evaluation))
            manifest = {"data_config":{"split":"split.json"}}
            reads = []
            reader = pd.read_parquet
            def checked(path, *args, **kwargs):
                if Path(path).name == "response.parquet":
                    reads.append(kwargs)
                return reader(path, *args, **kwargs)
            with patch("paiec.development_evidence.pd.read_parquet", side_effect=checked):
                selected = original_development_pairs(root, manifest, development, 17)
            self.assertEqual(len(selected), 1)
            self.assertEqual(set(selected[0].item_ids), set(ids))
            self.assertEqual(set(selected[0].responses.item_id), set(split.evaluation)|set(split.acquisition[:31]))
            for request in reads:
                self.assertTrue(any(key == "subject_id" for key,_,_ in request["filters"]))
                if "response" in request["columns"]:
                    self.assertTrue(any(key == "item_id" for key,_,_ in request["filters"]))
            full = [pair for pair in iter_pairs(load_benchmark(root / "payload/data", "demo")) if pair.subject_id == "original"]
            def predictor(subject, item, labeled):
                return (sum(entry[1] for entry in labeled)+2)/(len(labeled)+4)
            kwargs = dict(seed=17, acquisition_items=40, evaluation_items=120, shared_evidence=True)
            pd.testing.assert_frame_equal(evaluate(selected, predictor, **kwargs).frame,
                                          evaluate(full, predictor, **kwargs).frame)
            with self.assertRaisesRegex(ValueError, "only original development"):
                original_development_pairs(root, manifest, development.assign(benchmark="sealed"), 17)
            (folder / "response.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen development source changed"):
                original_development_pairs(root, manifest, development, 17)


if __name__ == "__main__":
    unittest.main()
