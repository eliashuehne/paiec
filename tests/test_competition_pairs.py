import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import pyarrow.parquet as pq

from paiec.competition_pairs import (PairMatrixConfig, benchmark_admission, build_pair_matrix,
                                     load_pair_matrix, parse_direction, parse_modalities,
                                     refresh_manifest)
from paiec.feature_preparation import read_jsonl

REVISION = "a" * 40


def write_benchmark(data, name, *, subjects, items, responses, response_type="binary",
                    granularity="item", modality="['text']",
                    scale='{"direction":"higher_is_better","kind":"discrete"}', trials=None):
    directory = data / name / "formatted_tables"
    directory.mkdir(parents=True)
    pd.DataFrame([{"item_id": item, "content": f"task {item} of {name}", "item_features": None}
                  for item in items]).to_parquet(directory / "items.parquet")
    pd.DataFrame([{"subject_id": subject, "display_name": subject, "normalized_name": subject,
                   "provider": "TestCo", "release_date": None, "access_date": None,
                   "harness": "h", "reasoning_effort": None, "harness_version": None,
                   "subject_features_extra": None} for subject in subjects]).to_parquet(
        directory / "subjects.parquet")
    rows = []
    for subject, item, value in responses:
        rows.append({"subject_id": subject, "item_id": item, "response_id": f"{subject}-{item}",
                     "response": value, "trial": (trials or {}).get((subject, item), 1)})
    pd.DataFrame(rows).to_parquet(directory / "response.parquet")
    pd.DataFrame([{"benchmark_id": name, "name": name, "modality": modality,
                   "response_type": response_type, "response_scale": scale,
                   "granularity": granularity, "n_subjects": len(subjects), "n_items": len(items),
                   "n_responses": len(rows), "coverage": 1.0, "max_trial": 1,
                   "has_reference_answer": True}]).to_parquet(directory / "benchmarks.parquet")


def corpus(root, *, response_type="binary", granularity="item", modality="['text']",
           scale='{"direction":"higher_is_better","kind":"discrete"}', rows_per_subject=100,
           labels=None):
    data = root / "payload" / "data"
    data.mkdir(parents=True)
    items = [f"i{index}" for index in range(rows_per_subject)]
    responses = []
    for subject in ("Strong Model", "Weak Model"):
        for index, item in enumerate(items):
            value = (labels or {}).get((subject, index), 1 if subject.startswith("Strong") and index % 2 else 0)
            responses.append((subject, item, value))
    write_benchmark(data, "bench", subjects=["Strong Model", "Weak Model"], items=items,
                    responses=responses, response_type=response_type, granularity=granularity,
                    modality=modality, scale=scale)
    (data / "corpus_manifest.json").write_text(json.dumps({
        "revision": REVISION, "corpus_scope": "paec-public-training",
        "eligible_dirs": ["bench/formatted_tables"]}))
    return data


def run(root, config=None, name="pairs"):
    config = config or PairMatrixConfig(dataset_revision=REVISION)
    return build_pair_matrix(root, config, root / "runs" / name)


class ParsingTests(unittest.TestCase):
    def test_direction_and_modalities_are_read_not_assumed(self):
        self.assertEqual(parse_direction('{"direction":"lower_is_better","kind":"discrete"}'),
                         "lower_is_better")
        self.assertEqual(parse_direction('{"direction":"unordered"}'), "unordered")
        self.assertEqual(parse_direction("not json"), "unordered")
        self.assertEqual(parse_direction(None), "unordered")
        self.assertEqual(parse_modalities("['text']"), ["text"])
        self.assertEqual(parse_modalities("['image', 'text']"), ["image", "text"])
        self.assertEqual(parse_modalities(None), [])

    def test_admission_records_its_reasons(self):
        config = PairMatrixConfig(dataset_revision=REVISION)
        admitted = benchmark_admission("b", {"response_type": "binary", "granularity": "item",
                                             "modality": "['text']",
                                             "response_scale": '{"direction":"higher_is_better"}'},
                                       [object()], config)
        self.assertTrue(admitted["admitted"])
        rejected = benchmark_admission("b", {"response_type": "ordinal", "granularity": "item",
                                             "modality": "['image', 'text']",
                                             "response_scale": "{}"}, [], config)
        self.assertFalse(rejected["admitted"])
        self.assertEqual(set(rejected["reasons"]), {"response_type_is_not_binary",
                                                    "modality_is_not_text_only",
                                                    "no_subject_reaches_the_item_threshold"})


class PairMatrixTests(unittest.TestCase):
    def test_cells_keep_native_labels_and_item_level_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus(root)
            summary = run(root)
            self.assertEqual(summary["benchmarks_admitted"], 1)
            self.assertEqual(summary["cells"], 2)
            self.assertEqual(summary["labelled_responses"], 200)
            matrix = load_pair_matrix(root / "runs/pairs")
            strong = [cell for cell in matrix["cells"] if "Strong" in cell["subject"]["normalized_name"]][0]
            weak = [cell for cell in matrix["cells"] if "Weak" in cell["subject"]["normalized_name"]][0]
            self.assertEqual(strong["rate"], .5)
            self.assertEqual(weak["rate"], 0.0)
            self.assertEqual(strong["label_sum"], 50)
            self.assertEqual(strong["direction"], "higher_is_better")
            self.assertEqual(strong["identity_tier"], "named")
            labels = matrix["labels"][strong["cell_index"]]
            self.assertEqual(int(labels["n_responses"].sum()), 100)
            items = pq.read_table(root / "runs/pairs/items.parquet").to_pandas()
            self.assertEqual(len(items), 100)
            self.assertEqual(items.iloc[0]["content"], "task i0 of bench")

    def test_lower_is_better_labels_are_preserved_and_direction_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus(root, scale='{"direction":"lower_is_better","kind":"discrete"}')
            summary = run(root)
            self.assertEqual(summary["directions"], {"lower_is_better": 2})
            self.assertFalse(summary["direction_flipped"])
            matrix = load_pair_matrix(root / "runs/pairs")
            self.assertTrue(all(cell["direction"] == "lower_is_better" for cell in matrix["cells"]))
            strong = [cell for cell in matrix["cells"] if "Strong" in cell["subject"]["normalized_name"]][0]
            self.assertEqual(strong["rate"], .5)

    def test_threshold_and_non_binary_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus(root, rows_per_subject=79)
            summary = run(root)
            self.assertEqual(summary["cells"], 0)
            self.assertIn("no_subject_reaches_the_item_threshold", summary["exclusion_reasons"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = corpus(root)
            directory = data / "bench" / "formatted_tables"
            table = pd.read_parquet(directory / "response.parquet").astype({"response": float})
            table.loc[0, "response"] = .5
            table.to_parquet(directory / "response.parquet")
            summary = run(root)
            self.assertEqual(summary["cells"], 2)
            self.assertEqual(summary["labelled_responses"], 199)

    def test_non_item_granularity_and_multi_modality_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus(root, granularity="pair", modality="['image', 'text']")
            summary = run(root)
            self.assertEqual(summary["benchmarks_admitted"], 0)
            self.assertEqual(summary["exclusion_reasons"],
                             {"granularity_is_not_item": 1, "modality_is_not_text_only": 1})

    def test_excluded_benchmarks_contribute_no_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = corpus(root)
            write_benchmark(data, "visual", subjects=["Strong Model"], items=[f"i{i}" for i in range(100)],
                            responses=[("Strong Model", f"i{i}", i % 2) for i in range(100)],
                            modality="['text', 'image']")
            (data / "corpus_manifest.json").write_text(json.dumps({
                "revision": REVISION, "corpus_scope": "paec-public-training",
                "eligible_dirs": ["bench/formatted_tables", "visual/formatted_tables"]}))
            summary = run(root)
            self.assertEqual(summary["benchmarks_admitted"], 1)
            self.assertEqual(summary["cells_by_benchmark"], {"bench": 2})
            matrix = load_pair_matrix(root / "runs/pairs")
            self.assertEqual({cell["benchmark"] for cell in matrix["cells"]}, {"bench"})

    def test_repeated_trials_stay_with_their_item(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = corpus(root)
            directory = data / "bench" / "formatted_tables"
            table = pd.read_parquet(directory / "response.parquet")
            extra = table.iloc[[0, 1]].copy()
            extra["response_id"] = ["x1", "x2"]
            extra["response"] = [0.0, 1.0]
            pd.concat([table, extra]).to_parquet(directory / "response.parquet")
            summary = run(root)
            self.assertEqual(summary["labelled_responses"], 202)
            matrix = load_pair_matrix(root / "runs/pairs")
            strong = [cell for cell in matrix["cells"] if "Strong" in cell["subject"]["normalized_name"]][0]
            labels = matrix["labels"][strong["cell_index"]]
            self.assertEqual(int(labels["n_responses"].sum()), 102)

    def test_manifest_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus(root)
            run(root)
            manifest = json.loads((root / "runs/pairs/manifest.json").read_text())
            self.assertIn("cells.jsonl", manifest["artifacts"])
            self.assertIn("cell_items.parquet", manifest["artifacts"])
            refresh_manifest(root / "runs/pairs")
            self.assertNotIn("manifest.json", json.loads(
                (root / "runs/pairs/manifest.json").read_text())["artifacts"])
            with self.assertRaises(ValueError):
                run(root)
            with self.assertRaises(ValueError):
                PairMatrixConfig(dataset_revision=REVISION, min_items=1)

    def test_revision_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus(root)
            with self.assertRaises(ValueError):
                build_pair_matrix(root, PairMatrixConfig(), root / "runs/pairs")


if __name__ == "__main__":
    unittest.main()
