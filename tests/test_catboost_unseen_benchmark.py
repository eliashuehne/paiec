import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from paiec.catboost_features import feature_names, feature_record
from paiec.catboost_unseen_benchmark import (UnseenBenchmarkConfig, _fold_identity,
                                             build_external_frame, normalize_name,
                                             select_benchmarks, subject_identity)
from paiec.feature_preparation import stable_hash
from paiec.head_data import task_key


def group(content):
    return stable_hash(task_key(content))

SUBJECT = {"normalized_name": "OpenAI GPT-4", "provider": "OpenAI", "release_date": "2023-03-14",
           "access_date": "", "harness": "HELM", "harness_version": "", "reasoning_effort": "",
           "subject_features_extra": ""}
ITEM = {"item_content": "What is 2 + 2?", "item_features": "", "interactors": "",
        "benchmark_id": "benchmark_deadbeef"}


def write_corpus(root):
    """Three directories: admitted, non-text, and one repeating panel content."""
    root.mkdir(parents=True)
    (root / "corpus_manifest.json").write_text(json.dumps({"eligible_dirs": [
        "good/formatted_tables", "imagey/formatted_tables", "repeat/formatted_tables",
        "shared/formatted_tables"]}))
    for name, modality, contents, subjects in [
            ("good", ["text"], ["What is 2 + 2?", "Name the capital of France."], ["OpenAI GPT-4"]),
            ("imagey", ["image", "text"], ["Look at this chart."], ["OpenAI GPT-4"]),
            ("repeat", ["text"], ["A panel question"], ["OpenAI GPT-4"]),
            ("shared", ["text"], ["Nobody here"], ["Some Other Model"])]:
        directory = root / name / "formatted_tables"
        directory.mkdir(parents=True)
        pd.DataFrame([{"modality": modality, "response_scale": json.dumps({"direction": "higher_is_better"}),
                       "response_type": "binary", "granularity": "item"}]).to_parquet(
            directory / "benchmarks.parquet")
        pd.DataFrame([{"item_id": str(index), "content": content}
                      for index, content in enumerate(contents)]).to_parquet(directory / "items.parquet")
        pd.DataFrame([{"subject_id": str(index), "normalized_name": value}
                      for index, value in enumerate(subjects)]).to_parquet(directory / "subjects.parquet")
    return root


class MatchingTests(unittest.TestCase):
    def test_names_collapse_case_and_punctuation_only(self):
        self.assertEqual(normalize_name("OpenAI GPT-4o"), "openai gpt 4o")
        self.assertEqual(normalize_name("  openai   gpt-4o "), "openai gpt 4o")
        self.assertEqual(normalize_name(""), "")
        self.assertNotEqual(normalize_name("GPT-4"), normalize_name("GPT-4o"))

    def test_identity_uses_all_supplied_fields(self):
        changed = dict(SUBJECT, harness="lm-eval")
        self.assertNotEqual(subject_identity(SUBJECT), subject_identity(changed))
        self.assertEqual(subject_identity(SUBJECT), subject_identity(dict(SUBJECT)))
        self.assertEqual(len(subject_identity(SUBJECT)), 8)

    def test_fold_identity_only_reads_training_rows(self):
        systems = {"a": dict(SUBJECT), "b": dict(SUBJECT, normalized_name="Other Model")}
        rows = [{"system_key": "a"}, {"system_key": "b"}]
        names, identities = _fold_identity(rows, systems, np.asarray([True, False]))
        self.assertIn("openai gpt 4", names)
        self.assertNotIn("other model", names)
        self.assertEqual(identities, {subject_identity(SUBJECT)})


class SelectionTests(unittest.TestCase):
    def config(self, **changes):
        value = {"version": "catboost_unseen_benchmark_v1",
                 "panel_run": "runs/competition_analysis/panel",
                 "data_root": "payload/data", "seed": 3, "min_pair_items": 20,
                 "rows_per_system": 10, "benchmark_cap": 50,
                 "max_panel_group_overlap": 0.0}
        value.update(changes)
        return UnseenBenchmarkConfig.model_validate(value)

    def test_filters_are_predeclared_and_outcome_blind(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = write_corpus(Path(temporary) / "data")
            panel_groups = {group("A panel question")}
            admitted, decisions = select_benchmarks(corpus, {"repeat": "development"},
                                                    {"openai gpt 4"}, panel_groups, self.config())
            reasons = {row["benchmark"]: row.get("reason") for row in decisions}
            self.assertEqual(reasons["repeat"], "sealed_role:development")
            self.assertEqual(reasons["imagey"], "not_text_only")
            self.assertEqual(reasons["shared"], "no_shared_model_name")
            self.assertEqual(reasons["good"], "admitted")
            self.assertEqual([benchmark.name for benchmark in admitted], ["good"])
            self.assertEqual(admitted[0].cap, 50)

    def test_panel_content_reuse_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = write_corpus(Path(temporary) / "data")
            admitted, decisions = select_benchmarks(corpus, {}, {"openai gpt 4"},
                                                    {group("What is 2 + 2?")}, self.config())
            self.assertEqual([row["benchmark"] for row in decisions if row["eligible"]
                              and row["benchmark"] == "good"], [])
            repeat = next(row for row in decisions if row["benchmark"] == "good")
            self.assertEqual(repeat["reason"], "panel_content_overlap")
            self.assertEqual(repeat["task_group_overlap"], .5)
            # "repeat" holds unrelated content and carries no sealed role here.
            self.assertEqual([benchmark.name for benchmark in admitted], ["repeat"])

    def test_overlap_tolerance_is_honoured(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = write_corpus(Path(temporary) / "data")
            admitted, _ = select_benchmarks(corpus, {}, {"openai gpt 4"},
                                            {group("What is 2 + 2?")},
                                            self.config(max_panel_group_overlap=.5))
            self.assertEqual([benchmark.name for benchmark in admitted], ["good", "repeat"])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            self.config(rows_per_system=0)
        with self.assertRaises(ValueError):
            self.config(max_panel_group_overlap=1.5)
        with self.assertRaises(ValueError):
            self.config(extra="nope")


class FrameContractTests(unittest.TestCase):
    def test_external_frame_uses_declared_columns_only(self):
        rows = [{"subject": SUBJECT, "item": ITEM}]
        groups = ("task_structure",)
        frame = build_external_frame(rows, feature_names(groups))
        self.assertEqual(list(frame.columns), feature_names(groups))
        self.assertEqual(frame.iloc[0]["task_numeric_span_count"], 2.)
        record = feature_record(SUBJECT, ITEM)
        self.assertNotIn("benchmark_id", record)
        with self.assertRaises(ValueError):
            build_external_frame([{"subject": SUBJECT, "item": {"item_content": ""}}],
                                 feature_names(groups))


if __name__ == "__main__":
    unittest.main()
