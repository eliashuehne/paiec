import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from feature_foundation_fixtures import corpus


class TaskStreamTests(unittest.TestCase):
    def module(self, name):
        try:
            return importlib.import_module("paiec.feature_foundation." + name)
        except ModuleNotFoundError:
            self.fail("Streaming " + name + " is missing")

    def test_unicode_and_long_input_counts_are_full_and_not_language_claims(self):
        m = self.module("tasks")
        raw = {"item_content": "中" * 3000 + "\nA. one\nB. two\n```python", "item_features": "", "interactors": "[]"}
        features = m.task_observables(raw)
        self.assertEqual(features["content_characters"], 3024)
        self.assertEqual(features["cjk_characters"], 3000)
        self.assertTrue(features["has_code_fence"])
        self.assertEqual(features["option_marker_count_hint"], 2)
        self.assertNotIn("language", features)
        self.assertNotIn("difficulty", features)

    def test_stream_accounts_for_all_records_and_never_reads_outcomes_or_protected_content(self):
        m = self.module("streaming")
        contracts = self.module("contracts")
        sources = self.module("sources")
        original = pq.ParquetFile.iter_batches

        def guarded(reader, *args, **kwargs):
            columns = kwargs.get("columns")
            if columns is None or set(columns) & {"response", "grading_criterion", "verifier", "display_name", "test_condition"}:
                raise AssertionError("Forbidden column read: " + str(columns))
            return original(reader, *args, **kwargs)

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config = contracts.FeatureFoundationConfig.model_validate(corpus(root))
            population = sources.inventory_population(config, Path(root))
            with patch.object(pq.ParquetFile, "iter_batches", guarded):
                result = m.stream_population(config, Path(root), Path(dest), population)
            self.assertEqual(result["items"], 6)
            self.assertEqual(result["responses"], 9)
            self.assertEqual(pq.ParquetFile(Path(dest) / "response_input_links.parquet").metadata.num_rows, 9)
            items = pq.read_table(Path(dest) / "items.parquet").to_pylist()
            protected = [r for r in items if r["benchmark"] == "sealed"]
            self.assertEqual(len(protected), 2)
            self.assertTrue(all(r["content"] is None and r["status"] == "withheld_protected" for r in protected))
            tasks = pq.read_table(Path(dest) / "tasks.parquet").to_pylist()
            self.assertEqual(len(tasks), 2)
            self.assertEqual({r["interactors"] for r in tasks}, {"[]", "python"})

    def test_cross_split_content_overlap_blocks_training_use_without_reassignment(self):
        m = self.module("streaming")
        contracts = self.module("contracts")
        sources = self.module("sources")
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            raw = corpus(root)
            raw["family_groups"] = []
            raw["frozen_roles"]["development"] = ["beta"]
            config = contracts.FeatureFoundationConfig.model_validate(raw)
            population = sources.inventory_population(config, Path(root))
            m.stream_population(config, Path(root), Path(dest), population)
            eligibility = pq.read_table(Path(dest) / "training_eligibility.parquet").to_pylist()
            alpha = [r for r in eligibility if r["benchmark"] == "alpha"]
            self.assertTrue(alpha)
            self.assertTrue(all(not r["eligible"] and r["reason"] == "cross_split_content" for r in alpha))
            self.assertEqual(next(r for r in population if r["benchmark"] == "alpha")["split"], "training")

    def test_legacy_monitor_bucket_is_not_training(self):
        m = self.module("tasks")
        from paiec.head_data import digest, task_key
        from paiec.expanded_data import monitor_key
        content = next("training question " + str(i) for i in range(200) if monitor_key(digest(task_key("training question " + str(i))), 20261023))
        self.assertEqual(m.effective_split(content, {"split": "training", "frozen_role": "training"}, 20261023), "validation")

    def test_structured_options_messages_and_asset_references_are_observed_without_loading(self):
        m = self.module("tasks")
        options = m.task_observables({"item_content": '{"question":"Which?","options":["a","b","c"]}', "item_features": "", "interactors": ""})
        self.assertEqual(options.get("structured_option_count_hint"), 3)
        self.assertEqual(options.get("section_count_hint"), 2)
        messages = m.task_observables({"item_content": '[{"role":"user","content":"Read context.txt"}]', "item_features": "asset_path=context.txt", "interactors": ""})
        self.assertEqual(messages.get("message_count_hint"), 1)
        self.assertTrue(messages.get("has_visible_asset_reference_hint"))

