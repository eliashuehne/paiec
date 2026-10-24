import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paiec.competition_data import (CompetitionDataConfig, visible_fields,
    verify_dataset_sources, build_dataset_tables, recorded_system_features)
from paiec.feature_preparation import read_jsonl, write_json, write_jsonl
from paiec.head_training import file_hash
from paiec.laya_predictor import load_questions
from paiec.schema import SUBJECT_FIELDS, ITEM_FIELDS


class CompetitionDataTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "runs/prepared/source"
        source.mkdir(parents=True)
        subject = {key: "" for key in SUBJECT_FIELDS}
        subject.update(normalized_name="Fixture 8x7B", harness="Recorded harness")
        items = [{**{key: "" for key in ITEM_FIELDS}, "item_content": f"Question {i}: 猫", "benchmark_id": "anonymous"} for i in range(8)]
        def record(item, expected):
            return {"state": json.dumps({"target_system": subject, "target_item": item}),
                    "questions": load_questions(), "expected": {"correct": expected}}
        write_jsonl(source / "training.jsonl", [record(item, bool(i % 2)) for i, item in enumerate(items)])
        write_jsonl(source / "development.jsonl", [record(dict(items[0], item_content="DEVELOPMENT ONLY"), "SECRET_DEV_OUTCOME")])
        split = root / "split.json"
        write_json(split, {"dataset_revision": "fixture"})
        manifest = {"dataset_revision": "fixture", "data_config": {"split": "split.json"},
                    "split_sha256": file_hash(split), "question": load_questions(),
                    "artifacts": {"source/" + p.name: file_hash(p) for p in source.iterdir()}}
        # A research path must not be accessed or treated as a dependency.
        manifest["artifacts"]["profile_bundle/catalog/profiles.jsonl"] = "does-not-exist"
        write_json(root / "runs/prepared/manifest.json", manifest)
        config = CompetitionDataConfig(prepared="runs/prepared", dataset_revision="fixture", roles=["training", "development"])
        return config, source

    def test_exact_input_characters_and_null_like_strings_are_preserved(self):
        subject = {key: "" for key in SUBJECT_FIELDS}
        item = {key: "" for key in ITEM_FIELDS}
        item.update(item_content="  None\n猫 ", item_features="null", interactors="None")
        self.assertEqual(visible_fields(subject, item), [subject, item])
        with self.assertRaises(ValueError):
            visible_fields(dict(subject, display_name="PRIVATE"), item)
        with self.assertRaises(ValueError):
            visible_fields(subject, dict(item, item_content=1))

    def test_dataset_route_has_no_profile_or_manual_annotation_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self.fixture(root)
            with patch("paiec.feature_preparation.CompactProfiles", side_effect=AssertionError("Research lookup")):
                inputs = verify_dataset_sources(root, config.model_dump())
                (root / "out").mkdir()
                report = build_dataset_tables(inputs, root / "out")
            self.assertEqual(report["training_targets"], 8)
            self.assertFalse(report["semantic_annotations_required"])
            self.assertEqual(report["external_profile_facts_used"], 0)
            for path in (root / "out").rglob("*.jsonl"):
                self.assertNotIn("SECRET_DEV_OUTCOME", path.read_text())
                if path.name != "training_targets.jsonl":
                    self.assertNotIn('"native_binary_value"', path.read_text())
            self.assertEqual([r["native_binary_value"] for r in read_jsonl(root / "out/training_targets.jsonl")],
                             [False, True] * 4)

    def test_source_changes_and_foreign_dataset_revision_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "revision"):
                verify_dataset_sources(root, dict(config.model_dump(), dataset_revision="OTHER"))
            inputs = verify_dataset_sources(root, config.model_dump())
            (source / "training.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                verify_dataset_sources(root, config.model_dump())
            (root / "out").mkdir()
            with self.assertRaisesRegex(ValueError, "Source changed"):
                build_dataset_tables(inputs, root / "out")

    def test_sealed_roles_and_external_registries_are_rejected(self):
        for value in [{"roles": ["training", "confirmation"]}, {"source_registry": "research/catalog.json"},
                      {"source_policy": "web_enriched"}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                CompetitionDataConfig.model_validate(value)

    def test_nominal_model_text_is_not_an_invented_capacity_or_tool_setting(self):
        subject = {key: "" for key in SUBJECT_FIELDS}
        subject.update(normalized_name="Fixture 8x7B", release_date="not known")
        result = recorded_system_features(subject)
        self.assertEqual(result["nominal_parameter_text"], ["8x7B"])
        self.assertIsNone(result["exact_parameter_count"])
        self.assertIsNone(result["active_parameter_count"])
        self.assertIsNone(result["supported_tools_from_documentation"])
        self.assertIsNone(result["dates_parseable_as_iso"]["release_date"])

    def test_hidden_target_fields_and_nonbinary_training_targets_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source = self.fixture(root)
            rows = list(read_jsonl(source / "training.jsonl"))
            rows[0]["expected"]["correct"] = "one"
            write_jsonl(source / "training.jsonl", rows)
            manifest_path = root / "runs/prepared/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["source/training.jsonl"] = file_hash(source / "training.jsonl")
            write_json(manifest_path, manifest)
            (root / "out").mkdir()
            inputs = verify_dataset_sources(root, config.model_dump())
            with self.assertRaisesRegex(ValueError, "raw binary"):
                build_dataset_tables(inputs, root / "out")
