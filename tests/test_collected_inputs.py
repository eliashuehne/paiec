from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from paiec.collected_inputs import CollectedStateBuilder, CollectedInputsConfig, FEATURE_FILES
from paiec.feature_preparation import write_json, write_jsonl
from paiec.head_training import file_hash
from paiec.schema import SUBJECT_FIELDS, ITEM_FIELDS
from paiec.collected_inputs_v2 import CollectedStateBuilderV2, compact_facts
from paiec.collected_inputs_v3 import CollectedInputWorkflowConfig, add_scope_notes
from paiec.system_profiles import Profile


class CollectedInputTests(unittest.TestCase):
    def fixture(self, root, *, confirmed=True):
        root.mkdir()
        subject = {k: "" for k in SUBJECT_FIELDS}
        subject.update(normalized_name="Model", subject_features_extra="reported_task_generation_size=512")
        item = {k: "" for k in ITEM_FIELDS}
        item.update(item_content="  Question 猫 [MASK]  ", benchmark_id="anonymous")
        write_jsonl(root / "systems.jsonl", [{"system_key": "s", "recorded": subject, "resolution": {
            "model_profile_id": "model.test", "harness_profile_id": "harness.test", "model_identity_basis": "base_model",
            "exact_evaluated_checkpoint_verified": False}}])
        write_jsonl(root / "reviewed_static_features.jsonl", [{"profile_id": "model.test", "entity_type": "model",
            "field": "architecture.total_parameters", "fact": {"value": 8000000000, "status": "documented"},
            "review_status": "reviewed", "review_disposition": "value_confirmed" if confirmed else "conflict_unresolved"}])
        write_jsonl(root / "typed_run_features.jsonl", [{"system_key": "s", "recorded": subject, "features": {
            "few_shot_count": {"value": 0, "status": "recorded"}, "enabled_tools": {"value": None, "status": "unknown"}}}])
        write_jsonl(root / "qualified_model_observations.jsonl", [{"profile_id": "model.test", "field": "documented_context_window",
            "value": "128k", "status": "documented_nominal"}])
        write_jsonl(root / "qualified_harness_observations.jsonl", [{"profile_id": "harness.test", "field": "optional_assistance",
            "value": ["Python"], "status": "documented_supported"}])
        write_json(root / "manifest.json", {"version": "feature_preparation_v1", "artifacts": {n: file_hash(root / n) for n in FEATURE_FILES}})
        return subject, item

    def test_every_collected_field_is_projected_without_promoting_support_to_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features"
            subject, item = self.fixture(path)
            builder = CollectedStateBuilder(path)
            before = deepcopy((subject, item))
            baseline = builder.build("s", subject, item, enriched=False)
            enriched = builder.build("s", subject, item, enriched=True)
            self.assertEqual(baseline["target_task"], enriched["target_task"])
            self.assertEqual(enriched["target_task"]["item_content"], "  Question 猫 [MASK]  ")
            self.assertNotIn("benchmark_id", enriched["target_task"])
            self.assertEqual((subject, item), before)
            facts = enriched["collected_information"]
            self.assertEqual(facts["documented_model_or_base"]["identity_basis"], "base_model")
            self.assertFalse(facts["documented_model_or_base"]["exact_evaluated_checkpoint_verified"])
            self.assertEqual(facts["documented_model_or_base"]["facts"]["architecture.total_parameters"]["value"], 8000000000)
            self.assertEqual(facts["recorded_run_settings"]["few_shot_count"]["value"], 0)
            self.assertIsNone(facts["recorded_run_settings"]["enabled_tools"]["value"])
            self.assertNotIn("expected", enriched)
            self.assertNotIn("evidence", str(facts))

    def test_mismatched_subject_private_fields_and_changed_tables_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features"
            subject, item = self.fixture(path)
            builder = CollectedStateBuilder(path)
            with self.assertRaisesRegex(ValueError, "identical visible"):
                builder.build("s", dict(subject, harness="different"), item, enriched=True)
            with self.assertRaises(ValueError):
                builder.build("s", dict(subject, display_name="private"), item, enriched=True)
            with (path / "reviewed_static_features.jsonl").open("a") as f:
                f.write(" ")
            with self.assertRaisesRegex(ValueError, "table changed"):
                CollectedStateBuilder(path)

    def test_unconfirmed_values_cannot_become_training_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features"
            self.fixture(path, confirmed=False)
            with self.assertRaisesRegex(ValueError, "Unconfirmed"):
                CollectedStateBuilder(path)
        with self.assertRaises(ValueError):
            CollectedInputsConfig(source_policy="provided_dataset_only")

    def test_compaction_keeps_zero_unknowns_and_documented_architecture_but_excludes_inferences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "features"
            subject, item = self.fixture(root)
            evidence = [{"source_id": "source", "locator": "config", "excerpt": "transformer"}]
            model = Profile.model_validate({"profile_id": "model.test", "entity_type": "model", "identity": {}, "specifications": {
                "architecture": {"backbone": {"value": "transformer", "status": "documented", "evidence": evidence},
                                 "layers": {"value": 32, "status": "inferred", "evidence": evidence}}}})
            harness = Profile.model_validate({"profile_id": "harness.test", "entity_type": "harness", "identity": {},
                "specifications": {"supported_tools": {"value": None, "status": "unknown"}}})
            write_jsonl(root / "supplemented_model_profiles.jsonl", [model.model_dump()])
            write_jsonl(root / "supplemented_harness_profiles.jsonl", [harness.model_dump()])
            import json
            manifest = json.loads((root / "manifest.json").read_text())
            for name in ["supplemented_model_profiles.jsonl", "supplemented_harness_profiles.jsonl"]:
                manifest["artifacts"][name] = file_hash(root / name)
            write_json(root / "manifest.json", manifest)
            builder = CollectedStateBuilderV2(root)
            state = builder.build("s", subject, item, enriched=True)
            c = state["collected_information"]
            self.assertEqual(c["recorded_run_settings"]["values_by_status"]["recorded"]["few_shot_count"], 0)
            self.assertIn("enabled_tools", c["recorded_run_settings"]["unknown_or_unconfirmed"])
            extras = c["documented_model_or_base"]["additional_previously_documented_profile_facts"]
            self.assertEqual(extras["architecture.backbone"], "transformer")
            self.assertNotIn("architecture.layers", extras)
            self.assertEqual(builder.excluded_inferred, 1)
            note = "Rounded language-model component; excludes vision encoder."
            add_scope_notes(state, builder, {"model.test": {"architecture.total_parameters": note}})
            self.assertEqual(c["documented_model_or_base"]["source_scope_qualifiers"]["architecture.total_parameters"], note)
            self.assertEqual(state["target_task"]["item_content"], item["item_content"])

    def test_all_prepared_input_versions_remain_explicit_and_research_only(self):
        for version in ["collected_inputs_v1", "collected_inputs_v2", "collected_inputs_v3"]:
            self.assertEqual(CollectedInputWorkflowConfig.model_validate({"version": version}).version, version)
        with self.assertRaises(ValueError):
            CollectedInputWorkflowConfig.model_validate({"version": "unknown"})
