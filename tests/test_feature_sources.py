import json
from pathlib import Path
import tempfile
import unittest

from paiec.feature_preparation import write_json, write_jsonl, read_jsonl
from paiec.feature_sources import supplement_models, supplement_harnesses
from paiec.head_training import file_hash
from paiec.system_profiles import Profile, ModelSpecifications, HarnessSpecifications


class SourceSupplementTests(unittest.TestCase):
    def test_harness_support_is_separate_from_actual_run_settings_and_model_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models, _ = self.fixture(root)
            supplement_models(root, models, root)
            model_bytes = (root / "supplemented_model_profiles.jsonl").read_bytes()
            model_audit = (root / "source_collection_audit.json").read_bytes()
            profile = Profile(profile_id="harness.test", entity_type="harness", specifications=HarnessSpecifications())
            write_jsonl(root / "harnesses.jsonl", [profile.model_dump(mode="json")])
            with self.assertRaisesRegex(ValueError, "wrong entity type"):
                supplement_harnesses(root, models, root)
            ledger = root / "research/feature_collection/assertions_v1.jsonl"
            assertion = next(read_jsonl(ledger))
            assertion.update(profile_id="harness.test", field="supported_tools", value=["English parser"])
            write_jsonl(ledger, [assertion])
            result = supplement_harnesses(root, {"path": str(root / "harnesses.jsonl")}, root)
            harness = next(read_jsonl(result["path"]))
            self.assertEqual(harness["specifications"]["supported_tools"]["value"], ["English parser"])
            self.assertNotIn("enabled_tools", harness["specifications"])
            self.assertEqual(model_bytes, (root / "supplemented_model_profiles.jsonl").read_bytes())
            self.assertEqual(model_audit, (root / "source_collection_audit.json").read_bytes())

    def fixture(self, root):
        catalog = root / "research/feature_collection"
        catalog.mkdir(parents=True)
        cache = root / "runs/feature_sources/primary_v1"
        cache.mkdir(parents=True)
        (cache / "card.md").write_text("Supported language: English.")
        source = {"source_id": "card", "sha256": file_hash(cache / "card.md"), "revision": "pinned"}
        write_jsonl(catalog / "sources.jsonl", [source])
        write_json(catalog / "snapshots.json", [dict(source, cache_file="card.md")])
        write_jsonl(catalog / "assertions_v1.jsonl", [{"profile_id": "model.test", "field": "documented_languages",
                    "value": ["English"], "status": "documented", "source_id": "card",
                    "quote": "Supported language: English.", "note": "Named artifact only"}])
        model = Profile(profile_id="model.test", entity_type="model", specifications=ModelSpecifications())
        write_jsonl(root / "models.jsonl", [model.model_dump(mode="json")])
        return {"path": str(root / "models.jsonl"), "profiles": 1}, cache

    def test_additions_preserve_the_original_catalog_and_validate_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models, cache = self.fixture(root)
            before = Path(models["path"]).read_bytes()
            result = supplement_models(root, models, root)
            self.assertEqual(result["source_collection"]["applied"], 1)
            self.assertEqual(next(read_jsonl(result["path"]))["specifications"]["documented_languages"]["value"], ["English"])
            self.assertEqual(before, Path(models["path"]).read_bytes())
            profile = next(read_jsonl(result["path"]))
            source = next(read_jsonl(root / "supplemental_sources.jsonl"))
            evidence = profile["specifications"]["documented_languages"]["evidence"][0]
            self.assertEqual(evidence["source_id"], source["source_id"])
            self.assertNotEqual(source["source_id"], "card")
            self.assertIn(file_hash(cache / "card.md"), source["source_id"])
            (cache / "card.md").write_text("Changed")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                supplement_models(root, models, root)

    def test_conflicting_documented_values_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models, _ = self.fixture(root)
            result = supplement_models(root, models, root)
            profile = next(read_jsonl(result["path"]))
            profile["specifications"]["documented_languages"]["value"] = ["German"]
            write_jsonl(root / "models.jsonl", [profile])
            result = supplement_models(root, models, root)
            self.assertEqual(len(result["source_collection"]["conflicts"]), 1)
            self.assertEqual(result["source_collection"]["applied"], 0)

    def test_registry_loads_bundles_and_rejects_storage_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models, _ = self.fixture(root)
            registry = root / "configs/pipelines/feature_sources_v1.json"
            registry.parent.mkdir(parents=True)
            bundle = {"catalog": "research/feature_collection",
                      "cache": "runs/feature_sources/primary_v1",
                      "assertions": "assertions_v1.jsonl"}
            write_json(registry, {"version": "feature_sources_v1", "bundles": [bundle]})
            result = supplement_models(root, models, root)
            self.assertEqual(result["source_collection"]["applied"], 1)
            self.assertEqual(len(result["source_collection"]["assertion_bundles"]), 1)
            for override in [{"cache": "research/cache"}, {"catalog": "../outside"},
                             {"assertions": "../outside.jsonl"}]:
                with self.subTest(override=override):
                    write_json(registry, {"bundles": [dict(bundle, **override)]})
                    with self.assertRaisesRegex(ValueError, "escapes"):
                        supplement_models(root, models, root)

    def test_every_supporting_quote_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models, _ = self.fixture(root)
            ledger = root / "research/feature_collection/assertions_v1.jsonl"
            assertion = next(read_jsonl(ledger))
            assertion["supporting_quotes"] = ["Unsupported extra claim"]
            write_jsonl(ledger, [assertion])
            with self.assertRaisesRegex(ValueError, "exact supporting text"):
                supplement_models(root, models, root)

    def test_nominal_context_remains_a_string_and_does_not_become_an_exact_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models, cache = self.fixture(root)
            catalog = root / "research/feature_collection"
            (cache / "card.md").write_text("Supported language: English. Context: 128K tokens.")
            source = {"source_id": "card", "sha256": file_hash(cache / "card.md"), "revision": "pinned"}
            write_jsonl(catalog / "sources.jsonl", [source])
            write_json(catalog / "snapshots.json", [dict(source, cache_file="card.md")])
            observation = {"profile_id": "model.test", "field": "documented_context_window",
                           "value": "128K", "status": "documented_nominal", "source_id": "card",
                           "quote": "128K tokens", "note": "Exact integer limit unestablished."}
            write_jsonl(catalog / "nominal.jsonl", [observation])
            registry = root / "configs/pipelines/sources.json"
            registry.parent.mkdir(parents=True)
            write_json(registry, {"bundles": [{"catalog": "research/feature_collection",
                "cache": "runs/feature_sources/primary_v1", "assertions": "assertions_v1.jsonl"}],
                "nominal_context_ledger": "research/feature_collection/nominal.jsonl"})
            result = supplement_models(root, models, root, source_registry="configs/pipelines/sources.json")
            profile = next(read_jsonl(result["path"]))
            self.assertIsNone(profile["specifications"]["max_combined_context_tokens"]["value"])
            self.assertEqual(next(read_jsonl(root / "qualified_model_observations.jsonl"))["value"], "128K")
            observation["value"] = 128000
            write_jsonl(catalog / "nominal.jsonl", [observation])
            with self.assertRaisesRegex(ValueError, "qualified string"):
                supplement_models(root, models, root, source_registry="configs/pipelines/sources.json")
