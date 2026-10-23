import json
from pathlib import Path
import tempfile
import unittest

from paiec.feature_preparation import write_json, write_jsonl, read_jsonl
from paiec.feature_sources import supplement_models
from paiec.head_training import file_hash
from paiec.system_profiles import Profile, ModelSpecifications


class SourceSupplementTests(unittest.TestCase):
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
