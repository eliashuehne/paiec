import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

from feature_foundation_fixtures import corpus


class FoundationContractsTests(unittest.TestCase):
    def module(self, name):
        try:
            return importlib.import_module("paiec.feature_foundation." + name)
        except ModuleNotFoundError:
            self.fail("Feature foundation " + name + " is not implemented")

    def test_default_recipe_includes_all_pinned_repository_enrichment(self):
        import json
        contracts = self.module("contracts")
        recipe = Path("configs/features/foundation_v1.json")
        config = contracts.FeatureFoundationConfig.model_validate_json(recipe.read_text())
        self.assertEqual(getattr(config, 'enrichment_scope', None), 'repository')
        self.assertEqual(config.enrichment_export, 'research/system_enrichment/snapshot')
        prefix = config.enrichment_export + '/'
        expected = {key: value for key, value in config.input_hashes.items() if key.startswith(prefix)}
        self.assertEqual(len(expected), 109)
        for relative, digest in expected.items():
            self.assertEqual(contracts.file_hash(relative), digest, relative)
        directory = Path(config.enrichment_export)
        self.assertEqual({str(path) for path in directory.rglob('*') if path.is_file()}, set(expected))

    def test_repository_enrichment_cannot_shadow_the_dataset_namespace(self):
        contracts = self.module('contracts')
        with tempfile.TemporaryDirectory() as root:
            raw = corpus(root)
            for prefix in ['payload', 'payload/data', 'payload/data/enrichment', './payload/', 'payload//data']:
                with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, 'namespace.*overlap|overlap.*namespace'):
                    contracts.FeatureFoundationConfig.model_validate(dict(raw, enrichment_scope='repository',
                                                                         enrichment_export=prefix))

    def test_repository_enrichment_is_loaded_separately_and_keeps_root_and_hash_guards(self):
        contracts, sources = self.module('contracts'), self.module('sources')
        from paiec.feature_foundation.build import build_bundle
        with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root/'checkout'
            relative = 'research/enrichment'
            enrichment = repo/relative
            (enrichment/'provenance').mkdir(parents=True)
            raw = corpus(data)
            raw.update(enrichment_scope='repository', enrichment_export=relative)
            for name in ['models.jsonl', 'harnesses.jsonl', 'sources.jsonl', 'catalog_systems.jsonl', 'runs.jsonl', 'provenance/evidence.json']:
                path = enrichment/name
                path.write_text('')
                raw['input_hashes'][relative+'/'+name] = contracts.file_hash(path)
            with patch.object(sources, 'repository_root', return_value=repo, create=True):
                config = contracts.FeatureFoundationConfig.model_validate(raw)
                out = root/'bundle'
                build_bundle(config, source_root=data, out=out)
                self.assertTrue((out/'provenance/evidence.json').exists())
                # Neither an adjacent checkout directory nor a changed archive is accepted.
                (enrichment/'models.jsonl').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'hash'):
                    sources.verify_sources(config, data)
                (enrichment/'models.jsonl').unlink()
                outside = root/'outside.jsonl'
                outside.write_text('')
                (enrichment/'models.jsonl').symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'root'):
                    sources.verify_sources(config, data)
                (enrichment/'models.jsonl').unlink()
                (enrichment/'models.jsonl').write_text('')
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    build_bundle(config, source_root=data, out=enrichment/'output')

    def test_full_population_has_no_scorer_threshold_and_preserves_protected_rows(self):
        sources = self.module("sources")
        contracts = self.module("contracts")
        with tempfile.TemporaryDirectory() as root:
            config = contracts.FeatureFoundationConfig.model_validate(corpus(root))
            rows = sources.inventory_population(config, Path(root))
            self.assertEqual(sum(r["items"] for r in rows), 6)
            self.assertEqual(sum(r["responses"] for r in rows), 9)
            self.assertEqual({r["benchmark"]: r["split"] for r in rows}, {"alpha": "training", "beta": "training", "sealed": "test"})
            self.assertTrue(next(r for r in rows if r["benchmark"] == "sealed")["protected"])
            self.assertEqual(next(r for r in rows if r["benchmark"] == "beta")["feature_status"], "partial_nontext")

    def test_only_metadata_is_needed_for_population_census(self):
        sources = self.module("sources")
        contracts = self.module("contracts")
        original = pq.read_table

        def guarded(path, **kwargs):
            if Path(path).name in {"items.parquet", "response.parquet", "subjects.parquet"}:
                raise AssertionError("Census must not read table values")
            return original(path, **kwargs)

        with tempfile.TemporaryDirectory() as root:
            config = contracts.FeatureFoundationConfig.model_validate(corpus(root))
            with patch("pyarrow.parquet.read_table", side_effect=guarded):
                self.assertEqual(len(sources.inventory_population(config, Path(root))), 3)

    def test_source_hash_change_and_escape_are_rejected(self):
        sources = self.module("sources")
        contracts = self.module("contracts")
        with tempfile.TemporaryDirectory() as root:
            raw = corpus(root)
            raw["input_hashes"]["payload/data/alpha/formatted_tables/items.parquet"] = "0" * 64
            config = contracts.FeatureFoundationConfig.model_validate(raw)
            with self.assertRaisesRegex(ValueError, "hash"):
                sources.verify_sources(config, Path(root))
            with self.assertRaises(ValueError):
                sources.safe_source(Path(root), "../outside")

    def test_frozen_role_conflicts_are_not_resolved_by_random_assignment(self):
        sources = self.module("sources")
        with self.assertRaisesRegex(ValueError, "conflict"):
            sources.assign_splits(["alpha", "beta"], {"training": ["alpha"], "confirmation": ["beta"]}, [["alpha", "beta"]], 12)

    def test_subject_key_uses_exact_visible_strings_and_no_private_fields(self):
        contracts = self.module("contracts")
        raw = {k: "" for k in contracts.SUBJECT_FIELDS}
        raw["normalized_name"] = " Example "
        self.assertEqual(contracts.project_subject(raw)["normalized_name"], " Example ")
        with self.assertRaises(ValueError):
            contracts.project_subject(dict(raw, display_name="private"))
        self.assertNotEqual(contracts.stable_key(raw), contracts.stable_key(dict(raw, normalized_name="Example")))

    def test_joins_reject_duplicate_feature_keys_and_preserve_repeated_base_rows(self):
        joins = self.module("joins")
        base = pd.DataFrame({"benchmark": ["a", "a", "b"], "item_id": ["i", "i", "i"]})
        features = pd.DataFrame({"benchmark": ["a", "b"], "item_id": ["i", "i"], "length": [2, 3]})
        joined = joins.safe_left_join(base, features, ("benchmark", "item_id"))
        self.assertEqual(joined["length"].tolist(), [2, 2, 3])
        with self.assertRaises(ValueError):
            joins.safe_left_join(base, pd.concat([features, features]), ("benchmark", "item_id"))


if __name__ == "__main__":
    unittest.main()
