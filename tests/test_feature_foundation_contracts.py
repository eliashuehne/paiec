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
