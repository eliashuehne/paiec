import importlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from feature_foundation_fixtures import corpus


class BundleTests(unittest.TestCase):
    def modules(self):
        try:
            build = importlib.import_module("paiec.feature_foundation.build")
            contracts = importlib.import_module("paiec.feature_foundation.contracts")
            return build, contracts
        except ModuleNotFoundError:
            self.fail("Complete bundle publication is missing")

    def test_bundle_has_full_counts_dictionary_and_explicit_review_shortfall(self):
        m, c = self.modules()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config = c.FeatureFoundationConfig.model_validate(corpus(root))
            bundle = m.build_bundle(config, source_root=Path(root), out=Path(dest) / "bundle")
            report = json.loads((bundle.directory / "coverage.json").read_text())
            self.assertEqual(report["population"]["items"], 6)
            self.assertEqual(report["population"]["responses"], 9)
            self.assertEqual(report["item_status"]["withheld_protected"], 2)
            self.assertFalse(report["review"]["complete"])
            self.assertTrue((bundle.directory / "report.html").exists())
            self.assertTrue(m.validate_bundle(bundle.directory)["valid"])
            dictionary = json.loads((bundle.directory / "feature_dictionary.json").read_text())
            self.assertTrue(any(r["name"] == "run.few_shot_count" and r["missing_behavior"] for r in dictionary))

    def test_deterministic_tables_do_not_depend_on_fixture_outcomes(self):
        m, c = self.modules()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            raw = corpus(root)
            config = c.FeatureFoundationConfig.model_validate(raw)
            a = m.build_bundle(config, source_root=Path(root), out=Path(dest) / "a")
            path = Path(root) / "payload/data/alpha/formatted_tables/response.parquet"
            table = pq.read_table(path)
            table = table.set_column(table.schema.get_field_index("response"), "response", pa.array([0, 1, 0]))
            pq.write_table(table, path)
            raw["input_hashes"][str(path.relative_to(root))] = c.file_hash(path)
            b = m.build_bundle(c.FeatureFoundationConfig.model_validate(raw), source_root=Path(root), out=Path(dest) / "b")
            self.assertEqual(a.manifest["logical_tables"], b.manifest["logical_tables"])
            self.assertNotEqual(a.receipt["input_hashes"], b.receipt["input_hashes"])

    def test_refuses_existing_output_and_tampered_completed_artifact(self):
        m, c = self.modules()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config = c.FeatureFoundationConfig.model_validate(corpus(root))
            out = Path(dest) / "bundle"
            m.build_bundle(config, source_root=Path(root), out=out)
            with self.assertRaises(FileExistsError):
                m.build_bundle(config, source_root=Path(root), out=out)
            (out / "feature_dictionary.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "hash"):
                m.validate_bundle(out)

    def test_refuses_input_output_overlap_and_final_manifest_on_failure(self):
        m, c = self.modules()
        with tempfile.TemporaryDirectory() as root:
            config = c.FeatureFoundationConfig.model_validate(corpus(root))
            with self.assertRaises(ValueError):
                m.build_bundle(config, source_root=Path(root), out=Path(root) / "payload/data/feature-output")

    def test_validator_checks_foreign_keys_even_when_byte_hashes_are_updated(self):
        m, c = self.modules()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config = c.FeatureFoundationConfig.model_validate(corpus(root))
            out = Path(dest) / "bundle"
            m.build_bundle(config, source_root=Path(root), out=out)
            path = out / "run_settings.parquet"
            rows = pq.read_table(path).to_pylist()
            rows[0]["system_key"] = "orphan"
            pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["artifacts"][path.name]["sha256"] = c.file_hash(path)
            # Leave logical digest untouched: either proof must reject this.
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                m.validate_bundle(out)

    def test_required_dictionary_cannot_be_dropped_from_the_manifest(self):
        m, c = self.modules()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config = c.FeatureFoundationConfig.model_validate(corpus(root))
            out = Path(dest) / "bundle"
            m.build_bundle(config, source_root=Path(root), out=out)
            manifest = json.loads((out / "manifest.json").read_text())
            del manifest["artifacts"]["feature_dictionary.json"]
            (out / "feature_dictionary.json").unlink()
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                m.validate_bundle(out)

