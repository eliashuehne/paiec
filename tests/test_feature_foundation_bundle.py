import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from feature_foundation_fixtures import corpus


class BundleTests(unittest.TestCase):
    def test_repository_review_ledger_is_included_without_changing_code_hash_root(self):
        m, c = self.modules()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            raw = corpus(root)
            raw["review_ledger"] = "reviews.jsonl"
            ledger = Path(root) / "reviews.jsonl"
            ledger.write_text("")
            original = m.safe_source
            def source(base, relative):
                return ledger if relative == "reviews.jsonl" else original(base, relative)
            with patch.object(m, "safe_source", side_effect=source):
                bundle = m.build_bundle(c.FeatureFoundationConfig.model_validate(raw), source_root=Path(root), out=Path(dest) / "bundle")
            self.assertIn("tasks.py", bundle.receipt["code_hashes"])
            self.assertIn("state_rendering.py", bundle.receipt["dependency_code_hashes"])
            self.assertIn("pyarrow", bundle.receipt["runtime_versions"])

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

    def test_validator_rejects_orphan_feature_and_eligibility_rows_with_valid_digests(self):
        m, c = self.modules()
        from paiec.feature_foundation.tables import write_table
        for name, column in [('task_features','task_key'), ('training_eligibility','item_key')]:
            with self.subTest(table=name), tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
                config = c.FeatureFoundationConfig.model_validate(corpus(root))
                out = Path(dest) / 'bundle'
                m.build_bundle(config, source_root=Path(root), out=out)
                path = out / (name+'.parquet')
                rows = pq.read_table(path).to_pylist()
                rows[0][column] = 'orphan'
                metadata = write_table(out, name, rows)
                manifest = json.loads((out/'manifest.json').read_text())
                manifest['logical_tables'][name] = metadata
                manifest['artifacts'][path.name] = {'sha256':c.file_hash(path), 'bytes':path.stat().st_size}
                (out/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'Orphan|orphan|eligibility'):
                    m.validate_bundle(out)

    def test_source_change_after_verification_cannot_publish_a_completed_bundle(self):
        m, c = self.modules()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config = c.FeatureFoundationConfig.model_validate(corpus(root))
            out = Path(dest) / 'bundle'
            def mutate(progress):
                if progress['benchmark'] == 'alpha':
                    path = Path(root)/'payload/data/beta/formatted_tables/items.parquet'
                    rows = pq.read_table(path).to_pylist(); rows[0]['content']='Changed after verification'
                    pq.write_table(pa.Table.from_pylist(rows), path)
            with self.assertRaisesRegex(ValueError, 'hash|changed'):
                m.build_bundle(config, source_root=Path(root), out=out, progress=mutate)
            self.assertTrue((out/'INCOMPLETE').exists())
            self.assertFalse((out/'manifest.json').exists())

    def test_unpinned_or_escaping_optional_enrichment_is_refused(self):
        m, c = self.modules()
        for escape in [False, True]:
            with self.subTest(escape=escape), tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
                raw = corpus(root)
                export = Path(root)/raw.get('enrichment_export','runs/shared_data/system_enrichment_v2')
                export.mkdir(parents=True)
                outside = Path(dest)/'outside.jsonl'; outside.write_text('')
                for name in ['models','harnesses','sources','catalog_systems','runs']:
                    (export/(name+'.jsonl')).write_text('')
                if escape:
                    (export/'models.jsonl').unlink(); (export/'models.jsonl').symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'root|hash-bound|pinned'):
                    m.build_bundle(c.FeatureFoundationConfig.model_validate(raw), source_root=Path(root), out=Path(dest)/'bundle')

    def test_validator_checks_composite_response_variant_reference(self):
        m, c = self.modules()
        from paiec.feature_foundation.tables import write_table
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            out = Path(dest)/'bundle'
            m.build_bundle(c.FeatureFoundationConfig.model_validate(corpus(root)),source_root=Path(root),out=out)
            rows = pq.read_table(out/'response_input_links.parquet').to_pylist()
            rows[0]['interactor_key']='NO_SUCH_INPUT_VARIANT'
            metadata=write_table(out,'response_input_links',rows)
            path=out/'response_input_links.parquet'; manifest=json.loads((out/'manifest.json').read_text())
            manifest['logical_tables']['response_input_links']=metadata
            manifest['artifacts'][path.name]={'sha256':c.file_hash(path),'bytes':path.stat().st_size}
            (out/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'variant'):
                m.validate_bundle(out)

