import json
from pathlib import Path
import tempfile
import unittest

from feature_foundation_fixtures import corpus
from paiec.feature_foundation.build import build_bundle
from paiec.feature_foundation.contracts import FeatureFoundationConfig
from paiec.feature_foundation.report import coverage_report
from paiec.feature_foundation.review import table_rows
from paiec.feature_foundation.tables import write_table


class ReportTests(unittest.TestCase):
    def test_first_public_dictionary_has_no_internal_revision_labels(self):
        from paiec.feature_foundation.report import feature_dictionary
        entries = feature_dictionary([])
        self.assertEqual({row['version'] for row in entries}, {'feature_foundation_v1'})
        recorded = [row for row in entries if row['name'].startswith('run.')]
        self.assertTrue(all('recorded_settings_v1' in row['definition'] for row in recorded))
        semantic = [row for row in entries if row['name'].startswith('semantic.')]
        self.assertTrue(all('task-taxonomy-v2' not in row['definition'] for row in semantic))

    def test_sparse_research_fields_and_protected_subjects_are_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config = FeatureFoundationConfig.model_validate(corpus(root))
            out = Path(dest) / 'bundle'
            build_bundle(config, source_root=Path(root), out=out)
            matches = list(table_rows(out / 'profile_matches.parquet'))
            for row in matches:
                if row['entity_type'] == 'model':
                    row.update(profile_id='p', status='matched')
            write_table(out, 'profile_matches', matches)
            # A matching profile exists, but does not supply this field.
            report = coverage_report(out, list(table_rows(out / 'population.parquet')),
                [{'entity_type':'model', 'field':'languages.documented', 'profile_id':'p'}], {})
            field = report['research_features']['model.languages.documented']
            self.assertEqual(field['missing_profile_field'], 1)
            self.assertEqual(sum(field.values()), report['distinct_processed_visible_systems'])
            for benchmark in report['per_benchmark']:
                slots = benchmark['research_features']['model.languages.documented']
                self.assertEqual(sum(slots.values()), benchmark['subjects'])
            sealed = next(r for r in report['per_benchmark'] if r['protected'])
            self.assertEqual(sealed['research_features']['model.languages.documented']['withheld_protected'], 1)

    def test_taxonomy_audit_reports_candidates_and_unknowns_per_facet(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config = FeatureFoundationConfig.model_validate(corpus(root))
            out = Path(dest) / 'bundle'
            build_bundle(config, source_root=Path(root), out=out)
            row = {'kind':'task', 'taxonomy_review':{'candidate_facets':{'domain':['programming']}}}
            (out / 'reviewed_examples.jsonl').write_text(json.dumps(row)+'\n')
            report = coverage_report(out, list(table_rows(out / 'population.parquet')), [], {})
            self.assertEqual(report['taxonomy_audit']['domain'], {'candidate':1, 'unknown':0, 'validated':0})
            self.assertEqual(report['taxonomy_audit']['language'], {'candidate':0, 'unknown':1, 'validated':0})
            self.assertEqual(report['task_feature_population']['test']['withheld_protected'], 2)
