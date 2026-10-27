"""Supplied-only lift views: exact trial inputs, no labels or role reassignment."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from feature_foundation_fixtures import corpus
from paiec.feature_foundation.build import build_bundle
from paiec.feature_foundation.contracts import FeatureFoundationConfig, file_hash


class LiftProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.raw = corpus(self.source)

    def module(self):
        try:
            return importlib.import_module('paiec.feature_foundation.lift')
        except ModuleNotFoundError:
            self.fail('The supplied-only lift projection is missing')

    def replace(self, name, rows):
        relative = f'payload/data/alpha/formatted_tables/{name}.parquet'
        path = self.source / relative
        pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
        self.raw['input_hashes'][relative] = file_hash(path)

    def rows(self, name):
        return pq.read_table(self.source / f'payload/data/alpha/formatted_tables/{name}.parquet').to_pylist()

    def build(self):
        return build_bundle(FeatureFoundationConfig.model_validate(self.raw), source_root=self.source,
                            out=self.root / 'bundle').directory

    def keys(self):
        return pd.DataFrame({'benchmark': ['alpha', 'alpha', 'sealed', 'absent'],
                             'subject_id': ['s1'] * 4, 'item_id': ['i1'] * 4, 'trial': [0, 1, 0, 0]})

    def test_trial_variants_keep_typed_settings_and_do_not_change_sample(self):
        module = self.module()
        subjects = self.rows('subjects')
        subjects[0]['subject_features_extra'] = 'n_shots=0;do_sample=false;enabled_tools=[];temperature=0'
        self.replace('subjects', subjects)
        responses = self.rows('response')
        responses[0]['interactors'] = ''
        responses[1]['interactors'] = 'python'
        self.replace('response', responses)
        bundle = self.build()
        keys = self.keys()
        sample = pd.concat([keys, keys.iloc[[0]]], ignore_index=True).assign(label=[1, 0, 1, 0, 1], fold=['0', '0', 'sealed', '1', '0'])
        original = sample.copy(deep=True)
        table, coverage = module.project_lift_features(bundle, sample)
        pd.testing.assert_frame_equal(sample, original)
        self.assertEqual(len(table), 4)
        first = table.iloc[0]
        self.assertEqual(first['recorded_few_shot_count'], 0)
        self.assertEqual(first['recorded_sampling_enabled'], False)
        self.assertEqual(first['recorded_enabled_tools'], '[]')
        self.assertEqual(first['recorded_few_shot_count_status'], 'known')
        self.assertTrue(pd.isna(first['recorded_top_p']))
        self.assertEqual(first['recorded_top_p_status'], 'unknown')
        self.assertEqual(table.iloc[:2]['task_has_supplied_interactors'].tolist(), [False, True])
        self.assertTrue(table.iloc[2:].filter(regex='^(recorded_|task_)').isna().all().all())
        self.assertEqual(coverage['task_resolution'], {'matched': 2, 'missing_response': 1, 'withheld_protected': 1})
        self.assertNotIn('label', table)
        self.assertNotIn('fold', table)
        self.assertFalse(any('research' in c or c.endswith('_key') or c in {'split', 'eligible'} for c in table.columns))
        altered = sample.assign(label=1 - sample.label, fold='sealed')
        changed, _ = module.project_lift_features(bundle, altered)
        pd.testing.assert_frame_equal(table, changed)

    def test_conflicting_setting_and_ambiguous_trial_are_missing(self):
        module = self.module()
        subjects = self.rows('subjects')
        subjects[0]['subject_features_extra'] = 'n_shots=5;n_shots=2'
        self.replace('subjects', subjects)
        responses = self.rows('response')
        responses.append({**responses[0], 'response_id': 'duplicate', 'interactors': 'other'})
        self.replace('response', responses)
        self.raw['expected_counts']['responses'] += 1
        table, coverage = module.project_lift_features(self.build(), self.keys().iloc[:2])
        self.assertTrue(table['recorded_few_shot_count'].isna().all())
        self.assertEqual(table['recorded_few_shot_count_status'].tolist(), ['conflict', 'conflict'])
        self.assertTrue(table.iloc[[0]].filter(regex='^task_').isna().all().all())
        self.assertEqual(coverage['task_resolution'], {'ambiguous_trial': 1, 'matched': 1})

    def test_duplicate_identical_trials_are_harmless_and_keys_are_strict(self):
        module = self.module()
        responses = self.rows('response')
        responses.append({**responses[0], 'response_id': 'duplicate'})
        self.replace('response', responses)
        self.raw['expected_counts']['responses'] += 1
        bundle = self.build()
        table, coverage = module.project_lift_features(bundle, self.keys().iloc[:1])
        self.assertEqual(len(table), 1)
        self.assertEqual(coverage['task_resolution'], {'matched': 1})
        for bad in [self.keys().drop(columns='trial'), self.keys().assign(trial=0.5),
                    self.keys().assign(subject_id=None), self.keys().assign(trial=True)]:
            with self.subTest(columns=bad.dtypes.to_dict()), self.assertRaises(ValueError):
                module.project_lift_features(bundle, bad)
        absent, gaps = module.project_lift_features(bundle, self.keys().iloc[3:])
        self.assertTrue(absent.filter(regex='^(recorded_|task_)').isna().all().all())
        self.assertEqual(gaps['task_resolution'], {'missing_response': 1})
        empty, _ = module.project_lift_features(bundle, self.keys().iloc[:0])
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty['recorded_few_shot_count'].dtype, pd.Int64Dtype())

    def test_rejects_incomplete_and_corrupt_consumed_tables(self):
        module = self.module()
        bundle = self.build()
        marker = bundle / 'INCOMPLETE'
        marker.touch()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            module.project_lift_features(bundle, self.keys())
        marker.unlink()
        with (bundle / 'task_features.parquet').open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash'):
            module.project_lift_features(bundle, self.keys())

    def test_export_rejects_unverified_dictionary_and_never_opens_research(self):
        module = self.module()
        bundle = self.build()
        before, _ = module.project_lift_features(bundle, self.keys())
        (bundle/'research_system_features.parquet').write_bytes(b'not read')
        after, _ = module.project_lift_features(bundle, self.keys())
        pd.testing.assert_frame_equal(before, after)
        (bundle/'feature_dictionary.json').write_text('[]')
        sample = self.root/'sample.parquet'
        self.keys().to_parquet(sample, index=False)
        with self.assertRaisesRegex(ValueError, 'hash'):
            module.export_lift_features(bundle, sample, self.root/'projection')

    def test_cli_plans_without_inputs_and_writes_fingerprinted_features(self):
        script = Path('scripts/lift/build_foundation_features.py')
        result = subprocess.run([sys.executable, str(script), '--bundle', str(self.root/'absent'),
                                 '--sample', str(self.root/'absent.parquet'), '--out', str(self.root/'out')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['execution_started'])
        self.assertFalse((self.root/'out').exists())
        bundle = self.build()
        sample = self.root / 'sample.parquet'
        self.keys().assign(label=1, fold='sealed').to_parquet(sample, index=False)
        out = self.root / 'out'
        result = subprocess.run([sys.executable, str(script), '--bundle', str(bundle), '--sample', str(sample),
                                 '--out', str(out), '--execute'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((out/'manifest.json').read_text())
        self.assertEqual(manifest['sample_sha256'], file_hash(sample))
        self.assertEqual(manifest['artifacts']['features.parquet']['sha256'], file_hash(out/'features.parquet'))
        self.assertEqual(manifest['sample_columns_read'], ['benchmark', 'subject_id', 'item_id', 'trial'])
        self.assertEqual(manifest['policy'], 'supplied_only')
        self.assertFalse((out/'INCOMPLETE').exists())
        self.assertEqual(pq.read_table(out/'features.parquet').num_rows, 4)
        self.assertTrue((out/'dictionary.json').exists())
        self.assertTrue(all(row['source_definition'] for row in json.loads((out/'dictionary.json').read_text())))
        result = subprocess.run([sys.executable, str(script), '--bundle', str(bundle), '--sample', str(sample),
                                 '--out', str(out), '--execute'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fresh', result.stderr)
