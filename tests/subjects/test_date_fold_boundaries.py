import unittest

import numpy as np
import pandas as pd

from paiec.lift import run_lift
from paiec.subject_dates import fill_release_dates


def dates():
    rows = []
    # Matching versions in training support the method; the sealed family disagrees.
    for subject, identity, family, date in [
        ('a', 'a', 'public', 100), ('b', 'b', 'public', 100),
        ('u', 'u', 'public', np.nan),
        ('sealed-a', 'sealed-a', 'private', 0),
        ('sealed-b', 'sealed-b', 'private', 5000),
        ('sealed-c', 'sealed-c', 'private', 10000),
        ('sealed-d', 'sealed-d', 'private', 15000),
    ]:
        rows.append(dict(subject_id=subject, identity_key=identity, family=family, line='',
                         version=1.0, log_params_b=1.0, provider_filled=family,
                         release_days=float(date), name_date_days=np.nan))
    return pd.DataFrame(rows)


class DateBoundaryTests(unittest.TestCase):
    def folds(self):
        return {'a': frozenset({'1'}), 'b': frozenset({'1'}), 'u': frozenset({'0'}),
                **{f'sealed-{s}': frozenset({'sealed'}) for s in 'abcd'}}

    def test_sealed_dates_cannot_select_methods_for_open_subjects(self):
        frame = dates()
        first = fill_release_dates(frame, self.folds(), accepted=None).set_index('subject_id').loc['u']
        frame.loc[frame['subject_id'].str.startswith('sealed'), 'release_days'] = 100
        second = fill_release_dates(frame, self.folds(), accepted=None).set_index('subject_id').loc['u']
        self.assertEqual(first['release_days_filled'], 100)
        pd.testing.assert_series_equal(first, second)

    def test_outer_fold_is_excluded_even_for_training_subjects(self):
        frame = dates().iloc[:3].copy()
        folds = self.folds()
        folds['a'] = frozenset({'2'})
        # The only reliable reference pair is in held-out fold 1.
        frame = pd.concat([frame, frame.iloc[[1]].assign(subject_id='c', identity_key='c')], ignore_index=True)
        folds['c'] = frozenset({'1'})
        frame.loc[frame['subject_id'] == 'a', 'release_days'] = np.nan
        filled = fill_release_dates(frame, folds, accepted=None, excluded_folds={'1'})
        self.assertTrue(pd.isna(filled.set_index('subject_id').loc['a', 'release_days_filled']))

    def test_empty_reference_set_leaves_missing_dates_missing(self):
        frame = dates().query("subject_id == 'u'")
        out = fill_release_dates(frame, self.folds(), accepted=None)
        self.assertTrue(pd.isna(out['release_days_filled'].iloc[0]))

    def test_lift_rebuilds_training_dates_without_outer_validation_fold(self):
        frame = dates().iloc[:3].copy()
        # Both training subjects are missing dates. The only dated subject is held out.
        frame.loc[frame['subject_id'] != 'a', 'release_days'] = np.nan
        frame = frame.assign(benchmark=['heldout', 'train', 'train'], fold=['0', '1', '1'],
                             item_id=['i0', 'i1', 'i2'], trial=1, label=[0, 1, 0],
                             release_days_filled=12345.0)
        seen = []
        def predict(train, test, columns, **kwargs):
            if test['fold'].iloc[0] == '0':
                seen.append(train['release_days_filled'].to_numpy())
            return np.full(len(test), 0.5)
        run_lift(frame, columns=['release_days_filled'], predict=predict)
        self.assertTrue(np.isnan(seen[0]).all())


class ExportDateBoundaryTests(unittest.TestCase):
    def test_subjects_omitted_from_sample_cannot_become_date_references(self):
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        from paiec.schema import SUBJECT_FIELDS
        root = Path(__file__).resolve().parents[2]
        fields = pd.DataFrame([
            {**dict.fromkeys(SUBJECT_FIELDS, ''), 'subject_id': subject,
             'normalized_name': f'Qwen3-7B-{subject}', 'release_date': date}
            for subject, date in [('a', '2024-01-01'), ('b', '2024-01-01'), ('u', None)]
        ])
        omitted = pd.DataFrame([
            {**dict.fromkeys(SUBJECT_FIELDS, ''), 'subject_id': f'omitted-{k}',
             'normalized_name': f'Llama3-7B-{k}', 'release_date': f'{2020 + 3 * k}-01-01'}
            for k in range(4)
        ])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pd.DataFrame({'subject_id': ['a', 'b', 'u'], 'fold': ['1', '1', '0']}).to_parquet(folder / 'sample.parquet')
            results = []
            for name, table in [('before', fields), ('after', pd.concat([fields, omitted]))]:
                table.to_parquet(folder / 'fields.parquet', index=False)
                command = [sys.executable, str(root / 'scripts/subjects/build_subject_features.py'),
                           '--fields', str(folder / 'fields.parquet'), '--sample', str(folder / 'sample.parquet'),
                           '--out', str(folder / name / 'subjects.parquet')]
                completed = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                out = pd.read_parquet(folder / name / 'subjects.parquet').set_index('subject_id')
                report = json.loads((folder / name / 'subjects_manifest.json').read_text())
                results.append((out.loc['u', 'release_days_filled'], report['date_backtest']))
            self.assertFalse(pd.isna(results[0][0]))
            self.assertEqual(results[0], results[1])
