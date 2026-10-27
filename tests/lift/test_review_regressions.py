"""Competition boundaries that the original budget tests did not exercise."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from paiec.lift import run_lift
from paiec.lift_budget import combine_budgets
from paiec.lift_report import compare, save_run


def unequal_pairs():
    parts = []
    for fold, count in [('0', 1), ('1', 2)]:
        for pair in range(count):
            n = 1000 if fold == '0' else 10
            parts.append(pd.DataFrame({
                'fold': fold, 'benchmark': 'b' + fold, 'subject_id': f's{fold}{pair}',
                'item_id': [f'i{k}' for k in range(n)], 'trial': 1,
                'label': int(fold == '1'),
            }))
    return pd.concat(parts, ignore_index=True)


class PairWeightingTests(unittest.TestCase):
    def scores(self, probability):
        return run_lift(
            unequal_pairs(), pair_weighted=True,
            predict=lambda train, test, columns, **kwargs: np.full(len(test), probability),
        )

    def test_each_pair_counts_once_despite_row_and_fold_sizes(self):
        base, candidate = self.scores(0.0), self.scores(0.5)
        summary = compare(base, candidate).summary
        self.assertAlmostEqual(summary['base_brier'], 2 / 3)
        self.assertAlmostEqual(summary['candidate_brier'], 0.25)
        self.assertAlmostEqual(summary['mean_delta_brier'], -5 / 12)
        self.assertEqual(base.pairs['rows'].tolist(), [1000, 10, 10])

    def test_budget_combination_keeps_pair_alcs(self):
        combined = combine_budgets({0: self.scores(0.0), 1: self.scores(1.0)})
        np.testing.assert_allclose(combined.pairs['brier'], [0.5, 0.5, 0.5])
        self.assertAlmostEqual(compare(combined, combined).summary['base_brier'], 0.5)

    def test_pair_scores_are_saved_for_auditing(self):
        comparison = compare(self.scores(0.0), self.scores(0.5))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            save_run(out, comparison, pd.DataFrame(), 'report', {})
            pairs = pd.read_csv(out / 'pairs.csv')
            self.assertEqual(len(pairs), 3)
            np.testing.assert_allclose(pairs['base_brier'], [0, 1, 1])


class VisibleEvidenceTests(unittest.TestCase):
    def frame(self):
        from paiec.schema import SUBJECT_FIELDS
        rows = []
        for subject, label in [('a', 0), ('b', 1)]:
            for pool in ['reveal', 'evaluate']:
                rows.append({**dict.fromkeys(SUBJECT_FIELDS, ''),
                             'benchmark': 'x', 'subject_id': subject, 'item_id': pool,
                             'trial': 1, 'label': label, 'pool': pool, 'order': 0, 'fold': '0'})
        return pd.DataFrame(rows)

    def test_identical_profiles_pool_observations_without_losing_duplicates(self):
        from paiec.lift_budget import budget_frame, shrunk_rate
        frame = budget_frame(self.frame(), 1)
        self.assertEqual(frame['pair_revealed_n'].tolist(), [2, 2])
        self.assertEqual(frame['pair_revealed_rate'].tolist(), [0.5, 0.5])
        np.testing.assert_allclose(shrunk_rate(pd.DataFrame({'label': [0, 1]}), frame), [0.5, 0.5])

    def test_different_harness_settings_keep_evidence_separate(self):
        from paiec.lift_budget import budget_frame
        frame = self.frame()
        frame.loc[frame['subject_id'] == 'b', 'harness'] = 'other harness'
        out = budget_frame(frame, 1)
        self.assertEqual(out['pair_revealed_rate'].tolist(), [0, 1])

    def test_missing_visible_fields_raise_instead_of_using_private_ids(self):
        from paiec.lift_budget import budget_frame
        with self.assertRaisesRegex(ValueError, 'subject fields'):
            budget_frame(self.frame().drop(columns='harness'), 1)

    def test_null_and_empty_fields_match(self):
        from paiec.lift_budget import budget_frame
        frame = self.frame()
        frame.loc[frame['subject_id'] == 'b', 'provider'] = None
        self.assertEqual(budget_frame(frame, 1)['pair_revealed_n'].tolist(), [2, 2])


class BudgetCoverageTests(unittest.TestCase):
    def test_budget_cannot_exceed_a_private_pairs_pool(self):
        from paiec.lift_budget import budget_frame
        frame = VisibleEvidenceTests().frame()
        # Even identical visible systems need their own allowed acquisition budgets.
        with self.assertRaisesRegex(ValueError, 'acquisition pool'):
            budget_frame(frame, 2)

    def test_full_budget_runs_reject_ineligible_pairs(self):
        from paiec.lift_budget import run_budgets
        frame = VisibleEvidenceTests().frame().drop(columns=['pool', 'order'])
        with self.assertRaisesRegex(ValueError, '80 distinct items'):
            run_budgets(frame, (0,))

    def test_sampling_keeps_acquisition_pool_before_capping_evaluation(self):
        from paiec import lift_sample
        rows = pd.DataFrame({
            'subject_id': 's', 'item_id': [f'i{k}' for k in range(100)],
            'trial': 1, 'label': np.arange(100) % 2,
        })
        out = lift_sample.sample_budget_benchmark(rows, 'b', cap=5)
        reveal = out[out['pool'] == 'reveal']
        evaluate = out[out['pool'] == 'evaluate']
        self.assertEqual(len(reveal), 31)
        self.assertEqual(reveal['order'].tolist(), list(range(31)))
        self.assertEqual(len(evaluate), 5)
        self.assertFalse(set(reveal['item_id']) & set(evaluate['item_id']))
        self.assertEqual(out['n_pair_items'].unique().tolist(), [100])

    def test_small_pairs_are_excluded_before_sampling(self):
        from paiec import lift_sample
        rows = pd.DataFrame({
            'subject_id': ['small'] * 79 + ['eligible'] * 80,
            'item_id': [f'i{k}' for k in range(79)] + [f'i{k}' for k in range(80)],
            'trial': 1, 'label': 1,
        })
        out = lift_sample.sample_budget_benchmark(rows, 'b', cap=5)
        self.assertEqual(out['subject_id'].unique().tolist(), ['eligible'])

    def test_prepared_pools_survive_fold_assignment_and_budget_runs(self):
        from paiec.lift_sample import build_sample, assign_folds
        from paiec.lift_budget import run_budgets
        from paiec.schema import SUBJECT_FIELDS
        rows = pd.DataFrame({
            'subject_id': 's', 'item_id': [f'i{k}' for k in range(100)], 'trial': 1, 'label': 1,
        })
        sample = assign_folds(build_sample([('a', rows), ('b', rows)], 5, budget_mode=True),
                              {'a': '0', 'b': '1'})
        sample = sample.assign(**dict.fromkeys(SUBJECT_FIELDS, ''))
        runs = run_budgets(sample, (0, 31), predictor='empirical_mean')
        self.assertEqual(runs[31].pairs['rows'].tolist(), [5, 5])
        np.testing.assert_allclose(runs[31].pairs['brier'], [0, 0])
        self.assertEqual(runs[0].pairs['brier'].tolist(), [0.25, 0.25])


class LiftCommandTests(unittest.TestCase):
    def test_command_loads_visible_fields_and_saves_pair_weighted_alc(self):
        import json
        import subprocess
        import sys
        from paiec.lift_sample import assign_folds, build_sample
        from paiec.schema import SUBJECT_FIELDS
        root = Path(__file__).resolve().parents[2]
        rows = pd.DataFrame({'subject_id': 's', 'item_id': [f'i{k}' for k in range(80)],
                             'trial': 1, 'label': 1})
        sample = assign_folds(build_sample([('a', rows), ('b', rows)], 5, budget_mode=True),
                              {'a': '0', 'b': '1'})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sample.to_parquet(folder / 'sample.parquet', index=False)
            pd.DataFrame([{'subject_id': 's', **dict.fromkeys(SUBJECT_FIELDS, '')}]).to_parquet(
                folder / 'fields.parquet', index=False)
            (folder / 'groups.json').write_text('{"groups": {}}')
            command = [sys.executable, str(root / 'scripts/lift/run_lift.py'),
                       '--sample', str(folder / 'sample.parquet'),
                       '--subject-fields', str(folder / 'fields.parquet'),
                       '--groups', str(folder / 'groups.json'), '--candidate', 'K',
                       '--out', str(folder / 'out'), '--name', 'integration']
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            record = json.loads(next((folder / 'out').glob('*/run.json')).read_text())
            self.assertEqual(record['summary']['weighting'], 'equal_pairs')
            self.assertEqual(record['summary']['candidate_brier'], 0)
            self.assertEqual(record['budgets'], [0, 1, 3, 7, 15, 31])
            self.assertEqual(len(pd.read_csv(next((folder / 'out').glob('*/pairs.csv')))), 2)


class PreparedPoolValidationTests(unittest.TestCase):
    def test_missing_acquisition_prefix_is_rejected(self):
        from paiec.lift_budget import budget_frame
        frame = VisibleEvidenceTests().frame()
        frame.loc[frame['pool'] == 'reveal', 'order'] = 100
        with self.assertRaisesRegex(ValueError, 'acquisition pool'):
            budget_frame(frame, 1)


class SampleRebuildTests(unittest.TestCase):
    def test_budget_eligibility_does_not_redraw_sealed_roles(self):
        import json
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for number in range(10):
                data = folder / 'data' / f'b{number}' / 'formatted_tables'
                data.mkdir(parents=True)
                n = 40 if number == 0 else 80
                pd.DataFrame([{'response_type': 'binary', 'granularity': 'item', 'release': 'public'}]).to_parquet(data / 'benchmarks.parquet')
                pd.DataFrame([{'subject_id': 's'}]).to_parquet(data / 'subjects.parquet')
                pd.DataFrame({'item_id': [f'i{k}' for k in range(n)], 'content': 'question', 'item_features': ''}).to_parquet(data / 'items.parquet')
                pd.DataFrame({'subject_id': 's', 'item_id': [f'i{k}' for k in range(n)], 'response': 1, 'trial': 1, 'interactors': ''}).to_parquet(data / 'responses.parquet')
            (folder / 'groups.json').write_text('{"groups": {}}')
            command = [sys.executable, str(root / 'scripts/lift/build_lift_sample.py'),
                       '--data', str(folder / 'data'), '--out', str(folder / 'out'),
                       '--groups', str(folder / 'groups.json'), '--folds', '2', '--cap', '5']
            old = subprocess.run([*command, '--legacy-sample'], text=True, capture_output=True)
            self.assertEqual(old.returncode, 0, old.stderr)
            original = json.loads((folder / 'out/sample_manifest.json').read_text())
            new = subprocess.run([*command, '--force'], text=True, capture_output=True)
            self.assertEqual(new.returncode, 0, new.stderr)
            rebuilt = json.loads((folder / 'out/sample_manifest.json').read_text())
            self.assertEqual(rebuilt['fold_of'], original['fold_of'])
            self.assertEqual(rebuilt['sealed'], original['sealed'])
            self.assertNotIn('b0', pd.read_parquet(folder / 'out/sample.parquet')['benchmark'].unique())
