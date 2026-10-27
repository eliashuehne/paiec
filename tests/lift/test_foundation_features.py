"""Foundation views plug into the established runner without changing its rows."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from feature_foundation_fixtures import corpus
from paiec.feature_foundation.build import build_bundle
from paiec.feature_foundation.contracts import FeatureFoundationConfig
from paiec.feature_foundation.lift import export_lift_features
from paiec.lift import load_features, fold_splits
from paiec.subject_features import build_subject_features, load_families
from paiec.subject_fields import subject_fields


class FoundationLiftIntegrationTests(unittest.TestCase):
    def test_export_joins_alongside_existing_subject_features_without_changing_folds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            bundle = build_bundle(FeatureFoundationConfig.model_validate(corpus(source)),
                                  source_root=source, out=root/'bundle')
            sample = pd.DataFrame({'benchmark': ['alpha', 'beta', 'sealed', 'alpha'],
                                   'subject_id': ['s1'] * 4, 'item_id': ['i1'] * 4,
                                   'trial': [0, 0, 0, 0], 'label': [1, 0, 1, 0],
                                   'fold': ['0', '1', 'sealed', '0'],
                                   'pool': ['evaluate', 'evaluate', 'reveal', 'evaluate'],
                                   'order': [-1, -1, 0, -1], 'n_pair_items': [80] * 4})
            sample.to_parquet(root/'sample.parquet', index=False)
            export_lift_features(bundle.directory, root/'sample.parquet', root/'projection')
            fields = subject_fields([pd.read_parquet(source/'payload/data/alpha/formatted_tables/subjects.parquet')])
            families = load_families('configs/subjects/model_families.json')
            build_subject_features(fields, families).to_parquet(root/'subjects.parquet', index=False)
            joined = load_features(sample, [root/'subjects.parquet', root/'projection/features.parquet'])
            pd.testing.assert_frame_equal(joined[sample.columns], sample)
            self.assertEqual(joined.identity.tolist(), ['Example 7B'] * 4)
            self.assertEqual(joined.loc[0, 'recorded_few_shot_count'], 5)
            self.assertEqual(joined.loc[0, 'task_content_characters'], 14)
            for fold, train, test in fold_splits(joined):
                self.assertNotIn('sealed', set(train.fold) | set(test.fold))
                self.assertEqual(set(test.fold), {fold})
