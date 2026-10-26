import unittest

import numpy as np
import pandas as pd

from paiec.subject_fields import COLUMNS, subject_fields


def subjects(**overrides):
    row = {"subject_id": "s1", "normalized_name": "gpt-x", "provider": "acme", "display_name": "GPT X"}
    return pd.DataFrame([{**row, **overrides}])


class SubjectFieldTests(unittest.TestCase):
    def test_one_row_per_subject_with_visible_fields_only(self):
        fields = subject_fields([subjects(), subjects(), subjects(subject_id="s2")])
        self.assertEqual(list(fields.columns), COLUMNS)
        self.assertEqual(fields["subject_id"].tolist(), ["s1", "s2"])
        self.assertNotIn("display_name", fields.columns)

    def test_empty_values_become_missing_without_display_name_fallback(self):
        fields = subject_fields([subjects(normalized_name=" ", provider=np.nan)])
        self.assertTrue(fields["normalized_name"].isna().all())
        self.assertTrue(fields["provider"].isna().all())

    def test_conflicting_fields_raise(self):
        with self.assertRaises(ValueError):
            subject_fields([subjects(), subjects(provider="other")])

    def test_no_tables_gives_empty_table(self):
        self.assertEqual(list(subject_fields([]).columns), COLUMNS)


if __name__ == "__main__":
    unittest.main()
