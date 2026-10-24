import unittest

import pandas as pd

from paiec.competition_panel import augment_to_minimum, CompetitionPanelConfig


class CompetitionPanelTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for scope in ["one", "two"]:
            for i in range(7):
                rows.append({"benchmark": "public", "subject_id": scope, "item_id": str(i),
                             "response_id": scope + str(i), "scope_key": scope,
                             "task_sha256": "task" + str(i), "response": i % 2})
        frame = pd.DataFrame(rows)
        return frame, frame[frame.item_id.isin(["0", "1"])].copy()

    def test_augmentation_preserves_old_rows_and_covers_unique_task_groups(self):
        candidates, retained = self.fixture()
        pointers, coverage = augment_to_minimum(candidates, retained, minimum=5, seed=1, maximum_added=10)
        self.assertEqual(len(pointers), 6)
        old = set(map(tuple, retained[["benchmark", "subject_id", "item_id", "response_id"]].itertuples(index=False, name=None)))
        self.assertFalse(old & set(pointers))
        self.assertTrue(all(r["retained_task_groups_after"] == 5 for r in coverage))

    def test_outcomes_and_table_order_cannot_change_added_pointers(self):
        candidates, retained = self.fixture()
        first, _ = augment_to_minimum(candidates, retained, minimum=5, seed=2, maximum_added=10)
        changed = candidates.iloc[::-1].copy()
        changed["response"] = "IGNORED_SECRET_LABEL"
        second, _ = augment_to_minimum(changed, retained.iloc[::-1], minimum=5, seed=2, maximum_added=10)
        self.assertEqual(first, second)

    def test_duplicate_question_variants_do_not_satisfy_the_minimum(self):
        candidates, retained = self.fixture()
        candidates.loc[candidates.item_id.isin(["2", "3", "4", "5", "6"]), "task_sha256"] = "same"
        with self.assertRaisesRegex(ValueError, "independent task groups"):
            augment_to_minimum(candidates, retained, minimum=5, seed=2, maximum_added=10)

    def test_missing_retained_trial_scope_changes_and_row_bound_fail_closed(self):
        candidates, retained = self.fixture()
        for frame, old, message in [(candidates.iloc[1:], retained, "retained training trial"),
                                    (candidates, retained[retained.scope_key.eq("one")], "scopes")]:
            with self.assertRaisesRegex(ValueError, message):
                augment_to_minimum(frame, old, minimum=5, seed=2, maximum_added=10)
        with self.assertRaisesRegex(ValueError, "row bound"):
            augment_to_minimum(candidates, retained, minimum=7, seed=2, maximum_added=1)
        with self.assertRaises(ValueError):
            CompetitionPanelConfig(minimum_task_groups=79)
