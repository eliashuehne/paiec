import unittest

import pandas as pd

from paiec.head_data import assign_roles, near_duplicates, pair_balanced_sample, role_counts, task_key


class HeadDataTests(unittest.TestCase):
    def test_role_counts_prioritize_multiple_independent_holdouts(self):
        self.assertEqual(list(role_counts(20).values()), [12, 4, 2, 2])
        self.assertEqual(list(role_counts(13).values()), [7, 2, 2, 2])
        with self.assertRaises(ValueError):
            role_counts(6)

    def test_family_randomization_is_reproducible_and_order_independent(self):
        families = {f"f{i}": {"domain": str(i % 3), "has_english": True} for i in range(13)}
        a, _ = assign_roles(families, 219)
        b, _ = assign_roles(dict(reversed(list(families.items()))), 219)
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(families))
        self.assertEqual({families[n]["domain"] for n, r in a.items() if r == "training"}, {"0", "1", "2"})

    def test_every_heldout_role_has_a_short_task_family(self):
        families = {f"f{i}": {"domain": str(i % 3), "has_english": True,
                              "has_short": i < 7} for i in range(13)}
        roles, _ = assign_roles(families, 219)
        for role in ("development", "calibration", "confirmation"):
            self.assertTrue(any(families[f]["has_short"] for f, r in roles.items() if r == role))

    def test_sampling_ignores_outcomes_and_weights_pairs_equally(self):
        rows = [{"benchmark": "b", "subject_id": s, "item_id": f"i{i}",
                 "response_id": f"{s}-{i}", "response": i % 2}
                for s, n in (("a", 2), ("b", 20)) for i in range(n)]
        frame = pd.DataFrame(rows)
        sample = pair_balanced_sample(frame, 12, 219)
        changed = frame.copy()
        changed.response = 1 - changed.response
        again = pair_balanced_sample(changed, 12, 219)
        self.assertEqual(sample.response_id.tolist(), again.response_id.tolist())
        self.assertEqual(sample.response_id.nunique(), 12)
        self.assertEqual(sample.groupby('subject_id').sample_weight.sum().tolist(), [6., 6.])

    def test_context_is_part_of_task_identity_and_chat_uses_last_user(self):
        a = task_key('{"question":"Who?", "context":"Alice", "options":["A","B"]}')
        b = task_key('{"question":"Who?", "context":"Bob", "options":["A","B"]}')
        self.assertNotEqual(a, b)
        self.assertEqual(task_key('[{"role":"system","content":"format"},{"role":"user","content":"WHO?"}]'), 'who')

    def test_near_screen_verifies_cross_role_matches_only(self):
        text = ' '.join(f'word{i}' for i in range(50))
        self.assertEqual(near_duplicates([text, text], ['training', 'development']), [(0, 1, 1.)])
        self.assertEqual(near_duplicates([text, text], ['training', 'training']), [])
        other = ' '.join(f'other{i}' for i in range(50))
        self.assertEqual(near_duplicates([text, other], ['training', 'development']), [])
