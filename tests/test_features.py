import math
import unittest

import numpy as np

from paiec.features import DIM, N_DENSE, SLOT, item_vector, subject_vector


def as_dict(vector):
    indices, values = vector
    return dict(zip(indices.tolist(), values.tolist()))


def item(content="", features="", interactors=""):
    return {"item_content": content, "item_features": features, "interactors": interactors,
            "benchmark_id": "b"}


def subject(**fields):
    base = {"normalized_name": "", "provider": "", "release_date": "", "access_date": "",
            "harness": "", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
    base.update(fields)
    return base


class ItemVectorTests(unittest.TestCase):
    def test_indices_stay_in_range_and_are_deterministic(self):
        first = item_vector(item("What is 2+2?\nA) 3\nB) 4", "task=math;lang=en"))
        second = item_vector(item("What is 2+2?\nA) 3\nB) 4", "task=math;lang=en"))
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(((first[0] >= 0) & (first[0] < DIM)).all())

    def test_answer_options_set_the_chance_rate(self):
        dense = as_dict(item_vector(item("Pick one:\nA) red\nB) blue\nC) green")))
        self.assertAlmostEqual(dense[SLOT["chance_rate"]], 1 / 3)
        self.assertEqual(dense[SLOT["has_options"]], 1.0)

    def test_empty_content_is_flagged(self):
        dense = as_dict(item_vector(item("")))
        self.assertEqual(dense[SLOT["empty_content"]], 1.0)

    def test_hashed_tokens_live_after_the_dense_slots(self):
        indices, _ = item_vector(item("hello world", "task=qa"))
        self.assertTrue((indices >= N_DENSE).any())


class SubjectVectorTests(unittest.TestCase):
    def test_release_year_and_missing_flags(self):
        dense = as_dict(subject_vector(subject(release_date="2025-03-01")))
        self.assertAlmostEqual(dense[SLOT["release_offset"]], 0.5)
        self.assertNotIn(SLOT["release_missing"], dense)
        self.assertEqual(dense[SLOT["name_missing"]], 1.0)

    def test_parameter_count_is_parsed_from_the_name(self):
        dense = as_dict(subject_vector(subject(normalized_name="Meta Llama 3.1 70B")))
        self.assertAlmostEqual(dense[SLOT["log_params"]], math.log10(70) / 2)

    def test_reasoning_effort_is_ordinal(self):
        dense = as_dict(subject_vector(subject(reasoning_effort="high")))
        self.assertAlmostEqual(dense[SLOT["effort_level"]], 1.0)
        self.assertNotIn(SLOT["effort_missing"], dense)


if __name__ == "__main__":
    unittest.main()
