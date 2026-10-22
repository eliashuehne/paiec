import unittest

from paiec.competition import competition_input, competition_labels
from paiec.schema import SUBJECT_FIELDS, ITEM_FIELDS


class CompetitionBoundaryTests(unittest.TestCase):
    def test_only_official_fields_and_strings_cross_boundary(self):
        subject, item = competition_input(
            {"normalized_name": None, "display_name": "SECRET", "provider": "p"},
            {"item_content": "task", "benchmark_id": "bbq", "reference_answer": "SECRET", "grading_criterion": "SECRET"},
        )
        self.assertEqual(tuple(subject), SUBJECT_FIELDS)
        self.assertEqual(tuple(item), ITEM_FIELDS)
        self.assertTrue(all(isinstance(value, str) for value in [*subject.values(), *item.values()]))
        self.assertEqual(subject["normalized_name"], "")
        self.assertNotIn("SECRET", str([subject, item]))
        self.assertRegex(item["benchmark_id"], r"^benchmark_[0-9a-f]{16}$")

    def test_ids_are_consistent_distinct_and_preserve_hosted_anonymous_ids(self):
        def item(name):
            return competition_input({}, {"benchmark_id": name})[1]
        self.assertEqual(item("bbq"), item("bbq"))
        self.assertNotEqual(item("bbq"), item("ai2d_test"))
        self.assertEqual(item("benchmark_816205")["benchmark_id"], "benchmark_816205")
        self.assertEqual(item("")["benchmark_id"], "")
        self.assertEqual(competition_input({}, item("bbq"))[1], item("bbq"))

    def test_supplied_text_is_preserved_even_if_it_mentions_a_benchmark(self):
        record = {"item_content": "A BBQ question", "item_features": "given details", "interactors": "tool", "benchmark_id": "bbq"}
        projected = competition_input({}, record)[1]
        for field in ("item_content", "item_features", "interactors"):
            self.assertEqual(projected[field], record[field])

    def test_acquired_labels_use_the_same_projection_and_ids(self):
        original = [[{}, {"benchmark_id": "bbq", "response": "SECRET"}], 1]
        labels = competition_labels([original])
        self.assertEqual(labels[0][0], competition_input(*original[0]))
        self.assertEqual(labels[0][1], 1)
        self.assertNotIn("SECRET", str(labels))
        with self.assertRaises(ValueError):
            competition_labels([[original[0], 2]])
