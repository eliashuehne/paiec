import unittest

import numpy as np

from paiec.groups import TagDesign, grouping_keys, parse_tags


class ParseTagTests(unittest.TestCase):
    def test_keeps_short_values_and_drops_hashes_and_long_text(self):
        tags = parse_tags("task=QA3;prompt_sha256=c754fe6ad18922d7bb4de4e4e35461ed;"
                          "note=" + "x" * 80 + ";lang= ja ;empty=")
        self.assertEqual(tags, {"task": "qa3", "lang": "ja"})


class GroupingKeyTests(unittest.TestCase):
    def test_repeating_keys_count_and_identifiers_do_not(self):
        revealed = [{"task": t, "id": f"item{k}"} for k, t in enumerate(["a", "b", "a", "b", "a", "c"])]
        self.assertEqual(grouping_keys(revealed), ["task"])

    def test_too_few_revealed_items_give_no_groups(self):
        self.assertEqual(grouping_keys([{"task": "a"}, {"task": "b"}, {"task": "a"}]), [])

    def test_a_single_value_is_not_a_grouping(self):
        self.assertEqual(grouping_keys([{"task": "a"}] * 6), [])


class DesignTests(unittest.TestCase):
    def test_empty_reveals_keep_only_the_pair_column(self):
        design = TagDesign.from_revealed([])
        self.assertEqual(design.matrix.shape, (0, 1))
        self.assertEqual(design.target_columns({"task": "a"}), ([], 0))

    def test_targets_cannot_add_grouping_keys_or_columns(self):
        design = TagDesign.from_revealed([{"task": task} for task in ["a", "b", "a", "b"]])
        original_columns = design.column_of.copy()
        self.assertEqual(design.target_columns({"task": "new", "language": "en"}), ([], 1))
        self.assertEqual(design.keys, ["task"])
        self.assertEqual(design.column_of, original_columns)

    def test_design_has_pair_column_and_one_column_per_tag_value(self):
        revealed = [{"task": t} for t in ["a", "b", "a", "b"]]
        design = TagDesign.from_revealed(revealed)
        self.assertEqual(design.keys, ["task"])
        self.assertTrue(np.array_equal(design.matrix[:, 0], np.ones(4)))
        self.assertEqual(design.matrix[:, 1:].sum(axis=1).tolist(), [1, 1, 1, 1])
        self.assertEqual(design.target_columns({"task": "a"}), ([design.column_of["task=a"]], 0))
        self.assertEqual(design.target_columns({"task": "z"}), ([], 1))
        self.assertEqual(design.target_columns({}), ([], 0))


if __name__ == "__main__":
    unittest.main()
