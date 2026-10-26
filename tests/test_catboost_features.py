import math
import unittest

from paiec.catboost_features import (CATALOG, DEFAULT_GROUPS, IDENTITY_GROUP, PROFILE_COLUMNS, PROFILE_GROUP,
                                     categorical_names, feature_names, feature_record, select,
                                     system_feature_record, task_feature_record)
from paiec.competition_fields import ITEM_FIELDS, SUBJECT_FIELDS


def fields(name="Model 8B", provider="Meta", harness="Lighteval", text="Question 猫?", extra_subject="a=1;b=2"):
    subject = {key: "" for key in SUBJECT_FIELDS}
    subject.update(normalized_name=name, provider=provider, harness=harness,
                   release_date="2024-07-23", subject_features_extra=extra_subject)
    item = {key: "" for key in ITEM_FIELDS}
    item.update(item_content=text, benchmark_id="benchmark_anonymous")
    return subject, item


class CatboostFeatureTests(unittest.TestCase):
    def test_catalog_is_the_declared_contract(self):
        # Profile columns are catalogued but need an external catalog at build time,
        # so they are absent from a record built from the supplied fields alone.
        self.assertEqual(set(feature_names((*DEFAULT_GROUPS, IDENTITY_GROUP, PROFILE_GROUP))), set(CATALOG))
        self.assertNotIn(IDENTITY_GROUP, DEFAULT_GROUPS)
        self.assertNotIn(PROFILE_GROUP, DEFAULT_GROUPS)
        self.assertEqual(len(feature_names()), len(set(feature_names())))
        record = feature_record(*fields())
        self.assertEqual(set(record), set(CATALOG) - set(PROFILE_COLUMNS))
        for name in feature_names():
            value = record[name]
            if CATALOG[name][1] == "categorical":
                self.assertIsInstance(value, str)
            else:
                self.assertIsInstance(value, float)
                self.assertTrue(math.isfinite(value), name)
        self.assertEqual(categorical_names(), ["provider", "harness", "size_class", "release_year_class"])

    def test_anonymous_benchmark_id_is_never_a_feature(self):
        subject, item = fields()
        self.assertNotIn("benchmark_id", CATALOG)
        self.assertEqual(feature_record(subject, item),
                         feature_record(subject, dict(item, benchmark_id="different_anonymous_id")))

    def test_foreign_or_nonstring_fields_are_rejected(self):
        subject, item = fields()
        with self.assertRaises(ValueError):
            feature_record(dict(subject, display_name="PRIVATE"), item)
        with self.assertRaises(ValueError):
            feature_record(subject, dict(item, reference_answer="42"))
        with self.assertRaises(ValueError):
            feature_record(dict(subject, provider=None), item)
        trimmed = {key: value for key, value in subject.items() if key not in {"access_date", "reasoning_effort"}}
        record = system_feature_record(trimmed)
        self.assertEqual(record["missing_access_date"], 1.0)
        self.assertEqual(record["missing_reasoning_effort"], 1.0)

    def test_nominal_capacity_does_not_invent_moe_parameters(self):
        moe = system_feature_record(dict(fields("Mixtral 8x7B")[0]))
        self.assertEqual(moe["size_class"], "moe_label")
        self.assertEqual(moe["nominal_parameters_known"], 0.0)
        self.assertEqual(moe["nominal_parameters_is_moe"], 1.0)
        plain = system_feature_record(fields("Llama 3.1 8B")[0])
        self.assertEqual(plain["nominal_parameters_known"], 1.0)
        self.assertGreater(plain["nominal_parameters_log"], 0)
        ambiguous = system_feature_record(fields("Model 8B 70B")[0])
        self.assertEqual(ambiguous["size_class"], "unknown")
        huge = feature_record(*fields("Model " + "9" * 500 + "B"))
        self.assertTrue(all(math.isfinite(value) for name, value in huge.items() if CATALOG[name][1] == "numeric"))

    def test_long_and_unicode_text_is_measured_without_clipping(self):
        short = feature_record(*fields(text="short question?"))
        long = feature_record(*fields(text="word " * 40000 + "尾巴"))
        self.assertGreater(long["task_characters_log"], short["task_characters_log"])
        self.assertGreater(long["task_words_log"], short["task_words_log"])
        self.assertEqual(long["task_has_cjk"], 1.0)
        for name in feature_names(("task_structure",)):
            self.assertTrue(math.isfinite(long[name]), name)
            if "fraction" in name:
                self.assertTrue(0 <= long[name] <= 1, name)

    def test_structure_flags_and_missing_categories(self):
        text = "Pick one:\nA. first\nB. second\n\n- bullet\n1. step\n```code```\nIs 2+2=4?"
        record = feature_record(dict(fields()[0], provider="", harness=""), fields(text=text)[1])
        self.assertEqual(record["provider"], "__missing__")
        self.assertEqual(record["missing_provider"], 1.0)
        self.assertEqual(record["missing_harness"], 1.0)
        self.assertEqual(record["task_has_code_fence"], 1.0)
        self.assertEqual(record["task_option_marker_count"], 2.0)
        self.assertEqual(record["task_bullet_count"], 1.0)
        self.assertEqual(record["task_numbered_item_count"], 1.0)
        self.assertEqual(record["task_ends_with_question"], 1.0)
        self.assertEqual(record["item_features_present"], 0.0)

    def test_select_projects_declared_groups_only(self):
        record = feature_record(*fields())
        project = select(record, ("task_structure",))
        self.assertEqual(set(project), set(feature_names(("task_structure",))))
        self.assertNotIn("provider", project)
        with self.assertRaises(ValueError):
            select(record, ("not_a_group",))
        with self.assertRaises(ValueError):
            task_feature_record({"item_content": "x", "reference_answer": "42"})


if __name__ == "__main__":
    unittest.main()
