import tempfile
import unittest
from pathlib import Path

from paiec.schema import (
    ITEM_FIELDS,
    SUBJECT_FIELDS,
    benchmark_dir,
    canonical_identity,
    clean,
    item_record,
    subject_record,
)


class CleanTests(unittest.TestCase):
    def test_missing_values_become_empty_strings(self):
        for value in (None, float("nan"), "nan", " None ", "", "  "):
            self.assertEqual(clean(value), "")

    def test_values_are_stripped(self):
        self.assertEqual(clean("  gpt-5  "), "gpt-5")

    def test_numbers_become_strings(self):
        self.assertEqual(clean(0.5), "0.5")


class RecordTests(unittest.TestCase):
    def test_subject_record_has_exactly_the_competition_fields(self):
        record = subject_record({"normalized_name": "m", "provider": "p"})
        self.assertEqual(tuple(record), SUBJECT_FIELDS)
        self.assertEqual(len(record), 8)

    def test_empty_normalized_name_falls_back_to_display_name(self):
        record = subject_record({"normalized_name": None, "display_name": "Model-X"})
        self.assertEqual(record["normalized_name"], "Model-X")

    def test_present_normalized_name_wins(self):
        record = subject_record(
            {"normalized_name": "model-x", "display_name": "Model-X (prompt A)"}
        )
        self.assertEqual(record["normalized_name"], "model-x")

    def test_item_record_has_exactly_the_competition_fields(self):
        record = item_record(
            {"content": "What is 2+2?", "item_features": None, "interactors": None},
            benchmark="toy",
        )
        self.assertEqual(tuple(record), ITEM_FIELDS)
        self.assertEqual(record["benchmark_id"], "toy")
        self.assertEqual(record["item_content"], "What is 2+2?")
        self.assertEqual(record["item_features"], "")


class CanonicalIdentityTests(unittest.TestCase):
    def test_benchmark_prefix_is_removed(self):
        self.assertEqual(canonical_identity("CMMLU / gemma3_27b_it"), "gemma3_27b_it")

    def test_prompt_annotation_is_removed(self):
        self.assertEqual(
            canonical_identity("UnifiedQA-T5-11B (RACE prompt)"), "unifiedqa-t5-11b"
        )

    def test_provider_path_keeps_its_tokens(self):
        # A provider path has no spaced slash, so nothing is prefixed away; the
        # separator becomes a space, which also matches the "google gemini"
        # spelling used elsewhere.
        self.assertEqual(
            canonical_identity("google/gemini-2.0-flash-lite-001"),
            "google gemini-2.0-flash-lite-001",
        )

    def test_case_and_spacing_are_normalised(self):
        self.assertEqual(canonical_identity("  OpenAI GPT-4  "), "openai gpt-4")

    def test_empty_names_stay_empty(self):
        self.assertEqual(canonical_identity(None), "")
        self.assertEqual(canonical_identity("nan"), "")


class BenchmarkDirTests(unittest.TestCase):
    def test_accepts_plain_and_manifest_spellings(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bbq" / "formatted_tables"
            target.mkdir(parents=True)
            (target / "items.parquet").touch()
            self.assertEqual(benchmark_dir(tmp, "bbq"), target)
            self.assertEqual(benchmark_dir(tmp, "bbq/formatted_tables"), target)

    def test_missing_benchmark_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                benchmark_dir(tmp, "absent")


if __name__ == "__main__":
    unittest.main()
