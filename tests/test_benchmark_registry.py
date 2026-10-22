import copy
import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from paiec.benchmark_registry import (
    BenchmarkDefinition,
    REGISTRY_DIRECTORY,
    load_benchmark_definition,
    load_registry,
)


def example():
    return load_benchmark_definition("afrimedqa").model_dump()


class RegistryTests(unittest.TestCase):
    def test_every_existing_suite_benchmark_has_a_reviewed_definition(self):
        root = REGISTRY_DIRECTORY.parents[1]
        suites = json.loads((root / "configs/suites.json").read_text())
        names = sorted({benchmark for suite in suites.values() for benchmark in suite["benchmarks"]})
        definitions = load_registry(names)
        self.assertEqual(set(definitions), set(names))
        self.assertTrue(all(value.review_status == "reviewed" for value in definitions.values()))

    def test_no_criteria_control_changes_only_the_criteria_field(self):
        definition = load_benchmark_definition("afrimedqa")
        treated = definition.question_schema()
        control = definition.question_schema(include_criteria=False)
        self.assertEqual(set(treated), {"correct"})
        self.assertEqual(set(control["correct"]), {"type", "instructions"})
        self.assertEqual(control["correct"], {key: value for key, value in treated["correct"].items() if key != "criteria"})
        self.assertIn("criteria", definition.question.model_dump())

    def test_criteria_require_exact_boolean_keys_and_nonempty_distinct_descriptions(self):
        bad_values = [
            {"yes": "yes", "no": "no"},
            {"true": "yes"},
            {"true": "yes", "false": "no", "unknown": "unknown"},
            {"true": "  ", "false": "no"},
            {"true": "same", "false": "same"},
            {"true": 1, "false": "no"},
        ]
        for criteria in bad_values:
            value = example()
            value["question"]["criteria"] = criteria
            with self.subTest(criteria=criteria), self.assertRaises(ValidationError):
                BenchmarkDefinition.model_validate(value)

    def test_unknown_fields_and_unsupported_schema_versions_are_rejected(self):
        for key, val in [("reference_answer", "A"), ("schema_version", 2), ("schema_version", True)]:
            value = example()
            value[key] = val
            with self.subTest(key=key, val=val), self.assertRaises(ValidationError):
                BenchmarkDefinition.model_validate(value)

    def test_blank_instructions_and_missing_review_provenance_are_rejected(self):
        value = example()
        value["question"]["instructions"] = "  "
        with self.assertRaises(ValidationError):
            BenchmarkDefinition.model_validate(value)
        value = example()
        value["evidence"].pop("dataset_revision")
        with self.assertRaises(ValidationError):
            BenchmarkDefinition.model_validate(value)

    def test_unknown_benchmark_does_not_fall_back_to_generic_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "no benchmark definition"):
                load_benchmark_definition("unreviewed_benchmark", directory)

    def test_path_like_benchmark_names_are_rejected_before_lookup(self):
        for name in ["../bbq", "bbq/other", "/tmp/bbq", "bbq.json"]:
            with self.subTest(name=name), self.assertRaises(ValidationError):
                load_benchmark_definition(name)

    def test_unreviewed_file_can_be_inspected_but_not_used_for_prediction(self):
        value = example()
        value["review_status"] = "needs_review"
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "afrimedqa.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "needs outcome review"):
                load_benchmark_definition("afrimedqa", directory)
            definition = load_benchmark_definition("afrimedqa", directory, require_reviewed=False)
            self.assertEqual(definition.review_status, "needs_review")
            with self.assertRaisesRegex(ValueError, "needs outcome review"):
                definition.question_schema()

    def test_wrong_benchmark_in_file_is_rejected(self):
        value = example()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "bbq.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "expected 'bbq'"):
                load_benchmark_definition("bbq", directory)

    def test_fingerprint_is_stable_but_changes_when_definition_changes(self):
        original = load_benchmark_definition("afrimedqa")
        self.assertEqual(original.fingerprint(), BenchmarkDefinition.model_validate(original.model_dump()).fingerprint())
        value = copy.deepcopy(original.model_dump())
        value["question"]["criteria"]["true"] += " Revised wording."
        changed = BenchmarkDefinition.model_validate(value)
        self.assertNotEqual(original.fingerprint(), changed.fingerprint())

    def test_duplicate_registry_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_registry(["bbq", "bbq"])


if __name__ == "__main__":
    unittest.main()
