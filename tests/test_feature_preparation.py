"""Synthetic tests for feature provenance, role isolation and outcome exclusion."""
import json
from pathlib import Path
import tempfile
import unittest

from paiec.competition import competition_input
from paiec.feature_preparation import (FeaturePreparationConfig, build_tasks,
    local_path, source_inputs, stable_hash, task_observables, write_jsonl)
from paiec.laya_predictor import load_questions
from paiec.schema import SUBJECT_FIELDS, ITEM_FIELDS


class FeaturePreparationTests(unittest.TestCase):
    def test_preparation_verification_rejects_changed_artifacts_and_path_escape(self):
        from paiec.feature_preparation import verify_inputs, write_json
        from paiec.head_training import file_hash
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, _, _ = self.fixture(root)
            prepared = Path(inputs["prepared"])
            (root / "split.json").write_text("{}")
            (prepared / "profile_bundle").mkdir()
            (prepared / "profile_bundle/policy.json").write_text("{}")
            manifest = {"data_config": {"split": "split.json"},
                        "split_sha256": file_hash(root / "split.json"),
                        "dataset_revision": "fixture", "renderer_sources": {},
                        "question": load_questions(),
                        "artifacts": {name: file_hash(prepared / name) for name in
                                      ["source/training.jsonl", "profile_bundle/policy.json"]}}
            write_json(prepared / "manifest.json", manifest)
            config = FeaturePreparationConfig(prepared="runs/prepared", roles=["training"])
            verify_inputs(root, config)
            manifest["artifacts"]["../../split.json"] = file_hash(root / "split.json")
            write_json(prepared / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_inputs(root, config)
            del manifest["artifacts"]["../../split.json"]
            write_json(prepared / "manifest.json", manifest)
            (prepared / "source/training.jsonl").write_text("modified")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_inputs(root, config)

    def fixture(self, root, roles=None):
        roles = roles or {"training": [("A task with 12 apples.", True)],
                          "monitor": [("Another question.", False)]}
        prepared = root / "runs/prepared"
        (prepared / "source").mkdir(parents=True)
        subject = {key: "" for key in SUBJECT_FIELDS}
        subject["normalized_name"] = "Test system"
        for role, examples in roles.items():
            rows = []
            for text, label in examples:
                item = {key: "" for key in ITEM_FIELDS}
                item.update(item_content=text, benchmark_id="synthetic")
                s, item = competition_input(subject, item)
                rows.append({"state": json.dumps({"target_system": s, "target_item": item}),
                             "questions": load_questions(), "expected": {"correct": label}})
            write_jsonl(prepared / f"source/{role}.jsonl", rows)
        out = root / "runs/output"
        out.mkdir()
        systems = out / "systems.jsonl"
        write_jsonl(systems, [{"system_key": stable_hash(subject)}])
        return {"prepared": str(prepared), "config": {"roles": list(roles)}}, {"path": str(systems)}, out

    def test_feature_outputs_do_not_depend_on_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, systems, out = self.fixture(root)
            before = build_tasks(inputs, systems, out)
            saved = {name: (out / name).read_bytes() for name in ["tasks.jsonl", "examples.jsonl"]}
            for role in inputs["config"]["roles"]:
                path = Path(inputs["prepared"]) / f"source/{role}.jsonl"
                row = json.loads(path.read_text())
                row["expected"] = {"correct": "SECRET_OUTCOME_MARKER"}
                write_jsonl(path, [row])
            after = build_tasks(inputs, systems, out)
            self.assertEqual(before, after)
            for name, value in saved.items():
                self.assertEqual(value, (out / name).read_bytes())
                self.assertNotIn(b"SECRET_OUTCOME_MARKER", value)
                self.assertNotIn(b'"expected"', value)

    def test_same_question_cannot_cross_roles_even_with_different_wrapping_space(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs, systems, out = self.fixture(Path(temporary), {
                "training": [("What is 2 + 2?", True)],
                "monitor": [("  What is 2 + 2?", False)]})
            # The source fixture itself must first conform to the input boundary.
            path = Path(inputs["prepared"]) / "source/monitor.jsonl"
            row = json.loads(path.read_text())
            state = json.loads(row["state"])
            state["target_item"]["item_content"] = "What is 2 + 2?"
            row["state"] = json.dumps(state)
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "spans preparation roles"):
                build_tasks(inputs, systems, out)

    def test_hidden_fields_are_rejected_at_source_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs, _, _ = self.fixture(Path(temporary))
            path = Path(inputs["prepared"]) / "source/training.jsonl"
            row = json.loads(path.read_text())
            state = json.loads(row["state"])
            state["target_system"]["display_name"] = "Private name"
            row["state"] = json.dumps(state)
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "competition boundary"):
                list(source_inputs(inputs["prepared"], ["training"]))

    def test_historical_missing_fields_remain_empty_without_identity_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs, _, _ = self.fixture(Path(temporary))
            path = Path(inputs["prepared"]) / "source/training.jsonl"
            row = json.loads(path.read_text())
            row["state"] = json.dumps({"target_system": {"harness": "Test"},
                                       "target_item": {"item_content": "A question"}})
            write_jsonl(path, [row])
            _, _, subject, item = next(source_inputs(inputs["prepared"], ["training"]))
            self.assertEqual(subject["normalized_name"], "")
            self.assertEqual(item["benchmark_id"], "")
            self.assertEqual(set(subject), set(SUBJECT_FIELDS))

    def test_only_unsealed_roles_and_local_outputs_are_accepted(self):
        for role in ["calibration", "confirmation"]:
            with self.assertRaises(ValueError):
                FeaturePreparationConfig(roles=["training", role])
        for value in ["/tmp/output", "runs/../payload", "payload", "runs"]:
            with self.assertRaises(ValueError):
                FeaturePreparationConfig(prepared=value)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runs").mkdir()
            (root / "runs/escape").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                local_path(root, "runs/escape/result")

    def test_task_observations_do_not_assert_language_or_difficulty(self):
        observed = task_observables({"item_content": "中文 12 apples", "interactors": "", "item_features": ""})
        self.assertTrue(observed["has_cjk_characters"])
        self.assertEqual(observed["numeric_spans"], 1)
        self.assertNotIn("difficulty", observed)
        self.assertNotIn("language", observed)


if __name__ == "__main__":
    unittest.main()
