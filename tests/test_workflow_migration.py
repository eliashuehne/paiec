import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paiec.data_lifecycle import retire_preview, restore_preview, file_inventory
from paiec.feature_preparation import write_json
from paiec.head_training import file_hash
from paiec.pipeline_workflows import WORKFLOWS, workflow_plan, execute_workflow
from paiec.release_workflow import release_gate, package_validated_candidate
from paiec.training_workflow import plan_native_job, dispatch_native_job


class WorkflowMigrationTests(unittest.TestCase):
    def assessment(self):
        inputs = {"policy_sha256": "p", "prepared_manifest_sha256": "d"}
        report = {"version": "frozen_policy_assessment_v1", "selected_policy_sha256": "p",
            "prepared_manifest_sha256": "d", "parameters_refitted": False, "evaluation_manifest_unchanged": True,
            "calibration_opened": False, "confirmation_opened": False, "supplemental_development_outcomes_used": False,
            "results": {k: {"brier_alc": v} for k, v in {"selected": .264, "constant": .25, "fixed_beta5": .239}.items()},
            "candidate_passes_development_gate": False}
        return inputs, report

    def test_release_rejects_failed_wrong_policy_and_sealed_results(self):
        inputs, assessment = self.assessment()
        self.assertFalse(release_gate(inputs, assessment)["candidate_passed"])
        for changed in [dict(assessment, selected_policy_sha256="other"), dict(assessment, calibration_opened=True),
                        dict(assessment, parameters_refitted=True), dict(assessment, candidate_passes_development_gate=True)]:
            with self.assertRaises(ValueError):
                release_gate(inputs, changed)
        assessment["results"]["selected"]["brier_alc"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            release_gate(inputs, assessment)

    def test_rejected_candidate_cannot_reach_packaging(self):
        inputs, assessment = self.assessment()
        with tempfile.TemporaryDirectory() as temporary:
            inputs.update(root=temporary, config={"mode": "release"})
            with patch("paiec.release_workflow.verify_release_inputs", return_value=inputs), \
                 patch("paiec.release_workflow.build_evidence_submission") as build:
                result = package_validated_candidate(inputs, assessment, temporary)
            build.assert_not_called()
            self.assertFalse(result["submission_built"])
            self.assertFalse(result["uploaded"])
            self.assertTrue((Path(temporary) / "release_decision.json").is_file())

    def test_passing_scores_still_require_the_starting_kit_and_runtime_checks(self):
        inputs, assessment = self.assessment()
        assessment["results"]["selected"]["brier_alc"] = .2
        assessment["candidate_passes_development_gate"] = True
        self.assertTrue(release_gate(inputs, assessment)["candidate_passed"])
        with tempfile.TemporaryDirectory() as temporary:
            inputs.update(root=temporary, config={"mode": "release", "selection": "runs/selection", "prepared": "runs/data"})
            with patch("paiec.release_workflow.verify_release_inputs", return_value=inputs), \
                 patch("paiec.release_workflow.build_evidence_submission", return_value={"archive_sha256": "zip"}), \
                 patch("paiec.release_workflow.subprocess.run", side_effect=RuntimeError("checker failed")), \
                 patch("paiec.release_workflow.qualify") as runtime:
                with self.assertRaisesRegex(RuntimeError, "checker failed"):
                    package_validated_candidate(inputs, assessment, temporary)
            runtime.assert_not_called()
            self.assertFalse((Path(temporary) / "summary.json").exists())

    def test_native_dispatch_binds_inputs_and_never_resumes_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs/execution").mkdir(parents=True)
            write_json(root / "configs/execution/profiles.json", {"local": {
                "executor": "local", "backend": "torch", "device": "mps", "python": "python"}})
            data = root / "runs/data/input.txt"
            data.parent.mkdir(parents=True)
            data.write_text("synthetic input")
            config = {"profile": "local", "job": {"task": "train-head-pilot", "out": "runs/unused"}}
            with patch("paiec.training_workflow.input_files", return_value=[Path("runs/data/input.txt")]):
                plan = plan_native_job(root, config, root / "runs/workflow")
            self.assertFalse(plan["execution_started"])
            self.assertEqual(plan["job"]["out"], "runs/workflow/job")
            data.write_text("changed")
            with patch("paiec.training_workflow.subprocess.run") as native:
                with self.assertRaisesRegex(ValueError, "changed after planning"):
                    dispatch_native_job(plan, root / "runs/workflow")
                native.assert_not_called()

    def test_new_assessment_cannot_bypass_explicit_launch_flag(self):
        config = {"mode": "assess", "prepared": "runs/data", "selection": "runs/selection", "selection_manifest_sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "allow-development"):
            execute_workflow({"workflow": "validation", "config": config, "opens_development_outcomes": False})

    def test_all_canonical_plans_are_side_effect_free(self):
        root = Path(__file__).resolve().parents[1]
        for name in WORKFLOWS:
            plan = workflow_plan(root, name, None, "runs/plan_only_migration_test/" + name)
            self.assertFalse(plan["execution_started"])
            self.assertFalse(Path(plan["out"]).exists())
            self.assertFalse(plan["uploads_submission"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            workflow_plan(root, "training", None, "runs/head_finetuning/compact_pilot_v3_release/new")

    def test_archive_is_byte_verified_and_restores_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preview = root / "runs/preview"
            preview.mkdir(parents=True)
            (preview / "SUPERSEDED.txt").write_text("Untrained input-development preview")
            (preview / "input.bin").write_bytes(b"synthetic bytes" * 100)
            (root / "runs/release").mkdir()
            before = file_inventory(preview)
            entry = {"path": "runs/preview", "replacement": "runs/release", "lifecycle": "retirable_preview"}
            archive = root / "runs/archive/preview"
            receipt = retire_preview(root, entry, archive)
            self.assertTrue(receipt["archive_verified"])
            self.assertFalse(preview.exists())
            self.assertEqual(file_hash(archive / "preview.tar.gz"), receipt["archive_sha256"])
            restore_preview(root, archive)
            self.assertEqual(file_inventory(preview), before)
            with self.assertRaises(FileExistsError):
                restore_preview(root, archive)
            (root / "configs").mkdir()
            write_json(root / "configs/live.json", {"prepared": "runs/preview"})
            with self.assertRaisesRegex(ValueError, "live recipes/checkpoints"):
                retire_preview(root, entry, root / "runs/archive/blocked")
            self.assertTrue(preview.exists())
