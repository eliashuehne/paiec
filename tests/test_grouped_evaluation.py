import json
from pathlib import Path
import tempfile
import unittest

from paiec.execution.jobs import JobSpec, ExecutionProfile, command, input_files
from paiec.feature_preparation import write_json, write_jsonl
from paiec.grouped_evaluation import GroupedEvaluationConfig, verify_requests, score_development
from paiec.head_training import file_hash


class GroupedEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        prepared = root / "runs/development"
        prepared.mkdir(parents=True)
        requests, records = [], []
        for pair in range(8):
            for item in range(120):
                requests.append({"subject": {"normalized_name": str(pair)}, "item": {"item_content": str(item)}})
                records.append({"role": "development", "pool": "evaluation", "exported": True,
                    "benchmark": "b" + str(pair // 4), "subject_id": str(pair), "item_id": str(item),
                    "label": item % 2, "row": len(records)})
        for i in range(3):
            requests.append(dict(requests[i]))
            records.append(dict(records[i], row=len(records), label=1 - records[i]["label"]))
        write_jsonl(prepared / "requests.jsonl", requests)
        write_jsonl(prepared / "scoring.jsonl", records)
        write_json(prepared / "inference_manifest.json", {"version": "grouped_development_requests_v1", "responses": 963,
            "contains_labels": False, "artifacts": {"requests.jsonl": file_hash(prepared / "requests.jsonl")}})
        write_json(prepared / "scoring_manifest.json", {"artifacts": {n: file_hash(prepared / n)
            for n in ["requests.jsonl", "scoring.jsonl", "inference_manifest.json"]}})
        config = GroupedEvaluationConfig(prepared="runs/development", training_job_id="job_" + "a" * 32,
            checkpoint_manifest_sha256="b" * 64)
        predictions = root / "runs/predictions"
        predictions.mkdir()
        write_jsonl(predictions / "predictions.jsonl", [{"row": i, "probability": .5} for i in range(963)])
        write_json(predictions / "report.json", {"status": "finished", "labels_available_to_inference": False,
            "checkpoint_manifest_sha256": config.checkpoint_manifest_sha256,
            "request_manifest_sha256": file_hash(prepared / "inference_manifest.json")})
        return prepared, config, predictions

    def test_trial_preserving_pair_weighted_scoring_and_calibration_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, config, predictions = self.fixture(root)
            result = score_development(prepared, predictions, root / "scores", checkpoint_manifest_sha256=config.checkpoint_manifest_sha256)
            self.assertAlmostEqual(result["brier_alc"], .25, places=12)
            self.assertEqual(result["n_responses"], 963)
            self.assertEqual(result["n_eval_inputs"], 960)
            self.assertEqual(result["n_pairs"], 8)
            self.assertEqual(result["mean_within_pair_auroc"], .5)
            self.assertFalse(result["calibration_fitted"])
            self.assertFalse(result["confirmation_opened"])

    def test_wrong_checkpoint_or_inconsistent_repeated_item_predictions_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, config, predictions = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "declared checkpoint"):
                score_development(prepared, predictions, root / "scores", checkpoint_manifest_sha256="different")
            values = [{"row": i, "probability": .5 if i < 960 else .7} for i in range(963)]
            write_jsonl(predictions / "predictions.jsonl", values)
            with self.assertRaisesRegex(ValueError, "inconsistent predictions"):
                score_development(prepared, predictions, root / "scores", checkpoint_manifest_sha256=config.checkpoint_manifest_sha256)

    def test_upload_contains_only_outcome_free_requests_and_bound_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, config, predictions = self.fixture(root)
            parent = root / config.training_output
            (parent / "output/final").mkdir(parents=True)
            (parent / "output/final/paiec_training.json").write_text("synthetic manifest")
            config = config.model_copy(update={"checkpoint_manifest_sha256": file_hash(parent / "output/final/paiec_training.json")})
            write_json(parent / "modal_receipt.json", {"job_id": config.training_job_id, "return_code": 0})
            recipe = root / "configs/recipe.json"
            recipe.parent.mkdir()
            write_json(recipe, config.model_dump())
            job = JobSpec(task="predict-grouped", out="runs/new", grouped_prediction_config="configs/recipe.json")
            files = input_files(root, job)
            self.assertEqual(set(files), {Path("configs/recipe.json"), Path("runs/development/requests.jsonl"), Path("runs/development/inference_manifest.json")})
            args = command(job, ExecutionProfile(executor="modal", backend="torch", device="cuda"), "python")
            self.assertNotIn("--train", args)
            self.assertEqual(args[args.index("--stage") + 1], "predict")
            with self.assertRaises(ValueError):
                JobSpec(task="predict-grouped", out="runs/new", evidence_mode="beta_mean")

    def test_private_fields_and_labels_rejected_before_remote_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, config, _ = self.fixture(root)
            write_jsonl(prepared / "requests.jsonl", [{"subject": {}, "item": {}, "label": 1}])
            write_json(prepared / "inference_manifest.json", {"version": "grouped_development_requests_v1", "responses": 963,
                "contains_labels": False, "artifacts": {"requests.jsonl": file_hash(prepared / "requests.jsonl")}})
            with self.assertRaisesRegex(ValueError, "only published"):
                verify_requests(root, config)

    def test_remote_training_job_paths_cannot_escape_the_declared_volume(self):
        for changed in [{"training_job_id": "../other"}, {"training_output": "../elsewhere"}, {"prepared": "/tmp/data"}]:
            with self.assertRaises(ValueError):
                GroupedEvaluationConfig.model_validate({"training_job_id": "job_" + "a" * 32,
                    "checkpoint_manifest_sha256": "b" * 64, **changed})
