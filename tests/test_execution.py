import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from paiec.execution.jobs import ExecutionProfile, JobSpec, command, input_files
from paiec.execution.transport import pack_inputs, unpack, import_tracking
from paiec.execution.modal_worker import execute, small_result_file

ROOT = Path(__file__).resolve().parents[1]


class ExecutionTests(unittest.TestCase):
    def test_cached_pilot_command_and_detached_submission_receipt(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock
        from paiec.execution.modal_adapter import launch
        job = JobSpec(task="train-head-pilot", out="runs/new", encoder_cache_gib=14)
        profile = self.profile("modal-l4-pilot")
        self.assertIn("--encoder-cache-gib", command(job,profile,"python"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ["src","scripts","configs"]:
                (root/name).mkdir()
            for name in ["pyproject.toml","requirements-modal-worker.lock"]:
                (root/name).write_text("")
            app = Mock(app_id="app-fixture")
            app.run.return_value = nullcontext()
            call = Mock(object_id="call-fixture")
            call.get.return_value = dict(results_path="results.zip",return_code=0)
            worker = Mock()
            worker.spawn.return_value = call
            volume = Mock()
            volume.batch_upload.return_value = nullcontext(Mock())
            with patch.dict(sys.modules,{"modal":SimpleNamespace(enable_output=nullcontext)}), \
                 patch("paiec.execution.modal_adapter.create_app",return_value=(app,worker,volume)), \
                 patch("paiec.execution.modal_adapter.fetch",side_effect=lambda v,p,d: d.mkdir()), \
                 patch("paiec.execution.modal_adapter.import_tracking"):
                launch(root,job,profile,[])
            app.run.assert_called_once_with(detach=True)
            receipt = json.loads(next((root/"runs/modal_jobs").glob("*.json")).read_text())
            self.assertTrue(receipt["detached"])
            self.assertEqual(receipt["function_call_id"],"call-fixture")

    @unittest.skipUnless(importlib.util.find_spec("mlflow"), "MLflow metric history import")
    def test_cloud_tracking_import_keeps_curve_steps_and_interrupted_status(self):
        from mlflow import MlflowClient
        from paiec.tracking import local_tracking_uri
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            record = dict(run_id="remote", name="pilot", params={}, metrics={"train/loss": .5},
                          metric_history={"train/loss": [dict(value=1., step=1, timestamp=1000),
                                                         dict(value=.5, step=2, timestamp=2000)]}, status="KILLED")
            (results / "remote_tracking.json").write_text(json.dumps([record]))
            mapping = import_tracking(root, results)
            client = MlflowClient(tracking_uri=local_tracking_uri(root))
            history = client.get_metric_history(mapping["remote"], "train/loss")
            self.assertEqual([(point.step, point.value) for point in history], [(1, 1.), (2, .5)])
            self.assertEqual(client.get_run(mapping["remote"]).info.status, "KILLED")
            self.assertEqual(import_tracking(root, results), mapping)

    def test_pilot_cloud_plan_uses_same_trainer_with_enriched_only_and_frozen_inventory(self):
        from paiec.head_pilot import load_config
        from paiec.execution.transport import sha256
        job = JobSpec(task="train-head-pilot", out="runs/new")
        self.assertEqual(job.variant, "enriched")
        profile = self.profile("modal-l4-pilot")
        cmd = command(job, profile, "python")
        self.assertIn("scripts/run_head_pilot.py", cmd)
        self.assertEqual(cmd[cmd.index("--variant") + 1], "enriched")
        self.assertEqual(cmd[cmd.index("--device") + 1], "cuda")
        self.assertIn("--download-parent", cmd)
        self.assertEqual(profile.timeout, 14400)
        files = input_files(ROOT, job)
        self.assertFalse(any(p.name in {"confirmation.jsonl", "calibration.jsonl", "traces.parquet"} for p in files))
        self.assertFalse(any(p.parts[0] == "payload" for p in files))
        config = load_config(ROOT / job.pilot_config)
        manifest = json.loads((ROOT / config.prepared / "manifest.json").read_text())
        # Every pinned preparation artifact must survive an isolated transport.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pack_inputs(ROOT, files, folder / "inputs.zip")
            unpack(folder / "inputs.zip", folder / "work")
            for name, expected in manifest["artifacts"].items():
                self.assertEqual(sha256(folder / "work" / config.prepared / name), expected)
        with self.assertRaises(ValueError):
            input_files(ROOT, job.model_copy(update={"seed": 1}))
        with self.assertRaises(ValueError):
            command(job, self.profile("local-mlx"), "python")
        with self.assertRaises(ValueError):
            JobSpec(task="train-head-pilot", out="runs/new", state_format="context_v1")

    def test_pilot_model_tensors_stay_on_volume_but_scores_are_downloaded(self):
        for name in ["enriched/best/model.safetensors", "enriched/latest/model.safetensors",
                     "enriched/best_decision.safetensors", "checkpoint/model.safetensors"]:
            self.assertFalse(small_result_file(Path(name)))
        for name in ["enriched/history.json", "enriched/best_predictions.csv", "comparison.json", "report.html"]:
            self.assertTrue(small_result_file(Path(name)))

    def profile(self, name):
        return ExecutionProfile.model_validate(
            json.loads((ROOT / "configs/execution/profiles.json").read_text())[name]
        )

    def test_modal_is_location_not_a_new_model_backend(self):
        local = self.profile("local-torch")
        cloud = self.profile("modal-l4")
        job = JobSpec(task="evaluate", out="runs/new", state_format="context_v1")
        a = command(job, local, "python")
        b = command(job, cloud, "python")
        self.assertEqual([x for x in a if x != "mps"], [x for x in b if x != "cuda"])
        self.assertIn("context_v1", b)
        with self.assertRaises(ValueError):
            ExecutionProfile(executor="modal", backend="mlx", device="cuda")
        with self.assertRaises(ValueError):
            command(
                JobSpec(task="train-head-smoke", out="runs/new"),
                self.profile("local-mlx"),
                "python",
            )

    def test_inputs_are_minimal_and_no_traces_or_labels_from_confirmation(self):
        files = input_files(ROOT, JobSpec(task="train-head-smoke", out="runs/new"))
        self.assertEqual(
            [p.name for p in files], ["provenance.parquet", "training.jsonl"]
        )
        evaluation = input_files(
            ROOT, JobSpec(task="evaluate", out="runs/new", suite="smoke")
        )
        self.assertEqual(len(evaluation), 7)
        self.assertNotIn("traces.parquet", [p.name for p in evaluation])
        with self.assertRaises(ValueError):
            JobSpec(task="evaluate", out="runs/../../elsewhere")
        with self.assertRaises(ValueError):
            JobSpec(task="train-head-smoke", out="runs/new", state_format="context_v1")

    def test_bundle_roundtrip_and_traversal_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input").write_text("actual data")
            hashes = pack_inputs(root, [Path("input")], root / "bundle.zip")
            unpack(root / "bundle.zip", root / "unpacked")
            self.assertEqual((root / "unpacked/input").read_text(), "actual data")
            self.assertEqual(
                json.loads((root / "unpacked/input_manifest.json").read_text()), hashes
            )
            with zipfile.ZipFile(root / "bad.zip", "w") as file:
                file.writestr("../escaped", "bad")
            with self.assertRaises(ValueError):
                unpack(root / "bad.zip", root / "bad")
            self.assertFalse((root / "escaped").exists())

    def test_worker_preserves_failed_job_logs_and_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            storage = root / "jobs"
            jobid = "job_" + "a" * 32
            base = storage / jobid
            base.mkdir(parents=True)
            pack_inputs(source, [], base / "inputs.zip")
            job = JobSpec(task="evaluate", out="runs/output")
            with patch(
                "paiec.execution.modal_worker.command",
                return_value=[
                    sys.executable,
                    "-c",
                    'print("failed fixture");raise SystemExit(7)',
                ],
            ):
                result = execute(
                    jobid,
                    job.model_dump(),
                    self.profile("modal-l4").model_dump(),
                    {},
                    storage=storage,
                    source=source,
                )
            self.assertEqual(result["return_code"], 7)
            self.assertIn("failed fixture", (base / "console.log").read_text())
            with zipfile.ZipFile(base / "results.zip") as archive:
                self.assertEqual(
                    json.loads(archive.read("status.json"))["return_code"], 7
                )

    def test_worker_timeout_still_packages_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            jobid = "job_" + "b" * 32
            base = root / jobid
            base.mkdir()
            pack_inputs(source, [], base / "inputs.zip")
            with patch(
                "paiec.execution.modal_worker.subprocess.run",
                side_effect=subprocess.TimeoutExpired("fixture", 1),
            ):
                result = execute(
                    jobid,
                    JobSpec(task="evaluate", out="runs/new").model_dump(),
                    self.profile("modal-l4").model_dump(),
                    {},
                    storage=root,
                    source=source,
                )
            self.assertEqual(result["return_code"], 124)
            self.assertTrue((base / "results.zip").exists())
