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
from paiec.execution.transport import pack_inputs, unpack
from paiec.execution.modal_worker import execute

ROOT = Path(__file__).resolve().parents[1]


class ExecutionTests(unittest.TestCase):
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
