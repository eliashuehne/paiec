from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paiec.feature_preparation import write_json


ROOT = Path(__file__).resolve().parents[1]


class GroupedCompletionTests(unittest.TestCase):
    def fixture(self, root, *, completed):
        spec = importlib.util.spec_from_file_location("grouped_completion_stage", ROOT / "scripts/complete_grouped_pilot.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = root
        job_id = "job_" + "a" * 32
        write_json(root / "launch.json", {"out": "runs/training", "pid": 1})
        (root / "configs/head_finetuning").mkdir(parents=True)
        (root / "configs/pipelines").mkdir(parents=True)
        if completed:
            (root / "runs/training/job/output/final").mkdir(parents=True)
            write_json(root / "runs/training/tracking_receipt.json", {})
            write_json(root / "runs/training/job/modal_receipt.json", {"job_id": job_id, "return_code": 0})
            write_json(root / "runs/training/job/output/report.json", {"status": "finished"})
            write_json(root / "runs/training/job/output/final/paiec_training.json", {"fixture": True})
        args = ["complete_grouped_pilot.py", "--training-launch", "launch.json", "--training-job-id", job_id,
                "--out", "runs/completion"]
        return module, args

    def test_completed_training_queues_one_bound_evaluation_and_never_restarts_fit(self):
        from subprocess import CompletedProcess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, args = self.fixture(root, completed=True)
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                self.assertNotIn("--train", command)
                if "scripts/run_pipeline.py" in command:
                    config = json.loads((root / command[command.index("--config") + 1]).read_text())
                    self.assertEqual(config["job"]["task"], "predict-grouped")
                    (root / "runs/workflows/grouped_development_v1/scores").mkdir(parents=True)
                    write_json(root / "runs/workflows/grouped_development_v1/scores/report.json",
                               {"brier_alc": .24, "constant_brier_alc": .25, "mean_within_pair_auroc": .6})
                return CompletedProcess(command, 0)
            with patch("sys.argv", args), patch.object(module.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                module.main()
                self.assertEqual(json.loads((root / "runs/completion/status.json").read_text())["phase"], "finished")
                self.assertEqual(len([c for c in calls if "scripts/run_pipeline.py" in c]), 1)
                with self.assertRaises(FileExistsError):
                    module.main()

    def test_failed_or_canceled_training_cannot_launch_an_evaluation(self):
        from subprocess import CompletedProcess
        for completed in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                module, args = self.fixture(root, completed=completed)
                if completed:
                    (root / "runs/training/STOP_AFTER_TRAINING").touch()
                with patch("sys.argv", args), patch.object(module.subprocess, "run", return_value=CompletedProcess([], 1, stdout="")) as run, \
                        redirect_stdout(io.StringIO()):
                    with self.assertRaises(InterruptedError if completed else RuntimeError):
                        module.main()
                    self.assertFalse(any("scripts/run_pipeline.py" in c.args[0] for c in run.call_args_list))
                    self.assertFalse((root / "configs/head_finetuning/grouped_development_v1.json").exists())

    def test_already_stopped_gpu_app_does_not_block_authorized_evaluation(self):
        from subprocess import CompletedProcess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, args = self.fixture(root, completed=True)
            (root / "runs/modal_jobs").mkdir()
            write_json(root / ("runs/modal_jobs/job_" + "a" * 32 + ".json"), {"app_id": "ap-fixture"})
            def run(command, **kwargs):
                if "stop" in command:
                    return CompletedProcess(command, 1, stdout="", stderr="App is already stopped. (Finished earlier).")
                if "scripts/run_pipeline.py" in command:
                    (root / "runs/workflows/grouped_development_v1/scores").mkdir(parents=True)
                    write_json(root / "runs/workflows/grouped_development_v1/scores/report.json",
                        {"brier_alc": .24, "constant_brier_alc": .25, "mean_within_pair_auroc": .6})
                return CompletedProcess(command, 0)
            with patch("sys.argv", args), patch.object(module.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                module.main()
            self.assertEqual(json.loads((root / "runs/completion/status.json").read_text())["phase"], "finished")
            self.assertEqual(json.loads((root / "runs/completion/gpu_cleanup.json").read_text())["stopped_apps"], ["ap-fixture"])
