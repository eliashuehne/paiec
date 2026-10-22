"""Local MLflow run records around the competition scoring harness.

Clients write directly to SQLite, so recording runs does not require a running
web server. The dashboard reads the same database and local artifact files.
"""

from contextlib import contextmanager
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings
import zipfile

import numpy as np

from .metrics import alc_weights

# Context tag MLflow uses to group runs that consumed the same data.
EVALUATION_SPLIT = "evaluation_split"


def local_tracking_uri(root: Path) -> str:
    return "sqlite:///" + str((root / "runs/mlflow/mlflow.db").resolve())


def git_context(root):
    def read(*args):
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    return {"commit": read("rev-parse", "HEAD"), "branch": read("branch", "--show-current"),
            "dirty": bool(read("status", "--porcelain"))}


def package_versions():
    return {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata.get("Name")}


def evaluation_split(sample):
    """Scored rows of a sample manifest, dropping acquisition positions."""
    return sample[sample["role"].eq("evaluation")].reset_index(drop=True)


def log_source(client, run_id, root):
    paths = sorted(set(
        [*root.glob("src/paiec/*.py"), *root.glob("scripts/*.py"), *root.glob("configs/**/*.json"),
         *root.glob("tests/test_*.py"), *root.glob("requirements*.lock"), root / "pyproject.toml"]
    ))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for source in paths:
                if source.is_file():
                    archive.write(source, str(source.relative_to(root)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        client.log_artifact(run_id, str(path), "reproducibility")
        client.set_tag(run_id, "source_snapshot_sha256", digest)


class TrackedRun:
    def __init__(self, client, run_id):
        self.client = client
        self.run_id = run_id

    def params(self, values):
        for key, value in values.items():
            self.client.log_param(self.run_id, key, value)

    def json(self, value, filename):
        self.client.log_dict(self.run_id, value, filename)

    def frame(self, frame, filename):
        self.client.log_text(self.run_id, frame.to_csv(index=False), filename)

    def dataset(self, frame, name, context=EVALUATION_SPLIT, targets=None):
        """Attach one logged dataset so the UI can group runs by shared sample.

        The dataset is the scored-input manifest, not the 23 GB payload: it is
        small, and its digest changes exactly when the sample changes. Failures
        are non-fatal, since the same manifest is also kept as an artifact.
        """
        try:
            from mlflow.data import from_pandas
            from mlflow.entities import DatasetInput, InputTag
        except ImportError:
            return None
        try:
            with warnings.catch_warnings():
                # from_pandas warns about integer columns because it assumes the
                # frame will be fed to a model. This manifest never is.
                warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")
                dataset = from_pandas(frame, name=name, targets=targets)
            self.client.log_inputs(self.run_id, datasets=[DatasetInput(
                dataset=dataset._to_mlflow_entity(),
                tags=[InputTag(key="mlflow.data.context", value=context)],
            )])
        except Exception as exc:  # dataset logging is a convenience, not evidence
            print(f"  dataset input not logged: {type(exc).__name__}: {exc}", flush=True)
            return None
        return dataset.digest

    def result(self, result, seconds, predictions):
        ece_by_budget = result.frame.groupby("budget")["ece"].mean()
        metrics = {
            "brier_alc": result.brier_alc,
            **{f"brier_{budget}": float(value) for budget, value in result.per_budget.items()},
            "ece_alc": float(np.dot(alc_weights(ece_by_budget.index), ece_by_budget.values)),
            "ece_budget_mean": result.ece,
            "n_pairs": len(result.pair_alc),
            "completed_eval_inputs": int(result.frame[result.frame.budget.eq(0)]["n_items"].sum()),
            "evaluation_seconds": seconds,
        }
        for key, value in metrics.items():
            self.client.log_metric(self.run_id, key, float(value))
        for benchmark, value in result.by_benchmark().items():
            self.client.log_metric(self.run_id, f"benchmark/{benchmark}/brier_alc", float(value))
        self.frame(result.frame, "scores/per_pair_budget.csv")
        self.frame(result.pair_alc.rename("brier_alc").reset_index(), "scores/per_pair.csv")
        self.frame(result.by_benchmark().rename("brier_alc").reset_index(), "scores/per_benchmark.csv")
        self.frame(predictions, "predictions.csv")
        self.json(metrics, "metrics.json")


class ExperimentTracker:
    def __init__(self, root: Path, *, experiment="PAIEC", enabled=True, tracking_uri=None):
        self.root = Path(root).resolve()
        self.client = None
        self.experiment_id = None
        self.uri = tracking_uri or local_tracking_uri(self.root)
        if not enabled:
            return
        if util.find_spec("mlflow") is None:
            print("MLflow tracking unavailable; install .[tracking] or use --no-tracking.", flush=True)
            return
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "true")
        from mlflow import MlflowClient
        (self.root / "runs/mlflow").mkdir(parents=True, exist_ok=True)
        self.client = MlflowClient(tracking_uri=self.uri)
        existing = self.client.get_experiment_by_name(experiment)
        if existing is None:
            directory = self.root / "runs/mlflow/artifacts" / hashlib.sha256(experiment.encode()).hexdigest()[:16]
            directory.mkdir(parents=True, exist_ok=True)
            self.experiment_id = self.client.create_experiment(experiment, artifact_location=directory.as_uri())
        else:
            self.experiment_id = existing.experiment_id

    @contextmanager
    def run(self, name, config, params, tags):
        if self.client is None:
            yield None
            return
        context = git_context(self.root)
        record = self.client.create_run(self.experiment_id, tags={
            "mlflow.runName": name, "mlflow.source.git.commit": context["commit"],
            "git_branch": context["branch"], "git_dirty": str(context["dirty"]).lower(),
            "evaluation_scope": "exploratory_public_data", **tags,
        })
        run = TrackedRun(self.client, record.info.run_id)
        try:
            run.params(params)
            run.json(config, "config.json")
            run.json({"python": sys.version, "packages": package_versions(), "git": context}, "reproducibility/environment.json")
            log_source(self.client, run.run_id, self.root)
            print(f"  MLflow run: {run.run_id}", flush=True)
            yield run
        except BaseException as exc:
            # Retain failed/interrupted runs without swallowing the original failure.
            try:
                self.client.log_text(run.run_id, f"{type(exc).__name__}: {exc}", "error.txt")
                self.client.set_terminated(run.run_id, "KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
            except Exception:
                pass
            raise
        else:
            self.client.set_terminated(run.run_id, "FINISHED")
