"""Optional MLflow receipt for the pair-matrix experiments.

Tracking is a record, never a prerequisite: a missing client leaves the run
files intact and reports that no receipt was written.
"""
import json
import os
from pathlib import Path

from .feature_preparation import write_json


def record(root, out, *, run_name, tags, params, metrics, artifacts=()):
    """Log one finished run and return its receipt metadata."""
    root, out = Path(root), Path(out)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        receipt = {"tracking": "mlflow_not_installed", "run_name": run_name}
        write_json(out / "tracking_receipt.json", receipt)
        return receipt
    from .tracking import local_tracking_uri
    mlflow.set_tracking_uri(local_tracking_uri(root))
    mlflow.set_experiment("PAIEC Competition Analysis")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({key: str(value) for key, value in tags.items()})
        for key, value in params.items():
            mlflow.log_param(key, value if isinstance(value, (int, float, str)) else json.dumps(value,
                                                                                              sort_keys=True))
        mlflow.log_metrics({key: value for key, value in metrics.items()
                            if isinstance(value, (int, float))})
        mlflow.log_artifacts(str(out), "analysis")
        for name in artifacts:
            path = Path(name)
            if path.is_file():
                mlflow.log_artifact(str(path), "source")
        receipt = {"tracking": "mlflow", "mlflow_run_id": run.info.run_id,
                   "experiment_id": run.info.experiment_id, "run_name": run_name}
        write_json(out / "tracking_receipt.json", receipt)
        return receipt


__all__ = ["record"]
