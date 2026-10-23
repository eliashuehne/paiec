"""Optional ZenML integration. Imported only by the dedicated pipeline command."""
import os
from pathlib import Path

from . import feature_preparation as features
from .tracking import local_tracking_uri

TRACKER = "paiec_mlflow"
STACK = "paiec_local"
EXPERIMENT = "PAIEC Feature Preparation"


def configure_environment(root):
    root = Path(root).resolve()
    # All ZenML configuration/credentials/database state stays project-local.
    os.environ["ZENML_CONFIG_PATH"] = str(root / "runs/zenml/config")
    os.environ["ZENML_REPOSITORY_PATH"] = str(root)
    os.environ["ZENML_ANALYTICS_OPT_IN"] = "false"
    os.environ["ZENML_AUTO_OPEN_DASHBOARD"] = "false"
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")


def setup_stack(root):
    from zenml.client import Client
    from zenml.enums import StackComponentType as Kind
    from zenml.integrations.mlflow import MlflowIntegration

    root = Path(root).resolve()
    if not MlflowIntegration.check_installation():
        raise RuntimeError("Install the dedicated pipeline environment with the pipeline extra")
    if not Client.is_repository_directory(root):
        Client.initialize(root)
    client = Client(root=root)
    uri = local_tracking_uri(root)
    from mlflow import MlflowClient
    mlflow_client = MlflowClient(tracking_uri=uri)
    if mlflow_client.get_experiment_by_name(EXPERIMENT) is None:
        mlflow_client.create_experiment(EXPERIMENT, artifact_location=(root / "runs/mlflow/artifacts/feature_preparation").as_uri())
    try:
        tracker = client.get_stack_component(Kind.EXPERIMENT_TRACKER, TRACKER)
    except KeyError:
        tracker = client.create_stack_component(TRACKER, "mlflow", Kind.EXPERIMENT_TRACKER,
                                                {"tracking_uri": uri, "experiment_name": EXPERIMENT})
    if tracker.configuration.get("tracking_uri") != uri or tracker.configuration.get("experiment_name") != EXPERIMENT:
        raise ValueError("Existing PAIEC tracker configuration differs; inspect before changing it")
    try:
        stack = client.get_stack(STACK)
    except KeyError:
        # Use explicitly local components rather than inheriting an arbitrary active stack.
        stack = client.create_stack(STACK, {Kind.ARTIFACT_STORE: "default", Kind.ORCHESTRATOR: "default",
                                           Kind.EXPERIMENT_TRACKER: tracker.id})
    if stack.components[Kind.EXPERIMENT_TRACKER][0].id != tracker.id:
        raise ValueError("Existing PAIEC stack uses a different tracker")
    client.activate_stack(stack.id)
    result = {"stack": STACK, "tracker": TRACKER, "tracking_uri": uri, "experiment": EXPERIMENT,
              "zenml_version": __import__("zenml").__version__,
              "mlflow_version": __import__("mlflow").__version__}
    state = root / "runs/zenml"
    state.mkdir(parents=True, exist_ok=True)
    features.write_json(state / "setup.json", result)
    return result


def create_pipeline():
    from .feature_steps import laya_feature_preparation
    return laya_feature_preparation
