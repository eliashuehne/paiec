"""Canonical workflow routing; importing this module never initializes ZenML."""
import json
from pathlib import Path

from .competition_data import CompetitionDataConfig
from .collected_inputs import CollectedInputsConfig
from .collected_inputs_v2 import CollectedInputsV2Config
from .input_workflow_router import InputWorkflowConfig, GroupedPreparationWorkflow
from .evidence_workflow import EvidenceWorkflowConfig
from .feature_preparation import FeaturePreparationConfig, local_path, write_json
from .release_workflow import ReleaseWorkflowConfig
from .training_workflow import TrainingWorkflowConfig
from .feature_foundation.contracts import FeatureFoundationConfig

WORKFLOWS = {
    "features": (FeatureFoundationConfig, "feature_foundation_v1.json", "paiec.feature_foundation_steps", "feature_foundation_preparation"),
    "inputs": (InputWorkflowConfig, "laya_collected_inputs_v3.json", "paiec.collected_input_steps", "laya_collected_input_preparation"),
    "research": (FeaturePreparationConfig, "feature_preparation_offline_v10.json", "paiec.feature_steps", "laya_feature_preparation"),
    "data": (CompetitionDataConfig, "competition_data_protocol_v2.json", "paiec.competition_steps", "competition_dataset_preparation"),
    "evidence": (EvidenceWorkflowConfig, "competition_evidence_workflow_v1.json", "paiec.evidence_steps", "competition_evidence_selection"),
    "training": (TrainingWorkflowConfig, "laya_training_workflow_v1.json", "paiec.training_steps", "laya_native_execution"),
    "validation": (ReleaseWorkflowConfig, "evidence_validation_workflow_v1.json", "paiec.release_steps", "competition_validation_release"),
    "release": (ReleaseWorkflowConfig, "evidence_release_workflow_v1.json", "paiec.release_steps", "competition_validation_release"),
}


def workflow_plan(root, workflow, config_path, out, *, source_root=None):
    root = Path(root).resolve()
    if source_root is not None and workflow != "features":
        raise ValueError("External --source-root is supported only for features")
    model, default, module, graph = WORKFLOWS[workflow]
    path = (root / (config_path or "configs/pipelines/" + default)).resolve()
    if not path.is_relative_to(root / "configs/pipelines"):
        raise ValueError("Workflow recipes must be under configs/pipelines/")
    config = model.model_validate_json(path.read_text())
    destination = local_path(root, out)
    if isinstance(config, GroupedPreparationWorkflow):
        from .grouped_data import GroupedTrainingConfig
        recipe_path = (root / config.recipe).resolve()
        if not recipe_path.is_relative_to(root / "configs/head_finetuning"):
            raise ValueError("Grouped recipes must be under configs/head_finetuning/")
        recipe = GroupedTrainingConfig.model_validate_json(recipe_path.read_text())
        if local_path(root, recipe.prepared) != destination / "tokens":
            raise ValueError("Use the grouped recipe's prepared directory parent as --out")
        for source_name in [recipe.inputs, recipe.dataset, recipe.qualification]:
            source = local_path(root, source_name)
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                raise ValueError("Keep grouped workflow outputs separate from source artifacts")
        module, graph = "paiec.grouped_steps", "laya_grouped_input_preparation"
    for name in ["prepared", "features", "selection", "saved_assessment"]:
        value = getattr(config, name, None)
        if value:
            source = local_path(root, value)
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                raise ValueError("Keep workflow outputs separate from immutable inputs")
    if workflow == "validation" and config.mode == "release":
        raise ValueError("Use the release workflow to build a package")
    if workflow == "release" and config.mode != "release":
        raise ValueError("Release requires an explicit release recipe")
    if workflow == "training":
        # Recipe inspection binds placement without reading outcome/model data.
        from .execution.jobs import ExecutionProfile, command, JobSpec
        profiles = json.loads((root / "configs/execution/profiles.json").read_text())
        profile = ExecutionProfile.model_validate(profiles[config.profile])
        job = JobSpec.model_validate(dict(config.job.model_dump(), out=str((destination / "job").relative_to(root))))
        if job.task == "train-grouped-enriched":
            from .grouped_data import GroupedTrainingConfig
            recipe = GroupedTrainingConfig.model_validate_json((root / job.grouped_config).read_text())
            for source_name in [recipe.prepared, recipe.qualification, recipe.inputs, recipe.dataset]:
                source = local_path(root, source_name)
                if destination.is_relative_to(source) or source.is_relative_to(destination):
                    raise ValueError("Keep grouped training outputs separate from immutable source artifacts")
        if job.task == "predict-grouped":
            from .grouped_evaluation import GroupedEvaluationConfig
            recipe = GroupedEvaluationConfig.model_validate_json((root / job.grouped_prediction_config).read_text())
            for source_name in [recipe.prepared, recipe.training_output]:
                source = local_path(root, source_name)
                if destination.is_relative_to(source) or source.is_relative_to(destination):
                    raise ValueError("Keep grouped evaluation outputs separate from training/scoring inputs")
        if job.task in {"train-head-pilot", "qualify-head-pilot", "train-encoder", "qualify-encoder"}:
            recipe = job.encoder_config if "encoder" in job.task else job.pilot_config
            prepared = local_path(root, json.loads((root / recipe).read_text())["prepared"])
            if destination.is_relative_to(prepared) or prepared.is_relative_to(destination):
                raise ValueError("Keep native job outputs separate from immutable prepared inputs")
        command(job, profile, "python")
        config = TrainingWorkflowConfig.model_validate(dict(config.model_dump(), job=job.model_dump()))
    return {"workflow": workflow, "root": str(root), "out": str(destination), "config": config.model_dump(),
            **({"source_root": str(Path(source_root or root).resolve()), "population_scope": "full_downloaded_public_release",
                "outcome_columns_read": False, "protected_content_read": False, "model_inference": False} if workflow == "features" else {}),
            "graph_module": module, "graph": graph, "execution_started": False,
            "stages": {"data": ["verify_sources", "prepare_dataset", "publish_dataset"],
                       "features": ["verify_feature_sources", "prepare_feature_tables", "publish_feature_report"],
                       "inputs": ["compile_collected_inputs", "check_native_input_visibility", "publish_laya_input_audit"],
                       "evidence": ["verify_dataset", "compare_fixed_controls", "select_on_training_families", "publish_candidate"],
                       "research": ["verify_inputs", "prepare_profiles_tasks_evidence", "publish_research"],
                       "training": ["verify_native_job", "execute_native_job", "publish_native_job"],
                       "validation": ["verify_candidate", "assess_or_review", "apply_gate", "publish_decision"],
                       "release": ["verify_candidate", "review_saved_assessment", "gate_package_qualify", "publish_decision"]}[workflow],
            "opens_development_outcomes": (workflow == "validation" and config.mode == "assess") or
                                          (workflow == "training" and config.job.task in {"train-head-pilot", "train-encoder", "predict-grouped"}),
            "may_launch_native_job": workflow == "training", "uploads_submission": False}


def execute_workflow(plan, *, allow_development=False):
    # A native trainer can evaluate development internally; launching the training
    # workflow is explicit. A separate assessment needs its own explicit flag.
    model, _, module, graph_name = WORKFLOWS[plan["workflow"]]
    config = model.model_validate(plan["config"])
    if isinstance(config, GroupedPreparationWorkflow):
        module, graph_name = "paiec.grouped_steps", "laya_grouped_input_preparation"
    if isinstance(config, ReleaseWorkflowConfig) and config.mode == "assess" and not allow_development:
        raise ValueError("New assessment requires --allow-development-evaluation; saved-result review does not")
    from .zenml_pipeline import configure_environment, setup_stack
    configure_environment(plan["root"])
    setup = setup_stack(plan["root"])
    if plan["workflow"] == "features":
        # On first repository initialization ZenML's compiler singleton can
        # retain the old default-stack config. Select the verified local stack
        # on that same client before compilation.
        from zenml.client import Client
        Client().activate_stack(setup["stack"])
    out = Path(plan["out"])
    if plan["workflow"] != "features":
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "resolved_config.json", plan["config"])
    import importlib
    graph = getattr(importlib.import_module(module), graph_name)
    graph(root=plan["root"], config=plan["config"], out=plan["out"],
          **({"source_root": plan["source_root"]} if plan["workflow"] == "features" else {}))
