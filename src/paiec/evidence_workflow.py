"""Training-only evidence workflow stages, independent of ZenML and MLflow."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from .competition_analysis import analyze_training_evidence, verify_competition_bundle
from .evidence_policy_analysis import analyze_evidence_policies
from .feature_preparation import local_path, relative_run_path, write_json
from .head_training import file_hash


class EvidenceWorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["competition_evidence_workflow_v1"] = "competition_evidence_workflow_v1"
    source_policy: Literal["provided_dataset_only"] = "provided_dataset_only"
    prepared: str

    @field_validator("prepared")
    @classmethod
    def prepared_path(cls, value):
        relative_run_path(value)
        return value


def verify_evidence_inputs(root, config):
    config = EvidenceWorkflowConfig.model_validate(config)
    prepared = local_path(root, config.prepared)
    verify_competition_bundle(prepared)
    return {"prepared": str(prepared), "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
            "config": config.model_dump()}


def recheck_evidence_inputs(inputs):
    prepared = Path(inputs["prepared"])
    if file_hash(prepared / "manifest.json") != inputs["prepared_manifest_sha256"]:
        raise ValueError("Prepared inputs changed during evidence workflow")
    verify_competition_bundle(prepared)
    return prepared


def compare_evidence_controls(inputs, out):
    return analyze_training_evidence(recheck_evidence_inputs(inputs), Path(out) / "fixed_controls")


def select_evidence_candidate(inputs, controls, out):
    prepared = recheck_evidence_inputs(inputs)
    if controls["prepared_manifest_sha256"] != inputs["prepared_manifest_sha256"]:
        raise ValueError("Control comparison belongs to different prepared inputs")
    if controls["excluded_small_retained_scopes"] or controls["fully_eligible_retained_scopes"] != controls["training_scopes"]:
        raise ValueError("Every training scope must cover all evidence budgets before selection")
    return analyze_evidence_policies(prepared, Path(out) / "selection")


def finalize_evidence_workflow(inputs, controls, selection, out):
    recheck_evidence_inputs(inputs)
    out = Path(out)
    for directory, expected in [("fixed_controls", controls), ("selection", selection)]:
        path = out / directory
        manifest = json.loads((path / "manifest.json").read_text())
        if "summary.json" not in manifest["artifacts"]:
            raise ValueError("Workflow stage summary is not hash-pinned")
        for name, digest in manifest["artifacts"].items():
            target = (path / name).resolve()
            if not target.is_relative_to(path.resolve()) or file_hash(target) != digest:
                raise ValueError("Evidence stage artifact changed before publication")
        actual = json.loads((path / "summary.json").read_text())
        # JSON/ZenML serialization turns numeric budget keys into strings.
        if actual != json.loads(json.dumps(expected)) or actual["prepared_manifest_sha256"] != inputs["prepared_manifest_sha256"]:
            raise ValueError("Evidence stage summary differs from its pipeline output")
    if (selection["original_development_outcomes_used"] or selection["monitor_outcomes_used"] or
            selection["sealed_roles_opened"] or controls["development_outcomes_used"] or controls["sealed_roles_opened"]):
        raise ValueError("Only training outcomes belong in candidate selection")
    report = {"version": "competition_evidence_workflow_v1", "source_policy": "provided_dataset_only",
              "prepared_manifest_sha256": inputs["prepared_manifest_sha256"],
              "status": "training_candidate_requires_independent_validation", "release_approved": False,
              "fixed_control_alc": controls["brier_alc_equal_visible_scope"],
              "selected_strength": selection["selected_strength"], "selected_strategy": selection["selected_strategy"],
              "nested_strategy_alc": selection["nested_strategy_alc_equal_scope"],
              "development_outcomes_used": False, "sealed_roles_opened": False,
              "native_laya_inference": False, "gpu_jobs": 0, "paid_calls": 0,
              "submission_built": False,
              "next_gate": "Explicit frozen-policy assessment; rejected candidates must not be released."}
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {
        str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file() and p != out / "manifest.json"}})
    return report
