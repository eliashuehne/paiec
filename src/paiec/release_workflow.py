"""Explicit assessment and fail-closed local release of an evidence candidate."""
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .competition_analysis import verify_competition_bundle
from .feature_preparation import local_path, relative_run_path, write_json
from .head_training import file_hash
from .policy_validation import assess_frozen_policy
from .submission_package import build_evidence_submission
from .submission_qualification import qualify


class ReleaseWorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["validation_release_workflow_v1"] = "validation_release_workflow_v1"
    mode: Literal["review", "assess", "release"] = "review"
    prepared: str
    selection: str
    selection_manifest_sha256: str
    saved_assessment: str | None = None
    saved_assessment_sha256: str | None = None

    @model_validator(mode="after")
    def bound_inputs(self):
        for value in [self.prepared, self.selection, self.saved_assessment]:
            if value:
                relative_run_path(value)
        for value in [self.selection_manifest_sha256, self.saved_assessment_sha256]:
            if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("Use a complete SHA-256 fingerprint")
        if self.mode != "assess" and (not self.saved_assessment or not self.saved_assessment_sha256):
            raise ValueError("Review/release requires a pinned saved assessment")
        return self


def verify_release_inputs(root, config):
    root = Path(root).resolve()
    config = ReleaseWorkflowConfig.model_validate(config)
    prepared, selection = local_path(root, config.prepared), local_path(root, config.selection)
    verify_competition_bundle(prepared)
    if file_hash(selection / "manifest.json") != config.selection_manifest_sha256:
        raise ValueError("Selected candidate manifest changed")
    manifest = json.loads((selection / "manifest.json").read_text())
    if not {"summary.json", "evidence_policy.json"} <= set(manifest["artifacts"]):
        raise ValueError("Selected candidate policy/summary must be pinned")
    for name, digest in manifest["artifacts"].items():
        path = (selection / name).resolve()
        if not path.is_relative_to(selection) or file_hash(path) != digest:
            raise ValueError("Selected candidate artifact changed")
    summary = json.loads((selection / "summary.json").read_text())
    if summary["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json"):
        raise ValueError("Candidate belongs to a different dataset")
    if file_hash(selection / "evidence_policy.json") != summary["policy_sha256"]:
        raise ValueError("Selected policy differs from its receipt")
    return {"root": str(root), "config": config.model_dump(), "policy_sha256": summary["policy_sha256"],
            "prepared_manifest_sha256": file_hash(prepared / "manifest.json")}


def assess_or_review_candidate(inputs, out):
    root, config = Path(inputs["root"]), ReleaseWorkflowConfig.model_validate(inputs["config"])
    if verify_release_inputs(root, config.model_dump()) != inputs:
        raise ValueError("Release inputs changed after planning")
    if config.mode == "assess":
        return assess_frozen_policy(root, local_path(root, config.prepared),
            local_path(root, config.selection) / "evidence_policy.json", inputs["policy_sha256"], Path(out) / "assessment")
    path = local_path(root, config.saved_assessment)
    if file_hash(path) != config.saved_assessment_sha256:
        raise ValueError("Saved assessment changed")
    return json.loads(path.read_text())


def release_gate(inputs, assessment):
    if (assessment.get("version") != "frozen_policy_assessment_v1" or
            assessment.get("selected_policy_sha256") != inputs["policy_sha256"] or
            assessment.get("prepared_manifest_sha256") != inputs["prepared_manifest_sha256"] or
            assessment.get("parameters_refitted") is not False or
            assessment.get("evaluation_manifest_unchanged") is not True or
            any(assessment.get(k) is not False for k in ["calibration_opened", "confirmation_opened", "supplemental_development_outcomes_used"])):
        raise ValueError("Assessment lacks isolated, unchanged candidate provenance")
    scores = {name: assessment["results"][name]["brier_alc"] for name in ["selected", "constant", "fixed_beta5"]}
    if any(type(v) not in {float, int} or not math.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError("Assessment scores must be finite Brier ALC values")
    passed = scores["selected"] <= min(.25, scores["fixed_beta5"]) + 1e-12
    if assessment.get("candidate_passes_development_gate") is not passed:
        raise ValueError("Saved gate differs from the predeclared comparison")
    return {"candidate_passed": passed, "scores": scores,
            "development_used_for_release_choice": True,
            "reason": "passed_predeclared_comparison" if passed else "candidate_rejected_no_package",
            "fallback": "No automatic replacement; freeze and validate a separately declared candidate."}


def package_validated_candidate(inputs, assessment, out):
    root, out = Path(inputs["root"]), Path(out)
    if verify_release_inputs(root, inputs["config"]) != inputs:
        raise ValueError("Candidate inputs changed before packaging")
    gate = release_gate(inputs, assessment)
    write_json(out / "release_decision.json", gate)
    release = inputs["config"]["mode"] == "release" and gate["candidate_passed"]
    report = {"version": "validation_release_result_v1", **gate, "release_approved": False,
              "submission_built": False, "uploaded": False, "hosted_execution_verified": False,
              "policy_sha256": inputs["policy_sha256"], "assessment_mode": inputs["config"]["mode"]}
    if release:
        package = out / "package"
        receipt = build_evidence_submission(local_path(root, inputs["config"]["selection"]), package)
        with (out / "starting_kit_check.log").open("w") as log:
            subprocess.run([sys.executable, str(root / "baseline/check_submission_zip.py"), str(package / "submission.zip")],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        qualification = qualify(local_path(root, inputs["config"]["prepared"]), package, out / "qualification")
        report.update(submission_built=True, local_release_qualified=True,
                      archive_sha256=receipt["archive_sha256"], runtime=qualification["runtime"])
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p != out / "manifest.json"}})
    return report
