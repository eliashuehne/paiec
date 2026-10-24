"""One frozen-development assessment of an already selected numerical policy."""
import json
from pathlib import Path

import pandas as pd

from .competition_analysis import verify_competition_bundle
from .development_evidence import original_development_pairs
from .evidence_policy import EvidencePolicy
from .evaluate import evaluate
from .feature_preparation import write_json
from .head_training import file_hash
from .predictors import Constant


def assess_frozen_policy(root, prepared, policy_path, policy_sha256, out):
    root, prepared, policy_path, out = Path(root).resolve(), Path(prepared), Path(policy_path), Path(out)
    _, contract = verify_competition_bundle(prepared)
    selected = EvidencePolicy.load(policy_path, expected_sha256=policy_sha256)
    frozen_state = json.loads(policy_path.read_text())
    if frozen_state["training_provenance"]["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json"):
        raise ValueError("Selected policy belongs to a different training panel")
    upstream = Path(contract["prepared"])
    if file_hash(upstream / "manifest.json") != contract["source_manifest_sha256"]:
        raise ValueError("Prepared source manifest changed")
    source_manifest = json.loads((upstream / "manifest.json").read_text())
    split_path = (root / source_manifest["data_config"]["split"]).resolve()
    if not split_path.is_relative_to(root) or file_hash(split_path) != contract["split_sha256"]:
        raise ValueError("Frozen split changed")
    split = json.loads(split_path.read_text())
    original = root / split["artifact_directory"]
    provenance_path = original / "provenance.parquet"
    if file_hash(provenance_path) != split["artifact_fingerprints"]["provenance.parquet"]:
        raise ValueError("Original frozen evaluation provenance changed")
    frame = pd.read_parquet(provenance_path, columns=["role", "exported", "jsonl_line", "benchmark", "subject_id", "item_id"])
    development = frame[frame.role.eq("development") & frame.exported].sort_values("jsonl_line")
    pairs = original_development_pairs(root, source_manifest, development, split["seed"])
    controls = {"selected": selected, "constant": Constant(), "empirical_mean": EvidencePolicy({
        "version": "evidence_policy_v1", "source_policy": "provided_dataset_only", "prior": .5, "strength": 0.}),
        "fixed_beta5": EvidencePolicy({"version": "evidence_policy_v1", "source_policy": "provided_dataset_only",
                                      "prior": .5, "strength": 5.})}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "frozen_assessment_plan.json", {"policy_sha256": policy_sha256, "seed": split["seed"],
        "controls": ["constant", "empirical_mean", "fixed_beta5"],
        "release_rule": "selected must beat constant and the predeclared fixed-beta5 control; otherwise recommend fixed-beta5 without refitting",
        "original_development_provenance_sha256": file_hash(provenance_path)})
    summaries = {}
    for name, predictor in controls.items():
        records = []
        result = evaluate(pairs, predictor, seed=split["seed"], acquisition_items=40,
                          evaluation_items=120, shared_evidence=True, prediction_records=records)
        result.frame.to_csv(out / (name + "_scores.csv"), index=False)
        pd.DataFrame(records).to_csv(out / (name + "_predictions.csv"), index=False)
        summaries[name] = {"brier_alc": result.brier_alc,
            "per_budget": {str(k): float(v) for k,v in result.per_budget.items()}, "scored_pairs": len(result.pair_alc)}
    if file_hash(policy_path) != policy_sha256:
        raise ValueError("Policy changed during assessment")
    passed = summaries["selected"]["brier_alc"] <= min(.25, summaries["fixed_beta5"]["brier_alc"]) + 1e-12
    report = {"version": "frozen_policy_assessment_v1", "source_policy": "provided_dataset_only",
        "prepared_manifest_sha256": file_hash(prepared / "manifest.json"), "selected_policy_sha256": policy_sha256,
        "selected_strength": selected.strength, "results": summaries, "original_development_pairs": len(pairs),
        "original_development_records": len(development), "evaluation_manifest_unchanged": True,
        "candidate_passes_development_gate": passed, "recommended_method": "selected" if passed else "fixed_beta5",
        "recommended_strength": selected.strength if passed else 5., "parameters_refitted": False,
        "frozen_development_used_for_release_choice": True, "supplemental_development_outcomes_used": False,
        "calibration_opened": False, "confirmation_opened": False, "native_laya_inference": False,
        "gpu_jobs": 0, "paid_calls": 0, "hosted_performance_established": False,
        "limitations": ["The original development set has been inspected in earlier research; it is not a fresh test set.",
            "Only predeclared controls are compared; no parameters are tuned to development outcomes.",
            "The recommendation uses development for model choice and is not an unbiased final estimate.",
            "Identical visible profiles share allowed evidence; private identities are never recovered."]}
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return report
