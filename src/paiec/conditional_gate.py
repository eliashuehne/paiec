"""Fail-closed initial release checks for the conditional control experiment."""
import math


def assess_conditional_controls(report):
    if (report.get("source_policy") != "provided_dataset_only" or report.get("training_families", 0) < 2 or
        report.get("original_development_outcomes_used") is not False or report.get("monitor_outcomes_used") is not False or
        report.get("sealed_roles_opened") is not False):
        raise ValueError("Candidate assessment requires isolated training-family validation")
    cold = report["zero_budget_brier_all_held_rows_equal_scope"]
    alc = report["brier_alc_on_common_eligible_scopes"]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in [*cold.values(), *alc.values()]):
        raise ValueError("Candidate assessment contains invalid scores")
    if cold.get("constant") != .25 or alc.get("constant") != .25 or "beta_k5" not in alc:
        raise ValueError("Matched fixed controls are missing")
    coverage_complete = report["full_budget_eligible_scopes"] == report["visible_scopes"] and report["excluded_small_scopes"] == 0
    candidates = {}
    for name in sorted(alc):
        if "/" not in name:
            continue
        model = name.split("/", 1)[0]
        if model not in cold:
            raise ValueError("Missing candidate cold-start measurement")
        reasons = []
        if cold[model] > cold["constant"]:
            reasons.append("cold_start_worse_than_constant_on_all_scopes")
        if alc[name] >= alc["beta_k5"]:
            reasons.append("does_not_improve_matched_smoothed_evidence_control")
        predictive_checks_passed = not reasons
        if not coverage_complete:
            reasons.append("full_budget_validation_does_not_cover_all_retained_scopes")
        candidates[name] = {"predictive_checks_passed": predictive_checks_passed,
                            "coverage_checks_passed": coverage_complete,
                            "initial_release_checks_passed": not reasons, "reasons": reasons}
    if not candidates:
        raise ValueError("No conditional candidates were assessed")
    return {"version": "conditional_candidate_assessment_v1", "candidates": candidates,
            "any_candidate_passed_initial_checks": any(r["initial_release_checks_passed"] for r in candidates.values()),
            "automatic_deployment_performed": False,
            "limitations": ["Passing these necessary checks would not establish hosted compatibility or statistical robustness.",
                            "Further unseen-system tests, sampling qualification and packaging checks are required."]}
