"""Fail-closed initial release checks for the CatBoost base predictor."""
import math

RESEARCH_POLICY = "provided_plus_research_profiles"


def assess_catboost_base(report, *, deployable=("catboost",), base="catboost"):
    if (report.get("source_policy") != "provided_dataset_only" or report.get("training_families", 0) < 2 or
            report.get("original_development_outcomes_used") is not False or
            report.get("monitor_outcomes_used") is not False or report.get("sealed_roles_opened") is not False):
        raise ValueError("Candidate assessment requires isolated training-family validation")
    cold = report["zero_budget_brier_all_held_rows_equal_scope"]
    alc = report["brier_alc_on_common_eligible_scopes"]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in [*cold.values(), *alc.values()]):
        raise ValueError("Candidate assessment contains invalid scores")
    if cold.get("constant") != .25 or alc.get("constant") != .25 or "beta_k5" not in alc:
        raise ValueError("Matched fixed controls are missing")
    if base not in cold:
        raise ValueError("The base predictor has no cold-start measurement")
    coverage_complete = (report["full_budget_eligible_scopes"] == report["visible_scopes"]
                         and report["excluded_small_scopes"] == 0)
    candidates = {}
    for name in sorted(alc):
        if "/" not in name:
            continue
        model, mode = name.split("/", 1)
        if model not in cold:
            raise ValueError("Missing candidate cold-start measurement")
        reasons = []
        if cold[model] >= cold["constant"]:
            reasons.append("cold_start_not_better_than_constant_on_all_scopes")
        if alc[name] >= alc["beta_k5"]:
            reasons.append("does_not_improve_matched_smoothed_evidence_control")
        predictive_checks_passed = not reasons
        if model not in deployable:
            # Ablations explain the feature groups; choosing one after seeing its
            # outer score would turn the validation into a tuning set.
            reasons.append("ablation_variant_not_a_release_candidate")
        if not coverage_complete:
            reasons.append("full_budget_validation_does_not_cover_all_retained_scopes")
        candidates[name] = {"predictive_checks_passed": predictive_checks_passed,
                            "coverage_checks_passed": coverage_complete,
                            "initial_release_checks_passed": not reasons, "reasons": reasons}
    return {"version": "catboost_base_assessment_v1",
            "base_predictor_out_of_family_cold_start_beats_constant": cold[base] < cold["constant"],
            "release_candidate_models": list(deployable),
            "ablation_variants_assessed_for_evidence_only": True,
            "candidates": candidates,
            "any_adaptation_candidate_passed_initial_checks": any(
                row["initial_release_checks_passed"] for row in candidates.values()),
            "automatic_deployment_performed": False,
            "limitations": ["Passing these necessary checks would not establish hosted compatibility or statistical robustness.",
                            "A base model must clear the cold-start gate before adaptation is worth fitting separately."]}


def assess_research_only_catboost_base(report, *, base="catboost"):
    """Same measurements, but the external profile source bars release.

    The predictive checks are still reported so the enrichment's effect is
    visible; the release verdict is closed because the profile facts come from
    external sources that remain unreviewed for competition eligibility.
    """
    if (report.get("source_policy") != RESEARCH_POLICY or report.get("training_families", 0) < 2 or
            report.get("original_development_outcomes_used") is not False or
            report.get("monitor_outcomes_used") is not False or report.get("sealed_roles_opened") is not False):
        raise ValueError("Research assessment requires isolated training-family validation")
    cold = report["zero_budget_brier_all_held_rows_equal_scope"]
    alc = report["brier_alc_on_common_eligible_scopes"]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in [*cold.values(), *alc.values()]):
        raise ValueError("Candidate assessment contains invalid scores")
    if cold.get("constant") != .25 or alc.get("constant") != .25 or "beta_k5" not in alc:
        raise ValueError("Matched fixed controls are missing")
    if base not in cold:
        raise ValueError("The base predictor has no cold-start measurement")
    coverage_complete = (report["full_budget_eligible_scopes"] == report["visible_scopes"]
                         and report["excluded_small_scopes"] == 0)
    candidates = {}
    for name in sorted(alc):
        if "/" not in name:
            continue
        model, _ = name.split("/", 1)
        if model not in cold:
            raise ValueError("Missing candidate cold-start measurement")
        reasons = []
        if cold[model] >= cold["constant"]:
            reasons.append("cold_start_not_better_than_constant_on_all_scopes")
        if alc[name] >= alc["beta_k5"]:
            reasons.append("does_not_improve_matched_smoothed_evidence_control")
        predictive_checks_passed = not reasons
        reasons.append("external_research_profiles_not_competition_eligible")
        if not coverage_complete:
            reasons.append("full_budget_validation_does_not_cover_all_retained_scopes")
        candidates[name] = {"predictive_checks_passed": predictive_checks_passed,
                            "coverage_checks_passed": coverage_complete,
                            "initial_release_checks_passed": False, "reasons": reasons}
    return {"version": "catboost_base_assessment_v1",
            "base_predictor_out_of_family_cold_start_beats_constant": cold[base] < cold["constant"],
            "release_candidate_models": [],
            "release_eligible": False,
            "release_blocked_by": ["external_research_profiles_not_competition_eligible"],
            "ablation_variants_assessed_for_evidence_only": True,
            "candidates": candidates,
            "any_adaptation_candidate_passed_initial_checks": False,
            "automatic_deployment_performed": False,
            "limitations": ["This variant exists to measure whether documented profile facts carry"
                            " information; it is not a submission candidate.",
                            "Profile facts describe documented configurations, not verified evaluated checkpoints."]}
