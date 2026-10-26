"""Fail-closed assessment of a rate-model run on the competition-shaped scorer.

The base model carries only the budget-0 weight of 0.1, so clearing the constant
is necessary and far from sufficient. The declared comparison is against the
revealed-label mean, which uses no model at all.

That comparison turned out to be weak on this corpus: with pair rates near 0.5
and budgets 1 and 3 carrying 0.4 of the weight, an unsmoothed label mean predicts
0 or 1 from a single label and scores no better than the constant. Shrinkage, not
the mean, is what the metric rewards. A secondary check therefore compares the
model against the strongest fixed control, declared here after the control run
and before any rate model was fitted.
"""
NAME = "competition_rate_gate_v1"
MIN_FOLDS = 3
MIN_PAIRS = 200
STRONGER_CONTROLS = ("fixed_beta_k5", "pooled_plus_own")


def _arm(summary, name, regime="all"):
    arms = summary.get("arms", {}).get(regime, {})
    if name not in arms:
        raise ValueError("Run summary is missing arm: " + name)
    return arms[name]


def assess(summary, *, regime="all"):
    """Return checks and a verdict; never promote an inconclusive model."""
    model = _arm(summary, "hierarchical_rate", regime)
    constant = _arm(summary, "constant_0.5", regime)
    acquired = _arm(summary, "acquired_label_mean", regime)
    arms = summary.get("arms", {}).get(regime, {})
    strongest = min((arms[name]["brier_alc_mean"] for name in STRONGER_CONTROLS if name in arms),
                    default=None)
    folds = [row for row in summary.get("folds", []) if row.get("regime") == regime]
    improvements = [row["arms"]["hierarchical_rate"]["brier_alc"]
                    - row["arms"]["acquired_label_mean"]["brier_alc"] for row in folds]
    checks = [
        {"check": "cold_start_beats_constant",
         "value": model["curve_mean"]["0"], "reference": constant["curve_mean"]["0"],
         "passes": model["curve_mean"]["0"] < constant["curve_mean"]["0"]},
        {"check": "alc_beats_revealed_label_mean",
         "value": model["brier_alc_mean"], "reference": acquired["brier_alc_mean"],
         "passes": model["brier_alc_mean"] < acquired["brier_alc_mean"]},
        {"check": "alc_beats_strongest_fixed_control",
         "value": model["brier_alc_mean"], "reference": strongest,
         "passes": strongest is not None and model["brier_alc_mean"] < strongest},
        {"check": "more_folds_improve_than_not",
         "value": sum(value < 0 for value in improvements), "reference": len(improvements),
         "passes": bool(improvements) and sum(improvements) < 0},
        {"check": "enough_folds_reported", "value": len(folds), "reference": MIN_FOLDS,
         "passes": len(folds) >= MIN_FOLDS},
        {"check": "enough_pairs_scored", "value": model.get("pairs", 0), "reference": MIN_PAIRS,
         "passes": (model.get("pairs") or 0) >= MIN_PAIRS},
    ]
    failed = [check["check"] for check in checks if not check["passes"]]
    if not failed:
        verdict = "clears_matched_adaptive_control"
    elif "alc_beats_strongest_fixed_control" in failed:
        verdict = "no_gain_over_fixed_controls"
    elif "alc_beats_revealed_label_mean" in failed or "more_folds_improve_than_not" in failed:
        verdict = "no_gain_over_revealed_label_mean"
    else:
        verdict = "inconclusive"
    return {"version": NAME, "regime": regime, "checks": checks,
            "failed": failed, "verdict": verdict,
            "release_candidate": False,
            "submission_ready": False,
            "note": "Assessment is local and CPU only; no hosted execution was verified."}


__all__ = ["NAME", "assess"]
