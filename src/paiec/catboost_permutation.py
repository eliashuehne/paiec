"""Permutation distribution for the scrambled-profile control.

``catboost_base_v5`` permuted the system-to-fact link once and showed that a
large part of the apparent ordering gain survives the scramble. One draw cannot
separate "seventeen more columns changed the boosting path" from "the facts
carry information", because a single permutation has no spread. This module
repeats the identical recipe over a declared list of permutations, records the
same panel and unseen-benchmark metrics for every draw, and compares the
unpermuted arm against the resulting null distribution.

A permutation reassigns documented profile records between supplied systems. It
therefore keeps every column's marginal distribution, every within-scope
constancy and the fold structure exactly as they were, and removes only the
correspondence between a system and its facts. Fitting stays on the panel,
evaluation rows never enter fitting, and the profile sources stay research-only.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from .catboost_analysis import evaluate_catboost_base
from .catboost_features import PROFILE_GROUP
from .catboost_unseen_benchmark import UnseenBenchmarkConfig, evaluate_unseen_benchmarks
from .feature_preparation import local_path, read_jsonl, stable_hash, write_json, write_jsonl
from .grouped_data import checked_artifacts
from .head_training import file_hash
from .system_profile_features import ProfileFacts

SEEDED_VERSION = "catboost_base_v5"
UNPERMUTED_VERSION = "catboost_base_v4"
METRIC_DIRECTION = {
    "panel_cold_equal_scope_brier": "lower",
    "panel_mean_within_scope_auroc": "higher",
    "panel_alc_none": "lower",
    "panel_alc_beta_mean": "lower",
    "panel_alc_logit_shift": "lower",
    "unseen_primary_equal_scope_brier": "lower",
    "unseen_pooled_equal_scope_brier": "lower",
    "unseen_pooled_auroc": "higher",
    "unseen_fold_mean_auroc": "higher",
    "unseen_fold_mean_brier": "lower",
}
# Declared before the permutations are fitted. The panel ordering metric is
# measured with one model per row and no cross-fold averaging, but the decision
# this experiment exists to serve is about transfer, so the unseen per-fold
# ordering metric is primary and every other value is reported beside it.
PRIMARY_METRIC = "unseen_fold_mean_auroc"


class ReusedReplicate(BaseModel):
    """A permutation that was already run and is folded into the distribution."""

    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int
    panel_run: str
    unseen_run: str


class PermutationConfig(BaseModel):
    """Predeclared permutation study: which recipe, which seeds, which arms."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["catboost_permutation_control_v1"] = "catboost_permutation_control_v1"
    recipe: str
    unseen_config: str
    permutation_seeds: list[int]
    reused_replicates: list[ReusedReplicate] = []
    observed_panel_run: str
    observed_unseen_run: str
    baseline_panel_run: str
    baseline_unseen_run: str
    alpha: float = 0.05

    @model_validator(mode="after")
    def coherent(self):
        if not self.permutation_seeds:
            raise ValueError("Declare at least one permutation seed")
        if len(set(self.permutation_seeds)) != len(self.permutation_seeds):
            raise ValueError("Permutation seeds must be unique")
        if any(type(seed) is not int for seed in self.permutation_seeds):
            raise ValueError("Permutation seeds must be integers")
        reused = [row.seed for row in self.reused_replicates]
        if len(set(reused)) != len(reused):
            raise ValueError("A reused replicate was declared twice")
        if set(reused) & set(self.permutation_seeds):
            raise ValueError("A seed cannot be both reused and freshly fitted")
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must lie strictly between 0 and 1")
        return self


def permuted_recipe(base, seed):
    """The declared recipe with the system-to-fact link permuted by ``seed``.

    The base recipe must be the unpermuted profile recipe, so a control can only
    ever be a scramble of a recipe that was actually measured.
    """
    if not isinstance(base, dict) or base.get("version") != UNPERMUTED_VERSION:
        raise ValueError(f"Permutations must scramble the {UNPERMUTED_VERSION} recipe")
    if "profile_permutation_seed" in base:
        raise ValueError("The base recipe is already a permutation control")
    for key in ("profile_catalog", "profile_policy"):
        if not isinstance(base.get(key), str):
            raise ValueError("The base recipe must declare its profile catalog and policy")
    groups = base.get("feature_groups")
    if not isinstance(groups, list) or PROFILE_GROUP not in groups:
        raise ValueError("The base recipe must select the profile group it scrambles")
    if type(seed) is not int:
        raise ValueError("Permutation seeds must be integers")
    return {**base, "version": SEEDED_VERSION, "profile_permutation_seed": seed}


def collect_metrics(panel, unseen):
    """The same panel and unseen-benchmark metrics for one control replicate."""
    cold = panel["zero_budget_brier_all_held_rows_equal_scope"]["catboost"]
    ordering = panel["cold_start_metrics"]["catboost"]["mean_within_scope_auroc"]
    alc = panel["brier_alc_on_common_eligible_scopes"]
    folds = [row["external_metrics"] for row in unseen["folds"] if row["external_metrics"]]
    if not folds:
        raise ValueError("Unseen evaluation has no eligible fold")
    return {
        "panel_cold_equal_scope_brier": cold,
        "panel_mean_within_scope_auroc": ordering,
        "panel_alc_none": alc["catboost/none"],
        "panel_alc_beta_mean": alc["catboost/beta_mean"],
        "panel_alc_logit_shift": alc["catboost/logit_shift"],
        "unseen_primary_equal_scope_brier": unseen["primary_equal_scope_brier"],
        "unseen_pooled_equal_scope_brier": unseen["pooled_row_metrics"]["equal_scope_brier"],
        "unseen_pooled_auroc": unseen["pooled_row_metrics"]["mean_within_scope_auroc"],
        "unseen_fold_mean_auroc": float(np.mean([fold["mean_within_scope_auroc"] for fold in folds])),
        "unseen_fold_mean_brier": float(np.mean([fold["equal_scope_brier"] for fold in folds])),
    }


def scope_ordering(unseen_run):
    """Within-scope AUROC per visible scope, from the saved row predictions.

    Scope-level values let the same question be asked on ninety-odd units
    instead of seven folds, without the cross-fold averaging that inflates the
    pooled figure.
    """
    groups = {}
    for row in read_jsonl(Path(unseen_run) / "row_predictions.jsonl"):
        groups.setdefault(row["scope_key"], []).append((int(row["label"]), float(row["prediction"])))
    values = {}
    for scope, pairs in groups.items():
        labels = np.asarray([label for label, _ in pairs], dtype=float)
        scores = np.asarray([score for _, score in pairs], dtype=float)
        positive, negative = int(labels.sum()), int(len(labels) - labels.sum())
        if not positive or not negative:
            continue
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        values[scope] = float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))
    return values


def paired_scope_ordering(observed, baseline, nulls, *, seed=20261025, resamples=4000):
    """Secondary check: does the ordering gain hold scope by scope?

    Every replicate scores the same rows, so each scope is a paired unit. The
    observed arm is compared with the mean permuted arm on each scope and the
    mean difference is bootstrapped over scopes. This is secondary to the
    predeclared primary metric; it exists to show where a residual would live.
    """
    scopes = set(observed["scope_auroc"])
    for arm in nulls.values():
        scopes &= set(arm["scope_auroc"])
    scopes &= set(baseline["scope_auroc"])
    scopes = sorted(scopes)
    if not scopes:
        raise ValueError("No scope is scored by every arm")
    rows = []
    for scope in scopes:
        null_values = [arm["scope_auroc"][scope] for arm in nulls.values()]
        rows.append({"scope_key": scope, "observed_auroc": observed["scope_auroc"][scope],
                     "baseline_auroc": baseline["scope_auroc"][scope],
                     "null_mean_auroc": float(np.mean(null_values)),
                     "null_min_auroc": float(np.min(null_values)),
                     "null_max_auroc": float(np.max(null_values)),
                     "delta_vs_null_mean": observed["scope_auroc"][scope] - float(np.mean(null_values)),
                     "baseline_delta_vs_null_mean": baseline["scope_auroc"][scope] - float(np.mean(null_values))})
    deltas = np.asarray([row["delta_vs_null_mean"] for row in rows])
    baseline_deltas = np.asarray([row["baseline_delta_vs_null_mean"] for row in rows])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(resamples, len(rows)))
    sampled, sampled_baseline = deltas[draws].mean(axis=1), baseline_deltas[draws].mean(axis=1)
    return {
        "rows": rows,
        "scopes": len(rows),
        "mean_delta_vs_null_mean": float(deltas.mean()),
        "mean_delta_ci95": [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))],
        "scopes_above_null_mean": int((deltas > 0).sum()),
        "scopes_above_every_permutation": int(sum(
            row["observed_auroc"] > row["null_max_auroc"] for row in rows)),
        "scopes_below_every_permutation": int(sum(
            row["observed_auroc"] < row["null_min_auroc"] for row in rows)),
        "mean_baseline_delta_vs_null_mean": float(baseline_deltas.mean()),
        "mean_baseline_delta_ci95": [float(np.quantile(sampled_baseline, .025)),
                                     float(np.quantile(sampled_baseline, .975))],
        "resamples": resamples, "bootstrap_seed": seed,
        "role": "secondary descriptive check, declared after the primary metric but before any replicate was fitted",
    }


def scored_row_signature(unseen_run):
    """Identity of the scored evaluation rows, independent of predictions.

    Permutations must never change which rows are eligible; if they did, the
    null distribution would not be comparable to the observed arm.
    """
    rows = sorted((row["benchmark"], row["subject_id"], row["item_id"], int(row["label"]))
                  for row in read_jsonl(Path(unseen_run) / "row_predictions.jsonl"))
    return stable_hash(rows), len(rows)


def _verify_run(directory):
    """Every declared artifact must still hash to its recorded value."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    checked_artifacts(directory, manifest["artifacts"])
    return directory


def read_arm(panel_run, unseen_run):
    """Metrics, scope ordering and integrity facts for a finished arm."""
    panel_run, unseen_run = _verify_run(panel_run), _verify_run(unseen_run)
    panel = json.loads((panel_run / "summary.json").read_text())
    unseen = json.loads((unseen_run / "summary.json").read_text())
    if unseen["panel_run"]["recipe_version"] != panel["recipe"]["version"]:
        raise ValueError("Unseen evaluation was not fitted from the declared panel run")
    signature, rows = scored_row_signature(unseen_run)
    return {"metrics": collect_metrics(panel, unseen), "scored_rows": rows,
            "scored_row_signature": signature, "scope_auroc": scope_ordering(unseen_run),
            "panel_recipe_sha256": file_hash(panel_run / "resolved_config.json"),
            "source_policy": panel.get("source_policy")}


def verified_replicate(root, recipe, unseen, seed, replicate):
    """Read one finished replicate and prove it is the declared permutation.

    A directory that merely exists is not evidence: the resolved recipes must be
    exactly the ones this study declares for that seed, and the evaluation must
    point back at that replicate's own panel run.
    """
    replicate = Path(replicate).resolve()
    panel_out, unseen_out = replicate / "panel", replicate / "unseen"
    if json.loads((panel_out / "resolved_config.json").read_text()) != permuted_recipe(recipe, seed):
        raise ValueError(f"Replicate {seed} does not resolve the declared permutation")
    recorded = json.loads((unseen_out / "resolved_config.json").read_text())
    if {key: value for key, value in recorded.items() if key != "panel_run"} != \
            {key: value for key, value in unseen.model_dump().items() if key != "panel_run"}:
        raise ValueError(f"Replicate {seed} evaluated a different unseen recipe")
    if local_path(root, recorded["panel_run"]) != panel_out:
        raise ValueError(f"Replicate {seed} scored a different panel run")
    return read_arm(panel_out, unseen_out)


def _record(folder, config):
    """Write the resolved configuration and a manifest over the finished folder."""
    write_json(folder / "resolved_config.json", config)
    write_json(folder / "manifest.json", {"artifacts": {
        str(path.relative_to(folder)): file_hash(path)
        for path in sorted(folder.rglob("*")) if path.is_file() and path.name != "manifest.json"}})


def run_replicate(root, recipe, unseen, seed, out, *, facts=None):
    """Fit one permutation control for the panel and score the unseen benchmarks.

    Resumable at replicate granularity: a finished and hash-verified replicate is
    reused rather than refitted, and a panel run without its evaluation continues
    from the evaluation.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    permuted = permuted_recipe(recipe, seed)
    panel_out, unseen_out = out / "panel", out / "unseen"
    unseen_config = UnseenBenchmarkConfig.model_validate(
        {**unseen.model_dump(), "panel_run": str(panel_out.relative_to(root))})
    if (unseen_out / "summary.json").is_file():
        _verify_run(panel_out), _verify_run(unseen_out)
        return "reused"
    facts = facts or ProfileFacts(root / permuted["profile_policy"], root=root)
    if not (panel_out / "summary.json").is_file():
        # An interrupted replicate leaves a partial folder; it is ours and has no
        # verified summary, so it is discarded rather than trusted.
        if panel_out.exists():
            shutil.rmtree(panel_out)
        evaluate_catboost_base(local_path(root, permuted["prepared"]), panel_out,
                               recipe=permuted, profile_facts=facts)
        _record(panel_out, permuted)
    else:
        _verify_run(panel_out)
        recorded = json.loads((panel_out / "resolved_config.json").read_text())
        if recorded != permuted:
            raise ValueError("Reused panel replicate resolves a different recipe")
    if unseen_out.exists():
        shutil.rmtree(unseen_out)
    evaluate_unseen_benchmarks(root, unseen_config, unseen_out)
    _record(unseen_out, unseen_config.model_dump())
    return "fitted"


def _gain(metric, value, baseline):
    """Signed improvement over the task-only arm; positive means better."""
    return float(value - baseline) if METRIC_DIRECTION[metric] == "higher" else float(baseline - value)


def compare_to_null(observed, baseline, nulls, *, alpha):
    """Distribution, path-only effect, residual and permutation p-values.

    ``p`` counts null draws at least as good as the observed arm and adds one, so
    with N permutations its floor is 1/(N+1). That floor is reported, because a
    single p-value from a small N cannot reach a conventional threshold.
    """
    seeds = sorted(nulls)
    comparison, distribution = {}, {}
    for metric, direction in METRIC_DIRECTION.items():
        values = [float(nulls[seed][metric]) for seed in seeds]
        observed_value, baseline_value = float(observed[metric]), float(baseline[metric])
        observed_gain = _gain(metric, observed_value, baseline_value)
        null_gains = [_gain(metric, value, baseline_value) for value in values]
        if direction == "higher":
            extreme = [seed for seed, value in zip(seeds, values) if value >= observed_value]
        else:
            extreme = [seed for seed, value in zip(seeds, values) if value <= observed_value]
        distribution[metric] = {
            "direction": direction, "values_by_seed": {str(seed): float(nulls[seed][metric]) for seed in seeds},
            "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min": float(np.min(values)), "max": float(np.max(values)),
            "q05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "q95": float(np.quantile(values, .95)),
        }
        comparison[metric] = {
            "direction": direction, "observed": observed_value, "baseline_task_structure_only": baseline_value,
            "null_mean": distribution[metric]["mean"],
            "observed_gain_over_baseline": observed_gain,
            "path_only_gain_null_mean": float(np.mean(null_gains)),
            "residual_gain": observed_gain - float(np.mean(null_gains)),
            "p_value": (1 + len(extreme)) / (1 + len(values)),
            "null_seeds_at_least_as_good": extreme,
        }
    primary = comparison[PRIMARY_METRIC]
    if primary["residual_gain"] <= 0:
        verdict = "observed_arm_no_better_than_permuted_control"
    elif primary["p_value"] <= alpha:
        verdict = "residual_ordering_gain_beyond_permuted_control"
    else:
        verdict = "residual_ordering_gain_within_permuted_control_range"
    return {"distribution": distribution, "comparison": comparison,
            "primary": {"metric": PRIMARY_METRIC, "alpha": alpha,
                        "p_value": primary["p_value"], "p_value_floor": 1 / (1 + len(seeds)),
                        "residual_gain": primary["residual_gain"], "verdict": verdict,
                        "permutations": len(seeds)}}


def main_arms(root, config):
    """The observed arm, the task-structure baseline and every null replicate.

    Reused replicates are accepted only when their resolved configuration is
    exactly what this module would have written for that seed, so a stale or
    edited run cannot enter the distribution.
    """
    root = Path(root).resolve()
    recipe = json.loads((root / config.recipe).read_text())
    unseen = UnseenBenchmarkConfig.model_validate(
        json.loads((root / config.unseen_config).read_text()))
    baseline_panel = local_path(root, config.baseline_panel_run)
    baseline_recipe = json.loads((baseline_panel / "resolved_config.json").read_text())
    if baseline_recipe.get("version") != "catboost_base_v3":
        raise ValueError("The baseline arm must be the declared task-structure recipe")
    if recipe.get("version") != UNPERMUTED_VERSION:
        raise ValueError("The observed arm must be the unpermuted profile recipe")
    out = {"observed": read_arm(local_path(root, config.observed_panel_run),
                                local_path(root, config.observed_unseen_run)),
           "baseline": read_arm(baseline_panel, local_path(root, config.baseline_unseen_run)),
           "nulls": {}, "reused": []}
    for row in config.reused_replicates:
        panel_run, unseen_run = local_path(root, row.panel_run), local_path(root, row.unseen_run)
        recorded = json.loads((panel_run / "resolved_config.json").read_text())
        if recorded != permuted_recipe(recipe, row.seed):
            raise ValueError(f"Reused replicate {row.seed} does not resolve the declared permutation")
        recorded_unseen = json.loads((unseen_run / "resolved_config.json").read_text())
        if {key: value for key, value in recorded_unseen.items() if key != "panel_run"} != \
                {key: value for key, value in unseen.model_dump().items() if key != "panel_run"}:
            raise ValueError(f"Reused replicate {row.seed} evaluated a different unseen recipe")
        if local_path(root, recorded_unseen["panel_run"]) != panel_run:
            raise ValueError(f"Reused replicate {row.seed} points at another panel run")
        out["nulls"][row.seed] = read_arm(panel_run, unseen_run)
        out["reused"].append(row.seed)
    signatures = {out["observed"]["scored_row_signature"], out["baseline"]["scored_row_signature"]}
    signatures |= {arm["scored_row_signature"] for arm in out["nulls"].values()}
    if len(signatures) != 1:
        raise ValueError("The arms did not score the same evaluation rows")
    if out["observed"]["source_policy"] != "provided_plus_research_profiles":
        raise ValueError("The observed arm is not the research-profile recipe")
    return recipe, unseen, out


def run_control(root, config, out, *, fit=True, allow_partial=False):
    """Fit every declared permutation, reuse finished ones, and report the null.

    ``fit=False`` only re-reads hash-verified replicates that already exist, so a
    finished distribution can be re-aggregated without refitting anything.
    ``allow_partial`` reports the declared replicates that are present and names
    the missing ones, so an interrupted study can still be read honestly.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    recipe, unseen, arms = main_arms(root, config)
    started = time.monotonic()
    missing = []
    out.mkdir(parents=True, exist_ok=True)
    facts = ProfileFacts(root / recipe["profile_policy"], root=root)
    status = {}
    for seed in config.permutation_seeds:
        replicate = out / f"seed_{seed:02d}"
        if seed in arms["nulls"]:
            status[seed] = "reused"
            continue
        if (replicate / "panel" / "summary.json").is_file() and \
                (replicate / "unseen" / "summary.json").is_file():
            arms["nulls"][seed] = verified_replicate(root, recipe, unseen, seed, replicate)
            status[seed] = "loaded"
            continue
        if not fit:
            if not allow_partial:
                raise ValueError(f"Replicate {seed} is not finished; it cannot be aggregated")
            missing.append(seed)
            continue
        phase = "unseen_evaluation" if (replicate / "panel" / "summary.json").is_file() else "panel_fitting"
        write_json(out / "progress.json", {"phase": phase, "seed": seed,
                                           "completed_fresh": sum(kind == "fitted" for kind in status.values()),
                                           "declared_fresh": len([s for s in config.permutation_seeds
                                                                  if s not in arms["reused"]]),
                                           "elapsed_seconds": time.monotonic() - started})
        status[seed] = run_replicate(root, recipe, unseen, seed, replicate, facts=facts)
        if not (replicate / "unseen" / "summary.json").is_file():
            raise ValueError(f"Replicate {seed} produced no evaluation summary")
        arms["nulls"][seed] = verified_replicate(root, recipe, unseen, seed, replicate)
    signatures = {arm["scored_row_signature"] for arm in arms["nulls"].values()}
    if len(signatures) != 1:
        raise ValueError("Permuted replicates did not score the same evaluation rows")
    if missing and not allow_partial:
        raise ValueError("Declared permutations are unfinished: " + ", ".join(map(str, sorted(missing))))
    comparison = compare_to_null(arms["observed"]["metrics"], arms["baseline"]["metrics"],
                                {seed: arm["metrics"] for seed, arm in arms["nulls"].items()},
                                alpha=config.alpha)
    scopes = paired_scope_ordering(arms["observed"], arms["baseline"], arms["nulls"])
    write_jsonl(out / "scope_ordering.jsonl", scopes.pop("rows"))
    report = {
        "version": "catboost_permutation_control_report_v1",
        "config": config.model_dump(),
        "study": {"recipe": config.recipe, "unseen_config": config.unseen_config,
                  "fresh_seeds": [seed for seed in config.permutation_seeds],
                  "reused_seeds": arms["reused"],
                  "unfinished_declared_seeds": sorted(missing),
                  "complete_as_declared": not missing,
                  "permutations_in_distribution": len(arms["nulls"]),
                  "primary_metric_declared_before_fitting": PRIMARY_METRIC,
                  "fitted_by_this_invocation": {str(seed): kind for seed, kind in
                                                sorted(status.items()) if kind == "fitted"}},
        "observed_arm": {"panel_run": config.observed_panel_run, "unseen_run": config.observed_unseen_run,
                         **{key: value for key, value in arms["observed"].items() if key != "scope_auroc"}},
        "baseline_arm": {"panel_run": config.baseline_panel_run, "unseen_run": config.baseline_unseen_run,
                         **{key: value for key, value in arms["baseline"].items() if key != "scope_auroc"}},
        "null_replicates": {str(seed): {key: value for key, value in arm.items() if key != "scope_auroc"}
                            for seed, arm in sorted(arms["nulls"].items())},
        **comparison,
        "secondary_paired_scope_ordering": scopes,
        "integrity": {"complete_as_declared": not missing,
                      "unfinished_declared_seeds": sorted(missing),
                      "evaluation_rows_identical_across_arms": True,
                      "evaluation_rows_per_arm": arms["observed"]["scored_rows"],
                      "scored_row_signature": arms["observed"]["scored_row_signature"],
                      "external_rows_used_for_fitting": 0,
                      "sealed_roles_opened": False,
                      "original_development_outcomes_used": False,
                      "calibration_fitted": False,
                      "profiles_research_only": True,
                      "reused_replicate_configs_match_declared_permutations": True},
        "limitations": [
            "A permutation keeps every column's marginal distribution and only removes the"
            " system-to-fact correspondence; repeated profile records can still be reassigned to"
            " their own system by chance, which makes the null conservative.",
            f"The p-value floor is 1/(N+1) = {1 / (1 + len(arms['nulls'])):.4f} with"
            f" {len(arms['nulls'])} permutations, so small p-values are only reachable with more draws.",
            "Ten metrics are reported from one study; only the predeclared primary metric carries the verdict.",
            "The paired scope check is secondary and descriptive; it is not a second hypothesis test.",
            *([f"The distribution covers {len(arms['nulls'])} of the declared"
               f" {len(config.permutation_seeds) + len(config.reused_replicates)} permutations;"
               " the missing declared seeds were not fitted."] if missing else []),
            "The unseen rows are public unused benchmarks and are development evidence once scored.",
            "No candidate is promoted, no calibration is fitted and no submission package is built.",
        ],
    }
    write_json(out / "permutation_distribution.json", report)
    write_json(out / "manifest.json", {"artifacts": {
        str(path.relative_to(out)): file_hash(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}})
    write_json(out / "progress.json", {"phase": "finished", "permutations": len(arms["nulls"]),
                                       "elapsed_seconds": time.monotonic() - started})
    return report


def artifact_digest(report):
    """Stable digest of the reported comparison, for receipts and tracking."""
    return hashlib.sha256(json.dumps(report["comparison"], sort_keys=True).encode()).hexdigest()
