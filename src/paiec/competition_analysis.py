"""Training-only acquisition diagnostics, without fitted models or native inference."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .feature_preparation import read_jsonl, write_json, write_jsonl
from .head_training import file_hash
from .metrics import BUDGETS, brier_alc


def verify_competition_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("version") != "competition_data_v1":
        raise ValueError("Use an explicitly dataset-only competition bundle")
    required = {"source_contract.json", "summary.json", "training_targets.jsonl",
                "evidence_plan_v1/training_row_index.jsonl", "evidence_plan_v1/evidence_episode_links.jsonl"}
    if not required <= set(manifest["artifacts"]):
        raise ValueError("Required competition artifacts are not hash-pinned")
    for name, digest in manifest["artifacts"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or file_hash(target) != digest:
            raise ValueError("Competition bundle artifact changed")
    summary = json.loads((path / "summary.json").read_text())
    contract = json.loads((path / "source_contract.json").read_text())
    if (summary.get("source_policy") != "provided_dataset_only" or summary.get("external_profile_facts_used") != 0 or
        contract["config"].get("source_policy") != "provided_dataset_only"):
        raise ValueError("Competition analysis cannot use research enrichment")
    return summary, contract


def numeric_evidence_predictions(support_labels):
    """Fixed controls, using this call's allowed support prefix only."""
    if any(type(y) not in {bool, int} or y not in (0, 1) for y in support_labels):
        raise ValueError("Evidence must retain binary native values")
    n, successes = len(support_labels), sum(support_labels)
    return {"constant": .5, "empirical_mean": float(successes / n) if n else .5,
            "beta_k5": float((successes + 2.5) / (n + 5))}


def analyze_training_evidence(prepared, out):
    prepared, out = Path(prepared), Path(out)
    source_summary, _ = verify_competition_bundle(prepared)
    source_index = list(read_jsonl(prepared / "evidence_plan_v1/training_row_index.jsonl"))
    index = {r["row_key"]: r for r in source_index}
    if len(source_index) != len(index):
        raise ValueError("Duplicate training source row")
    targets = {}
    for target in read_jsonl(prepared / "training_targets.jsonl"):
        key, label = target["row_key"], target["native_binary_value"]
        if key in targets or key not in index or type(label) not in {bool, int} or label not in (0, 1):
            raise ValueError("Invalid, duplicate or foreign training target")
        if index[key]["role"] != "training":
            raise ValueError("Only training-role outcomes are eligible")
        targets[key] = label
    if set(targets) != set(index):
        raise ValueError("Training target coverage is incomplete")
    episodes = list(read_jsonl(prepared / "evidence_plan_v1/evidence_episode_links.jsonl"))
    if len({e["scope_key"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate evidence episode scope")
    scored, excluded, predictions = [], [], []
    for episode in episodes:
        groups = episode["acquisition_task_groups"] + episode["query_task_groups"]
        if len(groups) != len(set(groups)):
            raise ValueError("Acquisition/query task overlap")
        if len(episode["acquisition_task_groups"]) != len(groups) // 2:
            raise ValueError("Episode differs from the declared 50/50 split")
        supports = episode["support_order_row_keys"]
        query = episode["query_row_keys"]
        if len(groups) < 80 or len(supports) < max(BUDGETS):
            excluded.append({"scope_key": episode["scope_key"], "retained_task_groups": len(groups),
                             "available_supports": len(supports), "reason": "insufficient_retained_items_for_full_protocol"})
            continue
        if len(supports) != len(set(supports)) or len(query) != len(set(query)) or set(supports) & set(query):
            raise ValueError("Duplicate or overlapping evidence pointers")
        for kind, keys, allowed in [("support", supports, set(episode["acquisition_task_groups"])),
                                    ("query", query, set(episode["query_task_groups"]))]:
            if not keys:
                raise ValueError("Episode has an empty side")
            for key in keys:
                row = index[key]
                if row["scope_key"] != episode["scope_key"] or row["task_group"] not in allowed:
                    raise ValueError("Evidence pointer violates scope or task split")
        labels = [targets[key] for key in supports]
        # Predictions use supports only; query targets enter the scoring expression.
        for budget in BUDGETS:
            for method, probability in numeric_evidence_predictions(labels[:budget]).items():
                error = float(np.mean([(probability - int(targets[key])) ** 2 for key in query]))
                scored.append({"scope_key": episode["scope_key"], "method": method, "budget": budget,
                               "brier": error, "query_rows": len(query), "retained_task_groups": len(groups)})
                predictions.append({"scope_key": episode["scope_key"], "method": method, "budget": budget,
                                    "probability": probability, "query_row_keys": query})
    if not scored:
        raise ValueError("No training scopes cover the complete competition protocol")
    curves, alc = {}, {}
    for method in sorted({r["method"] for r in scored}):
        curve = {budget: float(np.mean([r["brier"] for r in scored if r["method"] == method and r["budget"] == budget]))
                 for budget in BUDGETS}
        curves[method], alc[method] = curve, brier_alc(curve)
    coverage = defaultdict(set)
    for row in index.values():
        coverage[row["task_group"]].add(row["scope_key"])
    report = {"version": "training_evidence_diagnostic_v1", "source_policy": "provided_dataset_only",
              "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
              "training_rows": len(index), "training_scopes": len(episodes),
              "fully_eligible_retained_scopes": len(episodes) - len(excluded),
              "excluded_small_retained_scopes": len(excluded),
              "per_budget": curves, "brier_alc_equal_visible_scope": alc,
              "training_task_groups": len(coverage),
              "task_groups_by_scope_coverage": dict(sorted(Counter(len(v) for v in coverage.values()).items())),
              "scope_counts_are_private_subject_counts": False,
              "native_event_semantics_preserved": True, "native_inference_performed": False,
              "model_fitting_performed": False, "development_outcomes_used": False,
              "sealed_roles_opened": False, "generalization_result": False,
              "comparison_scope": "one training-only acquisition trajectory; retained scopes with at least 80 task groups",
              "limitations": ["One selected recorded trial per original system/item, not all native trials.",
                              "Equal visible-scope weighting; no private subject disambiguation.",
                              "Shared questions and systems make scopes statistically dependent.",
                              "This is not a frozen-development or hidden-competition result.",
                              "Small retained scopes are excluded and may omit many sampled configurations."],
              "implementation_sha256": file_hash(Path(__file__)),
              "source_record_counts": source_summary["records_by_role"]}
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "pair_budget_scores.jsonl", scored)
    write_jsonl(out / "pair_budget_predictions.jsonl", predictions)
    write_jsonl(out / "excluded_scopes.jsonl", excluded)
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return report
