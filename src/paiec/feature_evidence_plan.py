"""Label-free links for later training-only capability and evidence features.

No outcomes are read, summarized or fitted here. These are collection indices,
not an inference memory, learned capability profile or training-ready feature.
"""
from collections import Counter, defaultdict
from pathlib import Path

from .feature_preparation import read_jsonl, stable_hash, write_json, write_jsonl
from .head_training import file_hash


def prepare_evidence_plan(tasks, examples, out, *, seed=20261023, folds=5):
    if type(folds) is not int or folds < 2 or type(seed) is not int:
        raise ValueError("Use at least two folds and an integer seed")
    task_groups = {}
    for task in tasks:
        if task["task_key"] in task_groups:
            raise ValueError("Duplicate task")
        task_groups[task["task_key"]] = task["normalized_content_sha256"]
    rows, roles, seen = [], defaultdict(set), set()
    for example in examples:
        role = example["role"]
        if role not in {"training", "monitor", "development"}:
            raise ValueError("Sealed roles cannot enter an evidence plan")
        if example["task_key"] not in task_groups:
            raise ValueError("Unlinked task")
        group = task_groups[example["task_key"]]
        roles[group].add(role)
        identity = (role, example["source_line"])
        if identity in seen:
            raise ValueError("Duplicate source row")
        seen.add(identity)
        if role != "training":
            continue
        # Project explicit fields; never copy expected labels or private IDs.
        scope = stable_hash({"system_key": example["system_key"], "benchmark_id": example["benchmark_id"]})
        row = {"source_line": example["source_line"], "role": role,
               "system_key": example["system_key"], "benchmark_id": example["benchmark_id"],
               "scope_key": scope, "task_key": example["task_key"], "task_group": group}
        if type(row["source_line"]) is not int or row["source_line"] < 0:
            raise ValueError("Invalid source row")
        row["row_key"] = stable_hash(row)
        rows.append(row)
    if any(len(value) != 1 for value in roles.values()):
        raise ValueError("Normalized task overlaps preparation roles")
    groups = {r["task_group"] for r in rows}
    if len(groups) < folds:
        raise ValueError("Too few normalized training tasks for the requested folds")
    def order(kind, value):
        return stable_hash({"seed": seed, "kind": kind, "value": value})
    assignments = {key: i % folds for i, key in enumerate(sorted(groups, key=lambda key: order("fold", key)))}
    scoped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        row["fold"] = assignments[row["task_group"]]
        scoped[row["scope_key"]][row["task_group"]].append(row)
    pools, episodes = [], []
    too_small = 0
    for scope, grouped in sorted(scoped.items()):
        # One representative source row per normalized task, independent of label.
        representative = {group: min(values, key=lambda r: order("representative", r["row_key"]))["row_key"]
                          for group, values in grouped.items()}
        for fold in range(folds):
            candidates = sorted(group for group in grouped if assignments[group] != fold)
            pools.append({"scope_key": scope, "query_fold": fold,
                          "support_row_keys": [representative[g] for g in candidates],
                          "support_task_groups": candidates})
        ordered = sorted(grouped, key=lambda key: order("episode:" + scope, key))
        split = len(ordered) // 2
        support_groups, query_groups = ordered[:split], ordered[split:]
        too_small += int(not support_groups or not query_groups)
        support_order = sorted(support_groups, key=lambda key: order("support_order:" + scope, key))
        episodes.append({"scope_key": scope, "acquisition_task_groups": support_groups,
                         "query_task_groups": query_groups,
                         "query_row_keys": sorted(r["row_key"] for g in query_groups for r in grouped[g]),
                         "support_order_row_keys": [representative[g] for g in support_order[:31]],
                         "budgets": [{"requested": b, "available": min(b, len(support_order))}
                                     for b in [0, 1, 3, 7, 15, 31]]})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "training_row_index.jsonl", sorted(rows, key=lambda r: r["source_line"]))
    write_jsonl(out / "out_of_fold_support_pools.jsonl", pools)
    write_jsonl(out / "evidence_episode_links.jsonl", episodes)
    report = {"version": "feature_evidence_plan_v1", "seed": seed, "folds": folds,
              "training_rows": len(rows), "training_task_groups": len(groups),
              "visible_system_benchmark_scopes": len(scoped),
              "fold_group_counts": dict(Counter(assignments.values())),
              "out_of_fold_support_pools": len(pools), "episode_splits": len(episodes),
              "scopes_without_both_sides": too_small,
              "episode_split": "50/50 normalized task groups per visible scope; odd remainder goes to queries",
              "trial_grouping": "all available rows for a normalized task stay on the same side",
              "support_selection": "one outcome-blind representative row per normalized task",
              "eligible_source_role": "training", "query_outcomes_exported": False,
              "support_outcomes_exported": False, "outcomes_read": False,
              "empirical_capabilities_fitted": False, "evidence_correction_fitted": False,
              "semantic_retrieval_performed": False, "training_ready": False}
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return report
