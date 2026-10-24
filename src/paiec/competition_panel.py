"""Outcome-blind augmentation of the frozen training sample to protocol coverage.

No model, profile catalog, renderer, API or trace table is used. Existing source
rows are retained; only reviewed training-family questions can be added.
"""
from collections import Counter
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .competition import competition_input
from .competition_data import CompetitionDataConfig, verify_dataset_sources
from .expanded_data import ExpandedDataConfig, load_candidates, monitor_key
from .feature_preparation import local_path, read_jsonl, stable_hash, write_json, write_jsonl
from .head_training import file_hash
from .schema import subject_record, item_record


class CompetitionPanelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["competition_panel_v1"] = "competition_panel_v1"
    source_policy: Literal["provided_dataset_only"] = "provided_dataset_only"
    base_prepared: str = "runs/head_finetuning/enriched_30k_v1"
    split: str = "configs/head_finetuning/frozen_split_v1.json"
    dataset_revision: str = "54074453da7b9911be0f48e4c560be547bc84d21"
    augmentation_seed: int = 20261024
    minimum_task_groups: int = Field(default=80, ge=80)
    maximum_added_rows: int = Field(default=10000, ge=1)


def augment_to_minimum(candidates, retained, *, minimum, seed, maximum_added):
    """Select pointers only. Labels cannot influence counts, ranking or selection."""
    keys = ["benchmark", "subject_id", "item_id", "response_id"]
    columns = [*keys, "scope_key", "task_sha256"]
    if candidates.duplicated(keys[:3]).any() or retained.duplicated(keys[:3]).any():
        raise ValueError("Use one selected recorded trial per original system/item")
    available_keys = set(map(tuple, candidates[keys].itertuples(index=False, name=None)))
    retained_keys = set(map(tuple, retained[keys].itertuples(index=False, name=None)))
    if not retained_keys <= available_keys:
        raise ValueError("A retained training trial is not in the admitted candidate panel")
    if set(retained.scope_key) != set(candidates.scope_key):
        raise ValueError("Do not silently add or discard visible scopes during augmentation")
    retained_groups = retained.groupby("scope_key").task_sha256.agg(set).to_dict()
    selected, coverage = [], []
    for scope, pool in candidates[columns].groupby("scope_key", sort=True):
        existing = retained_groups[scope]
        needed = max(0, minimum - len(existing))
        options = pool[~pool.task_sha256.isin(existing)].copy()
        options["selection_rank"] = [stable_hash({"seed": seed, "scope": scope, "pointer": list(row)})
                                      for row in options[columns].itertuples(index=False, name=None)]
        options = options.sort_values("selection_rank").drop_duplicates("task_sha256")
        if len(options) < needed:
            raise ValueError(f"Scope {scope} has only {len(existing) + len(options)} eligible independent task groups")
        chosen = options.iloc[:needed]
        selected.extend(tuple(row) for row in chosen[keys].itertuples(index=False, name=None))
        coverage.append({"scope_key": scope, "retained_task_groups_before": len(existing),
                         "added_task_groups": needed, "retained_task_groups_after": len(existing) + needed})
    if len(selected) > maximum_added:
        raise ValueError("Required augmentation exceeds the declared row bound")
    return selected, coverage


def prepare_competition_panel(root, config, out):
    root, out = Path(root).resolve(), Path(out)
    config = CompetitionPanelConfig.model_validate(config)
    base = local_path(root, config.base_prepared)
    split_path = (root / config.split).resolve()
    if not split_path.is_relative_to(root / "configs/head_finetuning"):
        raise ValueError("Use the repository's frozen role split")
    contract = verify_dataset_sources(root, CompetitionDataConfig(prepared=config.base_prepared,
        dataset_revision=config.dataset_revision).model_dump())
    base_manifest = json.loads((base / "manifest.json").read_text())
    if file_hash(split_path) != contract["split_sha256"]:
        raise ValueError("Panel split differs from the base split")
    provenance_path = base / "enriched/provenance.parquet"
    if file_hash(provenance_path) != base_manifest["artifacts"]["enriched/provenance.parquet"]:
        raise ValueError("Base source provenance changed")
    provenance = pd.read_parquet(provenance_path, columns=["role", "jsonl_line", "family", "benchmark",
        "subject_id", "item_id", "response_id", "trial", "task_sha256"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "progress.json", {"phase": "reviewed_source_loading"})
    # Reuse the reviewed admission and outcome-blind trial draw, with the base seed.
    # This helper reads minimal public tables and frozen role metadata, not profiles.
    draw_seed = base_manifest["data_config"]["seed"]
    loader_config = ExpandedDataConfig(split=config.split, seed=draw_seed)
    candidates, tables, split, source_hashes = load_candidates(root, loader_config, out)
    candidates = candidates[~candidates.task_sha256.map(lambda key: monitor_key(key, draw_seed))].copy()
    reserved = set(provenance.loc[~provenance.role.eq("training"), "task_sha256"])
    if set(candidates.task_sha256) & reserved:
        raise ValueError("Candidate panel overlaps the base monitor or development role")
    scopes, subjects = {}, {}
    for benchmark, subject in candidates[["benchmark", "subject_id"]].drop_duplicates().itertuples(index=False, name=None):
        recorded = subject_record(tables[benchmark][0].loc[subject])
        visible_subject, visible_item = competition_input(recorded, {"benchmark_id": benchmark})
        scope = stable_hash({"system_key": stable_hash(visible_subject), "benchmark_id": visible_item["benchmark_id"]})
        scopes[(benchmark, subject)] = scope
        subjects[(benchmark, subject)] = visible_subject
    candidates["scope_key"] = [scopes[(b,s)] for b,s in candidates[["benchmark", "subject_id"]].itertuples(index=False, name=None)]
    retained = provenance.loc[provenance.role.eq("training")].sort_values("jsonl_line").copy()
    if retained.jsonl_line.tolist() != list(range(len(retained))):
        raise ValueError("Base training row order is ambiguous")
    retained["scope_key"] = [scopes[(b,s)] for b,s in retained[["benchmark", "subject_id"]].itertuples(index=False, name=None)]
    pointers, coverage = augment_to_minimum(candidates, retained, minimum=config.minimum_task_groups,
        seed=config.augmentation_seed, maximum_added=config.maximum_added_rows)
    keys = ["benchmark", "subject_id", "item_id", "response_id"]
    candidate_map = candidates.set_index(keys, drop=False)
    added = [candidate_map.loc[pointer].to_dict() for pointer in pointers]
    source_dir = out / "source"
    source_dir.mkdir()
    new_provenance = provenance.copy()
    new_provenance["origin"] = "base_retained"
    original_rows = list(read_jsonl(base / "source/training.jsonl"))
    if len(original_rows) != len(retained):
        raise ValueError("Base training source and provenance do not align")
    # Verify the regenerated admission pointers still describe every original input.
    for raw, record in zip(original_rows, retained.to_dict("records")):
        subject = subjects[(record["benchmark"], record["subject_id"])]
        parsed = json.loads(raw["state"])
        if parsed["target_system"] != subject:
            raise ValueError("Retained visible subject projection changed")
    additions, added_provenance = [], []
    for i, record in enumerate(added, start=len(original_rows)):
        subject = subjects[(record["benchmark"], record["subject_id"])]
        item = item_record(tables[record["benchmark"]][1].loc[record["item_id"]], record["benchmark"])
        item["interactors"] = record["interactors"]
        subject, item = competition_input(subject, item)
        additions.append({"state": json.dumps({"target_system": subject, "target_item": item}, ensure_ascii=False),
                          "questions": contract["question"], "expected": {"correct": bool(record["response"])}})
        added_provenance.append({**{key: record[key] for key in ["family", *keys, "trial", "task_sha256"]},
            "role": "training", "jsonl_line": i, "origin": "added_training"})
    write_jsonl(source_dir / "training.jsonl", [*original_rows, *additions])
    for role in ["monitor", "development"]:
        # Carry permitted inputs only; do not access or copy their expected values.
        write_jsonl(source_dir / (role + ".jsonl"), ({"state": row["state"], "questions": row["questions"]}
            for row in read_jsonl(base / "source" / (role + ".jsonl"))))
    if added_provenance:
        new_provenance = pd.concat([new_provenance, pd.DataFrame(added_provenance)], ignore_index=True)
    new_provenance.to_parquet(out / "family_provenance.parquet", index=False)
    write_jsonl(out / "coverage.jsonl", coverage)
    write_json(out / "source_file_hashes.json", source_hashes)
    after = [row["retained_task_groups_after"] for row in coverage]
    summary = {"version": "competition_panel_v1", "source_policy": "provided_dataset_only",
        "base_training_rows": len(original_rows), "added_training_rows": len(additions),
        "total_training_rows": len(original_rows) + len(additions), "visible_scopes": len(coverage),
        "minimum_task_groups_per_scope": min(after), "all_scopes_cover_full_protocol": min(after) >= 80,
        "base_source_records_retained": True, "monitor_and_development_inputs_unchanged": True,
        "monitor_and_development_outcomes_exported": False, "monitor_assignment_seed": draw_seed,
        "augmentation_seed": config.augmentation_seed, "outcome_blind_selection": True,
        "source_trials_redrawn": False, "external_profile_facts_used": 0, "native_inference": False,
        "gpu_jobs": 0, "paid_calls": 0}
    write_json(out / "summary.json", summary)
    write_json(out / "resolved_config.json", config.model_dump())
    write_json(out / "progress.json", {"phase": "finished", **summary})
    write_json(out / "manifest.json", {"version": "competition_panel_source_v1", "dataset_revision": config.dataset_revision,
        "data_config": {"split": config.split, "seed": draw_seed, "augmentation_seed": config.augmentation_seed},
        "split_sha256": contract["split_sha256"], "question": contract["question"],
        "base_source_manifest_sha256": contract["source_manifest_sha256"], "family_provenance_path": "family_provenance.parquet",
        "artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    return summary
