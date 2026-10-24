"""Dataset-only preparation for an executable competition predictor.

Only supplied fields and training targets enter the new route. Research model
cards, alias catalogs, manual annotations and demonstration scores are unused.
"""
from collections import Counter
from datetime import date
from html import escape
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .feature_preparation import local_path, read_jsonl, stable_hash, write_json, write_jsonl, task_observables
from .feature_evidence_plan import prepare_evidence_plan
from .head_data import task_key
from .head_training import file_hash
from .laya_predictor import load_questions
from .schema import SUBJECT_FIELDS, ITEM_FIELDS
from .state_rendering import split_settings


class CompetitionDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["competition_data_v1"] = "competition_data_v1"
    source_policy: Literal["provided_dataset_only"] = "provided_dataset_only"
    prepared: str = "runs/head_finetuning/enriched_30k_v1"
    dataset_revision: str = "54074453da7b9911be0f48e4c560be547bc84d21"
    roles: list[Literal["training", "monitor", "development"]] = ["training", "monitor", "development"]

    @model_validator(mode="after")
    def selected_roles(self):
        if not self.roles or "training" not in self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("Use unique unsealed roles including training")
        return self


def visible_fields(subject, item, *, allow_missing=False):
    """Preserve every supplied character; do not clean literal null-like text."""
    result = []
    for record, fields in [(subject, SUBJECT_FIELDS), (item, ITEM_FIELDS)]:
        if not isinstance(record, dict) or not set(record) <= set(fields) or (not allow_missing and set(record) != set(fields)):
            raise ValueError("Unexpected competition input fields")
        if any(type(v) is not str for v in record.values()):
            raise ValueError("Competition field values must be strings")
        result.append({key: record.get(key, "") for key in fields})
    return result


def verify_dataset_sources(root, config):
    root = Path(root).resolve()
    config = CompetitionDataConfig.model_validate(config)
    prepared = local_path(root, config.prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["dataset_revision"] != config.dataset_revision:
        raise ValueError("Dataset revision differs from the selected competition release")
    split = (root / manifest["data_config"]["split"]).resolve()
    if not split.is_relative_to(root) or file_hash(split) != manifest["split_sha256"]:
        raise ValueError("Frozen split changed")
    split_data = json.loads(split.read_text())
    if split_data["dataset_revision"] != config.dataset_revision:
        raise ValueError("Split belongs to another dataset release")
    selected = {}
    for role in config.roles:
        name = f"source/{role}.jsonl"
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared) or name not in manifest["artifacts"] or file_hash(path) != manifest["artifacts"][name]:
            raise ValueError("Selected source changed or is not pinned: " + role)
        selected[role] = {"path": str(path), "sha256": manifest["artifacts"][name]}
    if manifest["question"] != load_questions():
        raise ValueError("Active prediction question changed")
    # No other artifact path is opened, including profile_bundle and sealed roles.
    return {"root": str(root), "prepared": str(prepared), "config": config.model_dump(),
            "source_manifest_sha256": file_hash(prepared / "manifest.json"),
            "dataset_revision": config.dataset_revision, "split_sha256": manifest["split_sha256"],
            "sources": selected, "question": manifest["question"]}


def recorded_system_features(subject):
    """Transparent proxies from supplied strings; no outside model knowledge."""
    name = subject["normalized_name"]
    # Keep nominal parameter text literal. An MoE label is not an active/total count.
    nominal = re.findall(r"(?<![\w.])(?:\d+(?:\.\d+)?x)?\d+(?:\.\d+)?\s*[bBmM](?!\w)", name)
    dates = {}
    for key in ["release_date", "access_date"]:
        try:
            dates[key] = date.fromisoformat(subject[key]).isoformat()
        except ValueError:
            dates[key] = None
    return {"nominal_parameter_text": nominal, "exact_parameter_count": None,
            "active_parameter_count": None, "dates_parseable_as_iso": dates,
            "identity_supplied": bool(name),
            "recorded_settings": split_settings(subject["subject_features_extra"]),
            "supported_tools_from_documentation": None, "outside_sources_used": False}


def build_dataset_tables(inputs, out):
    out = Path(out)
    if not out.is_dir():
        raise ValueError("Create a fresh output directory before building tables")
    systems, tasks, rows, targets, groups = {}, {}, [], [], {}
    counts = Counter()
    for role, source in inputs["sources"].items():
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("Source changed during preparation")
        for index, raw in enumerate(read_jsonl(source["path"])):
            if raw["questions"] != inputs["question"]:
                raise ValueError("Source prediction question differs")
            state = json.loads(raw["state"])
            if set(state) != {"target_system", "target_item"}:
                raise ValueError("Unexpected source state fields")
            subject, item = visible_fields(state["target_system"], state["target_item"], allow_missing=True)
            system_key = stable_hash(subject)
            visible_item = {key: item[key] for key in ITEM_FIELDS if key != "benchmark_id"}
            key, group = stable_hash(visible_item), stable_hash(task_key(item["item_content"]))
            groups.setdefault(group, set()).add(role)
            if len(groups[group]) != 1:
                raise ValueError("Normalized task overlaps preparation roles")
            systems.setdefault(system_key, {"system_key": system_key, "recorded": subject,
                                           "observed": recorded_system_features(subject)})
            tasks.setdefault(key, {"task_key": key, "normalized_content_sha256": group,
                                   "recorded": visible_item, "observed": task_observables(visible_item),
                                   "semantic_annotations_required": False})
            row = {"role": role, "source_line": index, "system_key": system_key,
                   "task_key": key, "benchmark_id": item["benchmark_id"], "task_group": group,
                   "scope_key": stable_hash({"system_key": system_key, "benchmark_id": item["benchmark_id"]})}
            row["row_key"] = stable_hash(row)
            rows.append(row)
            counts[role] += 1
            if role == "training":
                expected = raw.get("expected")
                if not isinstance(expected, dict) or set(expected) != {"correct"}:
                    raise ValueError("Training source requires exactly one native binary target")
                label = expected["correct"]
                if type(label) not in {bool, int} or label not in (0, 1):
                    raise ValueError("Training source target is not a raw binary value")
                targets.append({"row_key": row["row_key"], "native_binary_value": label,
                                "semantics": "recorded_native_event; not_reinterpreted"})
    write_jsonl(out / "systems.jsonl", (systems[key] for key in sorted(systems)))
    write_jsonl(out / "tasks.jsonl", (tasks[key] for key in sorted(tasks)))
    write_jsonl(out / "examples.jsonl", rows)
    write_jsonl(out / "training_targets.jsonl", targets)
    evidence = prepare_evidence_plan(tasks.values(), rows, out / "evidence_plan_v1")
    report = {"version": "competition_data_v1", "source_policy": "provided_dataset_only",
              "records_by_role": dict(counts), "systems": len(systems), "task_inputs": len(tasks),
              "normalized_task_groups": len(groups), "training_targets": len(targets),
              "training_targets_separate_from_inputs": True, "targets_exported_for_other_roles": False,
              "supplied_characters": sum(sum(map(len, r["recorded"].values())) for r in tasks.values()),
              "truncation_performed": False, "external_profile_facts_used": 0,
              "manual_task_annotations_used": 0, "semantic_annotations_required": False,
              "benchmark_ids_used_as_model_features": False,
              "model_training_performed": False, "inference_performed": False,
              "gpu_jobs": 0, "paid_calls": 0, "evidence_plan": evidence,
              "prediction_runtime_ready": False, "hosted_execution_verified": False,
              "full_objective_achieved": False}
    write_json(out / "source_contract.json", inputs)
    write_json(out / "summary.json", report)
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Dataset-only competition preparation</title><h1>Dataset-only competition preparation</h1><p>Permitted supplied fields and separate training targets. No external profiles, teacher annotations, training or inference. Runtime and validation remain unfinished.</p><pre>' + escape(json.dumps(report, indent=2)) + '</pre>')
    write_json(out / "manifest.json", {"version": "competition_data_v1", "source_contract_sha256": file_hash(out / "source_contract.json"),
               "artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    return report
