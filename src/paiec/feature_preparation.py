"""Outcome-blind feature tables around an immutable compact-v3 preparation.

No native model, cloud client, annotation API or training code is invoked here.
New tables are research sidecars, not a replacement checkpoint input contract.
"""
from collections import Counter
from html import escape
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .competition import competition_input
from .compact_profiles import CompactProfiles
from .head_data import task_key
from .head_training import file_hash
from .laya_predictor import load_questions
from .schema import ITEM_FIELDS, SUBJECT_FIELDS
from .state_rendering import split_settings


class FeaturePreparationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["feature_preparation_v1"] = "feature_preparation_v1"
    prepared: str = "runs/head_finetuning/enriched_30k_v1"
    roles: list[Literal["training", "monitor", "development"]] = ["training", "monitor", "development"]

    @model_validator(mode="after")
    def valid_paths_and_roles(self):
        relative_run_path(self.prepared)
        if not self.roles or len(set(self.roles)) != len(self.roles) or "training" not in self.roles:
            raise ValueError("Use unique preparation roles including training")
        return self


def relative_run_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "runs":
        raise ValueError("Expected a repository-relative directory under runs/")
    return path


def local_path(root, value):
    path = (Path(root) / relative_run_path(value)).resolve()
    if not path.is_relative_to((Path(root) / "runs").resolve()):
        raise ValueError("Run path escapes repository storage")
    return path


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_inputs(prepared, roles):
    """Only the permitted state fields are yielded; expected labels are ignored."""
    question = load_questions()
    for role in roles:
        for index, row in enumerate(read_jsonl(Path(prepared) / "source" / f"{role}.jsonl")):
            if row["questions"] != question:
                raise ValueError("Source question differs from the active prediction task")
            state = json.loads(row["state"])
            if set(state) != {"target_system", "target_item"}:
                raise ValueError("Unexpected source state fields")
            subject, item = state["target_system"], state["target_item"]
            if not set(subject) <= set(SUBJECT_FIELDS) or not set(item) <= set(ITEM_FIELDS):
                raise ValueError("Source fields differ from the competition boundary")
            # Historical development states omit empty fields and benchmark ID.
            # Restore only empty values; do not recover identity from provenance.
            if any(not isinstance(value, str) for record in [subject, item] for value in record.values()):
                raise ValueError("Source fields must be strings")
            subject, item = competition_input(subject, item)
            yield role, index, subject, item


def verify_inputs(root, config):
    prepared = local_path(root, config.prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    root = Path(root).resolve()
    split = (root / manifest["data_config"]["split"]).resolve()
    if not split.is_relative_to(root) or file_hash(split) != manifest["split_sha256"]:
        raise ValueError("Frozen split changed")
    for name, expected in manifest["artifacts"].items():
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared) or file_hash(path) != expected:
            raise ValueError("Prepared artifact changed: " + name)
    for name, expected in manifest["renderer_sources"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError("Prepared renderer changed: " + name)
    if manifest["question"] != load_questions():
        raise ValueError("Active question changed")
    for role in config.roles:
        if f"source/{role}.jsonl" not in manifest["artifacts"]:
            raise ValueError("Source role is not covered by the immutable manifest")
    if "profile_bundle/policy.json" not in manifest["artifacts"]:
        raise ValueError("Prepared profile policy is not hash-pinned")
    return {"root": str(Path(root).resolve()), "prepared": str(prepared), "manifest_sha256": file_hash(prepared / "manifest.json"),
            "dataset_revision": manifest["dataset_revision"], "split_sha256": manifest["split_sha256"],
            "config": config.model_dump()}


def build_systems(inputs, out):
    """Match using visible fields only, using the original bundled policy."""
    prepared = Path(inputs["prepared"])
    profiles = CompactProfiles(prepared / "profile_bundle/policy.json", root=prepared)
    systems = {}
    for _, _, subject, _ in source_inputs(prepared, inputs["config"]["roles"]):
        key = stable_hash(subject)
        if key not in systems:
            systems[key] = {"system_key": key, "recorded": subject, "resolution": profiles.resolve(subject)}
    path = Path(out) / "systems.jsonl"
    write_jsonl(path, (systems[key] for key in sorted(systems)))
    return {"path": str(path), "systems": len(systems),
            "model_resolved": sum(bool(s["resolution"]["model_profile_id"]) for s in systems.values()),
            "harness_resolved": sum(bool(s["resolution"]["harness_profile_id"]) for s in systems.values())}


def build_profiles(inputs, systems, out, kind):
    if kind not in {"model", "harness"}:
        raise ValueError("Expected model or harness profiles")
    prepared = Path(inputs["prepared"])
    catalog = CompactProfiles(prepared / "profile_bundle/policy.json", root=prepared)
    keys = {r["resolution"][kind + "_profile_id"] for r in read_jsonl(systems["path"])} - {None}
    records = [catalog.profiles[key].model_dump(mode="json") for key in sorted(keys)]
    path = Path(out) / (kind + "_profiles.jsonl")
    write_jsonl(path, records)
    if kind == "harness":
        write_jsonl(Path(out) / "recorded_runs.jsonl", (
            {"system_key": row["system_key"], "recorded": row["recorded"],
             "extra_settings": split_settings(row["recorded"]["subject_features_extra"]) or [],
             "harness_profile_id": row["resolution"]["harness_profile_id"],
             "unknown_settings_are_disabled": False}
            for row in read_jsonl(systems["path"])))
    # Evidence IDs remain resolvable without fetching source documents.
    if kind == "model":
        write_jsonl(Path(out) / "profile_sources.jsonl", (
            catalog.sources[key].model_dump(mode="json") for key in sorted(catalog.sources)))
    return {"path": str(path), "profiles": len(records)}


def task_observables(item):
    """Deterministic measurements, not skill/difficulty or language assertions."""
    text = item["item_content"]
    return {"content_characters": len(text), "whitespace_words": len(text.split()),
            "numeric_spans": len(re.findall(r"\d+(?:\.\d+)?", text)),
            "has_code_fence": "```" in text,
            "has_cjk_characters": bool(re.search(r"[\u3400-\u9fff]", text)),
            "has_cyrillic_characters": bool(re.search(r"[\u0400-\u04ff]", text)),
            "has_arabic_characters": bool(re.search(r"[\u0600-\u06ff]", text)),
            "has_supplied_interactors": bool(item["interactors"]),
            "has_supplied_item_features": bool(item["item_features"])}


def build_tasks(inputs, systems, out):
    known_systems = {r["system_key"] for r in read_jsonl(systems["path"])}
    tasks, task_roles, counts = {}, {}, Counter()
    with (Path(out) / "examples.jsonl").open("w") as links:
        for role, index, subject, item in source_inputs(inputs["prepared"], inputs["config"]["roles"]):
            system_key = stable_hash(subject)
            if system_key not in known_systems:
                raise ValueError("Unresolved system linkage")
            task_hash = stable_hash(task_key(item["item_content"]))
            task_roles.setdefault(task_hash, set()).add(role)
            if len(task_roles[task_hash]) > 1:
                raise ValueError("A normalized task spans preparation roles")
            visible = {key: value for key, value in item.items() if key != "benchmark_id"}
            key = stable_hash(visible)
            if key not in tasks:
                tasks[key] = {"task_key": key, "normalized_content_sha256": task_hash,
                              "recorded": visible, "observed": task_observables(visible),
                              "assessment": {"status": "pending", "required_skills": None,
                                             "language": None, "reasoning_structure": None,
                                             "required_modalities": None, "difficulty": None}}
            links.write(json.dumps({"role": role, "source_line": index, "system_key": system_key,
                                    "task_key": key, "benchmark_id": item["benchmark_id"]}) + "\n")
            counts[role] += 1
    path = Path(out) / "tasks.jsonl"
    write_jsonl(path, (tasks[key] for key in sorted(tasks)))
    return {"path": str(path), "tasks": len(tasks), "normalized_contents": len(task_roles),
            "records_by_role": dict(counts), "labels_exported": False, "cross_role_task_overlap": 0}


FEATURE_PLAN = [
    {"bucket": "model", "status": "existing_sourced_facts", "next": "Review identity conflicts and missing specialization/language/limit facts."},
    {"bucket": "demonstrated_abilities", "status": "not_implemented", "next": "Estimate from eligible training outcomes with out-of-fold features; attach support counts."},
    {"bucket": "harness_and_run", "status": "existing_facts_and_recorded_settings", "next": "Map raw settings to typed enabled tools, prompting, selection and budgets; preserve unknowns."},
    {"bucket": "task_requirements", "status": "observables_only", "next": "Annotate skills, language, modalities and reasoning structure; review quality before bulk generation."},
    {"bucket": "system_task_compatibility", "status": "not_implemented", "next": "Derive interactions from verified capabilities and reviewed task requirements."},
    {"bucket": "revealed_evidence", "status": "not_implemented_in_this_pipeline", "next": "Prepare training-only support/query episodes at nested budgets, with query labels excluded."},
]


def finalize_bundle(inputs, systems, models, harnesses, tasks, out, *, collection=None):
    """Publish only a validated preparation, never an implicit training command."""
    out = Path(out)
    # Recheck inputs at publication so paths cannot silently point to changed data.
    fresh = verify_inputs(inputs["root"], FeaturePreparationConfig.model_validate(inputs["config"]))
    if fresh["manifest_sha256"] != inputs["manifest_sha256"]:
        raise ValueError("Preparation manifest changed during the pipeline")
    plan = {"feature_groups": FEATURE_PLAN, "new_laya_training_ready": False,
            "remaining": ["Reviewed task annotation and empirical capability builders",
                          "A versioned feature-to-Laya input contract and renderer",
                          "Matched controls and native token/visibility qualification"],
            "training_launched": False, "inference_performed": False, "api_calls": 0,
            "sealed_roles_opened": False, "existing_training_contract_changed": False}
    write_json(out / "feature_plan.json", plan)
    summary = {"version": "feature_preparation_v1", "inputs": inputs,
               "systems": systems, "models": models, "harnesses": harnesses, "tasks": tasks, "collection": collection, **plan}
    write_json(out / "summary.json", summary)
    report = '<!doctype html><meta charset="utf-8"><title>Laya feature preparation</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;line-height:1.5}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
    report += '<h1>Laya feature preparation</h1><p>Preparation complete. New feature-based Laya training is not ready. Existing checkpoints and inputs are unchanged.</p>'
    report += '<h2>Dataset</h2><pre>' + escape(json.dumps({"records_by_role": tasks["records_by_role"], "systems": systems["systems"], "models": models["profiles"], "harnesses": harnesses["profiles"], "unique_task_inputs": tasks["tasks"], "normalized_contents": tasks["normalized_contents"]}, indent=2)) + '</pre>'
    report += '<h2>Feature buckets</h2><table><tr><th>Bucket</th><th>Current status</th><th>Next work</th></tr>'
    for row in FEATURE_PLAN:
        report += '<tr>' + ''.join('<td>' + escape(row[key]) + '</td>' for key in ["bucket", "status", "next"]) + '</tr>'
    report += '</table><h2>Next gates</h2><ul>' + ''.join('<li>' + escape(item) + '</li>' for item in plan["remaining"]) + '</ul>'
    if collection:
        report += '<h2>Collection and review starter</h2><pre>' + escape(json.dumps(collection, indent=2)) + '</pre><p>Candidate annotations are agent-authored estimates with exact task spans. They have not been independently reviewed. Static facts reuse pinned sources; new source collection remains queued.</p>'
    (out / "report.html").write_text(report)
    write_json(out / "manifest.json", {"version": "feature_preparation_v1", "input_manifest_sha256": inputs["manifest_sha256"],
               "artifacts": {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return summary
