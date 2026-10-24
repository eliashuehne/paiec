"""Structured collection contracts and an outcome-blind, training-only review set."""
import json
from html import escape
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .feature_preparation import read_jsonl, write_json, write_jsonl, stable_hash, source_inputs
from .head_training import file_hash
from .compact_profiles import CompactProfiles
from .competition import competition_input
from .feature_review import audit_source_reviews, audit_recorded_setting_reviews, audit_identity_reviews


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class TaskClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    values: list[str] = Field(min_length=1)
    status: Literal["estimated"] = "estimated"
    evidence: list[Span] = Field(min_length=1)


class TaskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["task_assessment_v1"] = "task_assessment_v1"
    task_key: str = Field(pattern="^[0-9a-f]{64}$")
    method: Literal["codex_manual_v1"] = "codex_manual_v1"
    review_status: Literal["candidate", "reviewed", "rejected"] = "candidate"
    reviewer: str | None = None
    claims: dict[Literal["language", "domain", "skills", "reasoning_structure", "modalities",
                         "answer_format", "information_needs", "input_issues"], TaskClaim]
    # Empirical difficulty cannot be manufactured by a task annotator.
    difficulty: None = None

    @model_validator(mode="after")
    def review_metadata(self):
        if not self.claims:
            raise ValueError("An assessment needs grounded claims")
        if self.review_status != "candidate" and not self.reviewer:
            raise ValueError("A reviewed/rejected assessment needs a reviewer")
        return self

    def validate_against(self, task):
        if self.task_key != task["task_key"]:
            raise ValueError("Assessment task identity mismatch")
        text = task["recorded"]["item_content"]
        for claim in self.claims.values():
            for span in claim.evidence:
                if span.end <= span.start or text[span.start:span.end] != span.quote:
                    raise ValueError("Annotation evidence is not an exact task span")


def get_field(profile, dotted):
    value = profile["specifications"]
    for key in dotted.split("."):
        value = value[key]
    return value


def recover_recorded_run_features(records, profiles, fields):
    """Join only identical permitted descriptions, rejecting conflicting values."""
    matched = {}
    for profile in profiles:
        if profile.entity_type != "recorded_run":
            continue
        raw = profile.specifications.raw_supplied_subject
        if raw:
            key = stable_hash(competition_input(raw, {})[0])
            matched.setdefault(key, []).append(profile)
    output = []
    for key, record in sorted(records.items()):
        features = {}
        for field in fields:
            candidates = [getattr(p.specifications, field).model_dump(mode="json")
                          for p in matched.get(key, [])
                          if getattr(p.specifications, field).status in {"recorded", "documented"}]
            values = {stable_hash(fact["value"]) for fact in candidates}
            if len(values) == 1:
                fact = dict(candidates[0])
                fact["evidence"] = list({stable_hash(e): e for c in candidates for e in c["evidence"]}.values())
                features[field] = dict(fact, collection_status="recovered_from_exact_visible_match")
            else:
                features[field] = {"value": None, "status": "unknown", "evidence": [],
                                   "collection_status": "conflicting_visible_matches" if values else "not_recorded",
                                   "note": "No compatible verified setting; defaults are not actual run settings."}
        output.append({"system_key": key, "features": features, "recorded": record["recorded"]})
    return output


def build_collection_bundle(inputs, systems, models, harnesses, tasks, out):
    """Reuse pinned facts, create collection requests, and select 2 tasks/family.

    Family names are selection/audit metadata only, never annotation features.
    No target outcomes are read from provenance or passed to the annotator.
    """
    root, out = Path(inputs["root"]), Path(out)
    plan_path = root / "configs/pipelines/feature_collection_v1.json"
    plan = json.loads(plan_path.read_text())
    write_json(out / "collection_plan.json", plan)
    write_json(out / "task_assessment.schema.json", TaskAssessment.model_json_schema())
    records = {r["system_key"]: r for r in read_jsonl(systems["path"])}
    training_keys = {stable_hash(subject) for _, _, subject, _ in source_inputs(inputs["prepared"], ["training"])}
    scope = inputs["config"].get("collection_scope", "training_only")
    collection_keys = set(records) if scope == "all_selected_roles" else training_keys
    catalog = CompactProfiles(Path(inputs["prepared"]) / "profile_bundle/policy.json", root=inputs["prepared"])
    recovered = recover_recorded_run_features({key: records[key] for key in collection_keys},
                                              catalog.profiles.values(), plan["run_fields"])
    write_jsonl(out / "typed_run_features.jsonl", recovered)
    run_reviews = audit_recorded_setting_reviews(recovered, out)
    recovered_by_key = {r["system_key"]: r["features"] for r in recovered}
    queue = []
    for kind, result in [("model", models), ("harness", harnesses)]:
        used = {records[k]["resolution"][kind + "_profile_id"] for k in collection_keys} - {None}
        for profile in read_jsonl(result["path"]):
            if profile["profile_id"] not in used:
                continue
            for field in plan["static_fields"][kind]:
                fact = get_field(profile, field)
                queue.append({"entity_type": kind, "profile_id": profile["profile_id"], "field": field,
                              "fact": fact, "collection_status": "needs_source" if fact["status"] == "unknown" else "needs_applicability_review",
                              "priority": 1, "usage": "research_only"})
    for key in sorted(collection_keys):
        for field in plan["run_fields"]:
            queue.append({"entity_type": "recorded_run", "system_key": key, "field": field,
                          "collection_status": recovered_by_key[key][field]["collection_status"],
                          "fact": recovered_by_key[key][field], "priority": 1,
                          "usage": "research_only", "recorded": records[key]["recorded"]})
        for kind in ["model", "harness"]:
            if not records[key]["resolution"][kind + "_profile_id"]:
                queue.append({"entity_type": kind, "system_key": key, "field": "identity",
                              "collection_status": "unresolved_visible_identity", "priority": 1,
                              "usage": "research_only", "private_identity_lookup_allowed": False})
    write_jsonl(out / "collection_queue.jsonl", queue)
    source_review = audit_source_reviews(root, queue, out,
        source_registry=inputs["config"].get("source_registry", "configs/pipelines/feature_sources_v1.json"))
    identity_review = audit_identity_reviews(root, queue, records, out,
        ledger=inputs["config"].get("identity_review_ledger"))

    provenance = pd.read_parquet(Path(inputs["prepared"]) / "enriched/provenance.parquet",
                                 columns=["role", "jsonl_line", "family"])
    families = provenance[provenance.role.eq("training")].set_index("jsonl_line").family.to_dict()
    task_map = {t["task_key"]: t for t in read_jsonl(tasks["path"])}
    groups = {}
    for row in read_jsonl(out / "examples.jsonl"):
        if row["role"] == "training":
            groups.setdefault(families[row["source_line"]], {}).setdefault(row["task_key"], row["source_line"])
    selected = []
    for family, keys in sorted(groups.items()):
        keys = sorted(keys, key=lambda k: (len(task_map[k]["recorded"]["item_content"]), keys[k]))
        for key in dict.fromkeys([keys[0], keys[len(keys) // 2]]):
            selected.append({"family_for_audit_only": family, "selection": "shortest_and_median_length",
                             "task": task_map[key]})
    write_jsonl(out / "task_review_sample.jsonl", selected)
    seeds_path = root / "research/task_features/seed_v1.json"
    seeds = json.loads(seeds_path.read_text())
    candidates = []
    selected_keys = {r["task"]["task_key"] for r in selected}
    for seed in seeds:
        if seed["task_key"] not in selected_keys:
            raise ValueError("Seed annotations must belong to the training-only review sample")
        task = task_map[seed["task_key"]]
        text = task["recorded"]["item_content"]
        claims = {}
        for name, claim in seed["claims"].items():
            start = text.find(claim["quote"])
            if start < 0:
                raise ValueError("Seed evidence is not present in the task")
            claims[name] = TaskClaim(values=claim["values"], evidence=[Span(start=start, end=start+len(claim["quote"]), quote=claim["quote"])])
        assessment = TaskAssessment(task_key=seed["task_key"], claims=claims)
        assessment.validate_against(task)
        candidates.append(assessment.model_dump(mode="json"))
    write_jsonl(out / "task_annotation_candidates.jsonl", candidates)
    summary = {"collection_plan_sha256": file_hash(plan_path), "seed_sha256": file_hash(seeds_path),
               "collection_scope": scope, "systems_in_collection_scope": len(collection_keys),
               "collection_requests": len(queue), "missing_static_facts": sum(r["collection_status"] == "needs_source" for r in queue),
               "sample_tasks": len(selected), "candidate_annotations": len(candidates), "human_reviewed": 0,
               "selection_is_representative": False, "selection_note": "Length-stratified starter review only; not a representative quality estimate.",
               "source_snapshots_collected": models.get("source_collection", {}).get("source_snapshots", 0),
               "new_static_facts": sum(result.get("source_collection", {}).get("applied", 0) for result in [models, harnesses]),
               "annotation_api_calls": 0, "empirical_difficulty_fitted": False,
               "recovered_run_settings": sum(f["status"] != "unknown" for r in recovered for f in r["features"].values()),
               "unresolved_run_settings": sum(f["status"] == "unknown" for r in recovered for f in r["features"].values()),
               "training_ready": False}
    summary["source_review"] = source_review
    summary["run_setting_review"] = run_reviews
    summary["identity_review"] = identity_review
    summary["metadata_review_complete"] = all(review["complete"] for review in [source_review, run_reviews, identity_review])
    write_json(out / "collection_summary.json", summary)
    static = [r for r in queue if r["entity_type"] in {"model", "harness"} and r["field"] != "identity"]
    write_json(out / "coverage_audit.json", {
        "full_objective_achieved": False,
        "static_fields": {"required": len(static), "missing": sum(r["fact"]["status"] == "unknown" for r in static)},
        "source_review": source_review,
        "metadata_review_complete": summary["metadata_review_complete"],
        "unresolved_visible_identities": sum(r["collection_status"] == "unresolved_visible_identity" for r in queue),
        "identity_review": identity_review,
        "actual_run_fields": {"known": summary["recovered_run_settings"], "missing": summary["unresolved_run_settings"]},
        "run_setting_review": run_reviews,
        "task_annotations": {"task_inputs": len(task_map), "candidates": len(candidates), "independently_reviewed": 0},
        "demonstrated_abilities": "not_implemented",
        "learned_difficulty": "not_implemented",
        "system_task_compatibility": "not_implemented",
        "episodic_evidence_features": "not_implemented",
        "new_laya_input_contract": "not_implemented",
        "sealed_roles_opened": False,
    })
    html = '<!doctype html><meta charset="utf-8"><title>Feature collection review</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;line-height:1.5}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:16px;margin:12px 0}</style><h1>Feature collection review</h1><p>Training-only starter sample. Agent-authored candidates, not independently reviewed. No answers/outcomes or empirical difficulty are supplied.</p>'
    annotations = {r["task_key"]: r for r in candidates}
    for record in selected:
        task = record["task"]
        html += '<details><summary>' + escape(record["family_for_audit_only"] + ' / ' + task["task_key"][:12]) + '</summary><h3>Original task input</h3><pre>' + escape(task["recorded"]["item_content"]) + '</pre><h3>Candidate annotations</h3><pre>' + escape(json.dumps(annotations.get(task["task_key"], {"review_status": "unannotated"}), ensure_ascii=False, indent=2)) + '</pre></details>'
    (out / "collection_review.html").write_text(html)
    return summary
