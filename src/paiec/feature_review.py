"""Collection review records, kept separate from unknown predictor facts.

A source review can explain an unknown value without turning it into a known
value. Closing a review is distinct from establishing a capability or run setting.
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .feature_preparation import read_jsonl, stable_hash, write_json, write_jsonl
from .head_training import file_hash
from .state_rendering import split_settings
from .feature_sources import source_document_key


class CheckedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1)
    sha256: str = Field(pattern="^[a-f0-9]{64}$")
    reviewed_scope: str = Field(min_length=1)


class SourceReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["source_review_v1"] = "source_review_v1"
    entity_type: Literal["model", "harness"]
    profile_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    fact_value_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    disposition: Literal["value_confirmed", "not_established_in_reviewed_sources",
                         "identity_ambiguous", "conflict_unresolved"]
    rationale: str = Field(min_length=1)
    checked_documents: list[CheckedDocument] = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    independent_review: bool = False


class IdentityReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["identity_review_v1"] = "identity_review_v1"
    system_key: str = Field(pattern="^[a-f0-9]{64}$")
    entity_type: Literal["model", "harness"]
    checked_subject_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    disposition: Literal["not_supplied", "reported_private_checkpoint_unverified"]
    rationale: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    independent_review: bool = False


def audit_identity_reviews(root, queue, systems, out, *, ledger=None):
    root, out = Path(root).resolve(), Path(out)
    reviews = {}
    ledger_hash = None
    if ledger:
        path = (root / ledger).resolve()
        if not path.is_relative_to(root / "research"):
            raise ValueError("Identity review ledger escapes research storage")
        ledger_hash = file_hash(path)
        for raw in read_jsonl(path):
            review = IdentityReview.model_validate(raw)
            key = (review.system_key, review.entity_type)
            if key in reviews:
                raise ValueError("Duplicate identity review")
            reviews[key] = review
    rows = []
    for request in queue:
        if request["field"] != "identity":
            continue
        key = (request["system_key"], request["entity_type"])
        subject = systems[key[0]]["recorded"]
        if stable_hash(subject) != key[0]:
            raise ValueError("Identity review visible-subject hash mismatch")
        review = reviews.get(key)
        state = "pending" if not review else (
            "reviewed" if review.checked_subject_sha256 == stable_hash(subject) else "stale_subject_changed")
        if state == "reviewed" and review.disposition == "not_supplied":
            field = "normalized_name" if key[1] == "model" else "harness"
            if subject.get(field):
                raise ValueError("Identity is reported but review claims it was not supplied")
            settings = split_settings(subject.get("subject_features_extra", ""))
            aliases = ({"model_identifier", "source_model", "source_model_identifier", "reported_model", "reported_model_id"}
                       if key[1] == "model" else {"harness_identifier", "harness_name", "agent_harness"})
            if any(name in aliases and value.strip() for name, value in settings or []):
                raise ValueError("Extra settings report an identity the review did not account for")
        rows.append({"system_key": key[0], "entity_type": key[1], "field": "identity",
                     "review_status": state, "identity_resolved": False,
                     "checked_subject": subject, "review": review.model_dump(mode="json") if review else None})
    write_jsonl(out / "identity_review_status.jsonl", rows)
    summary = {"required_identity_requests": len(rows), "reviewed": sum(r["review_status"] == "reviewed" for r in rows),
               "pending": sum(r["review_status"] == "pending" for r in rows),
               "stale": sum(r["review_status"] == "stale_subject_changed" for r in rows),
               "unresolved": len(rows), "identities_invented": 0,
               "ledger_sha256": ledger_hash,
               "complete": all(r["review_status"] == "reviewed" for r in rows)}
    write_json(out / "identity_review_audit.json", summary)
    return summary


def fact_value_hash(fact):
    return stable_hash({"value": fact["value"], "status": fact["status"]})


def audit_recorded_setting_reviews(records, out):
    """Review the published run record, never substitute public-code defaults.

    These are mechanical reviews of available evidence, not independent human
    judgments or a claim that additional settings could never be recovered.
    Unrecognized containers/keys abstain so future formats need explicit review.
    """
    understood_keys = {"instruction_size", "protocol_reference", "serving_environment",
        "shot_size", "source_model", "model_identifier", "max_tokens", "structured_output",
        "base_model", "method", "checkpoint", "reported_model_path", "reported_task_generation_size",
        "n_shots", "selection", "source_study"}
    unknown_reasons = {
        "enabled_tools": "No verified enabled-tool list is recorded. Harness support, SQL output and serving environment do not establish enabled tools.",
        "few_shot_count": "No verified demonstration count is recorded. Instruction size, method names and harness defaults are not demonstration counts.",
        "temperature": "No verified sampling temperature is recorded. Provider or repository defaults are not actual run settings.",
        "max_output_tokens": "No verified output-token limit is established. A reported task generation size is retained literally and is not certified as a hard run limit.",
        "max_reasoning_tokens": "No separate reasoning-token budget is recorded. Output-token limits do not establish reasoning budgets.",
        "candidate_count": "No verified candidate count is recorded. Digits in method names or adaptation recipes do not establish candidate sampling counts.",
        "selection_method": "No verified candidate-selection rule is recorded. Harness functionality does not establish the rule used in this run.",
        "max_turns": "No verified maximum interaction-turn count is recorded. Single task text does not establish a one-turn execution limit.",
        "max_tool_calls": "No verified tool-call budget is recorded. A missing tool setting is not a zero budget.",
    }
    rows = []
    for record in records:
        subject = record["recorded"]
        if stable_hash(subject) != record["system_key"]:
            raise ValueError("Run review visible-subject identity mismatch")
        raw = subject.get("subject_features_extra", "")
        parsed = split_settings(raw) if raw else []
        unfamiliar = parsed is None or any(k not in understood_keys for k, _ in parsed)
        present_keys = {k for k, _ in parsed or []}
        for field, fact in record["features"].items():
            known = fact["status"] != "unknown"
            pending = unfamiliar or (not known and field not in unknown_reasons)
            if not known and present_keys & {"few_shot_count": {"shot_size", "n_shots"},
                                              "selection_method": {"selection"},
                                              "max_output_tokens": {"max_tokens"}}.get(field, set()):
                pending = True
            rows.append({"system_key": record["system_key"], "field": field,
                         "value_known": known, "review_status": "pending" if pending else "reviewed",
                         "method": "published_record_review_v1", "independent_review": False,
                         "fact_value_sha256": fact_value_hash(fact),
                         "checked_subject_sha256": stable_hash(subject),
                         "checked_fields": sorted(subject),
                         "disposition": "needs_format_review" if pending else ("recorded_value" if known else "not_established_in_published_record"),
                         "rationale": "Unrecognized recorded settings require review." if pending else (
                             "Recovered from a source-verified profile with an identical permitted subject description." if known else unknown_reasons[field]),
                         "scope": "Published subject and its exact-match recorded-run profile; external defaults cannot establish actual settings."})
    out = Path(out)
    write_jsonl(out / "run_setting_reviews.jsonl", rows)
    summary = {"required_run_fields": len(rows), "reviewed": sum(r["review_status"] == "reviewed" for r in rows),
               "pending": sum(r["review_status"] == "pending" for r in rows),
               "known_values": sum(r["value_known"] for r in rows),
               "unknown_values_promoted_to_known": 0, "independently_reviewed": 0,
               "complete": bool(rows) and all(r["review_status"] == "reviewed" for r in rows)}
    write_json(out / "run_setting_review_audit.json", summary)
    return summary


def audit_source_reviews(root, queue, out, *, source_registry):
    """Verify review scope/bytes and report pending work without filling values."""
    root, out = Path(root).resolve(), Path(out)
    registry = (root / source_registry).resolve()
    if not registry.is_relative_to(root / "configs/pipelines"):
        raise ValueError("Source registry escapes pipeline configuration storage")
    config = json.loads(registry.read_text()) if registry.exists() else {}
    reviews = {}
    ledger_hashes = []
    for bundle in config.get("review_ledgers", []):
        catalog, cache = (root / bundle["catalog"]).resolve(), (root / bundle["cache"]).resolve()
        if not catalog.is_relative_to(root / "research") or not cache.is_relative_to(root / "runs"):
            raise ValueError("Review bundle escapes collection storage")
        ledger = (catalog / bundle["reviews"]).resolve()
        if not ledger.is_relative_to(catalog):
            raise ValueError("Review ledger escapes collection storage")
        sources = {s["source_id"]: s for s in read_jsonl(catalog / "sources.jsonl")}
        snapshots = {s["source_id"]: s for s in json.loads((catalog / "snapshots.json").read_text())}
        ledger_hashes.append({"path": str(ledger.relative_to(root)), "sha256": file_hash(ledger)})
        for raw in read_jsonl(ledger):
            review = SourceReview.model_validate(raw)
            key = (review.entity_type, review.profile_id, review.field)
            if key in reviews:
                raise ValueError("Duplicate field source review")
            for document in review.checked_documents:
                source, snapshot = sources[document.source_id], snapshots[document.source_id]
                path = (cache / snapshot["cache_file"]).resolve()
                if not path.is_relative_to(cache):
                    raise ValueError("Review archive escapes collection storage")
                if source.get("eligibility") == "not_permitted":
                    raise ValueError("Review uses an ineligible source")
                if len({document.sha256, source["sha256"], snapshot["sha256"], file_hash(path)}) != 1:
                    raise ValueError("Reviewed source snapshot changed")
            reviews[key] = review
    rows, reviewed_features = [], []
    for request in queue:
        if "profile_id" not in request or "fact" not in request:
            continue
        key = (request["entity_type"], request["profile_id"], request["field"])
        fact = request["fact"]
        review = reviews.get(key)
        state = "pending"
        if review:
            if review.fact_value_sha256 != fact_value_hash(fact):
                state = "stale_value_changed"
            elif (fact["status"] == "unknown" and review.disposition == "value_confirmed") or (
                fact["status"] != "unknown" and review.disposition not in {"value_confirmed", "conflict_unresolved"}):
                raise ValueError("Review disposition contradicts fact status")
            else:
                state = "reviewed"
        exported_review = review.model_dump(mode="json") if review else None
        if exported_review:
            for document in exported_review["checked_documents"]:
                document["source_id"] = source_document_key(document["source_id"], document["sha256"])
        rows.append({"entity_type": key[0], "profile_id": key[1], "field": key[2],
                     "value_known": fact["status"] != "unknown", "review_status": state,
                     "value_confirmed": state == "reviewed" and review.disposition == "value_confirmed",
                     "review": exported_review})
        confirmed = state == "reviewed" and review.disposition == "value_confirmed"
        usable = dict(fact) if confirmed else {"value": None, "status": "unknown", "evidence": [],
            "note": "The source review does not confirm a usable value for this field; retain raw facts and the review separately."}
        reviewed_features.append({"entity_type": key[0], "profile_id": key[1], "field": key[2],
            "fact": usable, "review_status": state, "review_disposition": review.disposition if review else None,
            "usage": "research_only", "recorded_run_setting": False})
    write_jsonl(out / "source_review_status.jsonl", rows)
    write_jsonl(out / "reviewed_static_features.jsonl", reviewed_features)
    summary = {"required_static_fields": len(rows),
               "reviewed": sum(r["review_status"] == "reviewed" for r in rows),
               "pending": sum(r["review_status"] == "pending" for r in rows),
               "stale": sum(r["review_status"] == "stale_value_changed" for r in rows),
               "known_values": sum(r["value_known"] for r in rows),
               "confirmed_known_values": sum(r["value_confirmed"] for r in rows),
               "reviewed_conflicts": sum(r["review_status"] == "reviewed" and r["review"]["disposition"] == "conflict_unresolved" for r in rows),
               "independently_reviewed": sum(r["review_status"] == "reviewed" and r["review"]["independent_review"] for r in rows),
               "ledger_hashes": ledger_hashes,
               "acceptance_rule": "Complete source review; retain evidenced unknowns",
               "unknown_values_promoted_to_known": 0,
               "complete": bool(rows) and all(r["review_status"] == "reviewed" for r in rows)}
    write_json(out / "source_review_audit.json", summary)
    return summary
