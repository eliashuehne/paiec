"""Full research-export projection; no external fact enters supplied-only views."""
import json
from pathlib import Path

from .contracts import canonical_json

BASE_RELATIONS = {"base_model", "documented_harness_scoped_base_family"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_export(directory, *, required=True):
    directory = Path(directory)
    if not directory.is_dir():
        if required:
            raise FileNotFoundError("Full shared enrichment export is missing")
        return {"profiles": {}, "sources": {}, "catalog_mappings": [], "recorded_runs": []}
    profiles = {}
    for filename in ("models.jsonl", "harnesses.jsonl"):
        for profile in read_jsonl(directory / filename):
            if profile["profile_id"] in profiles:
                raise ValueError("Duplicate profile ID")
            profiles[profile["profile_id"]] = profile
    sources = {}
    for source in read_jsonl(directory / "sources.jsonl"):
        if source["source_id"] in sources:
            raise ValueError("Duplicate source ID")
        sources[source["source_id"]] = source
    return {"profiles": profiles, "sources": sources,
            "catalog_mappings": list(read_jsonl(directory / "catalog_systems.jsonl")),
            "recorded_runs": list(read_jsonl(directory / "runs.jsonl"))}


def project_fact(field, fact, relation, sources):
    evidence = fact.get("evidence", [])
    for ref in evidence:
        if ref["source_id"] not in sources:
            raise ValueError("Unresolvable fact source: " + ref["source_id"])
    known = fact.get("value") is not None and fact.get("status") in {"recorded", "documented"}
    reason = "documented_research_fact" if known else "unknown_or_unreviewed_fact"
    if known and not evidence:
        known, reason = False, "missing_evidence"
    if relation in BASE_RELATIONS and not field.startswith("architecture."):
        known, reason = False, "base_only_nonarchitecture"
    if any(sources[ref["source_id"]].get("eligibility") == "not_permitted" for ref in evidence):
        known, reason = False, "source_not_permitted"
    if fact.get("review_disposition") in {"identity_ambiguous", "value_rejected"}:
        known, reason = False, fact["review_disposition"]
    return {"field": field, "value_json": canonical_json(fact["value"]) if known else None,
            "status": "known" if known else "unknown", "reason": reason,
            "eligibility": "research_only", "historical_applicability": "unverified",
            "identity_basis": relation, "source_ids": [ref["source_id"] for ref in evidence],
            "review_disposition": fact.get("review_disposition", "not_reviewed")}


def feature_projection(rows, mode):
    if mode == "supplied_only":
        return []
    if mode != "research":
        raise ValueError("Unknown feature projection")
    return list(rows)


def resolve_enrichment(subjects, export, resolver=None):
    """Attach combined facts through the unchanged conservative matching policy."""
    matches, features, facts = [], [], []
    for pid, profile in sorted(export["profiles"].items()):
        for field, fact in sorted(profile["facts"].items()):
            # Validate every retained fact, including unmatched profiles.
            project_fact(field, fact, "inventory_only", export["sources"])
            facts.append({"profile_id": pid, "entity_type": profile["entity_type"], "field": field,
                          "fact": fact, "valid_from": profile.get("valid_from"), "valid_until": profile.get("valid_until"),
                          "eligibility": "research_only"})
    for subject in subjects:
        key = subject["system_key"]
        raw = {k: v for k, v in subject.items() if k != "system_key"}
        match = resolver.resolve(raw) if resolver else {"model_profile_id": None, "harness_profile_id": None, "model_identity_basis": "unresolved"}
        for entity in ("model", "harness"):
            pid = match[entity + "_profile_id"]
            basis = match.get("model_identity_basis", "unresolved") if entity == "model" else "supplied_harness_name"
            if pid and pid not in export["profiles"]:
                raise ValueError("Matched profile absent from full export")
            matches.append({"system_key": key, "entity_type": entity, "profile_id": pid,
                            "identity_basis": basis if pid else "unresolved", "status": "matched" if pid else "unresolved",
                            "exact_evaluated_checkpoint_verified": False, "eligibility": "research_only"})
            if pid:
                for field, fact in sorted(export["profiles"][pid]["facts"].items()):
                    features.append({"system_key": key, "profile_id": pid, "entity_type": entity,
                                     **project_fact(field, fact, basis, export["sources"])})
    return {"profile_matches": matches, "research_system_features": features, "facts": facts}
