"""Conservative research identity matching, independent of prediction renderers."""
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schema import SUBJECT_FIELDS
from .text import split_settings

IDENTIFIER_KEYS = {"model_identifier", "source_model", "model_id", "model_name"}
SCOPED_IDENTIFIER_KEYS = IDENTIFIER_KEYS | {"base_model", "checkpoint", "reported_model_path"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Alias(StrictModel):
    profile_id: str
    identifiers: list[str] = Field(min_length=1)


class Evidence(StrictModel):
    source_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class ScopedAlias(StrictModel):
    harness: str
    source_model: str
    profile_id: str
    basis: Literal["reported_model_family", "base_family"] = "reported_model_family"
    evidence: list[Evidence] = Field(min_length=1)
    note: str


class MatchingPolicy(StrictModel):
    version: Literal["profile_matching_v1"] = "profile_matching_v1"
    models: list[Alias]
    harnesses: list[Alias]
    scoped_aliases: list[ScopedAlias] = Field(default_factory=list)


class ScopedEvidenceSource(BaseModel):
    """Validate only the registry metadata consumed by a scoped identity rule."""
    model_config = ConfigDict(extra="ignore", strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision: str = Field(min_length=1)
    eligibility: Literal["inventory_source", "unreviewed"] = "unreviewed"
    source_type: Literal["benchmark_table", "model_card", "model_config",
                         "technical_report", "harness_documentation", "harness_code"]


def fingerprint(value):
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


class ProfileMatcher:
    def __init__(self, policy: MatchingPolicy, export: dict):
        self.policy = policy
        self.profiles = export["profiles"]
        self.sources = export["sources"]
        self.models = self._aliases(policy.models, "model")
        self.harnesses = self._aliases(policy.harnesses, "harness")
        keys = set()
        for rule in policy.scoped_aliases:
            key = (rule.harness.casefold(), rule.source_model.casefold())
            if key in keys or rule.profile_id not in self.profiles or self.profiles[rule.profile_id]["entity_type"] != "model":
                raise ValueError("Duplicate or invalid scoped model alias")
            keys.add(key)
            for evidence in rule.evidence:
                ScopedEvidenceSource.model_validate(self.sources.get(evidence.source_id, {}))
        self.metadata = {"version": policy.version, "usage": "research_only",
                         "catalog_sha256": fingerprint(export), "policy_sha256": fingerprint(policy.model_dump()),
                         "source_eligibility": "unreviewed for competition"}

    def _aliases(self, aliases, kind):
        result = {}
        for entry in aliases:
            if entry.profile_id not in self.profiles or self.profiles[entry.profile_id]["entity_type"] != kind:
                raise ValueError("Alias references a missing or wrong-kind profile")
            for identifier in entry.identifiers:
                key = identifier.strip().casefold()
                if key in result and result[key] != entry.profile_id:
                    raise ValueError("Ambiguous declared profile alias")
                result[key] = entry.profile_id
        return result

    def resolve(self, subject):
        # Never inspect display names, private IDs, benchmark identity or task text.
        subject = {key: subject.get(key, "") for key in SUBJECT_FIELDS}
        entries = split_settings(subject["subject_features_extra"]) or []
        explicit = [value.strip() for key, value in entries if key in IDENTIFIER_KEYS]
        base = [value.strip() for key, value in entries if key == "base_model"]
        normalized = subject["normalized_name"].strip()
        identifiers = explicit or base or ([normalized] if normalized else [])
        matches = {self.models.get(value.casefold()) for value in identifiers}
        matches.discard(None)
        canonical = self.models.get(normalized.casefold())
        if explicit and canonical:
            matches.add(canonical)
        model = next(iter(matches)) if len(matches) == 1 else None
        if (explicit or base) and any(value.casefold() not in self.models for value in identifiers):
            model = None
        match = {"model_profile_id": model,
                 "harness_profile_id": self.harnesses.get(subject["harness"].strip().casefold()),
                 "model_identity_basis": "explicit" if explicit else "base_model" if base else "normalized_name",
                 "exact_evaluated_checkpoint_verified": False, "alias_evidence": []}
        if model:
            return match
        scoped = [(key, value.strip()) for key, value in entries if key in SCOPED_IDENTIFIER_KEYS]
        if len(scoped) != 1 or scoped[0][0] != "source_model" or normalized or subject["harness_version"].strip():
            return match
        for rule in self.policy.scoped_aliases:
            if subject["harness"].strip().casefold() == rule.harness.casefold() and scoped[0][1].casefold() == rule.source_model.casefold():
                match.update(model_profile_id=rule.profile_id,
                             model_identity_basis="documented_harness_scoped_base_family" if rule.basis == "base_family" else "documented_harness_scoped_alias",
                             alias_evidence=[e.model_dump() for e in rule.evidence])
                break
        return match
