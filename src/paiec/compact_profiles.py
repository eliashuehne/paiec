"""Versioned compact research facts and harness-scoped identity evidence.

Legacy enrichment sources remain unchanged for existing checkpoint contracts.
Only documented facts enter text. Source evidence stays in the audit sidecar.
"""
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from .profile_enrichment import EnrichmentPolicy, ProfileEnricher, catalog_fingerprint
from .state_rendering import split_settings
from .system_profiles import Evidence, StrictModel, load_catalog


class ScopedAlias(StrictModel):
    harness: str
    source_model: str
    profile_id: str
    basis: Literal["reported_model_family", "base_family"] = "reported_model_family"
    evidence: list[Evidence] = Field(min_length=1)
    note: str


class CompactPolicy(EnrichmentPolicy):
    version: Literal["compact_profiles_v3"] = "compact_profiles_v3"
    scoped_aliases: list[ScopedAlias] = Field(default_factory=list)


class CompactProfiles(ProfileEnricher):
    """Reuse conservative legacy matching, adding explicitly reviewed scoped rules."""
    def __init__(self, policy_path, *, root=None):
        self.policy_path = Path(policy_path).resolve()
        self.policy = CompactPolicy.model_validate_json(self.policy_path.read_text())
        self.catalog = (Path(root or Path(__file__).resolve().parents[2]) / self.policy.catalog).resolve()
        profiles, sources = load_catalog(self.catalog)
        self.profiles = {p.profile_id: p for p in profiles}
        self.sources = {s.source_id: s for s in sources}
        self.models = self._aliases(self.policy.models, "model")
        self.harnesses = self._aliases(self.policy.harnesses, "harness")
        keys = set()
        for rule in self.policy.scoped_aliases:
            key = (rule.harness.casefold(), rule.source_model.casefold())
            if key in keys or self.profiles[rule.profile_id].entity_type != "model":
                raise ValueError("Duplicate or invalid scoped model alias")
            keys.add(key)
            for evidence in rule.evidence:
                source = self.sources[evidence.source_id]
                if source.eligibility == "not_permitted" or not source.sha256 or not source.revision:
                    raise ValueError("Scoped alias needs eligible, pinned primary evidence")
        self.metadata = {"version": self.policy.version, "usage": "research_only",
                         "catalog_sha256": catalog_fingerprint(self.catalog),
                         "policy_sha256": hashlib.sha256(self.policy_path.read_bytes()).hexdigest(),
                         "source_eligibility": "unreviewed for competition"}

    def resolve(self, subject):
        match = super().resolve(subject)
        match["alias_evidence"] = []
        if match["model_profile_id"]:
            return match
        entries = split_settings(subject.get("subject_features_extra", "")) or []
        explicit = [(key, value.strip()) for key, value in entries
                    if key in {"source_model", "model_identifier", "model_id", "model_name", "base_model", "checkpoint", "reported_model_path"}]
        # Multiple identifiers or a recognized canonical name must not be
        # disambiguated by a weaker study alias.
        if len(explicit) != 1 or explicit[0][0] != "source_model":
            return match
        canonical = subject.get("normalized_name", "").strip()
        if canonical:
            return match
        if subject.get("harness_version", "").strip():
            # These reviewed rules describe an unversioned recorded study.
            # A different explicitly supplied revision needs its own review.
            return match
        for rule in self.policy.scoped_aliases:
            if (subject.get("harness", "").strip().casefold() == rule.harness.casefold()
                    and explicit[0][1].casefold() == rule.source_model.casefold()):
                match.update(model_profile_id=rule.profile_id,
                             model_identity_basis=("documented_harness_scoped_base_family" if rule.basis == "base_family"
                                                   else "documented_harness_scoped_alias"),
                             alias_evidence=[e.model_dump() for e in rule.evidence])
                break
        return match

    def candidates(self, subject):
        match = self.resolve(subject)
        groups = {"model": [], "harness": []}
        labels = {"architecture.total_parameters": "parameters≈",
                  "architecture.active_parameters_per_token": "active≈",
                  "architecture.parameterization": "architecture=", "model_type": "type=",
                  "input_modalities": "inputs=", "output_modalities": "outputs=",
                  "architecture.backbone": "backbone=", "architecture.family": "family=",
                  "architecture.attention_types": "attention=", "architecture.layers": "layers=",
                  "architecture.hidden_size": "hidden=", "architecture.attention_heads": "heads=",
                  "architecture.key_value_heads": "KV heads=", "architecture.experts_total": "experts=",
                  "architecture.experts_active_per_token": "active experts="}
        pid = match["model_profile_id"]
        if pid:
            for field in self.policy.selected_model_fields:
                if match["model_identity_basis"] == "base_model" and not field.startswith("architecture."):
                    continue
                fact = self._fact_line(pid, field, field)
                if not fact:
                    continue
                original = self.profiles[pid].specifications
                for part in field.split("."):
                    original = getattr(original, part)
                value = original.value
                if field.endswith(("total_parameters", "active_parameters_per_token")):
                    value = f"{value / 1e9:g}B"
                elif isinstance(value, list):
                    value = "/".join(map(str, value))
                value = {"mixture_of_experts": "MoE"}.get(str(value), value)
                label = labels[field]
                if "language-model component" in original.note.casefold():
                    label = "language component " + label
                prefix = "Base model: " if match["model_identity_basis"] in {"base_model", "documented_harness_scoped_base_family"} else "Model: "
                groups["model"].append(dict(fact, kind="model", line=prefix + label + str(value)))
        pid = match["harness_profile_id"]
        if pid and (fact := self._fact_line(pid, "workflow", "workflow")):
            # Select complete documented list entries, never a clipped fragment
            # that could change the meaning of a supported workflow.
            for value in self.profiles[pid].specifications.workflow.value:
                groups["harness"].append(dict(fact, kind="harness", line="Harness may support: " + value))
        return groups, match

    def block(self, subject, count):
        groups, match = self.candidates(subject)
        heading = "Published capabilities (version unverified; tools not confirmed enabled)"
        selected = []
        # Give each known component a first fact before adding deeper dimensions.
        ordered = groups["model"][:1] + groups["harness"][:1] + groups["model"][1:] + groups["harness"][1:]
        for candidate in ordered:
            trial = heading + "\n" + "\n".join(f["line"] for f in selected + [candidate])
            if count(trial) <= self.policy.max_added_tokens:
                selected.append(candidate)
        text = heading + "\n" + "\n".join(f["line"] for f in selected) if selected else ""
        available = {key: bool(value) for key, value in groups.items()}
        delivered = {key: any(f["kind"] == key for f in selected) for key in groups}
        if any(available[k] and not delivered[k] for k in available):
            raise ValueError("Compact fact budget cannot fit one fact for each known component")
        return text, {**match, **self.metadata, "available": available, "delivered": delivered,
                      "selected_facts": selected, "tokens": count(text)}
