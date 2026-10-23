"""Opt-in, offline research enrichment using only supplied system identifiers.

Catalog evidence stays in audit metadata. Only an allowlist of documented facts
enters the state. No benchmark, subject ID, display name, item or label lookup.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import SUBJECT_FIELDS
from .state_rendering import split_settings
from .system_profiles import load_catalog

FORMAT = "enriched_v1"
CATALOG_FILES = ("profiles.jsonl", "sources.jsonl")
MODEL_FIELDS = (
    "architecture.backbone", "architecture.parameterization",
    "architecture.total_parameters", "architecture.active_parameters_per_token",
    "architecture.attention_types", "architecture.layers", "architecture.hidden_size",
    "architecture.attention_heads", "architecture.key_value_heads",
    "architecture.experts_total", "architecture.experts_active_per_token",
    "input_modalities", "output_modalities", "architecture.family", "model_type",
)
IDENTIFIER_KEYS = ("model_identifier", "source_model", "model_id", "model_name")


class ProfileAlias(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    profile_id: str
    identifiers: list[str] = Field(min_length=1)


class EnrichmentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["enriched_v1"] = FORMAT
    usage: Literal["research_only"] = "research_only"
    catalog: str
    max_added_tokens: int = Field(default=96, ge=1, le=256)
    selected_model_fields: list[str]
    models: list[ProfileAlias]
    harnesses: list[ProfileAlias]

    @model_validator(mode="after")
    def supported_fields(self):
        if len(set(self.selected_model_fields)) != len(self.selected_model_fields):
            raise ValueError("Duplicate enrichment fields")
        if set(self.selected_model_fields) - set(MODEL_FIELDS):
            raise ValueError("Only reviewed stable model properties may be rendered")
        return self


def catalog_fingerprint(directory):
    digest = hashlib.sha256()
    for name in CATALOG_FILES:
        digest.update(name.encode())
        digest.update((Path(directory) / name).read_bytes())
    return digest.hexdigest()


def native_budget(tokenizer, question, max_len, backend, head_max_len=192):
    if backend == "torch":
        from laya.common import build_head
    elif backend == "mlx":
        from laya_mlx.common import build_head
    else:
        raise ValueError("Unknown native tokenizer backend")
    head, _, _ = build_head(tokenizer, {"t": question["type"],
        "ins": question["instructions"], "crit": None}, head_max_len)
    room = max_len - len(head) - 1
    if room <= 0:
        raise ValueError("No room for state after the question")

    def count(text):
        return len(tokenizer(text.replace(tokenizer.mask_token, " "),
                             add_special_tokens=False)["input_ids"])
    return count, room


class ProfileEnricher:
    def __init__(self, policy_path, *, root=None, token_count, state_room):
        self.policy_path = Path(policy_path).resolve()
        self.policy = EnrichmentPolicy.model_validate_json(self.policy_path.read_text())
        root = Path(root or Path(__file__).resolve().parents[2])
        self.catalog = (root / self.policy.catalog).resolve()
        profiles, sources = load_catalog(self.catalog)
        self.profiles = {p.profile_id: p for p in profiles}
        self.sources = {s.source_id: s for s in sources}
        self.token_count, self.state_room = token_count, state_room
        self.models = self._aliases(self.policy.models, "model")
        self.harnesses = self._aliases(self.policy.harnesses, "harness")
        self.metadata = {
            "state_format": FORMAT, "usage": "research_only",
            "catalog_sha256": catalog_fingerprint(self.catalog),
            "policy_sha256": hashlib.sha256(self.policy_path.read_bytes()).hexdigest(),
            "max_added_tokens": self.policy.max_added_tokens,
            "source_eligibility": "external sources remain unreviewed for competition",
        }

    def _aliases(self, aliases, kind):
        result = {}
        for entry in aliases:
            if self.profiles[entry.profile_id].entity_type != kind:
                raise ValueError("Alias references the wrong profile kind")
            for identifier in entry.identifiers:
                key = identifier.strip().casefold()
                if key in result and result[key] != entry.profile_id:
                    raise ValueError("Ambiguous declared profile alias")
                result[key] = entry.profile_id
        return result

    def resolve(self, subject):
        # Never inspect display_name, benchmark/subject IDs or item fields.
        subject = {key: subject.get(key, "") for key in SUBJECT_FIELDS}
        entries = split_settings(subject.get("subject_features_extra", "")) or []
        explicit = [value.strip() for key, value in entries if key in IDENTIFIER_KEYS]
        base = [value.strip() for key, value in entries if key == "base_model"]
        normalized = subject.get("normalized_name", "").strip()
        identifiers = explicit or base or ([normalized] if normalized else [])
        matches = {self.models.get(value.casefold()) for value in identifiers}
        matches.discard(None)
        # Conflicting recognized canonical and explicit identifiers fail closed.
        canonical = self.models.get(normalized.casefold())
        if explicit and canonical:
            matches.add(canonical)
        model = next(iter(matches)) if len(matches) == 1 else None
        if (explicit or base) and any(value.casefold() not in self.models for value in identifiers):
            model = None
        return {"model_profile_id": model,
                "harness_profile_id": self.harnesses.get(subject.get("harness", "").strip().casefold()),
                "model_identity_basis": "explicit" if explicit else "base_model" if base else "normalized_name",
                "exact_evaluated_checkpoint_verified": False}

    def _fact_line(self, profile_id, field, label):
        fact = self.profiles[profile_id].specifications
        for key in field.split("."):
            fact = getattr(fact, key)
        if fact.status != "documented" or any(
            self.sources[e.source_id].eligibility == "not_permitted" for e in fact.evidence
        ):
            return None
        value = fact.value
        if field.endswith(("total_parameters", "active_parameters_per_token")):
            value = f"approximately {value / 1e9:g} billion"
            if "language-model component" in fact.note:
                label = "language component " + label
        elif isinstance(value, list):
            value = ", ".join(map(str, value))
        return {"line": f"{label}: {value}", "profile_id": profile_id,
                "field": field, "evidence": [e.model_dump() for e in fact.evidence]}

    def enrich(self, base_state, subject):
        match = self.resolve(subject)
        report = {**match, "selected_facts": [], "added_tokens": 0,
                  "skip_reason": None, **self.metadata}
        before = self.token_count(base_state)
        if before >= self.state_room:
            report["skip_reason"] = "original_state_has_no_spare_room"
            return base_state, report
        candidates = []
        if model := match["model_profile_id"]:
            for field in self.policy.selected_model_fields:
                if match["model_identity_basis"] == "base_model" and not field.startswith("architecture."):
                    continue
                fact = self._fact_line(model, field, field.removeprefix("architecture."))
                if fact:
                    candidates.append(("Published model properties (checkpoint unverified)", fact))
        if harness := match["harness_profile_id"]:
            fact = self._fact_line(harness, "workflow", "supported workflows")
            if fact:
                candidates.append(("Published harness capabilities (not confirmed enabled)", fact))
        # Append only complete facts that fit. No original character is removed,
        # and no metadata can displace task tokens or revealed evidence.
        sections = {}
        state = base_state
        for title, fact in candidates:
            trial = {key: list(value) for key, value in sections.items()}
            trial.setdefault(title, []).append(fact["line"])
            text = base_state + "".join("\n\n" + key + "\n" + "\n".join(value)
                                      for key, value in trial.items())
            after = self.token_count(text)
            if after <= self.state_room and after - before <= self.policy.max_added_tokens:
                sections, state = trial, text
                report["selected_facts"].append(fact)
                report["added_tokens"] = after - before
        if not report["selected_facts"]:
            report["skip_reason"] = "no_matching_documented_facts_fit"
        return state, report
