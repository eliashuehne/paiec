"""Research profile schema. Deliberately independent of prediction/state rendering."""

from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Generic, Literal, TypeVar
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(StrictModel):
    source_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class Fact(StrictModel, Generic[T]):
    value: T | None = None
    status: Literal["unknown", "recorded", "documented", "inferred"] = "unknown"
    evidence: list[Evidence] = Field(default_factory=list)
    note: str = "Not established from reviewed sources."

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "unknown":
            if self.value is not None or self.evidence:
                raise ValueError("Unknown facts have null values and no evidence")
        elif self.value is None or not self.evidence:
            raise ValueError(
                "Known/inferred facts require a value and supporting evidence"
            )
        return self


Backbone = Literal["transformer", "state_space", "recurrent", "hybrid", "other"]
AttentionDirection = Literal["causal", "bidirectional", "encoder_decoder", "other"]
Parameterization = Literal["dense", "mixture_of_experts", "hybrid", "other"]


class Architecture(StrictModel):
    family: Fact[str] = Field(default_factory=Fact[str])
    backbone: Fact[Backbone] = Field(default_factory=Fact[Backbone])
    attention_direction: Fact[AttentionDirection] = Field(
        default_factory=Fact[AttentionDirection]
    )
    parameterization: Fact[Parameterization] = Field(
        default_factory=Fact[Parameterization]
    )
    total_parameters: Fact[int] = Field(default_factory=Fact[int])
    active_parameters_per_token: Fact[int] = Field(default_factory=Fact[int])
    layers: Fact[int] = Field(default_factory=Fact[int])
    hidden_size: Fact[int] = Field(default_factory=Fact[int])
    attention_heads: Fact[int] = Field(default_factory=Fact[int])
    key_value_heads: Fact[int] = Field(default_factory=Fact[int])
    attention_types: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    experts_total: Fact[int] = Field(default_factory=Fact[int])
    experts_active_per_token: Fact[int] = Field(default_factory=Fact[int])
    positional_encoding: Fact[str] = Field(default_factory=Fact[str])
    tokenizer_identifier: Fact[str] = Field(default_factory=Fact[str])

    @model_validator(mode="after")
    def numeric_consistency(self):
        for key in [
            "total_parameters",
            "active_parameters_per_token",
            "layers",
            "hidden_size",
            "attention_heads",
            "key_value_heads",
            "experts_total",
            "experts_active_per_token",
        ]:
            if (value := getattr(self, key).value) is not None and value <= 0:
                raise ValueError(f"{key} must be positive when known")
        for active, total in [
            (self.active_parameters_per_token.value, self.total_parameters.value),
            (self.experts_active_per_token.value, self.experts_total.value),
            (self.key_value_heads.value, self.attention_heads.value),
        ]:
            if active is not None and total is not None and active > total:
                raise ValueError("Active/subset counts cannot exceed totals")
        return self


class ModelSpecifications(StrictModel):
    architecture: Architecture = Field(default_factory=Architecture)
    model_type: Fact[str] = Field(default_factory=Fact[str])
    base_model: Fact[str] = Field(default_factory=Fact[str])
    fine_tuning_domains: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    release_date: Fact[str] = Field(default_factory=Fact[str])
    max_input_tokens: Fact[int] = Field(default_factory=Fact[int])
    max_output_tokens: Fact[int] = Field(default_factory=Fact[int])
    max_combined_context_tokens: Fact[int] = Field(default_factory=Fact[int])
    input_modalities: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    output_modalities: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    documented_languages: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    supported_reasoning_settings: Fact[list[str]] = Field(
        default_factory=Fact[list[str]]
    )
    max_reasoning_tokens: Fact[int] = Field(default_factory=Fact[int])


class HarnessSpecifications(StrictModel):
    supported_tools: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    workflow: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    context_management: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    default_max_turns: Fact[int] = Field(default_factory=Fact[int])
    default_max_tool_calls: Fact[int] = Field(default_factory=Fact[int])
    default_max_retries: Fact[int] = Field(default_factory=Fact[int])
    default_timeout_seconds: Fact[float] = Field(default_factory=Fact[float])
    default_prompting_strategy: Fact[str] = Field(default_factory=Fact[str])


class RunSpecifications(StrictModel):
    model_profile_id: str | None = None
    harness_profile_id: str | None = None
    enabled_tools: Fact[list[str]] = Field(default_factory=Fact[list[str]])
    reasoning_effort: Fact[str] = Field(default_factory=Fact[str])
    temperature: Fact[float] = Field(default_factory=Fact[float])
    top_p: Fact[float] = Field(default_factory=Fact[float])
    max_output_tokens: Fact[int] = Field(default_factory=Fact[int])
    max_reasoning_tokens: Fact[int] = Field(default_factory=Fact[int])
    few_shot_count: Fact[int] = Field(default_factory=Fact[int])
    system_prompt: Fact[str] = Field(default_factory=Fact[str])
    prompt_template: Fact[str] = Field(default_factory=Fact[str])
    max_turns: Fact[int] = Field(default_factory=Fact[int])
    max_tool_calls: Fact[int] = Field(default_factory=Fact[int])
    max_retries: Fact[int] = Field(default_factory=Fact[int])
    timeout_seconds: Fact[float] = Field(default_factory=Fact[float])
    candidate_count: Fact[int] = Field(default_factory=Fact[int])
    selection_method: Fact[str] = Field(default_factory=Fact[str])
    quantization: Fact[str] = Field(default_factory=Fact[str])
    serving_configuration: Fact[str] = Field(default_factory=Fact[str])
    raw_supplied_subject: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def published_subject_only(self):
        from .schema import SUBJECT_FIELDS

        if set(self.raw_supplied_subject) - set(SUBJECT_FIELDS):
            raise ValueError("Raw subject may contain only published subject fields")
        return self


class Identity(StrictModel):
    name: Fact[str] = Field(default_factory=Fact[str])
    version: Fact[str] = Field(default_factory=Fact[str])
    provider: Fact[str] = Field(default_factory=Fact[str])
    resolution: Literal["exact_version", "reported_identifier_only", "unresolved"] = (
        "unresolved"
    )


class Profile(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    entity_type: Literal["model", "harness", "recorded_run"]
    identity: Identity = Field(default_factory=Identity)
    valid_from: str | None = None
    valid_until: str | None = None
    specifications: ModelSpecifications | HarnessSpecifications | RunSpecifications
    usage: Literal["research_only"] = "research_only"

    @model_validator(mode="after")
    def matching_specification(self):
        expected = {
            "model": ModelSpecifications,
            "harness": HarnessSpecifications,
            "recorded_run": RunSpecifications,
        }
        if not isinstance(self.specifications, expected[self.entity_type]):
            raise ValueError("Entity type and specification block disagree")
        for value in [self.valid_from, self.valid_until]:
            if value is not None:
                date.fromisoformat(value)
        if self.valid_from and self.valid_until and self.valid_from > self.valid_until:
            raise ValueError("Invalid validity interval")
        return self


class Source(StrictModel):
    source_id: str
    source_type: Literal[
        "benchmark_table",
        "model_card",
        "model_config",
        "technical_report",
        "harness_documentation",
        "harness_code",
    ]
    location: str
    revision: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    retrieved_on: str
    eligibility: Literal["inventory_source", "unreviewed", "not_permitted"] = (
        "unreviewed"
    )
    eligibility_note: str


def load_catalog(directory: Path):
    sources = [
        Source.model_validate_json(line)
        for line in (directory / "sources.jsonl").read_text().splitlines()
        if line.strip()
    ]
    profiles = [
        Profile.model_validate_json(line)
        for line in (directory / "profiles.jsonl").read_text().splitlines()
        if line.strip()
    ]
    source_ids = [source.source_id for source in sources]
    profile_ids = [profile.profile_id for profile in profiles]
    if len(set(source_ids)) != len(source_ids) or len(set(profile_ids)) != len(
        profile_ids
    ):
        raise ValueError("Duplicate source or profile ID")
    for profile in profiles:
        for block in [profile.identity, profile.specifications]:
            # Traverse dumped objects so every future nested fact is also checked.
            def visit(value):
                if isinstance(value, dict):
                    if {"status", "value", "evidence"} <= value.keys():
                        for evidence in value["evidence"]:
                            if evidence["source_id"] not in source_ids:
                                raise ValueError("Unresolved evidence source")
                    for child in value.values():
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)

            visit(block.model_dump())
        if isinstance(profile.specifications, RunSpecifications):
            for key, kind in [
                ("model_profile_id", "model"),
                ("harness_profile_id", "harness"),
            ]:
                reference = getattr(profile.specifications, key)
                if reference and not any(
                    p.profile_id == reference and p.entity_type == kind
                    for p in profiles
                ):
                    raise ValueError("Missing or mismatched run profile reference")
    return profiles, sources
