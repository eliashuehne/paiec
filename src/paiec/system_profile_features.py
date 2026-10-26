"""Documented model and harness profile features for the CatBoost experiments.

This module turns the offline research catalogs into declared columns. It reads
only documented facts whose sources are permitted, resolves a supplied subject
through the same fail-closed alias policy the Laya enrichment uses, and never
touches benchmark identity, items, outcomes or display names.

These columns come from external sources, so they remain research-only: a model
fitted with them is not a competition-eligible candidate, and the assessment
records that explicitly. Everything else about the experiment is unchanged.
"""
from __future__ import annotations

import math
from pathlib import Path

from .catboost_features import PROFILE_COLUMNS, PROFILE_GROUP
from .profile_enrichment import ProfileEnricher

MISSING = "__missing__"
NUMERIC_FACTS = (
    "architecture.total_parameters",
    "architecture.active_parameters_per_token",
    "architecture.layers",
    "architecture.hidden_size",
    "architecture.attention_heads",
    "architecture.key_value_heads",
    "architecture.experts_total",
    "architecture.experts_active_per_token",
)


class ProfileFacts:
    """Resolve supplied systems to documented facts, or to explicit missingness."""

    def __init__(self, policy_path, *, root=None):
        self.policy_path = Path(policy_path)
        self.enricher = ProfileEnricher(self.policy_path, root=root, token_count=len, state_room=10 ** 9)
        self.profiles, self.sources = self.enricher.profiles, self.enricher.sources

    @property
    def metadata(self):
        return dict(self.enricher.metadata)

    def value(self, profile_id, field):
        """One documented fact value, or None for unknown, absent or barred evidence."""
        if not profile_id:
            return None
        node = self.profiles[profile_id].specifications
        try:
            for key in field.split("."):
                node = getattr(node, key)
        except AttributeError:
            return None
        if node is None or getattr(node, "status", None) != "documented" or node.value is None:
            return None
        if any(self.sources[item.source_id].eligibility == "not_permitted" for item in node.evidence):
            return None
        return node.value

    def record(self, subject):
        """Declared profile columns for one supplied subject dictionary."""
        match = self.enricher.resolve(subject)
        model, harness = match["model_profile_id"], match["harness_profile_id"]
        total = self.value(model, "architecture.total_parameters")
        active = self.value(model, "architecture.active_parameters_per_token")
        layers = self.value(model, "architecture.layers")
        hidden = self.value(model, "architecture.hidden_size")
        heads = self.value(model, "architecture.attention_heads")
        kv_heads = self.value(model, "architecture.key_value_heads")
        experts = self.value(model, "architecture.experts_total")
        workflow = self.value(harness, "workflow")
        known = sum(self.value(model, field) is not None for field in NUMERIC_FACTS)

        def log(value, scale=1.0):
            return math.log1p(float(value) / scale) if value else 0.0

        def category(value, join=" | "):
            if isinstance(value, (list, tuple)):
                return join.join(map(str, value)) or MISSING
            return str(value) if value is not None else MISSING

        record = {
            "profile_model_found": float(model is not None),
            "profile_harness_found": float(harness is not None),
            "profile_documented_numeric_facts": float(known),
            "profile_total_parameters_log": log(total, 1e9),
            "profile_active_parameters_log": log(active, 1e9),
            "profile_active_parameter_share": (float(active) / float(total)
                                               if total and active else 0.0),
            "profile_layers": float(layers) if layers else 0.0,
            "profile_hidden_size_log": log(hidden),
            "profile_attention_heads_log": log(heads),
            "profile_key_value_head_share": (float(kv_heads) / float(heads) if heads and kv_heads else 0.0),
            "profile_experts_total_log": log(experts),
            "profile_family": category(self.value(model, "architecture.family")),
            "profile_model_type": category(self.value(model, "model_type")),
            "profile_parameterization": category(self.value(model, "architecture.parameterization")),
            "profile_input_modalities": category(self.value(model, "input_modalities")),
            "profile_output_modalities": category(self.value(model, "output_modalities")),
            "profile_workflow": category(workflow),
        }
        if tuple(record) != PROFILE_COLUMNS:
            raise ValueError("Profile record does not match the declared catalog contract")
        return record


def checked_profile_columns(groups):
    """Profile columns are selected only through the declared catalog group."""
    if PROFILE_GROUP not in groups:
        return ()
    return PROFILE_COLUMNS
