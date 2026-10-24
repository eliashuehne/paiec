"""Shared collected-state rendering for grouped training and inference."""
import json
from pathlib import Path

from .collected_inputs import FEATURE_FILES
from .collected_inputs_v2 import PROFILE_FILES, CollectedStateBuilderV2
from .collected_inputs_v3 import add_scope_notes
from .competition_fields import normalize_fields
from .feature_preparation import read_jsonl

SNAPSHOT_FILES = (*FEATURE_FILES, *PROFILE_FILES)
FORMAT = "collected_grouped_v1"
AGGREGATION = "unique_task_token_weighted_mean_logits_softmax"


class GroupedStateBuilder:
    def __init__(self, snapshot):
        self.builder = CollectedStateBuilderV2(snapshot)
        self.by_subject = {}
        for key, row in self.builder.systems.items():
            visible = json.dumps(row["recorded"], sort_keys=True)
            previous = self.by_subject.get(visible)
            if previous is not None and (self.builder.systems[previous]["resolution"] != row["resolution"] or
                    self.builder.runs[previous]["features"] != self.builder.runs[key]["features"]):
                raise ValueError("Conflicting enrichment for an identical supplied subject")
            self.by_subject[visible] = key
        self.qualifiers = {}
        for row in read_jsonl(Path(snapshot) / "reviewed_static_features.jsonl"):
            note = row["fact"].get("note", "")
            if row["fact"]["value"] is not None and any(term in note.casefold() for term in
                    ["rounded", "nominal", "approx", "language-model component", "excludes", "not exact", "not separately"]):
                self.qualifiers.setdefault(row["profile_id"], {})[row["field"]] = note

    def build(self, subject, item):
        subject, item = normalize_fields(subject, item)
        key = self.by_subject.get(json.dumps(subject, sort_keys=True))
        if key is not None:
            state = self.builder.build(key, subject, item, enriched=True)
            return add_scope_notes(state, self.builder, self.qualifiers)
        # An unseen complete run cannot inherit another run's recorded settings.
        # Keep all supplied strings; facts unavailable through a verified match
        # remain explicitly unbound, without private-identity/alias recovery.
        state = self.builder.build("", subject, item, enriched=False)
        empty = {"values_by_status": {}, "unknown_or_unconfirmed": []}
        state["collected_information"] = {
            "documented_model_or_base": {"identity_basis": "unresolved", "exact_evaluated_checkpoint_verified": False,
                "facts": dict(empty), "additional_documented_observations": {}, "additional_previously_documented_profile_facts": {}},
            "documented_harness_support_not_enabled_settings": {"facts": dict(empty),
                "additional_documented_observations": {}, "additional_previously_documented_profile_facts": {}},
            "recorded_run_settings": dict(empty)}
        return state
