"""Typed bounded writers with deterministic, compression-independent row digests."""
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import canonical_json

S = pa.string()
I = pa.int64()
B = pa.bool_()
L = pa.list_(S)


def schema(*fields):
    return pa.schema(fields)


SCHEMAS = {
    "population": schema(("benchmark", S), ("directory", S), ("subjects", I), ("items", I), ("responses", I),
                         ("family", S), ("split", S), ("frozen_role", S), ("protected", B), ("exposure", S),
                         ("direction", S), ("modalities", L), ("feature_status", S)),
    "systems": schema(("system_key", S), *[(k, S) for k in ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra")]),
    "subject_links": schema(("benchmark", S), ("subject_id", S), ("system_key", S), ("status", S)),
    "items": schema(("benchmark", S), ("item_id", S), ("item_key", S), ("source_row", I), ("split", S),
                    ("content", S), ("item_features", S), ("status", S), ("group_key", S)),
    "tasks": schema(("task_key", S), ("item_content", S), ("item_features", S), ("interactors", S)),
    "task_features": schema(("task_key", S), *[(k, I) for k in ("content_characters", "whitespace_words", "lines", "paragraphs", "numeric_spans", "cjk_characters", "cyrillic_characters", "arabic_characters", "option_marker_count_hint", "structured_option_count_hint", "section_count_hint", "message_count_hint")],
                            *[(k, B) for k in ("has_code_fence", "has_math_marker_hint", "has_table_pipe_hint", "has_supplied_item_features", "has_supplied_interactors", "has_visible_asset_reference_hint")], ("structure_hint", S)),
    "item_input_links": schema(("benchmark", S), ("item_id", S), ("item_key", S), ("interactor_key", S),
                               ("task_key", S), ("group_key", S), ("split", S), ("feature_status", S), ("interactors_recorded", B)),
    "response_input_links": schema(("benchmark", S), ("source_row", I), ("response_id", S), ("subject_id", S), ("item_id", S),
                                   ("trial", I), ("system_key", S), ("item_key", S), ("interactor_key", S), ("status", S)),
    "training_eligibility": schema(("benchmark", S), ("item_key", S), ("task_key", S), ("split", S), ("eligible", B), ("reason", S)),
    "content_groups": schema(("group_key", S), ("split_mask", I), ("item_rows", I)),
    "setting_observations": schema(("system_key", S), ("source_field", S), ("source_path", S), ("occurrence", I),
                                    ("raw_key", S), ("raw_value", S), ("field", S), ("status", S), ("reason", S), ("value_json", S), ("source_scope", S), ("observation_id", S)),
    "run_settings": schema(("system_key", S), ("field", S), ("value_type", S), ("value_json", S), ("status", S),
                           ("reason", S), ("source_scope", S), ("observation_ids", L), ("policy_version", S)),
    "profile_matches": schema(("system_key", S), ("entity_type", S), ("profile_id", S), ("identity_basis", S),
                               ("status", S), ("exact_evaluated_checkpoint_verified", B), ("eligibility", S)),
    "research_system_features": schema(("system_key", S), ("profile_id", S), ("entity_type", S), ("field", S), ("value_json", S),
                                        ("status", S), ("reason", S), ("eligibility", S), ("historical_applicability", S),
                                        ("identity_basis", S), ("source_ids", L), ("review_disposition", S)),
}


class TableWriter:
    def __init__(self, directory, name, *, batch_size=4096):
        self.path = Path(directory) / (name + ".parquet")
        self.schema = SCHEMAS[name]
        self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.buffer = []
        self.count = 0
        self.digest = hashlib.sha256(self.schema.to_string().encode())
        self.batch_size = batch_size

    def add(self, row):
        if set(row) != set(self.schema.names):
            raise ValueError("Row columns do not match table schema: " + str(self.path.name))
        self.digest.update(canonical_json(row).encode() + b"\n")
        self.buffer.append(row)
        self.count += 1
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer = []

    def close(self):
        self.flush()
        self.writer.close()
        return {"rows": self.count, "logical_sha256": self.digest.hexdigest(), "schema": self.schema.to_string()}


def write_table(directory, name, rows, batch_size=4096):
    writer = TableWriter(directory, name, batch_size=batch_size)
    try:
        for row in rows:
            writer.add(row)
    finally:
        result = writer.close()
    return result
