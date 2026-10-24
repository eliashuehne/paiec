"""Preserve meaning-changing source qualifiers alongside compact facts."""
import json
from pathlib import Path
from typing import Literal

from .collected_inputs import CollectedInputsConfig
from .collected_inputs_v2 import CollectedInputsV2Config, CollectedStateBuilderV2, prepare_collected_inputs_v2
from .feature_preparation import local_path, read_jsonl, write_json
from .head_training import file_hash


class CollectedInputsV3Config(CollectedInputsV2Config):
    version: Literal["collected_inputs_v3"] = "collected_inputs_v3"


def add_scope_notes(state, builder, qualifiers):
    if "collected_information" not in state:
        return state
    resolution = builder.by_subject[json.dumps(state["target_system_recorded"], sort_keys=True)]
    for key, pid in [("documented_model_or_base", resolution["model_profile_id"]),
                     ("documented_harness_support_not_enabled_settings", resolution["harness_profile_id"])]:
        if pid in qualifiers:
            state["collected_information"][key]["source_scope_qualifiers"] = qualifiers[pid]
    return state


def prepare_collected_inputs_v3(root, config, out):
    config = CollectedInputsV3Config.model_validate(config)
    root, out = Path(root), Path(out)
    features = local_path(root, config.features)
    builder = CollectedStateBuilderV2(features)
    qualifiers = {}
    for row in read_jsonl(features / "reviewed_static_features.jsonl"):
        note = row["fact"].get("note", "")
        if row["fact"]["value"] is not None and any(term in note.casefold() for term in
                ["rounded", "nominal", "approx", "language-model component", "excludes", "not exact", "not separately"]):
            qualifiers.setdefault(row["profile_id"], {})[row["field"]] = note
    report = prepare_collected_inputs_v2(root, dict(config.model_dump(), version="collected_inputs_v2"), out)
    for role in ["training", "monitor", "development"]:
        path = out / "collected" / (role + ".jsonl")
        temporary = path.with_suffix(".rewrite")
        with temporary.open("w") as stream:
            for row in read_jsonl(path):
                row["state"] = add_scope_notes(row["state"], builder, qualifiers)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    for path in out.glob("collected_*_example.json"):
        row = json.loads(path.read_text())
        row["state"] = add_scope_notes(row["state"], builder, qualifiers)
        write_json(path, row)
    report.update(version="collected_inputs_v3", source_scope_qualifiers_preserved=sum(map(len, qualifiers.values())))
    write_json(out / "preparation_summary.json", report)
    contract = json.loads((out / "input_contract.json").read_text())
    contract.update(version="collected_state_v3", renderer_sha256=file_hash(Path(__file__)),
        parent_renderer_sources={name: file_hash(Path(__file__).with_name(name)) for name in ["collected_inputs.py", "collected_inputs_v2.py"]})
    write_json(out / "input_contract.json", contract)
    write_json(out / "input_manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "input_manifest.json"}})
    return report


class CollectedInputWorkflowConfig:
    """Keep all versioned input recipes executable through the same command."""
    @classmethod
    def model_validate(cls, value):
        models = {"collected_inputs_v1": CollectedInputsConfig, "collected_inputs_v2": CollectedInputsV2Config,
                  "collected_inputs_v3": CollectedInputsV3Config}
        if value.get("version") not in models:
            raise ValueError("Unknown collected-input recipe version")
        return models[value["version"]].model_validate(value)

    @classmethod
    def model_validate_json(cls, value):
        return cls.model_validate(json.loads(value))
