"""Compact lossless feature projection; preserves the separately audited v1."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Literal

from .collected_inputs import CollectedInputsConfig, CollectedStateBuilder, prepare_collected_inputs
from .feature_preparation import local_path, read_jsonl, write_json
from .head_training import file_hash
from .system_profiles import Profile

PROFILE_FILES = ("supplemented_model_profiles.jsonl", "supplemented_harness_profiles.jsonl")


class CollectedInputsV2Config(CollectedInputsConfig):
    version: Literal["collected_inputs_v2"] = "collected_inputs_v2"


def flatten_facts(value, prefix=""):
    if isinstance(value, dict):
        if "value" in value and "status" in value:
            yield prefix, value
        else:
            for key, child in value.items():
                yield from flatten_facts(child, prefix + "." + key if prefix else key)


def compact_facts(fields):
    grouped = defaultdict(dict)
    unknown = []
    for field, fact in sorted(fields.items()):
        if fact["value"] is None:
            unknown.append(field)
        else:
            grouped[fact["status"]][field] = fact["value"]
    return {"values_by_status": dict(grouped), "unknown_or_unconfirmed": unknown}


class CollectedStateBuilderV2(CollectedStateBuilder):
    def __init__(self, features):
        super().__init__(features)
        features = Path(features)
        manifest = json.loads((features / "manifest.json").read_text())
        self.additional, self.excluded_inferred = defaultdict(dict), 0
        for name in PROFILE_FILES:
            if name not in manifest["artifacts"] or file_hash(features / name) != manifest["artifacts"][name]:
                raise ValueError("Documented profile facts are not pinned")
            for raw in read_jsonl(features / name):
                profile = Profile.model_validate(raw)
                for field, fact in flatten_facts(profile.specifications.model_dump()):
                    if field in self.static[profile.profile_id] or fact["value"] is None:
                        continue
                    if fact["status"] == "documented" and fact["evidence"]:
                        self.additional[profile.profile_id][field] = {"value": fact["value"], "status": fact["status"]}
                    else:
                        self.excluded_inferred += 1
        self.by_subject = {json.dumps(r["recorded"], sort_keys=True): r["resolution"] for r in self.systems.values()}

    def compact_state(self, state):
        if "collected_information" not in state:
            return state
        resolution = self.by_subject[json.dumps(state["target_system_recorded"], sort_keys=True)]
        collected = state["collected_information"]
        for key, pid in [("documented_model_or_base", resolution["model_profile_id"]),
                         ("documented_harness_support_not_enabled_settings", resolution["harness_profile_id"])]:
            section = collected[key]
            section["facts"] = compact_facts(section["facts"])
            section["additional_previously_documented_profile_facts"] = {
                key: fact["value"] for key, fact in sorted(self.additional.get(pid, {}).items())}
        collected["recorded_run_settings"] = compact_facts(collected["recorded_run_settings"])
        return state

    def build(self, system_key, subject, item, *, enriched):
        return self.compact_state(super().build(system_key, subject, item, enriched=enriched))


def prepare_collected_inputs_v2(root, config, out):
    config = CollectedInputsV2Config.model_validate(config)
    root, out = Path(root), Path(out)
    features = local_path(root, config.features)
    builder = CollectedStateBuilderV2(features)
    base_config = dict(config.model_dump(), version="collected_inputs_v1")
    report = prepare_collected_inputs(root, base_config, out)
    for role in ["training", "monitor", "development"]:
        path = out / "collected" / (role + ".jsonl")
        temporary = path.with_suffix(".rewrite")
        with temporary.open("w") as stream:
            for row in read_jsonl(path):
                row["state"] = builder.compact_state(row["state"])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    for path in out.glob("collected_*_example.json"):
        row = json.loads(path.read_text())
        row["state"] = builder.compact_state(row["state"])
        write_json(path, row)
    for name in PROFILE_FILES:
        shutil.copyfile(features / name, out / "collection_snapshot" / name)
    report.update(version="collected_inputs_v2", additional_documented_profile_values_available=sum(map(len, builder.additional.values())),
        inferred_additional_profile_values_excluded=builder.excluded_inferred,
        compact_projection_preserves_reviewed_values_and_unknowns=True)
    write_json(out / "preparation_summary.json", report)
    contract = json.loads((out / "input_contract.json").read_text())
    contract.update(version="collected_state_v2", renderer_sha256=file_hash(Path(__file__)),
        parent_renderer_sha256=file_hash(Path(__file__).with_name("collected_inputs.py")),
        source_quality="Scoped reviews plus separately identified, previously documented profile facts; inferred extras excluded.")
    write_json(out / "input_contract.json", contract)
    write_json(out / "input_manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "input_manifest.json"}})
    return report
