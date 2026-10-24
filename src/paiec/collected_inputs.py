"""Full, unshortened Laya inputs from reviewed collection tables; no fitting."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .competition_analysis import verify_competition_bundle
from .competition_data import visible_fields
from .feature_preparation import local_path, read_jsonl, write_json, write_jsonl
from .head_training import file_hash
from .laya_predictor import load_questions

FEATURE_FILES = ("systems.jsonl", "reviewed_static_features.jsonl", "typed_run_features.jsonl",
                 "qualified_model_observations.jsonl", "qualified_harness_observations.jsonl")


class CollectedInputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["collected_inputs_v1"] = "collected_inputs_v1"
    source_policy: Literal["research_only"] = "research_only"
    prepared: str = "runs/competition_pipeline/protocol_panel_v2"
    features: str = "runs/feature_pipeline/offline_evidence_preparation_v1"
    max_len: int = Field(default=2048, ge=512, le=8192)
    head_max_len: Literal[192] = 192
    native_python: str = ".venv-laya/bin/python"


def verify_feature_tables(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("version") != "feature_preparation_v1" or not set(FEATURE_FILES) <= set(manifest["artifacts"]):
        raise ValueError("Use a pinned reviewed feature-preparation bundle")
    for name in FEATURE_FILES:
        if file_hash(folder / name) != manifest["artifacts"][name]:
            raise ValueError("Collected feature table changed: " + name)
    return manifest


class CollectedStateBuilder:
    """Exact visible-subject joins; research facts never become actual settings."""
    def __init__(self, features):
        features = Path(features)
        verify_feature_tables(features)
        self.systems = self.unique(read_jsonl(features / "systems.jsonl"), "system_key")
        self.runs = self.unique(read_jsonl(features / "typed_run_features.jsonl"), "system_key")
        self.static = defaultdict(dict)
        self.static_rows = list(read_jsonl(features / "reviewed_static_features.jsonl"))
        for row in self.static_rows:
            fields = self.static[row["profile_id"]]
            if row["field"] in fields:
                raise ValueError("Duplicate reviewed entity field")
            fact = row["fact"]
            confirmed = row["review_status"] == "reviewed" and row["review_disposition"] == "value_confirmed"
            if fact["value"] is not None and not confirmed:
                raise ValueError("Unconfirmed static value cannot enter Laya")
            fields[row["field"]] = {"value": fact["value"], "status": fact["status"]}
        self.observations = defaultdict(dict)
        for name in FEATURE_FILES[-2:]:
            for row in read_jsonl(features / name):
                fields = self.observations[row["profile_id"]]
                key = row["field"]
                # Separate entries are retained rather than silently overwriting.
                fields.setdefault(key, []).append({"value": row["value"], "status": row["status"]})

    @staticmethod
    def unique(rows, key):
        result = {}
        for row in rows:
            if row[key] in result:
                raise ValueError("Duplicate collected subject")
            result[row[key]] = row
        return result

    def build(self, system_key, subject, item, *, enriched):
        subject, item = visible_fields(subject, item)
        state = {"target_system_recorded": subject,
                 "target_task": {key: item[key] for key in ["item_content", "item_features", "interactors"]}}
        if not enriched:
            return state
        collected, run = self.systems.get(system_key), self.runs.get(system_key)
        if not collected or not run or collected["recorded"] != subject or run["recorded"] != subject:
            raise ValueError("Collected facts lack an identical visible-subject match")
        resolution = collected["resolution"]
        model, harness = resolution["model_profile_id"], resolution["harness_profile_id"]
        settings = {}
        for field, fact in sorted(run["features"].items()):
            if fact["value"] is not None and fact["status"] != "recorded":
                raise ValueError("Run settings must be recorded, not documentation defaults")
            settings[field] = {"value": fact["value"], "status": fact["status"]}
        state["collected_information"] = {
            "documented_model_or_base": {"identity_basis": resolution["model_identity_basis"],
                "exact_evaluated_checkpoint_verified": resolution["exact_evaluated_checkpoint_verified"],
                "facts": self.static.get(model, {}), "additional_documented_observations": self.observations.get(model, {})},
            "documented_harness_support_not_enabled_settings": {
                "facts": self.static.get(harness, {}), "additional_documented_observations": self.observations.get(harness, {})},
            "recorded_run_settings": settings}
        return state


def prepare_collected_inputs(root, config, out):
    root, out = Path(root).resolve(), Path(out)
    config = CollectedInputsConfig.model_validate(config)
    dataset, features = local_path(root, config.prepared), local_path(root, config.features)
    for source in [dataset, features]:
        if out.resolve().is_relative_to(source) or source.is_relative_to(out.resolve()):
            raise ValueError("Input export must be separate from immutable source bundles")
    verify_competition_bundle(dataset)
    builder = CollectedStateBuilder(features)
    systems = {r["system_key"]: r["recorded"] for r in read_jsonl(dataset / "systems.jsonl")}
    tasks = {r["task_key"]: r["recorded"] for r in read_jsonl(dataset / "tasks.jsonl")}
    rows = list(read_jsonl(dataset / "examples.jsonl"))
    target_rows = list(read_jsonl(dataset / "training_targets.jsonl"))
    targets = {r["row_key"]: r["native_binary_value"] for r in target_rows}
    training_keys = {r["row_key"] for r in rows if r["role"] == "training"}
    if len(target_rows) != len(targets) or set(targets) != training_keys or any(type(y) not in {bool, int} or y not in (0, 1) for y in targets.values()):
        raise ValueError("Only complete native training targets may be exported")
    out.mkdir(parents=True, exist_ok=True)
    counts, used_models, used_harnesses = Counter(), set(), set()
    question = load_questions()
    links = []
    for arm in ["recorded", "collected"]:
        folder = out / arm
        folder.mkdir(exist_ok=False)
        handles = {role: (folder / (role + ".jsonl")).open("w") for role in ["training", "monitor", "development"]}
        try:
            for row in rows:
                subject = systems[row["system_key"]]
                item = dict(tasks[row["task_key"]], benchmark_id=row["benchmark_id"])
                state = builder.build(row["system_key"], subject, item, enriched=arm == "collected")
                record = {"state": state, "questions": question}
                if row["role"] == "training":
                    record["expected"] = {"correct": targets[row["row_key"]]}
                    if "Lauren is a cartoonist" in item["item_content"] and subject["normalized_name"] == "Meta Llama 3.1 8B":
                        write_json(out / (arm + "_cartoonist_example.json"), record)
                handles[row["role"]].write(json.dumps(record, ensure_ascii=False) + "\n")
                if arm == "recorded":
                    counts[row["role"]] += 1
                    links.append({k: row[k] for k in ["row_key", "role", "system_key", "task_key", "scope_key", "task_group"]})
        finally:
            for handle in handles.values():
                handle.close()
    for key in {r["system_key"] for r in rows}:
        resolution = builder.systems[key]["resolution"]
        used_models.add(resolution["model_profile_id"])
        used_harnesses.add(resolution["harness_profile_id"])
    known = [r for r in builder.static_rows if r["fact"]["value"] is not None]
    snapshot = out / "collection_snapshot"
    snapshot.mkdir(exist_ok=False)
    for name in FEATURE_FILES:
        shutil.copyfile(features / name, snapshot / name)
    write_jsonl(out / "row_links.jsonl", links)
    report = {"version": "collected_inputs_v1", "source_policy": "research_only", "records_by_role": dict(counts),
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "feature_manifest_sha256": file_hash(features / "manifest.json"),
        "confirmed_static_values_available": len(known),
        "confirmed_static_values_bound_to_exported_subjects": sum(r["profile_id"] in used_models | used_harnesses for r in known),
        "recorded_run_values_bound_to_exported_subjects": sum(f["value"] is not None for key in {r["system_key"] for r in rows}
            for f in builder.runs[key]["features"].values()),
        "documented_observation_entries_bound_to_exported_subjects": sum(len(v) for pid in used_models | used_harnesses
            for v in builder.observations.get(pid, {}).values()),
        "full_supplied_task_text_preserved": True, "all_reviewed_fields_exported_for_matched_entities": True,
        "model_harness_support_separate_from_actual_run_settings": True,
        "targets_separate_from_state": True, "nontraining_outcomes_exported": False,
        "task_annotations_used": 0, "unreviewed_task_annotations_excluded": True,
        "revealed_examples_used": False, "native_token_audit_completed": False, "training_ready": False,
        "max_len": config.max_len, "head_max_len": config.head_max_len,
        "historical_renderer_changed": False, "gpu_jobs": 0, "paid_calls": 0}
    write_json(out / "input_contract.json", {"version": "collected_state_v1", "question": question,
        "serialization": "laya.common.serialize_state", "max_len": config.max_len, "head_max_len": config.head_max_len,
        "renderer_sha256": file_hash(Path(__file__)), "source_policy": "research_only",
        "native_training_schema": ["state", "questions", "expected"], "revealed_examples_used": False})
    write_json(out / "preparation_summary.json", report)
    write_json(out / "input_manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "input_manifest.json"}})
    return report
