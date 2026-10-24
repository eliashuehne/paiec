"""Enriched-only grouped corpus, stored as bounded-memory native token views.

Only training labels are opened. Family/scope metadata controls loss weights and
never enters the encoder. Native imports are confined to offline preparation.
"""
from collections import Counter, defaultdict
from itertools import zip_longest
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .feature_preparation import local_path, read_jsonl, write_json
from .head_training import file_hash
from .grouped_checkpoint import renderer_fingerprints, snapshot_collection
from .grouped_inputs import GroupedStateBuilder, FORMAT, AGGREGATION, SNAPSHOT_FILES
from .laya_predictor import load_questions


class GroupedTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["grouped_enriched_training_v1"] = "grouped_enriched_training_v1"
    source_policy: Literal["research_only"] = "research_only"
    inputs: str = "runs/laya_inputs/collected_v3"
    input_manifest_sha256: str = "5c7d46b2bab1ddd729b7b448a7bdf7ff7851ab8ce67711c9c8377041f134bc9b"
    dataset: str = "runs/competition_pipeline/protocol_panel_v2"
    prepared: str = "runs/grouped_enriched/prepared_v1/tokens"
    qualification: str = "runs/workflows/multilingual_qualification_v1/job/output/qualification.json"
    qualification_sha256: str = "2b6b8c2406dc2ceb36a72ef72c4daf28ff29a7743c453d2cf6e147b7691d762a"
    model: Literal["multilingual"] = "multilingual"
    max_len: Literal[8192] = 8192
    head_max_len: Literal[256] = 256
    overlap_tokens: Literal[128] = 128
    max_views: Literal[32] = 32
    upper_layers: Literal[2] = 2
    seed: int = 0
    epochs: Literal[1] = 1
    effective_batch: int = Field(default=64, ge=1, le=64)
    learning_rate: float = Field(default=1e-4, gt=0, le=.001, allow_inf_nan=False)
    encoder_learning_rate: float = Field(default=1e-5, gt=0, le=.00005, allow_inf_nan=False)
    weight_decay: float = Field(default=.01, ge=0, allow_inf_nan=False)
    grad_clip: float = Field(default=1., gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def paths(self):
        for name in ["inputs", "dataset", "prepared", "qualification"]:
            path = Path(getattr(self, name))
            if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("runs",):
                raise ValueError("Grouped artifacts must be repository-relative under runs/")
        for digest in [self.input_manifest_sha256, self.qualification_sha256]:
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Expected a SHA256 fingerprint")
        return self

    def input_contract(self):
        return {key: getattr(self, key) for key in ["max_len", "head_max_len", "overlap_tokens", "max_views"]}


def checked_artifacts(folder, artifacts):
    folder = Path(folder).resolve()
    for name, digest in artifacts.items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder) or not target.is_file() or file_hash(target) != digest:
            raise ValueError("Grouped artifact changed or escaped its bundle: " + name)


def verify_qualification(root, config):
    path = local_path(root, config.qualification)
    if file_hash(path) != config.qualification_sha256:
        raise ValueError("Multilingual qualification report changed")
    report = json.loads(path.read_text())
    if report.get("status") != "passed" or report.get("version") != "multilingual_qualification_v1":
        raise ValueError("A passed multilingual qualification is required")
    for key in ["model", "inputs", "max_len", "head_max_len", "overlap_tokens", "max_views", "upper_layers", "seed"]:
        if report["config"][key] != getattr(config, key):
            raise ValueError("Grouped recipe differs from qualification: " + key)
    arms = [a for a in report["arms"] if a["upper_layers"] == config.upper_layers]
    if len(arms) != 1 or not arms[0]["passed"]:
        raise ValueError("Selected encoder policy did not qualify")
    checked_artifacts(root, report["training_source_sha256"])
    return report


def family_scope_weights(rows):
    counts = Counter((r["validation_family"], r["scope_key"]) for r in rows)
    scopes = defaultdict(set)
    for family, scope in counts:
        scopes[family].add(scope)
    # Mean weight one: equal mass per family, then per system/scope within it.
    return [len(rows) / (len(scopes) * len(scopes[r["validation_family"]]) *
            counts[r["validation_family"], r["scope_key"]]) for r in rows]


class TokenCorpus:
    """Read one original example at a time; views are not independent samples."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.records = list(read_jsonl(self.folder / "groups.jsonl"))
        size = (self.folder / "tokens.bin").stat().st_size
        if not size or size % 4 or not self.records:
            raise ValueError("Invalid grouped token store")
        self.tokens = np.memmap(self.folder / "tokens.bin", mode="r", dtype="<i4")
        end, seen = 0, set()
        for r in self.records:
            if r["role"] != "training" or r["row_key"] in seen or type(r["label"]) is not int or r["label"] not in (0, 1):
                raise ValueError("Only unique native training targets belong in this corpus")
            seen.add(r["row_key"])
            weights = r["weights"]
            if (len(r["views"]) != len(weights) or not weights or len(weights) > 32 or
                any(not math.isfinite(w) or w <= 0 for w in weights) or abs(sum(weights) - 1.) > 1e-6 or
                not math.isfinite(r["sample_weight"]) or r["sample_weight"] <= 0):
                raise ValueError("Invalid grouped training weights")
            for view in r["views"]:
                length = view["length"]
                if (view["offset"] != end or not 1 <= length <= 8192 or len(view["markers"]) != 2 or
                    not all(type(m) is int and 0 <= m < length for m in view["markers"]) or
                    view["markers"][0] >= view["markers"][1]):
                    raise ValueError("Invalid grouped token offsets or option markers")
                end += length
        if end != len(self.tokens) or (self.tokens < 0).any():
            raise ValueError("Grouped token store has missing or trailing tokens")

    def __len__(self):
        return len(self.records)

    def group(self, index):
        r = self.records[index]
        target = [0., 1.] if r["label"] else [1., 0.]
        return {"views": [{"ids": self.tokens[v["offset"]:v["offset"] + v["length"]].tolist(),
                           "markers": v["markers"], "qtype": 2, "target": target} for v in r["views"]],
                "weights": r["weights"], "label": r["label"]}


def verify_prepared(root, config):
    config = GroupedTrainingConfig.model_validate(config)
    verify_qualification(root, config)
    path = local_path(root, config.prepared)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("version") != "grouped_enriched_tokens_v1" or manifest.get("config") != config.model_dump():
        raise ValueError("Grouped preparation belongs to another recipe")
    required = {"tokens.bin", "groups.jsonl", "questions.json", "summary.json", "reload_fixtures.json", "collection_snapshot/manifest.json",
                *["collection_snapshot/" + n for n in SNAPSHOT_FILES]}
    if not required <= set(manifest["artifacts"]):
        raise ValueError("Incomplete grouped preparation")
    checked_artifacts(path, manifest["artifacts"])
    if manifest["renderer_sources"] != renderer_fingerprints() or json.loads((path / "questions.json").read_text()) != load_questions():
        raise ValueError("Grouped preparation renderer or question changed")
    return path, manifest


def prepare_grouped(root, config):
    from transformers import AutoTokenizer
    from .conditional_analysis import training_panel
    from .multilingual_recipe import pinned_multilingual_parent, encode_group
    config = GroupedTrainingConfig.model_validate(config)
    root = Path(root).resolve()
    qualification = verify_qualification(root, config)
    source, out = local_path(root, config.inputs), local_path(root, config.prepared)
    for folder in [source, local_path(root, config.dataset)]:
        if out.is_relative_to(folder) or folder.is_relative_to(out):
            raise ValueError("Keep grouped preparations separate from immutable sources")
    if file_hash(source / "input_manifest.json") != config.input_manifest_sha256:
        raise ValueError("Collected input export changed")
    checked_artifacts(source, json.loads((source / "input_manifest.json").read_text())["artifacts"])
    report = json.loads((source / "preparation_summary.json").read_text())
    dataset = local_path(root, config.dataset)
    if file_hash(dataset / "manifest.json") != report["dataset_manifest_sha256"]:
        raise ValueError("Training family source differs from collected inputs")
    _, rows, targets, provenance = training_panel(dataset)
    links = [r for r in read_jsonl(source / "row_links.jsonl") if r["role"] == "training"]
    if len(links) != len(rows) or any(a["row_key"] != b["row_key"] for a, b in zip(links, rows)):
        raise ValueError("Collected row links differ from training provenance")
    weights = family_scope_weights(rows)
    parent = pinned_multilingual_parent()
    for name, digest in qualification["checkpoint_sha256"].items():
        if file_hash(parent / name) != digest:
            raise ValueError("Pinned multilingual parent differs from qualification")
    tok = AutoTokenizer.from_pretrained(parent / "tokenizer", local_files_only=True)
    question = load_questions()
    if json.loads((source / "input_contract.json").read_text())["question"] != question:
        raise ValueError("Collected question differs from active training question")
    out.mkdir(parents=True, exist_ok=False)
    snapshot_collection(source / "collection_snapshot", out / "collection_snapshot")
    builder = GroupedStateBuilder(out / "collection_snapshot")
    offset, count, multi, max_views, max_length = 0, 0, 0, 0, 0
    fixtures = []
    with (out / "tokens.bin").open("wb") as binary, (out / "groups.jsonl").open("w") as index:
        for row, metadata, weight in zip_longest(read_jsonl(source / "collected/training.jsonl"), rows, weights):
            if row is None or metadata is None or weight is None:
                raise ValueError("Training row counts differ")
            label = row["expected"]["correct"]
            if type(label) not in {bool, int} or label not in (0, 1) or int(label) != targets[metadata["row_key"]] or row["questions"] != question:
                raise ValueError("Grouped training target/question differs from native source")
            subject = row["state"]["target_system_recorded"]
            item = dict(row["state"]["target_task"], benchmark_id="")
            if builder.build(subject, item) != row["state"]:
                raise ValueError("Shared training/inference state differs from collected export")
            group = encode_group(tok, row["state"], question, label, config.max_len, config.head_max_len,
                                 config.overlap_tokens, config.max_views)
            views = []
            for view in group["views"]:
                values = np.asarray(view["ids"], dtype="<i4")
                values.tofile(binary)
                views.append({"offset": offset, "length": len(values), "markers": view["markers"]})
                offset += len(values)
                max_length = max(max_length, len(values))
            record = {"role": "training", "row_key": metadata["row_key"], "family": metadata["validation_family"],
                      "scope_key": metadata["scope_key"], "label": int(label), "sample_weight": weight,
                      "views": views, "weights": group["weights"], "layout": group["layout"]}
            index.write(json.dumps(record) + "\n")
            if count < 2 or (len(views) > 1 and not any(f["multi_view"] for f in fixtures)):
                fixtures.append({"subject": subject, "item": item, "group_index": count, "multi_view": len(views) > 1})
            count += 1
            multi += len(views) > 1
            max_views = max(max_views, len(views))
            if count % 2000 == 0:
                print(json.dumps({"prepared_original_examples": count, "multi_view_examples": multi}), flush=True)
    write_json(out / "questions.json", question)
    write_json(out / "reload_fixtures.json", fixtures)
    summary = {"version": "grouped_enriched_preparation_v1", "state_format": FORMAT, "aggregation": AGGREGATION,
        "original_training_examples": count, "multi_view_examples": multi, "maximum_views": max_views,
        "maximum_sequence": max_length, "token_count": offset, "training_families": len({r["validation_family"] for r in rows}),
        "training_scopes": len({r["scope_key"] for r in rows}), "weighting": "equal family, then scope; mean original-example weight one",
        "all_training_rows_retained": count == report["records_by_role"]["training"], "all_states_match_inference_builder": True,
        "task_tokens_omitted": 0, "model_weights_loaded": False, "nontraining_outcomes_opened": False,
        "model_inference_performed": False, "research_only": True, "training_launched": False,
        "family_provenance": provenance, "quality_result": False}
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", {"version": "grouped_enriched_tokens_v1", "config": config.model_dump(),
        "renderer_sources": renderer_fingerprints(), "parent_sha256": qualification["checkpoint_sha256"],
        "artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    TokenCorpus(out)
    return summary
