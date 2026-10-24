"""Pinned multilingual recipe and lossless, grouped long-input preparation."""
from collections import Counter
import copy
import gzip
from importlib.metadata import version
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .feature_preparation import local_path, read_jsonl, write_json, write_jsonl, stable_hash
from .head_training import file_hash
from .laya_runtime import DEFAULT_CHECKPOINT_REVISION, MODEL_REPOSITORY


class MultilingualRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["multilingual_qualification_v1"] = "multilingual_qualification_v1"
    inputs: str = "runs/laya_inputs/collected_v3"
    capacity_audit: str = "runs/laya_inputs/collected_v3_multilingual_8192_audit"
    prepared: str = "runs/multilingual_qualification/prepared_v1"
    model: Literal["multilingual"] = "multilingual"
    checkpoint_revision: Literal["7b928d828b7b0e022f929d9bd2e44165aa270148"] = DEFAULT_CHECKPOINT_REVISION
    max_len: Literal[8192] = 8192
    head_max_len: Literal[256] = 256
    overlap_tokens: int = Field(default=128, ge=0, le=512)
    seed: int = 0
    upper_layers: Literal[2] = 2
    learning_rate: float = Field(default=.0003, gt=0, le=.001)
    encoder_learning_rate: float = Field(default=.00001, gt=0, le=.00005)
    weight_decay: float = .01
    grad_clip: float = 1.
    synthetic_updates: int = Field(default=80, ge=10, le=80)
    real_updates: int = Field(default=60, ge=10, le=60)
    max_views: int = Field(default=32, ge=2, le=32)


def token_spans(count, capacity, overlap):
    if count < 1 or capacity <= overlap or overlap < 0:
        raise ValueError("Invalid task-window capacity/overlap")
    spans, start, covered = [], 0, 0
    while start < count:
        end = min(count, start + capacity)
        spans.append({"start": start, "end": end, "new_tokens": end - covered})
        covered = end
        if end == count:
            break
        start = end - overlap
    if sum(s["new_tokens"] for s in spans) != count:
        raise ValueError("Task tokens were omitted or counted twice")
    return spans


def pinned_multilingual_parent(*, download=False):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_REPOSITORY, revision=DEFAULT_CHECKPOINT_REVISION,
        local_files_only=not download, allow_patterns=["multilingual/" + p for p in
            ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"]])) / "multilingual"


def encode_group(tok, state, question, label, max_len, head_max_len=256, overlap=128, max_views=32):
    from laya.common import serialize_state, encode_text, state_room, build_sequence
    from laya.train import to_internal
    q = to_internal("correct", question["correct"])
    room = state_room(tok, q, max_len, head_max_len)
    ids = encode_text(tok, serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    if len(ids) <= room:
        windows, spans, weights = [ids], [], [1.]
        task_tokens, prefix_tokens = None, None
    else:
        context = copy.deepcopy(state)
        task = context["target_task"]["item_content"]
        context["target_task"]["item_content"] = "[Complete task supplied in overlapping token windows below]"
        prefix = serialize_state(context) + "\nTarget task window:\n"
        prefix_ids = encode_text(tok, prefix.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        task_ids = encode_text(tok, task.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        spans = token_spans(len(task_ids), room - len(prefix_ids), overlap)
        if len(spans) > max_views:
            raise ValueError("Input exceeds declared view bound; it must not be dropped")
        windows = [prefix_ids + task_ids[s["start"]:s["end"]] for s in spans]
        weights = [s["new_tokens"] / len(task_ids) for s in spans]
        task_tokens, prefix_tokens = len(task_ids), len(prefix_ids)
    encoded = []
    for window in windows:
        if len(window) > room:
            raise ValueError("Grouped view would be truncated")
        sequence, markers, _stats, truncation = build_sequence(tok, None, q, max_len, head_max_len,
            state_ids=window, return_stats=True, return_truncation_stats=True)
        if truncation["truncated"] or len(markers) != 2:
            raise ValueError("Native window or yes/no markers were lost")
        encoded.append({"ids": sequence, "markers": markers, "qtype": 2,
                        "target": [0., 1.] if label else [1., 0.]})
    return {"views": encoded, "weights": weights, "label": int(label), "layout": {
        "views": len(encoded), "source_state_tokens": len(ids), "task_tokens": task_tokens,
        "shared_prefix_tokens": prefix_tokens, "task_spans": spans, "all_task_tokens_covered": True,
        "one_loss_per_original_example": True}}


def verify_qualification_inputs(root, config):
    config = MultilingualRecipe.model_validate(config)
    path = local_path(root, config.prepared)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["config"] != config.model_dump() or manifest["version"] != "multilingual_qualification_inputs_v1":
        raise ValueError("Qualification preparation belongs to a different recipe")
    for name, digest in manifest["artifacts"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or file_hash(target) != digest:
            raise ValueError("Qualification fixture changed: " + name)
    return path, manifest


def prepare_qualification(root, config):
    from transformers import AutoTokenizer
    config = MultilingualRecipe.model_validate(config)
    root = Path(root)
    inputs = local_path(root, config.inputs)
    manifest = json.loads((inputs / "input_manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        if file_hash(inputs / name) != digest:
            raise ValueError("Collected training input changed")
    audit_path = local_path(root, config.capacity_audit)
    audit = json.loads((audit_path / "summary.json").read_text())
    if (audit["model"] != "multilingual" or audit["max_len"] != 8192 or
        audit["input_manifest_sha256"] != file_hash(inputs / "input_manifest.json")):
        raise ValueError("Use the exact multilingual capacity audit for these inputs")
    tok = AutoTokenizer.from_pretrained(pinned_multilingual_parent() / "tokenizer", local_files_only=True)
    if file_hash(pinned_multilingual_parent() / "tokenizer/tokenizer.json") != audit["tokenizer_sha256"]:
        raise ValueError("Pinned tokenizer changed")
    question = json.loads((inputs / "input_contract.json").read_text())["question"]
    visibility = {(r["role"], r["row"]): r for r in read_jsonl(audit_path / "row_visibility.jsonl") if r["arm"] == "collected"}
    out = local_path(root, config.prepared)
    out.mkdir(parents=True, exist_ok=False)
    layouts, real, stress, lengths = [], [], {}, []
    class_counts = Counter()
    all_rows = []
    for role in ["training", "monitor", "development"]:
        for i, row in enumerate(read_jsonl(inputs / "collected" / (role + ".jsonl"))):
            count = visibility[role, i]["state_tokens"]
            if visibility[role, i]["truncated"]:
                group = encode_group(tok, row["state"], question, row.get("expected", {}).get("correct", 0),
                    8192, config.head_max_len, config.overlap_tokens, config.max_views)
                if group["layout"]["source_state_tokens"] != count:
                    raise ValueError("Native tokenizer disagrees with prior visibility audit")
                layout = group["layout"]
            else:
                layout = {"views": 1, "source_state_tokens": count, "all_task_tokens_covered": True,
                          "one_loss_per_original_example": True}
            layouts.append({"role": role, "row": i, **layout})
            if role == "training":
                all_rows.append((count, i, row))
            if len(layouts) % 2000 == 0:
                write_json(out / "progress.json", {"rows_planned": len(layouts)})
    seen = set()
    for count, i, row in sorted(all_rows, key=lambda r: (r[0], r[1])):
        label = int(row["expected"]["correct"])
        key = stable_hash(row["state"])
        if class_counts[label] < 8 and key not in seen:
            real.append({"kind": "real_training_canary", "source_row": i, "state": row["state"], "label": label})
            class_counts[label] += 1
            seen.add(key)
        if len(real) == 16:
            break
    if class_counts != {0: 8, 1: 8}:
        raise ValueError("Insufficient distinct short canary examples in both classes")
    for budget in [4096, 8192]:
        eligible = [r for r in all_rows if r[0] + audit["head_tokens"] + 1 <= budget]
        count, i, row = max(eligible, key=lambda r: r[0])
        stress[str(budget)] = {"kind": "long_training_fixture", "source_row": i, "state": row["state"],
                               "label": int(row["expected"]["correct"])}
    count, i, row = max(all_rows, key=lambda r: r[0])
    stress["multi_view"] = {"kind": "long_training_fixture", "source_row": i, "state": row["state"],
                            "label": int(row["expected"]["correct"])}
    fixture = {"question": question, "real_canary": real, "stress": stress,
               "input_manifest_sha256": file_hash(inputs / "input_manifest.json"), "tokenizer_sha256": audit["tokenizer_sha256"]}
    write_json(out / "fixtures.json", fixture)
    write_jsonl(out / "row_layouts.jsonl", layouts)
    report = {"version": "multilingual_qualification_preparation_v1", "rows_by_role": dict(Counter(r["role"] for r in layouts)),
        "multi_view_rows_by_role": dict(Counter(r["role"] for r in layouts if r["views"] > 1)),
        "maximum_views": max(r["views"] for r in layouts), "all_rows_retained": True, "all_task_tokens_covered": True,
        "aggregation": "unique-task-token-weighted mean of view logits; one native binary loss per original example",
        "cross_window_reasoning_guaranteed": False, "real_canary_examples": 16,
        "real_canary_sampling": "short distinct training states, eight per native class; diagnostic only",
        "nontraining_labels_opened": False, "quality_result": False, "model_weights_loaded": False,
        "input_manifest_sha256": fixture["input_manifest_sha256"], "capacity_audit_sha256": file_hash(audit_path / "summary.json"),
        "tokenizer_sha256": fixture["tokenizer_sha256"], "packages": {n: version(n) for n in ["laya", "transformers", "tokenizers"]}}
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"version": "multilingual_qualification_inputs_v1", "config": config.model_dump(),
        "artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return report
