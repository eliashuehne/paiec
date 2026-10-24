"""Native tokenizer/target conversion only: never loads a Laya model."""
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

from .feature_preparation import read_jsonl, write_json
from .head_training import file_hash


def audit_native_inputs(prepared, out, *, model="english", max_len=None, head_max_len=None):
    from laya.common import build_sequence, build_head, state_room, serialize_state, render_options
    from laya.train import to_internal, items_from_rows
    from transformers import AutoTokenizer
    from .head_pilot_training import cached_parent
    from .laya_runtime import DEFAULT_CHECKPOINT_REVISION
    from .laya_runtime import MODEL_REPOSITORY
    from huggingface_hub import snapshot_download

    prepared, out = Path(prepared), Path(out)
    manifest = json.loads((prepared / "input_manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared.resolve()) or file_hash(path) != digest:
            raise ValueError("Prepared Laya input changed before token audit")
    contract = json.loads((prepared / "input_contract.json").read_text())
    if model not in {"english", "multilingual"}:
        raise ValueError("Use an explicit original pinned Laya checkpoint")
    if model == "multilingual":
        parent = Path(snapshot_download(MODEL_REPOSITORY, revision=DEFAULT_CHECKPOINT_REVISION, local_files_only=True,
            allow_patterns=["multilingual/rl_agent_config.json", "multilingual/encoder/config.json", "multilingual/tokenizer/*"])) / "multilingual"
    else:
        parent = cached_parent(DEFAULT_CHECKPOINT_REVISION)
    agent_config = json.loads((parent / "rl_agent_config.json").read_text())
    encoder_config = json.loads((parent / "encoder/config.json").read_text())
    max_len = contract["max_len"] if max_len is None else max_len
    head_max_len = contract["head_max_len"] if head_max_len is None else head_max_len
    if max_len > encoder_config["max_position_embeddings"]:
        raise ValueError("Requested audit exceeds the pinned encoder position limit")
    tok = AutoTokenizer.from_pretrained(parent / "tokenizer", local_files_only=True)
    q = to_internal("correct", contract["question"]["correct"])
    head, markers, head_stats = build_head(tok, q, head_max_len)
    if len(markers) != 2 or head_stats["options_distinct"] != 2:
        raise ValueError("Native yes/no option spans collapsed")
    rooms = {budget: state_room(tok, q, budget, head_max_len) for budget in sorted({512, 1024, 2048, 4096, 8192, max_len})}
    out.mkdir(parents=True, exist_ok=False)
    summaries = {}
    targets_checked = set()
    with (out / "row_visibility.jsonl").open("w") as visibility:
        for arm in ["recorded", "collected"]:
            for role in ["training", "monitor", "development"]:
                counts, complete = Counter(), Counter()
                for i, row in enumerate(read_jsonl(prepared / arm / (role + ".jsonl"))):
                    if row["questions"] != contract["question"] or (role != "training" and "expected" in row):
                        raise ValueError("Question changed or nontraining targets entered export")
                    ids, native_markers, stats, truncation = build_sequence(tok, row["state"], q,
                        max_len, head_max_len, return_stats=True, return_truncation_stats=True)
                    if len(native_markers) != 2 or not ids:
                        raise ValueError("Native question is not representable")
                    counts["rows"] += 1
                    counts["overflow_rows"] += truncation["truncated"]
                    counts["state_tokens"] += truncation["state_tokens"]
                    counts["mask_token_replacement_rows"] += tok.mask_token in serialize_state(row["state"])
                    for budget, room in rooms.items():
                        complete[budget] += truncation["state_tokens"] <= room
                    if role == "training":
                        label = row["expected"]["correct"]
                        key = arm, int(label)
                        if key not in targets_checked:
                            items, skipped = items_from_rows(tok, [row], max_len, head_max_len)
                            if skipped or len(items) != 1 or items[0]["target"] != ([0., 1.] if label else [1., 0.]):
                                raise ValueError("Native target order differs from false/true labels")
                            targets_checked.add(key)
                    visibility.write(json.dumps({"arm": arm, "role": role, "row": i, **truncation}) + "\n")
                    if (i + 1) % 2000 == 0:
                        write_json(out / "progress.json", {"arm": arm, "role": role, "rows_checked": i + 1})
                summaries[arm + "/" + role] = {**dict(counts), "complete_state_rows_by_total_budget": dict(complete)}
    report = {"version": "collected_input_native_audit_v2", "input_manifest_sha256": file_hash(prepared / "input_manifest.json"),
        "model": model, "shipped_max_len": agent_config["max_len"], "shipped_head_max_len": agent_config["head_max_len"],
        "encoder_max_position_embeddings": encoder_config["max_position_embeddings"],
        "audit_overrides_export_context": max_len != contract["max_len"] or head_max_len != contract["head_max_len"] or model != "english",
        "packages": {name: version(name) for name in ["laya", "transformers", "tokenizers"]},
        "native_common_sha256": file_hash(__import__("laya.common", fromlist=["x"]).__file__),
        "native_train_sha256": file_hash(__import__("laya.train", fromlist=["x"]).__file__),
        "tokenizer_sha256": file_hash(parent / "tokenizer/tokenizer.json"),
        "encoder_config_sha256": file_hash(parent / "encoder/config.json"),
        "checkpoint_revision": DEFAULT_CHECKPOINT_REVISION, "head_tokens": len(head), "state_room": rooms[max_len],
        "native_option_texts": render_options(q), "native_head_preview": tok.decode(head),
        "max_len": max_len, "head_max_len": head_max_len, "results": summaries, "native_schema_valid": True,
        "target_conversion_checks": len(targets_checked), "question_criteria_unchanged": True,
        "full_visibility_at_requested_limit": all(s["overflow_rows"] == 0 for s in summaries.values()),
        "training_ready": False, "new_format_learning_gain_established": False,
        "blockers": ["Use an explicit capacity/packing policy before training any overflowing row.",
                     "This is a new input contract, not a replacement for historical checkpoint renderers.",
                     "Research enrichment has not been approved for competition use."],
        "native_model_loaded": False, "model_forward_passes": 0, "training_updates": 0,
        "nontraining_outcomes_read": False, "revealed_examples_used": False, "network_downloads": False}
    write_json(out / "summary.json", report)
    return report
