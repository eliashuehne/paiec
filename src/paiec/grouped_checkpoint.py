"""Self-contained grouped checkpoints; no implicit catalog or format upgrades."""
import json
from pathlib import Path
import shutil

from .grouped_inputs import FORMAT, AGGREGATION, SNAPSHOT_FILES
from .feature_preparation import write_json
from .head_training import file_hash, normalize_tokenizer_export
from .laya_predictor import load_questions

RENDERER_FILES = ["grouped_inputs.py", "grouped_engine.py", "grouped_checkpoint.py", "grouped_predictor.py",
                  "collected_inputs.py", "collected_inputs_v2.py", "collected_inputs_v3.py", "multilingual_recipe.py",
                  "competition_fields.py", "competition_data.py", "feature_preparation.py", "system_profiles.py",
                  "encoder_training.py", "grouped_data.py", "grouped_learning.py", "schema.py",
                  "laya_predictor.py", "head_training.py"]


def renderer_fingerprints():
    folder = Path(__file__).parent
    return {"src/paiec/" + name: file_hash(folder / name) for name in RENDERER_FILES}


def snapshot_collection(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in SNAPSHOT_FILES:
        shutil.copyfile(source / name, destination / name)
    write_json(destination / "manifest.json", {"version": "feature_preparation_v1",
        "artifacts": {name: file_hash(destination / name) for name in SNAPSHOT_FILES}})


def save_grouped_checkpoint(model, tok, native_config, destination, snapshot, *, contract, training):
    from laya.train import save_checkpoint
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Grouped checkpoints never overwrite an existing checkpoint")
    save_checkpoint(model, tok, dict(native_config, max_len=contract["max_len"], head_max_len=contract["head_max_len"]), str(destination))
    normalize_tokenizer_export(destination)
    snapshot_collection(snapshot, destination / "collection_snapshot")
    write_json(destination / "questions.json", load_questions())
    bound = dict(contract, state_format=FORMAT, aggregation=AGGREGATION, source_policy="research_only",
        model="multilingual", question_sha256=file_hash(Path(__file__).resolve().parents[2] / "configs/laya_question.json"),
        renderer_sources=renderer_fingerprints(), snapshot="collection_snapshot", probability="raw_grouped_softmax", evidence_mode="none")
    manifest = {"version": "grouped_enriched_checkpoint_v1", "model": "multilingual", "input_contract": bound,
        "training": training, "artifact_sha256": {str(p.relative_to(destination)): file_hash(p)
            for p in sorted(destination.rglob("*")) if p.is_file()}}
    write_json(destination / "paiec_training.json", manifest)
    return manifest


def verify_grouped_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads((checkpoint / "paiec_training.json").read_text())
    c = manifest["input_contract"]
    if (manifest.get("version") != "grouped_enriched_checkpoint_v1" or manifest.get("model") != "multilingual" or
        c.get("state_format") != FORMAT or c.get("aggregation") != AGGREGATION or c.get("source_policy") != "research_only" or
        c.get("probability") != "raw_grouped_softmax" or c.get("evidence_mode") != "none"):
        raise ValueError("Unsupported grouped checkpoint/input contract")
    required = {"model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "questions.json",
                "collection_snapshot/manifest.json", *["collection_snapshot/" + n for n in SNAPSHOT_FILES]}
    if not required <= set(manifest["artifact_sha256"]):
        raise ValueError("Grouped checkpoint is not self-contained")
    for name, digest in manifest["artifact_sha256"].items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint) or file_hash(path) != digest:
            raise ValueError("Grouped checkpoint artifact changed: " + name)
    if c["renderer_sources"] != renderer_fingerprints():
        raise ValueError("Grouped checkpoint renderer changed")
    question = json.loads((checkpoint / "questions.json").read_text())
    if question != load_questions() or c["question_sha256"] != file_hash(Path(__file__).resolve().parents[2] / "configs/laya_question.json"):
        raise ValueError("Grouped training/inference question differs")
    if (c.get("snapshot") != "collection_snapshot" or c.get("max_len") != 8192 or c.get("head_max_len") != 256 or
            c.get("overlap_tokens") != 128 or c.get("max_views") != 32):
        raise ValueError("Grouped checkpoint differs from qualified window settings")
    return manifest
