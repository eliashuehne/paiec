"""Small synthetic file inventories for transport and integrity tests.

These tests inspect paths and hashes, not Parquet content or model behavior.
Nothing here reads or copies real training, development, or sealed data.
"""

import json
from pathlib import Path

from paiec.head_pilot import HeadPilotConfig
from paiec.head_training import file_hash
from paiec.laya_predictor import load_questions


def write_file(root, name, contents):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def write_json(root, name, value):
    return write_file(root, name, json.dumps(value))


def write_inventory_fixture(root):
    directory = "runs/head_finetuning/fixture_data"
    split = {"artifact_directory": directory, "artifact_fingerprints": {}}
    for name in ["training.jsonl", "development.jsonl", "provenance.parquet",
                 "confirmation.jsonl", "calibration.jsonl"]:
        path = write_file(root, f"{directory}/{name}", "synthetic inventory entry\n")
        split["artifact_fingerprints"][name] = file_hash(path)
    write_json(root, "configs/head_finetuning/frozen_split_v1.json", split)
    write_json(root, "configs/suites.json", {"smoke": {"benchmarks": ["toy_a", "toy_b"]}})
    write_json(root, "payload/data/corpus_manifest.json", {"revision": "fixture"})
    for benchmark in ["toy_a", "toy_b"]:
        for name in ["subjects.parquet", "items.parquet", "response.parquet", "traces.parquet"]:
            write_file(root, f"payload/data/{benchmark}/formatted_tables/{name}", "inventory only\n")


def write_pilot_fixture(root):
    write_inventory_fixture(root)
    config = HeadPilotConfig()
    write_json(root, "configs/head_finetuning/enrichment_pilot_v1.json", config.model_dump())
    artifacts = {}
    for variant in ["baseline", "enriched"]:
        for role in ["training", "development"]:
            name = f"{variant}/{role}.jsonl"
            path = write_json(root, f"{config.prepared}/{name}", {"state": "synthetic", "expected": 1})
            artifacts[name] = file_hash(path)
    renderer = "src/paiec/fixture_renderer.py"
    write_file(root, renderer, "# synthetic renderer contract\n")
    write_json(root, f"{config.prepared}/manifest.json", {
        "config": config.model_dump(), "artifacts": artifacts,
        "split_sha256": file_hash(Path(root) / config.split),
        "question": load_questions(),
        "renderer_sources": {renderer: file_hash(Path(root) / renderer)},
    })
    return config
