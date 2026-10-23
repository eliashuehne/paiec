"""Prepared, matched head-only pilots. Preparation never loads model weights."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .head_training import file_hash
from .profile_enrichment import EnrichmentPolicy, catalog_fingerprint

VARIANTS = {"baseline": "context_v1", "enriched": "enriched_v1"}


def variant_format(config, variant):
    if config.input_pipeline == "compact_v3":
        return "compact_v3" if variant == "baseline" else "compact_enriched_v3"
    if config.input_pipeline == "task_v2":
        return "task_v2" if variant == "baseline" else "task_enriched_v2"
    return VARIANTS[variant]


class HeadPilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    split: str = "configs/head_finetuning/frozen_split_v1.json"
    policy: str = "configs/head_finetuning/enrichment_v1.json"
    prepared: str = "runs/head_finetuning/enrichment_pilot_v1"
    output: str = "runs/head_finetuning/enrichment_pilot_v1_training"
    seed: int = 0
    epochs: int = Field(default=4, ge=1, le=4)
    samples: int = Field(default=5000, ge=1, le=5000)
    micro_batch: int = Field(default=2, ge=1, le=4)
    accumulation: int = Field(default=32, ge=1, le=64)
    evaluation_batch: int = Field(default=2, ge=1, le=4)
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-3)
    weight_decay: float = Field(default=.01, ge=0)
    grad_clip: float = Field(default=1., gt=0)
    max_len: int = Field(default=1024, ge=512, le=2048)
    head_max_len: Literal[192] = 192
    device: Literal["mps", "cuda"] = "mps"
    precision: Literal["float32"] = "float32"
    input_pipeline: Literal["legacy", "task_v2", "compact_v3"] = "legacy"

    @model_validator(mode="after")
    def safe_output_paths(self):
        if self.input_pipeline != "compact_v3" and self.max_len > 1024:
            raise ValueError("Historical pilot recipes are bounded to 1024 tokens")
        for name in ["prepared", "output"]:
            path = Path(getattr(self, name))
            if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("runs",):
                raise ValueError("Pilot artifacts must use repository-relative paths inside runs/")
        if self.prepared == self.output:
            raise ValueError("Preparation and training need separate directories")
        return self

    @property
    def effective_batch(self):
        return self.micro_batch * self.accumulation

    @property
    def updates(self):
        return self.epochs * math.ceil(self.samples / self.effective_batch)


def load_config(path):
    return HeadPilotConfig.model_validate_json(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest_state(state):
    return hashlib.sha256(state.encode()).hexdigest()


def epoch_windows(count, effective_batch, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return [order[i:i + effective_batch] for i in range(0, count, effective_batch)]


def pair_weights(manifest):
    counts = manifest.groupby(["benchmark", "subject_id"]).row_id.transform("size")
    pairs = len(manifest[["benchmark", "subject_id"]].drop_duplicates())
    return (len(manifest) / (pairs * counts)).tolist()


def snapshot_profiles(policy_path, destination, root):
    if json.loads(Path(policy_path).read_text()).get("version") == "compact_profiles_v3":
        from .compact_profiles import CompactPolicy
        policy = CompactPolicy.model_validate_json(Path(policy_path).read_text())
    else:
        policy = EnrichmentPolicy.model_validate_json(Path(policy_path).read_text())
    target = Path(destination) / "profile_bundle"
    (target / "catalog").mkdir(parents=True)
    for name in ["profiles.jsonl", "sources.jsonl"]:
        shutil.copyfile(Path(root) / policy.catalog / name, target / "catalog" / name)
    bundled = policy.model_dump()
    bundled["catalog"] = "profile_bundle/catalog"
    write_json(target / "policy.json", bundled)
    return target / "policy.json"


def input_contract(prepared, variant, config):
    contract = {"state_format": variant_format(config, variant), "max_len": config.max_len,
                "head_max_len": config.head_max_len, "use_labels": False,
                "question_sha256": file_hash(Path(__file__).resolve().parents[2] / "configs/laya_question.json"),
                "renderer_sources": {name: file_hash(Path(__file__).resolve().parents[2] / name) for name in
                    ["src/paiec/laya_predictor.py", "src/paiec/state_rendering.py",
                     "src/paiec/profile_enrichment.py", "src/paiec/input_contract.py"]}}
    if config.input_pipeline == "task_v2":
        for name in ["src/paiec/task_inputs.py", "src/paiec/task_predictor.py", "src/paiec/competition.py"]:
            contract["renderer_sources"][name] = file_hash(Path(__file__).resolve().parents[2] / name)
    if config.input_pipeline == "compact_v3":
        for name in ["compact_inputs.py", "compact_profiles.py", "compact_predictor.py", "predictor_factory.py",
                     "task_inputs.py", "task_predictor.py", "competition.py", "system_profiles.py", "schema.py"]:
            path = "src/paiec/" + name
            contract["renderer_sources"][path] = file_hash(Path(__file__).resolve().parents[2] / path)
    if variant == "enriched" or config.input_pipeline == "compact_v3":
        contract.update(policy_file="profile_bundle/policy.json",
                        policy_sha256=file_hash(prepared / "profile_bundle/policy.json"),
                        catalog_sha256=catalog_fingerprint(prepared / "profile_bundle/catalog"))
    return contract


def verify_prepared(root, config):
    prepared = Path(root) / config.prepared
    manifest = json.loads((prepared / "manifest.json").read_text())
    # Placement and a fresh output directory can change independently of the
    # frozen inputs and scientific recipe. Record both in run provenance.
    prepared_config = HeadPilotConfig.model_validate(manifest["config"])
    if prepared_config.model_copy(update={"output": config.output, "device": config.device}) != config:
        raise ValueError("Prepared configuration differs; prepare a new version")
    if manifest["split_sha256"] != file_hash(Path(root) / config.split):
        raise ValueError("Frozen split changed")
    for name, expected in manifest["artifacts"].items():
        if file_hash(prepared / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    from .laya_predictor import load_questions
    if manifest["question"] != load_questions():
        raise ValueError("Active question differs from preparation")
    for name, expected in manifest["renderer_sources"].items():
        if file_hash(Path(root) / name) != expected:
            raise ValueError(f"Input renderer changed since preparation: {name}")
    return prepared, manifest
