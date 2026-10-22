"""Bounded, training-only head smoke check. Native training imports stay lazy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HeadSmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    split: str = "configs/head_finetuning/frozen_split_v1.json"
    model: Literal["english"] = "english"
    seed: int = 0
    samples: int = Field(default=256, ge=1, le=256)
    updates: int = Field(default=10, ge=1, le=10)
    micro_batch: int = Field(default=4, ge=1, le=4)
    accumulation: int = Field(default=16, ge=1, le=16)
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-3)
    weight_decay: float = Field(default=.01, ge=0)
    grad_clip: float = Field(default=1., gt=0)
    device: Literal["mps", "cpu", "cuda"] = "mps"
    max_len: Literal[512] = 512
    head_max_len: Literal[192] = 192


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_smoke_rows(provenance, count, seed):
    """Label-blind, deterministic pair-balanced selection of fully visible training rows."""
    frame = provenance.loc[(provenance.role == "training") & provenance.exported &
                           ~provenance.truncated].copy()
    groups = [group.sample(frac=1, random_state=seed).to_dict("records")
              for _, group in frame.groupby(["benchmark", "subject_id"], sort=True)]
    rng = random.Random(seed)
    rng.shuffle(groups)
    selected = []
    while len(selected) < count and any(groups):
        for group in groups:
            if group and len(selected) < count:
                selected.append(group.pop())
    if len(selected) != count:
        raise ValueError("Insufficient fully visible training rows")
    # Equal pair mass in this deliberately short-input smoke distribution.
    counts = {}
    for row in selected:
        key = (row["benchmark"], row["subject_id"])
        counts[key] = counts.get(key, 0) + 1
    for row in selected:
        row["smoke_weight"] = count / (len(counts) * counts[(row["benchmark"], row["subject_id"])])
    return selected


def freeze_for_head(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("head.", "type_emb.", "scorer.")))
    model.train()
    model.encoder.eval()
    model.act_head.eval()
    return [p for p in model.parameters() if p.requires_grad]


def tensor_hashes(model):
    groups = {key: hashlib.sha256() for key in ("encoder", "act_head", "decision")}
    for name, value in sorted(model.state_dict().items()):
        if name == "temperature":
            continue
        group = "encoder" if name.startswith("encoder.") else (
            "act_head" if name.startswith("act_head.") else "decision")
        groups[group].update(name.encode())
        groups[group].update(value.detach().cpu().contiguous().numpy().tobytes())
    return {name: digest.hexdigest() for name, digest in groups.items()}


def weighted_ce(logits, targets, weights, denominator):
    """Each microbatch contributes to the SAME optimizer-window denominator."""
    import torch
    return (-(targets * torch.log_softmax(logits, -1)).sum(-1) * weights).sum() / denominator


def normalize_tokenizer_export(checkpoint):
    """Apply native load-time compatibility fixes BEFORE recording artifact hashes."""
    from laya.agent import _fix_tokenizer_config
    _fix_tokenizer_config(str(checkpoint))
