"""Enriched frozen/upper-encoder experiments, independent of historical pilots.

Native imports are lazy. Operational batch choices come from a saved CUDA
qualification; all arms keep the same data, optimizer windows and precision.
"""
from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
import random
import resource
import time
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .expanded_data import verify_expanded
from .head_pilot import read_rows, write_json, epoch_windows
from .head_training import file_hash, tensor_hashes, weighted_ce


class NonFiniteTrainingGradient(RuntimeError):
    pass


class ComputeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    precision: Literal["float32", "float16", "bfloat16"]
    micro_batch: int = Field(ge=1, le=64)
    grouped: bool = True
    head_checkpointing: bool = False
    fp32_scorer: bool = False


class EncoderTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prepared: str = "runs/head_finetuning/enriched_30k_v1"
    seed: int = 0
    epochs: int = Field(default=1, ge=1, le=3)
    samples: int = Field(default=30000, ge=1, le=50000)
    effective_batch: int = Field(default=64, ge=8, le=128)
    upper_layers: int = Field(default=2, ge=1, le=4)
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-3)
    encoder_learning_rate: float = Field(default=1e-5, gt=0, le=5e-5)
    weight_decay: float = Field(default=.01, ge=0)
    grad_clip: float = Field(default=1., gt=0)
    max_len: Literal[2048] = 2048
    head_max_len: Literal[192] = 192

    @model_validator(mode="after")
    def paths(self):
        path = Path(self.prepared)
        if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("runs",):
            raise ValueError("Prepared data must be inside runs/")
        return self


def set_training_policy(model, upper_layers):
    """Freeze the prefix/action head; train selected actual ModernBERT blocks."""
    layers = model.encoder.layers
    if upper_layers > len(layers) or upper_layers < 0:
        raise ValueError("Invalid upper-layer count")
    prefixes = ("head.", "type_emb.", "scorer.")
    encoder_prefixes = tuple(f"encoder.layers.{i}." for i in range(len(layers) - upper_layers, len(layers)))
    if upper_layers:
        encoder_prefixes += ("encoder.final_norm.",)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes + encoder_prefixes))
    model.train()
    model.encoder.eval()
    for layer in list(layers)[len(layers) - upper_layers:] if upper_layers else []:
        layer.train()
    if upper_layers:
        model.encoder.final_norm.train()
    model.act_head.eval()
    return [p for p in model.parameters() if p.requires_grad]


def grouped_batches(window, lengths, micro_batch, *, grouped=True):
    order = sorted(window, key=lambda i: (lengths[i], i)) if grouped else list(window)
    return [order[i:i + micro_batch] for i in range(0, len(order), micro_batch)]


def collate(items, pad_id, max_len, padding=64):
    from laya.common import collate_items
    from torch.nn.functional import pad
    batch = collate_items([items], pad_id)
    length = batch["input_ids"].shape[1]
    if length > max_len:
        raise ValueError("Native sequence exceeds context limit")
    padded = min(max_len, math.ceil(length / padding) * padding)
    for key, value in [("input_ids", pad_id), ("attention_mask", 0)]:
        batch[key] = pad(batch[key], (0, padded - length), value=value)
    return batch


def keep_scorer_float32(model):
    """Avoid quantizing the difference between large, nearly equal option logits.

    This changes only the compute context, not module names, weights or export
    architecture. Reapplied after loading when the experiment requests it.
    """
    import torch
    if getattr(model.scorer, "_paiec_fp32_forward", False):
        return
    original = model.scorer.forward
    def stable_forward(values):
        with torch.autocast("cuda", enabled=False):
            return original(values.float())
    model.scorer.forward = stable_forward
    model.scorer._paiec_fp32_forward = True


def forward(model, batch, device, precision, upper_layers, fp32_scorer=False):
    import torch
    from laya.train import _forward
    if fp32_scorer:
        keep_scorer_float32(model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(precision)
    context = torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
    with context:
        return _forward(model, batch, device, False, upper_layers == 0)


def optimizer_for(model, config):
    import torch
    head, encoder = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if name.startswith("encoder.") else head).append(parameter)
    groups = [{"params": head, "lr": config.learning_rate}]
    if encoder:
        groups.append({"params": encoder, "lr": config.encoder_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay, fused=True)


def make_scaler(settings):
    import torch
    return torch.amp.GradScaler("cuda", init_scale=1024., growth_interval=2000) if settings["precision"] == "float16" else None


def step_optimizer(optimizer, scaler):
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()


def encode_data(root, config, *, download=False, use_cache=True):
    """Native token/mask and exact renderer checks on every prepared record."""
    from laya.train import items_from_rows, encode_item
    from .head_pilot_training import cached_parent
    from .compact_inputs import CompactStateBuilder
    from .compact_profiles import CompactProfiles
    from .head_pilot import digest_state
    from transformers import AutoTokenizer
    prepared = Path(root) / config.prepared
    manifest = verify_expanded(root, prepared)
    from .laya_runtime import DEFAULT_CHECKPOINT_REVISION
    if manifest["checkpoint_revision"] != DEFAULT_CHECKPOINT_REVISION:
        raise ValueError("Expanded parent differs from pinned original")
    parent = cached_parent(manifest["checkpoint_revision"], allow_download=download)
    tok = AutoTokenizer.from_pretrained(parent / "tokenizer", local_files_only=True)
    cache_manifest = prepared / "native_tokens_manifest.json"
    if use_cache and cache_manifest.is_file():
        return load_token_cache(root, config, parent, tok, manifest)
    builder = CompactStateBuilder(tok, manifest["state_room"], CompactProfiles(prepared / "profile_bundle/policy.json", root=prepared))
    sidecar = pd.read_parquet(prepared / "enriched/provenance.parquet")
    data = {}
    for role in ["training", "monitor", "development"]:
        rows = read_rows(prepared / "enriched" / (role + ".jsonl"))
        sources = read_rows(prepared / "source" / (role + ".jsonl"))
        frame = sidecar[sidecar.role.eq(role)].sort_values("jsonl_line").reset_index(drop=True)
        if len(frame) != len(rows) or len(sources) != len(rows):
            raise ValueError("Expanded provenance count differs")
        # Labels stay loss/scorer side; the model encoder reads state/questions only.
        items, skipped = items_from_rows(tok, rows, config.max_len, config.head_max_len)
        if skipped or len(items) != len(rows):
            raise ValueError(f"Native conversion failed: {role} {skipped}")
        encoded = [encode_item(tok, item, config.max_len, config.head_max_len) for item in items]
        for row, source, record, native in zip(rows, sources, frame.to_dict("records"), encoded):
            ids = builder.ids(row["state"])
            if digest_state(row["state"]) != record["state_sha256"] or digest_state(json.dumps(ids)) != record["effective_input_sha256"]:
                raise ValueError("Native retained input changed")
            if row["questions"] != manifest["question"] or row["expected"] != source["expected"]:
                raise ValueError("Question/label changed")
            parsed = json.loads(source["state"])
            if builder.build(parsed["target_system"], parsed["target_item"]).enriched != row["state"]:
                raise ValueError("Runtime enrichment differs from training")
            if len(native["ids"]) > config.max_len or len(native["markers"]) != 2:
                raise ValueError("Native options/context changed")
        data[role] = dict(rows=rows, frame=frame, encoded=encoded)
        print(f"Native checks passed: {role} / {len(rows)}", flush=True)
    if len(data["training"]["rows"]) != config.samples:
        raise ValueError("Training count differs from recipe")
    return parent, tok, manifest, data


def save_token_cache(root, config):
    """Do the expensive CPU preparation locally before renting a GPU."""
    import gzip
    from importlib.metadata import version
    parent, tok, manifest, data = encode_data(root, config, use_cache=False)
    prepared = Path(root) / config.prepared
    target = prepared / "native_tokens.jsonl.gz"
    if target.exists() or (prepared / "native_tokens_manifest.json").exists():
        raise FileExistsError("Native token preparation is immutable")
    with gzip.open(target, "wt") as handle:
        for role, values in data.items():
            for encoded in values["encoded"]:
                handle.write(json.dumps(dict(role=role, encoded=encoded)) + "\n")
    write_json(prepared / "native_tokens_manifest.json", dict(
        prepared_manifest_sha256=file_hash(prepared / "manifest.json"),
        token_file_sha256=file_hash(target), max_len=config.max_len, head_max_len=config.head_max_len,
        packages={name: version(name) for name in ["laya", "transformers", "tokenizers"]},
        parent_weights_sha256=file_hash(parent / "model.safetensors"),
        rows={role: len(values["encoded"]) for role, values in data.items()},
        full_native_render_token_parity=True, model_weights_loaded=False, training_performed=False))
    return dict(bytes=target.stat().st_size, rows={role: len(values["encoded"]) for role, values in data.items()})


def load_token_cache(root, config, parent, tok, manifest):
    import gzip
    from importlib.metadata import version
    from laya.train import items_from_rows, encode_item
    prepared = Path(root) / config.prepared
    cache = json.loads((prepared / "native_tokens_manifest.json").read_text())
    if cache["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json") or cache["token_file_sha256"] != file_hash(prepared / "native_tokens.jsonl.gz"):
        raise ValueError("Native token cache changed")
    if (cache["max_len"], cache["head_max_len"]) != (config.max_len, config.head_max_len):
        raise ValueError("Native token context changed")
    if any(version(name) != expected for name, expected in cache["packages"].items()):
        raise ValueError("Native token package version changed")
    if file_hash(parent / "model.safetensors") != cache["parent_weights_sha256"]:
        raise ValueError("Token cache parent changed")
    sidecar = pd.read_parquet(prepared / "enriched/provenance.parquet")
    data = {role: dict(rows=read_rows(prepared / "enriched" / (role + ".jsonl")),
                      frame=sidecar[sidecar.role.eq(role)].sort_values("jsonl_line").reset_index(drop=True), encoded=[])
            for role in ["training", "monitor", "development"]}
    with gzip.open(prepared / "native_tokens.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            data[row["role"]]["encoded"].append(row["encoded"])
    for role, values in data.items():
        if len(values["rows"]) != len(values["encoded"]) or len(values["encoded"]) != cache["rows"][role]:
            raise ValueError("Native token count changed")
        longest = max(range(len(values["encoded"])), key=lambda i: len(values["encoded"][i]["ids"]))
        indices = sorted(set([0, longest, *random.Random(config.seed).sample(range(len(values["rows"])), min(32, len(values["rows"])))]))
        items, skipped = items_from_rows(tok, [values["rows"][i] for i in indices], config.max_len, config.head_max_len)
        if skipped or len(items) != len(indices):
            raise ValueError("Cached native fixture conversion failed")
        for i, item in zip(indices, items):
            if encode_item(tok, item, config.max_len, config.head_max_len) != values["encoded"][i]:
                raise ValueError("Cached native fixture differs")
        print(f"Verified native token cache: {role} / {len(values['encoded'])}", flush=True)
    return parent, tok, manifest, data


def train_window(model, optimizer, data, window, tok, config, settings, upper_layers, scaler=None):
    if settings["precision"] == "float16" and scaler is None:
        raise ValueError("FP16 training requires explicit gradient scaling")
    import torch
    device = torch.device("cuda")
    weights = data["frame"].sample_weight.to_numpy()
    denominator = float(weights[window].sum())
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    lengths = [len(item["ids"]) for item in data["encoded"]]
    for indices in grouped_batches(window, lengths, settings["micro_batch"], grouped=settings.get("grouped", True)):
        batch = collate([data["encoded"][i] for i in indices], tok.pad_token_id, config.max_len)
        logits = forward(model, batch, device, settings["precision"], upper_layers, settings.get("fp32_scorer", False))
        loss = weighted_ce(logits, batch["target"].to(device), torch.as_tensor(weights[indices], device=device, dtype=torch.float32), denominator)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        total += loss.detach()
    if scaler is not None:
        scaler.unscale_(optimizer)
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip, error_if_nonfinite=False)
    if not torch.isfinite(total).item() or not torch.isfinite(norm).item():
        raise NonFiniteTrainingGradient("Non-finite loss/gradient; no optimizer step applied")
    return float(total), float(norm)


def predict(model, tok, data, config, settings, upper_layers):
    import torch
    model.eval()
    encoded = data["encoded"]
    probabilities = np.empty(len(encoded), dtype=np.float64)
    # Group globally for evaluation and restore original item/trial order.
    batches = grouped_batches(list(range(len(encoded))), [len(v["ids"]) for v in encoded], settings["micro_batch"])
    with torch.no_grad():
        for indices in batches:
            batch = collate([encoded[i] for i in indices], tok.pad_token_id, config.max_len)
            logits = forward(model, batch, torch.device("cuda"), settings["precision"], upper_layers, settings.get("fp32_scorer", False))
            probabilities[indices] = logits.softmax(-1)[:, 1].cpu().numpy()
    return probabilities


def train_arm(root, out, config, settings, upper_layers, encoded_data=None):
    import torch
    from laya.train import load_checkpoint
    from .head_pilot_training import save_pilot_checkpoint
    from .head_evaluation import score_frozen_development
    from .laya_runtime import local_checkpoint_files
    from .tracking import ExperimentTracker
    prepared = Path(root) / config.prepared
    arm = "frozen" if upper_layers == 0 else f"upper_{upper_layers}"
    out = Path(out) / arm
    out.mkdir(parents=True, exist_ok=False)
    parent, tok, manifest, data = encoded_data or encode_data(root, config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _, native = load_checkpoint(str(parent))
    if native.get("option_layout", "sequential") != "sequential":
        raise ValueError("Sequential option layout required")
    before = tensor_hashes(model)
    model.cuda()
    model.head_checkpointing = settings.get("head_checkpointing", False)
    parameters = set_training_policy(model, upper_layers)
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    write_json(out / "trainable_parameters.json", dict(names=trainable_names, count=sum(p.numel() for p in parameters)))
    # Frozen-prefix hash separately excludes the intentionally trainable upper blocks.
    import hashlib
    def frozen_hash():
        digest = hashlib.sha256()
        for name, value in sorted(model.named_parameters()):
            if not value.requires_grad:
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    frozen_before = frozen_hash()
    optimizer = optimizer_for(model, config)
    scaler = make_scaler(settings)
    tracker = ExperimentTracker(Path(root), experiment="PAIEC Encoder Adaptation")
    with tracker.run(f"enriched 30k / {arm}", dict(config=config.model_dump(), settings=settings,
                      prepared_manifest_sha256=file_hash(prepared / "manifest.json")),
                     dict(arm=arm, trainable_parameters=sum(p.numel() for p in parameters),
                          samples=config.samples, precision=settings["precision"], backend="torch", device="cuda"),
                     dict(run_type="enriched_encoder_experiment", source_eligibility="research_only", confirmation_scored="false")) as tracked:
        history, monitor_history = [], []
        initial_p = predict(model, tok, data["monitor"], config, settings, upper_layers)
        monitor_labels = np.array([r["expected"]["correct"] for r in data["monitor"]["rows"]], dtype=float)
        write_json(out / "initial_monitor.json", dict(brier=float(np.average((initial_p-monitor_labels)**2, weights=data["monitor"]["frame"].sample_weight))))
        updates = 0
        start = time.perf_counter()
        for epoch in range(config.epochs):
            set_training_policy(model, upper_layers)
            for window in epoch_windows(config.samples, config.effective_batch, config.seed, epoch):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                step_start = time.perf_counter()
                loss, norm = train_window(model, optimizer, data["training"], window, tok, config, settings, upper_layers, scaler)
                step_optimizer(optimizer, scaler)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                updates += 1
                record = dict(update=updates, epoch=epoch+1, loss=loss, grad_norm=norm,
                    presentations=len(window), seconds=time.perf_counter()-step_start,
                    grad_scale=scaler.get_scale() if scaler else 1.,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved())
                history.append(record)
                with (out / "history.jsonl").open("a") as handle:
                    handle.write(json.dumps(record)+"\n")
                if tracked:
                    for key in ["loss", "seconds", "grad_norm", "cuda_peak_allocated_bytes"]:
                        tracked.client.log_metric(tracked.run_id, "train/"+key, record[key], step=updates)
                if updates % 25 == 0:
                    print(json.dumps(dict(arm=arm, **record)), flush=True)
            monitor_p = predict(model, tok, data["monitor"], config, settings, upper_layers)
            monitor_history.append(dict(epoch=epoch+1, brier=float(np.average((monitor_p-monitor_labels)**2, weights=data["monitor"]["frame"].sample_weight))))
            write_json(out / "monitor_history.json", monitor_history)
            data["monitor"]["frame"][["row_id", "benchmark", "subject_id", "item_id"]].assign(probability=monitor_p).to_csv(out / "monitor_predictions.csv", index=False)
            # One epoch is the initial frozen recipe; no checkpoint selection on development.
        after = tensor_hashes(model)
        if frozen_hash() != frozen_before or before["act_head"] != after["act_head"]:
            raise RuntimeError("Frozen parameters changed")
        if (before["encoder"] == after["encoder"]) != (upper_layers == 0) or before["decision"] == after["decision"]:
            raise RuntimeError("Intended modules did not change")
        p = predict(model, tok, data["development"], config, settings, upper_layers)
        # Repeated native inputs must receive a single probability at scorer boundary.
        frame = data["development"]["frame"]
        p = pd.Series(p).groupby(frame.state_sha256).transform("mean").to_numpy()
        labels = [int(r["expected"]["correct"]) for r in data["development"]["rows"]]
        result, predictions = score_frozen_development(frame, labels, p)
        predictions.to_csv(out / "predictions.csv", index=False)
        result.frame.to_csv(out / "scores.csv", index=False)
        provenance = dict(parent_revision=manifest["checkpoint_revision"], device="cuda", precision=settings["precision"],
            upper_layers=upper_layers, trainable_parameters=sum(v.numel() for v in parameters),
            prepared_manifest_sha256=file_hash(prepared / "manifest.json"), parent_weights_sha256=file_hash(parent / "model.safetensors"),
            tensor_hashes_before=before, tensor_hashes_after=after, frozen_parameter_sha256=frozen_before,
            training_policy="head and selected upper encoder blocks/final norm; frozen action head and prefix",
            epoch=config.epochs, development_brier_alc=result.brier_alc, settings=settings)
        save_pilot_checkpoint(model, tok, native, out / "best", prepared, config, "enriched", provenance)
        training_manifest = json.loads((out / "best/paiec_training.json").read_text())
        training_manifest["run_type"] = "enriched_encoder_experiment"
        write_json(out / "best/paiec_training.json", training_manifest)
        _, identity = local_checkpoint_files(str(out / "best"), "english")
        # Export ordinary native weights; no adapter runtime or inference shim needed.
        fixtures = dict(data["training"], encoded=data["training"]["encoded"][:4])
        expected = predict(model, tok, fixtures, config, settings, upper_layers)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler, parameters, model
        gc.collect()
        torch.cuda.empty_cache()
        model, _, _ = load_checkpoint(str(out / "best"))
        if tensor_hashes(model) != after:
            raise RuntimeError("Reloaded weights differ")
        model.cuda()
        actual = predict(model, tok, fixtures, config, settings, upper_layers)
        drift = float(np.max(np.abs(expected-actual)))
        if drift > 1e-5:
            raise RuntimeError("Reloaded probabilities differ")
        report = dict(status="finished", arm=arm, brier_alc=result.brier_alc, updates=updates,
            presentations=sum(v["presentations"] for v in history), settings=settings,
            upper_layers=upper_layers, epochs=config.epochs, training_monitor=monitor_history,
            training_development_export_seconds=time.perf_counter()-start,
            peak_cuda_allocated_bytes=max(v["cuda_peak_allocated_bytes"] for v in history),
            peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            reload_max_probability_difference=drift, checkpoint_identity=identity["checkpoint_revision"],
            tensor_hashes_before=before, tensor_hashes_after=after,
            frozen_parameters_unchanged=True, constant_brier_alc=.25,
            calibration_opened=False, confirmation_opened=False, resume_supported=False,
            raw_head_no_revealed_evidence=True, mlflow_run_id=tracked.run_id if tracked else None)
        write_json(out / "report.json", report)
        if tracked:
            tracked.result(result, report["training_development_export_seconds"], predictions)
            tracked.json(report, "training_report.json")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(report), flush=True)
    return report
