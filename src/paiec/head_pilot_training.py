"""Shared PyTorch MPS/CUDA pilot, entered only by the explicit --train action."""
from __future__ import annotations

import gc
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .head_evaluation import score_frozen_development
from .head_pilot import (read_rows, write_json, verify_prepared, epoch_windows,
                         pair_weights, VARIANTS, digest_state, variant_format)
from .head_training import (file_hash, freeze_for_head, tensor_hashes, weighted_ce,
                            normalize_tokenizer_export)
from .laya_runtime import DEFAULT_CHECKPOINT_REVISION, MODEL_REPOSITORY
from .tracking import ExperimentTracker

MEMORY_POLICY = {"mps_memory_fraction": .7, "padding_multiple": 128,
                 "cache_trim": "every optimizer update and every 16 development batches"}


def collate_pilot(items, pad_id, max_len):
    """Bound MPS graph shapes without changing tokens, positions or target masks."""
    from laya.common import collate_items
    from torch.nn.functional import pad
    batch = collate_items([items], pad_id)
    length = batch["input_ids"].shape[1]
    multiple = MEMORY_POLICY["padding_multiple"]
    padded = min(max_len, ((length + multiple - 1) // multiple) * multiple)
    if length > max_len:
        raise ValueError("Native sequence exceeds the configured context")
    if padded > length:
        batch["input_ids"] = pad(batch["input_ids"], (0, padded - length), value=pad_id)
        batch["attention_mask"] = pad(batch["attention_mask"], (0, padded - length), value=0)
    return batch


def cached_parent(revision, *, allow_download=False):
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(MODEL_REPOSITORY, revision=revision,
        local_files_only=not allow_download, allow_patterns=["model.safetensors", "rl_agent_config.json",
                                               "encoder/config.json", "tokenizer/*"]))
    for name in ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"]:
        if not (directory / name).is_file():
            raise ValueError(f"Missing cached parent artifact: {name}")
    return directory


def preflight(root, config, *, allow_download=False):
    """Validate native serialization and local resources without loading weights."""
    import torch
    from transformers import AutoTokenizer
    from .profile_enrichment import ProfileEnricher, native_budget
    from .laya_predictor import build_state

    prepared, manifest = verify_prepared(root, config)
    available = torch.backends.mps.is_available() if config.device == "mps" else torch.cuda.is_available()
    if not available:
        raise RuntimeError(f"Pilot requires {config.device}; no silent device fallback")
    if manifest["checkpoint_revision"] != DEFAULT_CHECKPOINT_REVISION:
        raise ValueError("Pilot parent differs from the pinned original checkpoint")
    parent = cached_parent(manifest["checkpoint_revision"], allow_download=allow_download)
    tok = AutoTokenizer.from_pretrained(parent / "tokenizer", local_files_only=True)
    count, room = native_budget(tok, manifest["question"]["correct"], config.max_len, "torch")
    if config.input_pipeline == "compact_v3":
        from .compact_profiles import CompactProfiles
        from .compact_inputs import CompactStateBuilder
        enricher = CompactProfiles(prepared / "profile_bundle/policy.json", root=prepared)
        compact_builder = CompactStateBuilder(tok, room, enricher)
    else:
        enricher = ProfileEnricher(prepared / "profile_bundle/policy.json", root=prepared,
                                  token_count=count, state_room=room)
    frame = pd.read_parquet(prepared / "baseline/provenance.parquet")
    for variant in VARIANTS:
        sidecar = pd.read_parquet(prepared / variant / "provenance.parquet")
        if (sidecar.groupby("effective_input_sha256").role.nunique() > 1).any():
            raise ValueError("Identical effective inputs overlap training and development")
    split = json.loads((root / config.split).read_text())
    original_data = root / split["artifact_directory"]
    checks = {}
    for role in ["training", "development"]:
        baseline = read_rows(prepared / "baseline" / (role + ".jsonl"))
        enriched = read_rows(prepared / "enriched" / (role + ".jsonl"))
        baseline_frame = frame[frame.role.eq(role)].sort_values("jsonl_line")
        if len(baseline) != len(enriched) or len(baseline) != len(baseline_frame):
            raise ValueError("Paired records do not align")
        source_path = original_data / (role + ".jsonl")
        if file_hash(source_path) != split["artifact_fingerprints"][role + ".jsonl"]:
            raise ValueError("Frozen input used for runtime parity changed")
        original = read_rows(source_path)
        # Full native Torch-vs-preparation token checks, not a forward pass.
        for variant, rows in [("baseline", baseline), ("enriched", enriched)]:
            sidecar = pd.read_parquet(prepared / variant / "provenance.parquet")
            sidecar = sidecar[sidecar.role.eq(role)].sort_values("jsonl_line")
            for row, expected in zip(rows, sidecar.to_dict("records")):
                if row["questions"] != manifest["question"] or type(row["expected"]["correct"]) is not bool:
                    raise ValueError("Invalid question or binary training target")
                ids = tok(row["state"].replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
                if len(ids) != expected["state_tokens"] or digest_state(json.dumps(ids[:room])) != expected["effective_input_sha256"]:
                    raise ValueError("PyTorch native input differs from prepared token sequence")
        compact_details = []
        for base, extra, source in zip(baseline, enriched, original):
            if base["expected"] != extra["expected"]:
                raise ValueError("Enrichment changed labels")
            if config.input_pipeline != "compact_v3" and not extra["state"].startswith(base["state"]):
                raise ValueError("Enrichment changed labels or original text")
            parsed = json.loads(source["state"])
            if config.input_pipeline == "compact_v3":
                pair = compact_builder.build(parsed["target_system"], parsed["target_item"])
                if pair.baseline != base["state"]:
                    raise ValueError("Compact baseline differs from prepared input")
                rendered = pair.enriched
                compact_details.append(pair.enrichment)
            elif config.input_pipeline == "task_v2":
                from .task_inputs import TaskStateBuilder
                base_rendered = TaskStateBuilder(tok, room).build(parsed["target_system"], parsed["target_item"]).state
                if base_rendered != base["state"]:
                    raise ValueError("Task-v2 baseline differs from prepared input")
                rendered, _ = enricher.enrich(base_rendered, parsed["target_system"])
            else:
                rendered = build_state(parsed["target_system"], parsed["target_item"], [],
                                       state_format="enriched_v1", enricher=enricher)
            if rendered != extra["state"]:
                raise ValueError("Runtime renderer differs from prepared training state")
        suffixes = [extra["state"][len(base["state"]):] for base, extra in zip(baseline, enriched)]
        checks[role] = {"responses": len(baseline), "native_token_parity": "all records passed",
                       "runtime_render_parity": "all records passed",
                       "model_property_rows": sum("Published model properties" in s for s in suffixes),
                       "harness_capability_rows": sum("Published harness capabilities" in s for s in suffixes)}
        if compact_details:
            checks[role].update(model_property_rows=sum(d["delivered"]["model"] for d in compact_details),
                                harness_capability_rows=sum(d["delivered"]["harness"] for d in compact_details),
                                matched_task_payload="all records passed")
    free = shutil.disk_usage(root).free
    if free < 10 * 1024**3:
        raise RuntimeError("Need at least 10 GiB free for the two variants' best/latest checkpoints")
    return {"status": "ready", "training_performed": False, "model_inference": False,
            "device": config.device, "precision": "float32", "checks": checks,
            "free_disk_bytes": free, "parent": str(parent),
            "parent_weights_sha256": file_hash(parent / "model.safetensors"),
            "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
            "cross_role_effective_input_conflicts": 0,
            "optimizer_updates_per_variant": config.updates,
            "effective_batch": config.effective_batch,
            "confirmation_opened": False, "calibration_opened": False}


def trim_device(device):
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    else:
        # CUDA's allocator can reuse its buffers; retain the cache between steps.
        torch.cuda.synchronize(device)


def device_memory(device):
    import torch
    if device.type == "mps":
        return {"mps_driver_bytes": torch.mps.driver_allocated_memory(),
                "mps_tensor_bytes": torch.mps.current_allocated_memory()}
    return {"cuda_allocated_bytes": torch.cuda.memory_allocated(device),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}


def memory_policy(device):
    if device == "mps":
        return MEMORY_POLICY
    return {"padding_multiple": MEMORY_POLICY["padding_multiple"],
            "cache_trim": "CUDA allocator cache retained; synchronize for timing",
            "precision": "float32; TF32 disabled; autocast disabled"}


def save_pilot_checkpoint(model, tokenizer, native_config, directory, prepared,
                          config, variant, provenance):
    from safetensors.torch import save_file
    directory.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(directory / "model.safetensors"))
    model.encoder.config.save_pretrained(directory / "encoder")
    tokenizer.save_pretrained(directory / "tokenizer")
    normalize_tokenizer_export(directory)
    native_config = dict(native_config, max_len=config.max_len,
                         head_max_len=config.head_max_len, temperature=[1., 1., 1.])
    native_config.pop("temperature_by_options", None)
    write_json(directory / "rl_agent_config.json", native_config)
    contract = json.loads((prepared / variant / "input_contract.json").read_text())
    if variant == "enriched" or config.input_pipeline == "compact_v3":
        shutil.copytree(prepared / "profile_bundle", directory / "profile_bundle", dirs_exist_ok=True)
    artifact_hashes = {str(p.relative_to(directory)): file_hash(p)
                      for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "paiec_training.json"}
    manifest = dict(model="english", run_type="head_enrichment_pilot", variant=variant,
                    config=config.model_dump(), input_contract=contract,
                    calibration="none; raw temperature 1", **provenance,
                    artifact_sha256=artifact_hashes)
    write_json(directory / "paiec_training.json", manifest)


def train_variant(root, config, variant, *, encoder_cache_gib=0):
    import torch
    from laya.train import load_checkpoint, items_from_rows, encode_item, _forward
    from safetensors.torch import save_file, load_file
    from .laya_runtime import local_checkpoint_files

    prepared, preparation = verify_prepared(root, config)
    out = root / config.output / variant
    out.mkdir(parents=True, exist_ok=False)
    parent = cached_parent(preparation["checkpoint_revision"])
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    model, tok, native = load_checkpoint(str(parent))
    if native.get("option_layout", "sequential") != "sequential":
        raise ValueError("Pilot requires the original sequential option layout")
    before = tensor_hashes(model)
    rows = {role: read_rows(prepared / variant / (role + ".jsonl")) for role in ["training", "development"]}
    manifest = pd.read_parquet(prepared / variant / "provenance.parquet")
    frames = {role: manifest[manifest.role.eq(role)].sort_values("jsonl_line").reset_index(drop=True)
              for role in rows}
    train_items, skipped = items_from_rows(tok, rows["training"], config.max_len, config.head_max_len)
    if skipped or len(train_items) != config.samples:
        raise ValueError(f"Training item conversion failed: {skipped}")
    encoded = [encode_item(tok, item, config.max_len, config.head_max_len) for item in train_items]
    unique = list(dict.fromkeys(row["state"] for row in rows["development"]))
    lookup = {state: index for index, state in enumerate(unique)}
    # Dummy labels for encoding; real development outcomes enter only the scorer.
    dev_rows = [dict(state=state, questions=preparation["question"], expected={"correct": False}) for state in unique]
    dev_items, skipped = items_from_rows(tok, dev_rows, config.max_len, config.head_max_len)
    if skipped or len(dev_items) != len(unique):
        raise ValueError(f"Development item conversion failed: {skipped}")
    dev_encoded = [encode_item(tok, item, config.max_len, config.head_max_len) for item in dev_items]
    development_labels = [int(row["expected"]["correct"]) for row in rows["development"]]
    weights = pair_weights(frames["training"])
    device = torch.device(config.device)
    if encoder_cache_gib and device.type != "cuda":
        raise ValueError("Pilot feature caching currently requires CUDA")
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(MEMORY_POLICY["mps_memory_fraction"])
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed_all(config.seed)
    model.to(device)
    params = freeze_for_head(model)
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    parameter_counts = {"trainable": sum(p.numel() for p in params), "total": sum(p.numel() for p in model.parameters())}
    tracker = ExperimentTracker(root, experiment="PAIEC Head Fine-tuning")
    provenance = {"prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
                  "encoder_cache_gib": encoder_cache_gib,
                  "split_sha256": preparation["split_sha256"], "parent_revision": preparation["checkpoint_revision"],
                  "parent_weights_sha256": file_hash(parent / "model.safetensors"),
                  "memory_policy": memory_policy(config.device),
                  "device": config.device,
                  "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "Apple MPS",
                  "training_inputs_sha256": file_hash(prepared / variant / "training.jsonl"),
                  "development_inputs_sha256": file_hash(prepared / variant / "development.jsonl")}

    def evaluate():
        model.eval()
        values = []
        with torch.no_grad():
            for first in range(0, len(dev_encoded), config.evaluation_batch):
                batch = collate_pilot(dev_encoded[first:first + config.evaluation_batch], tok.pad_token_id, config.max_len)
                logits = _forward(model, batch, device, False, True)
                values.extend(logits.softmax(-1)[:, 1].cpu().tolist())
                del batch, logits
                if (first // config.evaluation_batch + 1) % 16 == 0:
                    trim_device(device)
        trim_device(device)
        probabilities = [values[lookup[row["state"]]] for row in rows["development"]]
        result, predictions = score_frozen_development(frames["development"], development_labels, probabilities)
        return result, predictions

    with tracker.run(f"head pilot / {variant} / seed {config.seed}" + (" / encoder cache" if encoder_cache_gib else ""),
                     {"config": config.model_dump(), "provenance": provenance, "preparation": preparation},
                     {"run_type": "head_enrichment_pilot", "variant": variant, "state_format": variant_format(config, variant),
                      "max_len": config.max_len, "seed": config.seed, "precision": config.precision,
                      "backend": "torch", "device": config.device,
                      "encoder_cache_gib": encoder_cache_gib,
                      "catalog_sha256": preparation["enrichment"]["catalog_sha256"], **parameter_counts},
                     {"run_type": "head_enrichment_pilot", "confirmation_scored": "false",
                      "calibration_fitted": "false", "source_eligibility": "research_only"}) as tracked:
        provenance["mlflow_run_id"] = tracked.run_id if tracked else None
        initial, original_predictions = evaluate()
        encoder_cache = None
        if encoder_cache_gib:
            from .encoder_cache import FrozenEncoderCache, validate_cache
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(device)
            limit = min(int(encoder_cache_gib * 1024**3), free_bytes - 4 * 1024**3)
            encoder_cache = FrozenEncoderCache(model.encoder, limit,
                identity={"parent_weights_sha256": provenance["parent_weights_sha256"],
                          "prepared_manifest_sha256": provenance["prepared_manifest_sha256"],
                          "precision": "float32; TF32 off"})
            provenance["encoder_cache_limit_bytes"] = limit
            fixture_ids = list(dict.fromkeys([min(range(len(encoded)), key=lambda i: len(encoded[i]["ids"])),
                max(range(len(encoded)), key=lambda i: len(encoded[i]["ids"])),
                *epoch_windows(len(encoded), config.effective_batch, config.seed, 0)[0]]))
            batches = [collate_pilot([encoded[i] for i in fixture_ids[j:j+2]], tok.pad_token_id, config.max_len)
                       for j in range(0, len(fixture_ids), 2)]
            validation = validate_cache(model, encoder_cache, batches, device)
            cached_initial, cached_predictions = evaluate()
            validation["development_max_probability_difference"] = float(np.max(np.abs(
                original_predictions.probability - cached_predictions.probability)))
            validation["development_brier_difference"] = abs(initial.brier_alc-cached_initial.brier_alc)
            if validation["development_max_probability_difference"] > 1e-5 or validation["development_brier_difference"] > 1e-6:
                raise RuntimeError("Cached full-development predictions differ")
            write_json(out / "cache_validation.json", validation)
            if tracked:
                tracked.json(validation, "cache_validation.json")
            print("Encoder cache validation: " + json.dumps(validation), flush=True)
        history, epoch_results = [], []
        best_score, best_epoch, update = float("inf"), None, 0
        start_time = time.perf_counter()
        for epoch in range(config.epochs):
            freeze_for_head(model)
            for window in epoch_windows(len(encoded), config.effective_batch, config.seed, epoch):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                step_start = time.perf_counter()
                denominator = sum(weights[i] for i in window)
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.
                for first in range(0, len(window), config.micro_batch):
                    indices = window[first:first + config.micro_batch]
                    batch = collate_pilot([encoded[i] for i in indices], tok.pad_token_id, config.max_len)
                    logits = _forward(model, batch, device, False, True)
                    loss = weighted_ce(logits, batch["target"].to(device),
                        torch.tensor([weights[i] for i in indices], device=device), denominator)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    loss.backward()
                    loss_value += float(loss.detach().cpu())
                if any(p.grad is not None for p in model.encoder.parameters()):
                    raise RuntimeError("Frozen encoder received gradients")
                norm = float(torch.nn.utils.clip_grad_norm_(params, config.grad_clip, error_if_nonfinite=True))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                del batch, logits, loss
                trim_device(device)
                update += 1
                record = {"epoch": epoch + 1, "update": update, "loss": loss_value,
                          "grad_norm": norm, "presentations": len(window),
                          "seconds": time.perf_counter() - step_start,
                          **device_memory(device)}
                if encoder_cache:
                    record.update(encoder_cache_bytes=encoder_cache.bytes,
                                  encoder_cache_hits=encoder_cache.hits, encoder_cache_misses=encoder_cache.misses)
                    write_json(out / "cache_stats.json", encoder_cache.stats())
                history.append(record)
                write_json(out / "history.json", history)
                if tracked:
                    for key in ["loss", "grad_norm", "seconds", *device_memory(device),
                                *(["encoder_cache_bytes", "encoder_cache_hits", "encoder_cache_misses"] if encoder_cache else [])]:
                        tracked.client.log_metric(tracked.run_id, "train/" + key, record[key], step=update)
                if update % 10 == 0:
                    print(json.dumps(dict(variant=variant, **record)), flush=True)
            result, predictions = evaluate()
            epoch_results.append({"epoch": epoch + 1, "brier_alc": result.brier_alc})
            if result.brier_alc < best_score:
                best_score, best_epoch = result.brier_alc, epoch + 1
                decision = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
                            if key.startswith(("head.", "type_emb.", "scorer."))}
                save_file(decision, str(out / "best_decision.safetensors"))
                predictions.to_csv(out / "best_predictions.csv", index=False)
                result.frame.to_csv(out / "best_scores.csv", index=False)
                del decision
            current = tensor_hashes(model)
            if not all(torch.isfinite(p).all().item() for p in params):
                raise RuntimeError("Non-finite decision weights")
            if current["encoder"] != before["encoder"] or current["act_head"] != before["act_head"]:
                raise RuntimeError("A frozen module changed")
            save_pilot_checkpoint(model, tok, native, out / "latest", prepared, config, variant,
                                  dict(provenance, epoch=epoch + 1, development_brier_alc=result.brier_alc))
            write_json(out / "epochs.json", epoch_results)
            if tracked:
                tracked.client.log_metric(tracked.run_id, "development/brier_alc", result.brier_alc, step=epoch + 1)
            print(f"{variant}: epoch {epoch + 1}/{config.epochs}, development Brier ALC={result.brier_alc:.6f}", flush=True)
        if update != config.updates:
            raise RuntimeError("Optimizer updates differ from the prepared budget")
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        best_decision = load_file(str(out / "best_decision.safetensors"))
        state = model.state_dict()
        for key, value in best_decision.items():
            state[key].copy_(value.to(device))
        del best_decision, state
        best_result, best_predictions = evaluate()
        if abs(best_result.brier_alc - best_score) > 1e-6:
            raise RuntimeError("Selected head no longer reproduces its development score")
        after = tensor_hashes(model)
        if before["encoder"] != after["encoder"] or before["act_head"] != after["act_head"] or before["decision"] == after["decision"]:
            raise RuntimeError("Head-only training integrity check failed")
        save_pilot_checkpoint(model, tok, native, out / "best", prepared, config, variant,
                              dict(provenance, epoch=best_epoch, development_brier_alc=best_score))
        _, identity = local_checkpoint_files(str(out / "best"), "english")
        fixtures = sorted({min(range(len(encoded)), key=lambda i: len(encoded[i]["ids"])),
                           max(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))})

        def fixture_probabilities():
            with torch.no_grad():
                return [_forward(model, collate_pilot([encoded[i]], tok.pad_token_id, config.max_len),
                                 device, False, True).softmax(-1).cpu().tolist()[0] for i in fixtures]

        cache_stats = encoder_cache.stats() if encoder_cache else None
        if encoder_cache:
            encoder_cache.close()
            encoder_cache = None
        original_probabilities = fixture_probabilities()
        del params, model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
        model, _, _ = load_checkpoint(str(out / "best"))
        if tensor_hashes(model) != after:
            raise RuntimeError("Reloaded checkpoint tensors differ")
        model.to(device).eval()
        reloaded_probabilities = fixture_probabilities()
        drift = max(abs(a - b) for old, new in zip(original_probabilities, reloaded_probabilities)
                    for a, b in zip(old, new))
        if drift > 1e-6:
            raise RuntimeError("Reloaded probabilities differ")
        write_json(out / "reload_check.json", {"fixture_indices": fixtures,
                   "max_probability_difference": drift, "reloaded_tensor_hashes": after})
        report = dict(status="finished", variant=variant, best_epoch=best_epoch,
            encoder_cache=cache_stats,
            brier_alc=best_score, initial_raw_brier_alc=initial.brier_alc, constant_brier_alc=.25,
            updates=update, presentations=sum(row["presentations"] for row in history),
            training_and_development_seconds=time.perf_counter() - start_time,
            parameter_counts=parameter_counts, tensor_hashes_before=before, tensor_hashes_after=after,
            memory_policy=memory_policy(config.device), device=config.device,
            checkpoint_identity=identity["checkpoint_revision"],
            reload_max_probability_difference=drift,
            confirmation_scored=False, calibration_fitted=False, resume_supported=False,
            n_pairs=len(frames["development"][["benchmark", "subject_id"]].drop_duplicates()),
            n_eval_inputs=len(frames["development"][["benchmark", "subject_id", "item_id"]].drop_duplicates()),
            n_responses=len(frames["development"]), unique_development_states=len(unique),
            mlflow_run_id=tracked.run_id if tracked else None)
        write_json(out / "report.json", report)
        if tracked:
            tracked.result(best_result, report["training_and_development_seconds"], best_predictions)
            tracked.json(report, "training_report.json")
            tracked.json(json.loads((out / "best/paiec_training.json").read_text()), "checkpoint/paiec_training.json")
            for name in ["epochs.json", "history.json", "best_predictions.csv", "best_scores.csv", "reload_check.json"]:
                tracked.client.log_artifact(tracked.run_id, str(out / name))
            for path in (prepared / "profile_bundle").rglob("*"):
                if path.is_file():
                    tracked.client.log_artifact(tracked.run_id, str(path), "profile_bundle")
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
    return report
