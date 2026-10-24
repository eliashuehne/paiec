"""Bounded L4 throughput, numerical fidelity and isolated learning controls."""
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from .encoder_training import (encode_data, set_training_policy, train_window, predict,
                               optimizer_for, make_scaler, step_optimizer, NonFiniteTrainingGradient)
from .head_pilot import write_json, epoch_windows
from .head_training import file_hash, tensor_hashes


def zero_canonical_weight_hash(model):
    """Ignore signed-zero bit changes, preserving exact bits of every nonzero value.

    Fused AdamW with LR=0 can turn -0 into +0. A CPU positive control reproduces
    this with exactly zero numerical drift. Final training frozen hashes remain
    the original strict byte hashes; this exception is only for disposable probes.
    """
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name == "temperature":
            continue
        values = value.detach().cpu().contiguous().numpy().copy()
        values[values == 0] = 0
        digest.update(name.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def gradient_sample(model):
    import torch
    parts = []
    for _, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            values = parameter.grad.detach().flatten()
            stride = max(1, values.numel() // 256)
            parts.append(values[::stride][:256].float().cpu())
        elif not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError("Frozen parameter received a gradient")
    return torch.cat(parts).numpy()


def learning_controls(parent, tok, config, settings, upper_layers, data, question):
    import torch
    from laya.train import load_checkpoint, items_from_rows, encode_item
    torch.manual_seed(config.seed)
    model, _, _ = load_checkpoint(str(parent))
    model.cuda()
    model.head_checkpointing = settings.get("head_checkpointing", False)
    set_training_policy(model, upper_layers)
    canary_config = config.model_copy(update={"learning_rate": 3e-4, "encoder_learning_rate": 1e-5})
    optimizer = optimizer_for(model, canary_config)
    scaler = make_scaler(settings)
    def synthetic(count, offset):
        rows = [dict(state=f"Isolated learning diagnostic {i+offset}. The target system's recorded success signal is {'YES' if i%2 else 'NO'}.",
                     questions=question, expected={"correct": bool(i%2)}) for i in range(count)]
        items, skipped = items_from_rows(tok, rows, config.max_len, config.head_max_len)
        if skipped:
            raise RuntimeError("Synthetic control did not encode")
        return dict(rows=rows, frame=pd.DataFrame({"sample_weight": [1.]*count}),
                    encoded=[encode_item(tok, item, config.max_len, config.head_max_len) for item in items])
    train, held = synthetic(32, 0), synthetic(32, 1000)
    labels = np.array([r["expected"]["correct"] for r in held["rows"]], dtype=float)
    history = []
    for update in range(1, 81):
        set_training_policy(model, upper_layers)
        loss, _ = train_window(model, optimizer, train, list(range(32)), tok, canary_config, settings, upper_layers, scaler)
        step_optimizer(optimizer, scaler)
        optimizer.zero_grad(set_to_none=True)
        if update % 10 == 0:
            p = predict(model, tok, held, config, settings, upper_layers)
            row = dict(update=update, loss=loss, brier=float(np.mean((p-labels)**2)), accuracy=float(np.mean((p>.5)==labels)))
            history.append(row)
            print("Synthetic control: " + json.dumps(dict(upper_layers=upper_layers, **row)), flush=True)
            if row["accuracy"] >= .95 and row["brier"] <= .08:
                break
    synthetic_passed = history[-1]["accuracy"] >= .95 and history[-1]["brier"] <= .08
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    if not synthetic_passed:
        return dict(passed=False, synthetic=history, note="No real training authorized by a failed learning control")
    # Independent model and short real training-only fixtures: measure ability to fit,
    # never report their in-sample score as development/competition performance.
    torch.manual_seed(config.seed)
    model, _, _ = load_checkpoint(str(parent))
    model.cuda()
    model.head_checkpointing = settings.get("head_checkpointing", False)
    set_training_policy(model, upper_layers)
    optimizer = optimizer_for(model, canary_config)
    scaler = make_scaler(settings)
    indices = sorted(range(len(data["training"]["encoded"])), key=lambda i: len(data["training"]["encoded"][i]["ids"]))[:32]
    tiny = dict(rows=[data["training"]["rows"][i] for i in indices],
                encoded=[data["training"]["encoded"][i] for i in indices],
                frame=pd.DataFrame({"sample_weight": [1.]*32}))
    labels = np.array([r["expected"]["correct"] for r in tiny["rows"]], dtype=float)
    initial_p = predict(model, tok, tiny, config, settings, upper_layers)
    initial = float(np.mean((initial_p-labels)**2))
    real_history = []
    for update in range(1, 61):
        set_training_policy(model, upper_layers)
        loss, _ = train_window(model, optimizer, tiny, list(range(32)), tok, canary_config, settings, upper_layers, scaler)
        step_optimizer(optimizer, scaler)
        optimizer.zero_grad(set_to_none=True)
        if update % 10 == 0:
            p = predict(model, tok, tiny, config, settings, upper_layers)
            row = dict(update=update, loss=loss, brier=float(np.mean((p-labels)**2)))
            real_history.append(row)
            print("Tiny overfit: " + json.dumps(dict(upper_layers=upper_layers, **row)), flush=True)
            if row["brier"] <= min(.10, initial*.6):
                break
    passed = real_history[-1]["brier"] <= min(.10, initial*.6)
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return dict(passed=passed, synthetic=history, real_training_only_initial_brier=initial,
                real_training_only=real_history, quality_result=False,
                controls_discarded=True, final_training_restarts_original_parent=True)


def qualify(root, out, config, *, download=False):
    import torch
    from laya.train import load_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("Qualification requires CUDA")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    parent, tok, manifest, data = encode_data(root, config, download=download)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    window = epoch_windows(config.samples, config.effective_batch, config.seed, 0)[0]
    longest = sorted(range(config.samples), key=lambda i: len(data["training"]["encoded"][i]["ids"]), reverse=True)
    fixture = dict(data["training"], encoded=[data["training"]["encoded"][i] for i in window])
    results, checks = [], []
    candidates = [dict(precision="float32", micro_batch=2, grouped=False, head_checkpointing=False),
                  dict(precision="float32", micro_batch=4, grouped=True, head_checkpointing=False),
                  dict(precision="float32", micro_batch=8, grouped=True, head_checkpointing=False),
                  *[dict(precision="float16", micro_batch=n, grouped=True, head_checkpointing=False, fp32_scorer=True) for n in [4, 8, 16, 32]]]
    if not torch.cuda.is_bf16_supported():
        candidates = [c for c in candidates if c["precision"] != "bfloat16"]
    for upper in [0, config.upper_layers]:
        torch.manual_seed(config.seed)
        model, _, _ = load_checkpoint(str(parent))
        before = tensor_hashes(model)
        numerical_before = zero_canonical_weight_hash(model)
        model.cuda()
        set_training_policy(model, upper)
        optimizer = optimizer_for(model, config)
        # Zero-LR optimizer work allocates real AdamW state and measures its cost.
        # It never changes model weights; all probe models are discarded.
        for group in optimizer.param_groups:
            group["lr"] = 0.
        reference_p = predict(model, tok, fixture, config, candidates[0], upper)
        model.eval()
        train_window(model, optimizer, data["training"], window, tok, config, candidates[0], upper)
        reference_grad = gradient_sample(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for candidate in candidates:
            record = dict(upper_layers=upper, settings=candidate)
            scaler = make_scaler(candidate)
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                p = predict(model, tok, fixture, config, candidate, upper)
                model.eval()
                train_window(model, optimizer, data["training"], window, tok, config, candidate, upper, scaler)
                grad = gradient_sample(model)
                cosine = float(np.dot(grad, reference_grad) / (np.linalg.norm(grad)*np.linalg.norm(reference_grad)))
                drift = float(np.max(np.abs(p-reference_p)))
                mean_drift = float(np.mean(np.abs(p-reference_p)))
                limit = {"float32": 1e-4, "bfloat16": .03, "float16": .005}[candidate["precision"]]
                numerical_passed = drift < limit and mean_drift < .01 and cosine > .99
                step_optimizer(optimizer, scaler)
                optimizer.zero_grad(set_to_none=True)
                times = []
                for repeat in range(3):
                    set_training_policy(model, upper)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    train_window(model, optimizer, data["training"], window, tok, config, candidate, upper, scaler)
                    step_optimizer(optimizer, scaler)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    if repeat:
                        times.append(time.perf_counter()-started)
                set_training_policy(model, upper)
                # Longest-sequence stress test, including actual optimizer state.
                train_window(model, optimizer, data["training"], longest[:candidate["micro_batch"]], tok, config, candidate, upper, scaler)
                step_optimizer(optimizer, scaler)
                optimizer.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated()
                capacity_passed = peak < .85*torch.cuda.get_device_properties(0).total_memory
                record.update(status="passed" if numerical_passed and capacity_passed else "rejected",
                    median_seconds_per_64=float(np.median(times)), max_probability_difference=drift,
                    mean_probability_difference=mean_drift, sampled_gradient_cosine=cosine,
                    peak_allocated_bytes=peak, reserved_bytes=torch.cuda.memory_reserved(),
                    longest_sequence_passed=True, numerical_passed=numerical_passed, capacity_passed=capacity_passed)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                record["status"] = "out_of_memory"
            except NonFiniteTrainingGradient:
                optimizer.zero_grad(set_to_none=True)
                record["status"] = "non_finite_gradient_rejected"
            results.append(record)
            write_json(out / "probe_progress.json", results)
            print("GPU probe: " + json.dumps(record), flush=True)
        after = tensor_hashes(model)
        numerical_after = zero_canonical_weight_hash(model)
        if numerical_before != numerical_after:
            raise RuntimeError("Zero-LR probes changed nonzero model weights")
        checks.append(dict(upper_layers=upper, numeric_weights_unchanged=True,
            byte_hashes_unchanged=before == after, tensor_hashes_before=before, tensor_hashes_after=after,
            signed_zero_canonical_hash=numerical_after,
            note="Fused zero-LR updates may normalize signed zero; every nonzero bit must remain identical"))
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    viable = []
    for candidate in candidates:
        matching = [r for r in results if r["settings"] == candidate and r["status"] == "passed"]
        if len(matching) == 2:
            viable.append((sum(r["median_seconds_per_64"] for r in matching), candidate))
    if not viable:
        raise RuntimeError("No common numerically/capacity-qualified settings")
    settings = min(viable, key=lambda row: row[0])[1]
    controls = {str(upper): learning_controls(parent, tok, config, settings, upper, data, manifest["question"])
                for upper in [0, config.upper_layers]}
    report = dict(status="passed" if all(c["passed"] for c in controls.values()) else "failed_learning_control",
        settings=settings, results=results, zero_lr_probe_checks=checks, learning_controls=controls,
        gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        precision_policy="FP32 stored weights/optimizer; selected autocast policy; FP32 final scorer when selected; TF32 disabled",
        numerical_checks="64 mixed-length inputs, sampled clipped gradients, longest-input capacity; no quality claim",
        prepared_manifest_sha256=file_hash(Path(root) / config.prepared / "manifest.json"),
        training_source_sha256={name: file_hash(Path(root) / name) for name in [
            "src/paiec/encoder_training.py", "src/paiec/encoder_qualification.py", "scripts/run_encoder_experiment.py"]},
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        config=config.model_dump(), config_sha256=__import__("hashlib").sha256(json.dumps(config.model_dump(), sort_keys=True).encode()).hexdigest(),
        qualification_does_not_train_final_checkpoint=True, calibration_opened=False, confirmation_opened=False)
    write_json(out / "qualification.json", report)
    if report["status"] != "passed":
        raise RuntimeError("Learning controls failed; inspect before training")
    return report
