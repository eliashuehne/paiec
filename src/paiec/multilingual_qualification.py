"""Bounded, disposable multilingual CUDA controls; never a full pilot."""
import gc
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np

from .encoder_training import set_training_policy, optimizer_for, make_scaler, step_optimizer, collate, forward
from .encoder_qualification import gradient_sample, zero_canonical_weight_hash
from .feature_preparation import write_json
from .grouped_learning import backward_group
from .head_training import file_hash
from .multilingual_recipe import MultilingualRecipe, pinned_multilingual_parent, verify_qualification_inputs, encode_group

SETTINGS = {"precision": "float16", "micro_batch": 1, "fp32_scorer": True, "head_checkpointing": True}


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def qualify_multilingual(root, out, config, *, download=False):
    import torch
    from laya.train import load_checkpoint, save_checkpoint
    config = MultilingualRecipe.model_validate(config)
    prepared, manifest = verify_qualification_inputs(root, config.model_dump())
    fixture = json.loads((prepared / "fixtures.json").read_text())
    parent = pinned_multilingual_parent(download=download)
    if not torch.cuda.is_available():
        raise RuntimeError("Multilingual hardware qualification requires CUDA")
    if file_hash(parent / "tokenizer/tokenizer.json") != fixture["tokenizer_sha256"]:
        raise ValueError("Worker tokenizer differs from prepared fixtures")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def deadline():
        if time.monotonic() - start > 1050:
            raise TimeoutError("Bounded qualification time exhausted; no full training follows")
    def load(upper):
        torch.manual_seed(config.seed)
        model, tok, cfg = load_checkpoint(str(parent))
        model.cuda()
        model.head_checkpointing = True
        model.encoder.gradient_checkpointing_enable()
        set_training_policy(model, upper)
        return model, tok, dict(cfg, max_len=8192, head_max_len=256)
    def groups(tok, records, budget=8192):
        return [encode_group(tok, r["state"], fixture["question"], r["label"], budget,
            config.head_max_len, config.overlap_tokens, config.max_views) for r in records]
    def logits_for(model, tok, view, upper, precision="float16"):
        batch = collate([view], tok.pad_token_id, 8192)
        return forward(model, batch, torch.device("cuda"), precision, upper, fp32_scorer=True)
    def probabilities(model, tok, items, upper, precision="float16"):
        model.eval()
        result = []
        with torch.no_grad():
            for item in items:
                mean = sum(w * logits_for(model, tok, v, upper, precision).reshape(2) for w, v in zip(item["weights"], item["views"]))
                result.append(float(mean.softmax(-1)[1]))
        return np.array(result)
    def backward(model, tok, items, upper, optimizer, scaler, *, training=True):
        if training:
            set_training_policy(model, upper)
        else:
            model.eval()
        optimizer.zero_grad(set_to_none=True)
        loss = 0.
        for item in items:
            loss += backward_group(lambda v: logits_for(model, tok, v, upper), item["views"], item["weights"],
                item["label"], divisor=len(items), scaler=scaler, cuda=True)
        if scaler:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
        if not np.isfinite(loss) or not torch.isfinite(norm).item():
            raise RuntimeError("Non-finite qualification loss/gradient")
        return loss, float(norm)
    results = []
    for upper in [0, config.upper_layers]:
        deadline()
        model, tok, cfg = load(upper)
        optimizer, scaler = optimizer_for(model, config), make_scaler(SETTINGS)
        for group in optimizer.param_groups:
            group["lr"] = 0.
        original = zero_canonical_weight_hash(model)
        numerical_items = groups(tok, fixture["real_canary"][:1])
        reference = probabilities(model, tok, numerical_items, upper, "float32")
        mixed = probabilities(model, tok, numerical_items, upper)
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        view = numerical_items[0]["views"][0]
        logits = logits_for(model, tok, view, upper, "float32")
        torch.nn.functional.cross_entropy(logits, torch.tensor([numerical_items[0]["label"]], device="cuda")).backward()
        reference_gradient = gradient_sample(model)
        backward(model, tok, numerical_items, upper, optimizer, scaler, training=False)
        mixed_gradient = gradient_sample(model)
        cosine = float(np.dot(reference_gradient, mixed_gradient) / (np.linalg.norm(reference_gradient) * np.linalg.norm(mixed_gradient)))
        encoder_grad = sum(float(p.grad.detach().abs().sum()) for n, p in model.named_parameters()
                           if n.startswith("encoder.") and p.grad is not None)
        head_grad = sum(float(p.grad.detach().abs().sum()) for n, p in model.named_parameters()
                        if not n.startswith("encoder.") and p.grad is not None)
        optimizer.zero_grad(set_to_none=True)
        numerical = {"maximum_probability_drift": float(np.max(np.abs(reference - mixed))),
                     "sampled_gradient_cosine": cosine, "encoder_gradient_l1": encoder_grad, "head_gradient_l1": head_grad}
        numerical["passed"] = numerical["maximum_probability_drift"] < .01 and cosine > .98 and head_grad > 0 and (encoder_grad > 0 if upper else encoder_grad == 0)
        step_optimizer(optimizer, scaler)
        optimizer.zero_grad(set_to_none=True)
        capacity = []
        for name, record in fixture["stress"].items():
            deadline()
            budget = int(name) if name.isdigit() else 8192
            item = groups(tok, [record], budget)
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                began = time.perf_counter()
                loss, norm = backward(model, tok, item, upper, optimizer, scaler)
                step_optimizer(optimizer, scaler)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                record = {"fixture": name, "views": len(item[0]["views"]), "maximum_sequence": max(len(v["ids"]) for v in item[0]["views"]),
                    "forward_backward_optimizer_seconds": time.perf_counter() - began,
                    "peak_cuda_allocated_bytes": peak, "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "loss": loss, "gradient_norm": norm, "passed": peak < .85 * torch.cuda.get_device_properties(0).total_memory}
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
                record = {"fixture": name, "passed": False, "reason": "out_of_memory"}
            capacity.append(record)
            write_json(out / "progress.json", {"phase": "capacity", "upper_layers": upper, "checks": capacity})
            print("Capacity: " + json.dumps({"upper_layers": upper, **record}), flush=True)
        unchanged = zero_canonical_weight_hash(model) == original
        del model, optimizer, scaler
        gc.collect(); torch.cuda.empty_cache()
        arm = {"upper_layers": upper, "numerical": numerical, "capacity": capacity, "zero_lr_probe_weights_numerically_unchanged": unchanged}
        if not numerical["passed"] or not all(r["passed"] for r in capacity) or not unchanged:
            arm.update(passed=False, learning_skipped="Hardware/numerical gate failed")
            results.append(arm)
            continue
        controls = {}
        for kind, max_updates in [("synthetic", config.synthetic_updates), ("real_training_canary", config.real_updates)]:
            deadline()
            model, tok, cfg = load(upper)
            frozen_before = frozen_hash(model)
            optimizer, scaler = optimizer_for(model, config), make_scaler(SETTINGS)
            if kind == "synthetic":
                def synthetic(offset):
                    languages = ["An isolated training diagnostic.", "Ein isolierter Lerntest.", "这是独立的学习检查。"]
                    return [{"state": {"target_task": {"item_content": languages[i % 3] +
                        f" Case {offset+i}. The target system's recorded success signal is {'YES' if i%2 else 'NO'}."}}, "label": i % 2} for i in range(32)]
                training, scoring = groups(tok, synthetic(0)), groups(tok, synthetic(1000))
            else:
                training = scoring = groups(tok, fixture["real_canary"])
            labels = np.array([g["label"] for g in scoring])
            initial = float(np.mean((probabilities(model, tok, scoring, upper) - labels) ** 2))
            history = []
            for update in range(1, max_updates + 1):
                deadline()
                loss, _norm = backward(model, tok, training, upper, optimizer, scaler)
                step_optimizer(optimizer, scaler)
                optimizer.zero_grad(set_to_none=True)
                if update % 10 == 0 or update == max_updates:
                    p = probabilities(model, tok, scoring, upper)
                    brier, accuracy = float(np.mean((p - labels) ** 2)), float(np.mean((p > .5) == labels))
                    passed = brier <= (.08 if kind == "synthetic" else min(.10, initial * .6)) and (accuracy >= .95 if kind == "synthetic" else True)
                    history.append({"update": update, "loss": loss, "brier": brier, "accuracy": accuracy, "passed": passed})
                    print("Learning: " + json.dumps({"upper_layers": upper, "control": kind, **history[-1]}), flush=True)
                    write_json(out / "progress.json", {"phase": kind, "upper_layers": upper, "history": history})
                    if passed:
                        break
            control = {"initial_brier": initial, "history": history, "passed": history[-1]["passed"],
                       "frozen_parameters_unchanged": frozen_hash(model) == frozen_before, "quality_result": False}
            if kind == "real_training_canary":
                before = probabilities(model, tok, scoring[:4], upper)
                with tempfile.TemporaryDirectory() as temporary:
                    save_checkpoint(model, tok, cfg, temporary)
                    reloaded, reload_tok, reload_cfg = load_checkpoint(temporary)
                    reloaded.cuda(); reloaded.head_checkpointing = True
                    after = probabilities(reloaded, reload_tok, scoring[:4], upper)
                    drift = float(np.max(np.abs(before - after)))
                    control.update(export_reload_maximum_drift=drift, export_reload_passed=drift <= .005,
                                   exported_max_len=reload_cfg["max_len"], exported_head_max_len=reload_cfg["head_max_len"])
                    del reloaded
            controls[kind] = control
            del model, optimizer, scaler
            gc.collect(); torch.cuda.empty_cache()
        arm.update(controls=controls, passed=all(c["passed"] and c["frozen_parameters_unchanged"] and c.get("export_reload_passed", True) for c in controls.values()))
        results.append(arm)
        write_json(out / "arm_results.json", results)
    report = {"version": "multilingual_qualification_v1", "status": "passed" if all(r["passed"] for r in results) else "failed",
        "checkpoint_sha256": {name: file_hash(parent / name) for name in
            ["model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json", "rl_agent_config.json"]},
        "config": config.model_dump(), "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
        "gpu": torch.cuda.get_device_name(), "arms": results, "seconds": time.monotonic() - start,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_source_sha256": {str(p.relative_to(root)): file_hash(p) for p in [Path(__file__), Path(__file__).with_name("multilingual_recipe.py"), Path(__file__).with_name("grouped_learning.py")]},
        "controls_discarded": True, "full_pilot_launched": False, "quality_result": False,
        "original_development_outcomes_used": False, "sealed_roles_opened": False, "paid_teacher_calls": 0,
        "source_policy": "research_only", "cross_window_reasoning_guaranteed": False}
    write_json(out / "qualification.json", report)
    return report
