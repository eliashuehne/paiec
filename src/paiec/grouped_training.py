"""One enriched multilingual training arm with one loss per original record.

No development scoring, checkpoint selection, baseline arm, or revealed evidence
is implicit in this entry point. A new GPU launch remains a separate action.
"""
import gc
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from .encoder_training import set_training_policy, optimizer_for, make_scaler, step_optimizer
from .feature_preparation import write_json
from .grouped_checkpoint import save_grouped_checkpoint
from .grouped_data import GroupedTrainingConfig, TokenCorpus, verify_prepared, verify_qualification
from .grouped_engine import TorchGroupedEngine
from .grouped_learning import backward_group
from .head_pilot import epoch_windows
from .head_training import file_hash, tensor_hashes
from .multilingual_qualification import frozen_hash
from .multilingual_recipe import pinned_multilingual_parent


def backward_window(engine, groups, sample_weights, *, scaler=None):
    """Accumulate the exact weighted grouped objective for one optimizer step."""
    if len(groups) != len(sample_weights) or not groups or any(not math.isfinite(w) or w <= 0 for w in sample_weights):
        raise ValueError("Expected one positive sample weight per original example")
    denominator = sum(sample_weights)
    return sum(backward_group(engine.view_logits, g["views"], g["weights"], g["label"],
        divisor=denominator / weight, scaler=scaler, cuda=engine.device.type == "cuda")
        for g, weight in zip(groups, sample_weights))


def train_grouped(root, config, out, *, download=False, tracking=True):
    import torch
    from laya.train import load_checkpoint
    from .grouped_predictor import make_grouped_laya_predictor
    from .tracking import ExperimentTracker
    root, out = Path(root).resolve(), Path(out).resolve()
    config = GroupedTrainingConfig.model_validate(config)
    prepared, manifest = verify_prepared(root, config.model_dump())
    qualification = verify_qualification(root, config)
    if out.is_relative_to(prepared) or prepared.is_relative_to(out):
        raise ValueError("Keep training output separate from immutable prepared inputs")
    if out.exists():
        raise FileExistsError("Grouped training never resumes or overwrites an existing job")
    if not torch.cuda.is_available():
        raise RuntimeError("Full grouped training requires the qualified Torch/CUDA path")
    corpus = TokenCorpus(prepared)
    if len(corpus) != json.loads((prepared / "summary.json").read_text())["original_training_examples"]:
        raise ValueError("Grouped training corpus count changed")
    parent = pinned_multilingual_parent(download=download)
    for name, digest in qualification["checkpoint_sha256"].items():
        if file_hash(parent / name) != digest or manifest["parent_sha256"][name] != digest:
            raise ValueError("Training parent differs from qualified original multilingual weights")
    packages = {name: version(name) for name in ["torch", "transformers", "tokenizers", "laya", "safetensors"]}
    # These native APIs and numerical controls were qualified at these versions.
    expected = {"torch": "2.14.1", "transformers": "5.20.2", "tokenizers": "0.23.3", "laya": "0.4.13", "safetensors": "0.8.0"}
    if any(packages[n].split("+")[0] != v for n, v in expected.items()):
        raise ValueError("Native packages differ from multilingual qualification; qualify the changed environment first")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, tok, native_config = load_checkpoint(str(parent))
    model.cuda()
    model.head_checkpointing = True
    model.encoder.gradient_checkpointing_enable()
    trainable = set_training_policy(model, config.upper_layers)
    before, frozen_before = tensor_hashes(model), frozen_hash(model)
    engine = TorchGroupedEngine(model, tok, device="cuda", precision="float16", max_len=config.max_len, upper_layers=config.upper_layers)
    optimizer, scaler = optimizer_for(model, config), make_scaler({"precision": "float16"})
    out.mkdir(parents=True, exist_ok=False)
    tracker = ExperimentTracker(root, experiment="PAIEC grouped enriched training", enabled=tracking)
    start, updates, examples, view_calls = time.monotonic(), 0, 0, 0
    write_json(out / "resolved_config.json", config.model_dump())
    with tracker.run("multilingual-grouped-enriched", config.model_dump(),
            {"max_len": config.max_len, "upper_layers": config.upper_layers, "original_examples": len(corpus),
             "effective_batch": config.effective_batch, "epochs": config.epochs},
            {"mode": "grouped_enriched_training", "source_policy": "research_only", "development_scoring": "false"}) as tracked:
        with (out / "history.jsonl").open("w", buffering=1) as history:
            for epoch in range(config.epochs):
                for window in epoch_windows(len(corpus), config.effective_batch, config.seed, epoch):
                    set_training_policy(model, config.upper_layers)
                    optimizer.zero_grad(set_to_none=True)
                    # Materialize only this optimizer window, never the full token corpus.
                    groups = [corpus.group(i) for i in window]
                    weights = [corpus.records[i]["sample_weight"] for i in window]
                    began = time.monotonic()
                    loss = backward_window(engine, groups, weights, scaler=scaler)
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
                    if not math.isfinite(loss) or not torch.isfinite(norm).item():
                        raise RuntimeError("Non-finite grouped training loss/gradient; stopping without a completed checkpoint")
                    step_optimizer(optimizer, scaler)
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    examples += len(groups)
                    # Multi-view backward replays each view once for bounded graph memory.
                    view_calls += sum(len(g["views"]) * (2 if len(g["views"]) > 1 else 1) for g in groups)
                    record = {"update": updates, "epoch": epoch + 1, "loss": loss, "gradient_norm": float(norm),
                        "original_examples": examples, "view_forward_calls": view_calls, "seconds": time.monotonic() - began}
                    history.write(json.dumps(record) + "\n")
                    write_json(out / "progress.json", record)
                    print(json.dumps(record), flush=True)
                    if tracked:
                        for key in ["loss", "gradient_norm", "original_examples", "view_forward_calls"]:
                            tracked.client.log_metric(tracked.run_id, key, float(record[key]), step=updates)
                    del groups
        after = tensor_hashes(model)
        if frozen_hash(model) != frozen_before or before["act_head"] != after["act_head"]:
            raise ValueError("Grouped training changed a frozen parameter")
        if before["decision"] == after["decision"] or before["encoder"] == after["encoder"]:
            raise ValueError("The selected decision/upper-encoder parameters did not change")
        fixtures = json.loads((prepared / "reload_fixtures.json").read_text())
        reference = [engine.predict_group(corpus.group(f["group_index"])) for f in fixtures]
        checkpoint = out / "final"
        training = {"config": config.model_dump(), "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
            "qualification_sha256": config.qualification_sha256, "parent_sha256": qualification["checkpoint_sha256"],
            "packages": packages, "updates": updates, "examples": examples, "parameter_hashes_before": before,
            "parameter_hashes_after": after, "frozen_parameters_unchanged": True,
            "precision": "float16 autocast; float32 scorer", "device": torch.cuda.get_device_name(),
            "training_sources": {"src/paiec/" + n: file_hash(Path(__file__).with_name(n))
                for n in ["grouped_training.py", "grouped_data.py", "grouped_learning.py"]}}
        save_grouped_checkpoint(model, tok, native_config, checkpoint, prepared / "collection_snapshot",
                                contract=config.input_contract(), training=training)
        # Release the training graphs/optimizer before a real native export/reload check.
        del optimizer, scaler, trainable, model
        engine.close()
        gc.collect()
        torch.cuda.empty_cache()
        predictor = make_grouped_laya_predictor(checkpoint_path=checkpoint, device="cuda", dtype="float16")
        drift = []
        for fixture, expected_probability in zip(fixtures, reference):
            group = predictor.prepare_input(fixture["subject"], fixture["item"])
            native = corpus.group(fixture["group_index"])
            for a, b in zip(group["views"], native["views"], strict=True):
                if a["ids"] != b["ids"] or a["markers"] != b["markers"]:
                    raise ValueError("Exported inference tokens/markers differ from training")
            if group["weights"] != native["weights"]:
                raise ValueError("Exported inference aggregation differs from training")
            drift.append(abs(predictor(fixture["subject"], fixture["item"]) - expected_probability))
        predictor.close()
        if not drift or max(drift) > .005:
            raise ValueError("Grouped export/reload probability drift failed")
        result = {"version": "grouped_enriched_training_result_v1", "status": "finished", "updates": updates,
            "original_examples": examples, "view_forward_calls": view_calls, "seconds": time.monotonic() - start,
            "checkpoint": str(checkpoint), "export_reload_maximum_probability_drift": max(drift),
            "training_inference_token_parity": True, "nontraining_outcomes_opened": False, "development_scored": False,
            "quality_result": False, "enriched_only": True, "revealed_evidence_used": False, "training": training}
        write_json(out / "report.json", result)
        if tracked:
            tracked.json(result, "report.json")
            tracked.json(json.loads((checkpoint / "paiec_training.json").read_text()), "paiec_training.json")
    return result
