#!/usr/bin/env python3
"""Run at most ten head-only updates; no development/calibration/test scoring."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.head_training import (HeadSmokeConfig, file_hash, select_smoke_rows,
                                 freeze_for_head, tensor_hashes, weighted_ce, normalize_tokenizer_export)
from paiec.laya_runtime import checkpoint_files, local_checkpoint_files
from paiec.tracking import ExperimentTracker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/head_finetuning/smoke_v1.json")
    ap.add_argument("--device", choices=["mps", "cpu", "cuda"], default=None)
    args = ap.parse_args()
    config = HeadSmokeConfig.model_validate_json(Path(args.config).read_text())
    if args.device:
        config = HeadSmokeConfig.model_validate(dict(config.model_dump(), device=args.device))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    import pandas as pd
    import torch
    from laya.train import load_checkpoint, items_from_rows, encode_item, _forward
    from laya.common import collate_items
    from safetensors.torch import save_file
    split_path = ROOT / config.split
    split = json.loads(split_path.read_text())
    data = ROOT / split["artifact_directory"]
    # Verify only the artifacts this smoke consumes. Never open held-out inputs/labels.
    for name in ("training.jsonl", "provenance.parquet"):
        if file_hash(data / name) != split["artifact_fingerprints"][name]:
            raise ValueError(f"Frozen artifact changed: {name}")
    selected = select_smoke_rows(pd.read_parquet(data / "provenance.parquet"), config.samples, config.seed)
    rows_all = [json.loads(line) for line in (data / "training.jsonl").read_text().splitlines()]
    rows = [rows_all[int(record["jsonl_line"])] for record in selected]
    if any(row["questions"] != split["input"]["question_definition"] for row in rows):
        raise ValueError("Training question differs from frozen input")
    pd.DataFrame(selected).to_parquet(out / "smoke_manifest.parquet", index=False)
    (out / "training.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tracker = ExperimentTracker(ROOT, experiment="PAIEC Head Fine-tuning")
    with tracker.run("head-only smoke / 256 rows / 10 updates", config.model_dump(),
                     {**config.model_dump(), "run_type": "training_smoke", "precision": "float32",
                      "split_sha256": file_hash(split_path)},
                     {"run_type": "training_smoke", "confirmation_scored": "false",
                      "quality_evaluation": "false"}) as tracked:
        write = lambda name, value: (out / name).write_text(json.dumps(value, indent=2))
        write("config.json", config.model_dump())
        parent, source = checkpoint_files(config.model, split["input"]["checkpoint_revision"])
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        model, tok, cfg = load_checkpoint(str(parent))
        if cfg.get("option_layout", "sequential") != "sequential":
            raise ValueError("Expected sequential checkpoint")
        device = torch.device(config.device)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no silent device fallback")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent fallback")
        items, skipped = items_from_rows(tok, rows, config.max_len, config.head_max_len)
        if skipped or len(items) != config.samples:
            raise ValueError(f"Invalid smoke items: {skipped}")
        encoded = [encode_item(tok, it, config.max_len, config.head_max_len) for it in items]
        # Recheck actual native serialization rather than relying only on the sidecar.
        from laya.common import build_head
        for row, item in zip(rows, items):
            from laya.train import encode_state
            head, _, _ = build_head(tok, item["q"], config.head_max_len)
            full_state = encode_state(tok, row["state"], 1000000)
            if len(full_state) + len(head) + 1 > config.max_len:
                raise ValueError("Smoke row unexpectedly truncates")
        before = tensor_hashes(model)
        model.to(device)
        params = freeze_for_head(model)
        parameter_counts = {"trainable": sum(p.numel() for p in params),
                            "total": sum(p.numel() for p in model.parameters())}
        print(f"Parameter counts: {parameter_counts}", flush=True)
        write("parent.json", source)
        write("parameter_counts.json", parameter_counts)
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        weights = [record["smoke_weight"] for record in selected]
        order = list(range(len(items)))
        rng = random.Random(config.seed)
        cursor = len(order)
        history = []
        train_start = time.perf_counter()
        for step in range(config.updates):
            window = []
            for _ in range(config.micro_batch * config.accumulation):
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor = 0
                window.append(order[cursor])
                cursor += 1
            denominator = sum(weights[i] for i in window)
            optimizer.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            loss_value = 0.
            for start in range(0, len(window), config.micro_batch):
                indices = window[start:start + config.micro_batch]
                batch = collate_items([[encoded[i] for i in indices]], tok.pad_token_id)
                logits = _forward(model, batch, device, False, True)
                loss = weighted_ce(logits, batch["target"].to(device),
                                   torch.tensor([weights[i] for i in indices], device=device), denominator)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                loss_value += float(loss.detach().cpu())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, config.grad_clip, error_if_nonfinite=True))
            if any(p.grad is not None for p in model.encoder.parameters()):
                raise RuntimeError("Encoder received gradients")
            optimizer.step()
            if device.type == "mps":
                torch.mps.synchronize()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = {"update": step + 1, "loss": loss_value, "grad_norm": grad_norm,
                      "seconds": time.perf_counter() - step_start,
                      "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
            if device.type == "mps":
                record.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                              mps_driver_bytes=torch.mps.driver_allocated_memory())
            if device.type == "cuda":
                record.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
            history.append(record)
            write("history.json", history)
            print(json.dumps(record), flush=True)
            if tracked:
                for key, value in record.items():
                    if key != "update":
                        tracked.client.log_metric(tracked.run_id, "train/" + key, value, step=step + 1)
        training_seconds = time.perf_counter() - train_start
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        model.eval()
        after = tensor_hashes(model)
        assert before["encoder"] == after["encoder"], "Encoder changed"
        assert before["act_head"] == after["act_head"], "Auxiliary head changed"
        assert before["decision"] != after["decision"], "Decision head unchanged"
        if not all(torch.isfinite(p).all().item() for p in params):
            raise RuntimeError("Non-finite trained weights")
        fixture_indices = sorted(set([min(range(len(encoded)), key=lambda i: len(encoded[i]["ids"])),
                                      max(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))]))
        fixtures = [rows[i] for i in fixture_indices]
        def predictions(current):
            with torch.no_grad():
                return [_forward(current, collate_items([[encoded[i]]], tok.pad_token_id), device, False, True)
                        .softmax(-1).cpu().tolist()[0] for i in fixture_indices]
        original_predictions = predictions(model)
        write("training_integrity.json", {"before": before, "after": after,
              "training_seconds": training_seconds, "fixture_indices": fixture_indices,
              "pre_save_probabilities": original_predictions})
        checkpoint = out / "checkpoint"
        checkpoint.mkdir()
        # FP32, unlike upstream save_checkpoint's implicit FP16 conversion.
        model.to("cpu")
        save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(checkpoint / "model.safetensors"))
        model.encoder.config.save_pretrained(checkpoint / "encoder")
        tok.save_pretrained(checkpoint / "tokenizer")
        normalize_tokenizer_export(checkpoint)
        cfg.update(max_len=config.max_len, head_max_len=config.head_max_len, temperature=[1., 1., 1.])
        cfg.pop("temperature_by_options", None)
        (checkpoint / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
        manifest = {"model": config.model, "run_type": "training_smoke", "parent": source,
                    "mlflow_run_id": tracked.run_id if tracked else None,
                    "split_sha256": file_hash(split_path), "sample_sha256": file_hash(out / "smoke_manifest.parquet"),
                    "config": config.model_dump(), "calibration": "none; raw temperature 1",
                    "artifact_sha256": {str(p.relative_to(checkpoint)): file_hash(p)
                                        for p in sorted(checkpoint.rglob("*")) if p.is_file()}}
        (checkpoint / "paiec_training.json").write_text(json.dumps(manifest, indent=2))
        del params, model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        reloaded, _, _ = load_checkpoint(str(checkpoint))
        reloaded_hashes = tensor_hashes(reloaded)
        assert reloaded_hashes == after, "Saved tensors differ"
        reloaded.to(device).eval()
        reloaded_predictions = predictions(reloaded)
        difference = max(abs(a - b) for x, y in zip(original_predictions, reloaded_predictions) for a, b in zip(x, y))
        assert difference <= 1e-6, f"Reload prediction drift: {difference}"
        _, identity = local_checkpoint_files(str(checkpoint), config.model)
        report = {"status": "passed", "purpose": "training machinery only; no quality claim",
                  "updates": config.updates, "unique_training_rows": len(rows),
                  "example_presentations": config.updates * config.micro_batch * config.accumulation,
                  "training_seconds": training_seconds, "tensor_hashes_before": before,
                  "tensor_hashes_after": after, "reloaded_tensor_hashes": reloaded_hashes,
                  "reload_max_probability_difference": difference,
                  "fixture_probabilities": reloaded_predictions,
                  "parameter_counts": parameter_counts, "checkpoint_identity": identity["checkpoint_revision"],
                  "confirmation_scored": False, "development_scored": False, "calibration_fitted": False,
                  "resume_supported": False}
        write("report.json", report)
        write("reload_fixtures.json", fixtures)
        if tracked:
            for name in ("config.json", "parent.json", "parameter_counts.json", "history.json", "report.json", "smoke_manifest.parquet", "reload_fixtures.json"):
                tracked.client.log_artifact(tracked.run_id, str(out / name))
            tracked.json(manifest, "checkpoint/paiec_training.json")
            tracked.params({"checkpoint_identity": identity["checkpoint_revision"], **parameter_counts})
            write("mlflow_receipt.json", {"run_id": tracked.run_id, "experiment_id": tracker.experiment_id})
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
