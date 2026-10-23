#!/usr/bin/env python3
"""Bounded CUDA batch probe; never applies optimizer updates or exports weights.

Can run inside an existing Modal container. An optional training PID is paused
and always resumed, with an independent watchdog as a second safeguard.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--tf32", action="store_true", help="probe TF32 matmul with FP32 stored weights")
    parser.add_argument("--cache-encoder", action="store_true", help="prototype GPU feature caching for the frozen encoder")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    import torch
    import pandas as pd
    from laya.train import load_checkpoint, items_from_rows, encode_item, _forward
    from paiec.head_pilot import load_config, read_rows, epoch_windows, pair_weights
    from paiec.head_pilot_training import cached_parent, collate_pilot
    from paiec.head_training import freeze_for_head, weighted_ce

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an existing CUDA worker")
    config = load_config(root / "configs/head_finetuning/enrichment_pilot_v1.json")
    prepared = root / config.prepared
    manifest = json.loads((prepared / "manifest.json").read_text())
    rows = read_rows(prepared / "enriched/training.jsonl")
    frame = pd.read_parquet(prepared / "enriched/provenance.parquet")
    frame = frame[frame.role.eq("training")].sort_values("jsonl_line").reset_index(drop=True)
    selected = epoch_windows(len(rows), config.effective_batch, config.seed, 0)[0]
    longest = frame.state_tokens.nlargest(config.effective_batch).index.tolist()
    weights = pair_weights(frame)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.75)
    watchdog = None
    paused = False
    results = []
    def expire(_signum, _frame):
        raise TimeoutError("Batch probe exceeded its 180-second runtime bound")
    signal.signal(signal.SIGALRM, expire)
    try:
        if args.pause_pid:
            command = Path(f"/proc/{args.pause_pid}/cmdline").read_bytes()
            if args.pause_pid <= 1 or b"scripts/run_head_pilot.py" not in command:
                raise ValueError("Refusing to pause an unrelated process")
            # Resume even if this probing process is killed unexpectedly.
            code = f"import os,signal,time; time.sleep(210); os.kill({args.pause_pid},signal.SIGCONT)"
            watchdog = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            os.kill(args.pause_pid, signal.SIGSTOP)
            paused = True
        signal.alarm(180)
        parent = cached_parent(manifest["checkpoint_revision"])
        model, tok, _ = load_checkpoint(str(parent))
        model.to("cuda")
        freeze_for_head(model)
        items, skipped = items_from_rows(tok, [rows[i] for i in selected], config.max_len, config.head_max_len)
        long_items, long_skipped = items_from_rows(tok, [rows[i] for i in longest], config.max_len, config.head_max_len)
        if skipped or long_skipped or len(items) != 64 or len(long_items) != 64:
            raise ValueError("Benchmark input conversion failed")
        encoded = [encode_item(tok, item, config.max_len, config.head_max_len) for item in items]
        long_encoded = [encode_item(tok, item, config.max_len, config.head_max_len) for item in long_items]
        denominator = sum(weights[i] for i in selected)
        probability_check = None
        cache_info = None
        if args.cache_encoder:
            from types import SimpleNamespace
            original_forward = model.encoder.forward
            cache = {}
            def cached_forward(input_ids, attention_mask, **kwargs):
                if kwargs or model.encoder.training or any(p.requires_grad for p in model.encoder.parameters()):
                    raise RuntimeError("Probe cache only supports the frozen sequential encoder")
                keys = [tuple(ids[:int(mask.sum())].tolist()) for ids, mask in zip(input_ids, attention_mask)]
                missing = [i for i, key in enumerate(keys) if key not in cache]
                if missing:
                    with torch.no_grad():
                        hidden = original_forward(input_ids=input_ids[missing], attention_mask=attention_mask[missing]).last_hidden_state
                    for offset, i in enumerate(missing):
                        cache[keys[i]] = hidden[offset, :len(keys[i])].detach().clone()
                    del hidden
                h = torch.zeros((*input_ids.shape, model.encoder.config.hidden_size), device=input_ids.device, dtype=next(model.encoder.parameters()).dtype)
                for i, key in enumerate(keys):
                    h[i, :len(key)] = cache[key]
                return SimpleNamespace(last_hidden_state=h)
            # Check probability and head-gradient equivalence with identical RNG.
            batch = collate_pilot(encoded[:2], tok.pad_token_id, config.max_len)
            model.eval()
            with torch.no_grad():
                reference = _forward(model, batch, torch.device("cuda"), False, True).softmax(-1).cpu()
            model.encoder.forward = cached_forward
            with torch.no_grad():
                candidate = _forward(model, batch, torch.device("cuda"), False, True).softmax(-1).cpu()
            probability_check = dict(n_records=2, maximum_probability_difference=float((reference-candidate).abs().max()))
            freeze_for_head(model)
            rng = torch.cuda.get_rng_state()
            model.encoder.forward = original_forward
            loss = _forward(model, batch, torch.device("cuda"), False, True).square().mean()
            loss.backward()
            gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            model.encoder.forward = cached_forward
            torch.cuda.set_rng_state(rng)
            loss = _forward(model, batch, torch.device("cuda"), False, True).square().mean()
            loss.backward()
            drift = max(float((p.grad-gradients[name]).abs().max()) for name,p in model.named_parameters() if name in gradients)
            model.zero_grad(set_to_none=True)
            probability_check["maximum_head_gradient_difference"] = drift
            if probability_check["maximum_probability_difference"] > 1e-5 or drift > 1e-4:
                raise RuntimeError("Cached encoder failed equivalence checks")
            del batch, loss, reference, candidate, gradients
            overhead = config.max_len-manifest["state_room"]
            estimates = {}
            for role in ["training","development"]:
                sidecar = pd.read_parquet(prepared / "enriched/provenance.parquet")
                selected_rows = sidecar[sidecar.role.eq(role)]
                tokens = sum(min(config.max_len, int(n)+overhead) for n in selected_rows.state_tokens)
                estimates[role] = tokens * model.encoder.config.hidden_size * 4
            cache_info = dict(dtype="float32", hidden_size=model.encoder.config.hidden_size,
                estimated_full_corpus_cache_bytes=estimates,
                note="Size estimate uses prepared state lengths plus native question overhead; measured speed covers one warmed training window, not all epochs")
        if args.tf32:
            model.eval()
            with torch.no_grad():
                batch = collate_pilot(encoded, tok.pad_token_id, config.max_len)
                reference = _forward(model, batch, torch.device("cuda"), False, True).softmax(-1).cpu()
                torch.backends.cuda.matmul.allow_tf32 = True
                candidate = _forward(model, batch, torch.device("cuda"), False, True).softmax(-1).cpu()
            probability_check = dict(n_records=64,
                maximum_probability_difference=float((reference - candidate).abs().max()),
                mean_probability_difference=float((reference - candidate).abs().mean()))
            del batch, reference, candidate
            freeze_for_head(model)

        def window(micro_batch, sorted_lengths=False):
            order = list(range(64))
            if sorted_lengths:
                order.sort(key=lambda i: len(encoded[i]["ids"]))
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            padded_tokens = 0
            for first in range(0, 64, micro_batch):
                indices = order[first:first + micro_batch]
                batch = collate_pilot([encoded[i] for i in indices], tok.pad_token_id, config.max_len)
                padded_tokens += batch["input_ids"].numel()
                logits = _forward(model, batch, torch.device("cuda"), False, True)
                loss = weighted_ce(logits, batch["target"].to("cuda"),
                    torch.tensor([weights[selected[i]] for i in indices], device="cuda"), denominator)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite benchmark loss")
                loss.backward()
                float(loss.detach().cpu())  # Match the trainer's loss logging synchronization.
                del batch, logits, loss
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
            torch.cuda.synchronize()
            result = dict(micro_batch=micro_batch, accumulation=64 // micro_batch,
                ordering="length_sorted_within_window" if sorted_lengths else "input_order",
                seconds=time.perf_counter() - start, padded_tokens=padded_tokens,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved())
            model.zero_grad(set_to_none=True)
            return result

        window(2)  # Warm-up excluded from measurements.
        for micro_batch, sorted_lengths in [(2, False), (8, False), (8, True), (16, True), (32, True), (64, False)]:
            try:
                result = window(micro_batch, sorted_lengths)
                # A longest-input batch tests capacity beyond the timed mixed window.
                batch = collate_pilot(long_encoded[:micro_batch], tok.pad_token_id, config.max_len)
                torch.cuda.reset_peak_memory_stats()
                logits = _forward(model, batch, torch.device("cuda"), False, True)
                loss = logits.square().mean()
                loss.backward()
                torch.cuda.synchronize()
                result["longest_batch_peak_bytes"] = torch.cuda.max_memory_allocated()
                del batch, logits, loss
                model.zero_grad(set_to_none=True)
                results.append(result)
            except torch.cuda.OutOfMemoryError:
                results.append(dict(micro_batch=micro_batch, ordering="sorted" if sorted_lengths else "input_order", status="out_of_memory"))
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                break
            print(json.dumps(results[-1]), flush=True)
        report = dict(status="finished", gpu=torch.cuda.get_device_name(), precision="float32; TF32 " + ("on" if args.tf32 else "off"),
            effective_batch=64, timing_windows=1, optimizer_updates_applied=0,
            probability_check=probability_check,
            encoder_cache=cache_info,
            records="first frozen training window; longest-input capacity probes", results=results)
        print("BATCH_BENCHMARK_RESULT=" + json.dumps(report), flush=True)
    finally:
        signal.alarm(0)
        if paused:
            os.kill(args.pause_pid, signal.SIGCONT)
        if watchdog:
            watchdog.terminate()
            watchdog.wait(timeout=5)


if __name__ == "__main__":
    main()
