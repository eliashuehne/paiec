#!/usr/bin/env python3
"""Expanded enriched encoder experiments with independent token/probe/train stages."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.encoder_training import EncoderTrainingConfig, ComputeSettings, encode_data, save_token_cache, train_arm
from paiec.head_pilot import write_json
from paiec.head_training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_finetuning/enriched_30k_encoder_v1.json")
    parser.add_argument("--stage", choices=["prepare-tokens", "qualify", "train"], required=True)
    parser.add_argument("--out")
    parser.add_argument("--qualification", help="saved passed qualification report; required for training")
    parser.add_argument("--download-parent", action="store_true")
    args = parser.parse_args()
    config = EncoderTrainingConfig.model_validate_json((ROOT / args.config).read_text())
    if args.stage == "prepare-tokens":
        print(json.dumps(save_token_cache(ROOT, config)), flush=True)
        return
    if not args.out:
        parser.error("Use a fresh --out directory")
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("Experiment output exists")
    if args.stage == "qualify":
        from paiec.encoder_qualification import qualify
        print(json.dumps(qualify(ROOT, out, config, download=args.download_parent)), flush=True)
        return
    if not args.qualification:
        parser.error("Training requires a passed --qualification report")
    qualification = json.loads((ROOT / args.qualification).read_text())
    config_hash = hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode()).hexdigest()
    if qualification["status"] != "passed" or qualification["config_sha256"] != config_hash:
        raise ValueError("Qualification failed or recipe changed")
    if qualification["prepared_manifest_sha256"] != file_hash(ROOT / config.prepared / "manifest.json"):
        raise ValueError("Qualification inputs changed")
    for name, expected in qualification["training_source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Training code changed since qualification: " + name)
    import torch
    if torch.cuda.get_device_name() != qualification["gpu"]:
        raise ValueError("Qualified GPU differs")
    out.mkdir(parents=True)
    settings = ComputeSettings.model_validate(qualification["settings"]).model_dump()
    write_json(out / "run_plan.json", dict(config=config.model_dump(), settings=settings, qualification=qualification,
        arms=["frozen", f"upper_{config.upper_layers}"], both_enriched=True))
    data = encode_data(ROOT, config, download=args.download_parent)
    reports = [train_arm(ROOT, out, config, settings, upper, data) for upper in [0, config.upper_layers]]
    write_json(out / "comparison.json", dict(runs=reports, adapted_minus_frozen=reports[1]["brier_alc"]-reports[0]["brier_alc"],
        note="Same enriched 30k inputs and precision; original development; raw predictions ignore revealed outcomes."))
    import pandas as pd
    table = pd.DataFrame([{key: r[key] for key in ["arm", "brier_alc", "updates", "presentations", "peak_cuda_allocated_bytes", "peak_host_rss_kib", "training_development_export_seconds"]} for r in reports])
    html = '<!doctype html><meta charset="utf-8"><title>Enriched 30k encoder experiment</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:16px}td,th{padding:8px}pre{white-space:pre-wrap}</style>'
    html += '<h1>Enriched 30k: frozen versus upper encoder training</h1><p>Both arms start from the original pinned English parent and use identical enriched records, optimizer windows and precision. Original development remains unchanged; final calibration/confirmation remain sealed. These raw heads ignore revealed-label budgets.</p>'
    html += table.to_html(index=False) + '<pre>' + escape((out / "comparison.json").read_text()) + '</pre>'
    (out / "report.html").write_text(html)


if __name__ == "__main__":
    main()
