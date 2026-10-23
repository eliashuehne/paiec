#!/usr/bin/env python3
"""Check the prepared experiment; actual training requires --train."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.head_pilot import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_finetuning/enrichment_pilot_v1.json")
    parser.add_argument("--variant", choices=["baseline", "enriched", "both"], default="both")
    parser.add_argument("--train", action="store_true", help="explicitly start gradient updates and development inference")
    parser.add_argument("--out", help="fresh training output directory; prepared inputs and training settings stay fixed")
    parser.add_argument("--device", choices=["mps", "cuda"], help="explicit placement override; no fallback")
    parser.add_argument("--download-parent", action="store_true", help="allow fetching only the pinned original parent into the model cache")
    parser.add_argument("--encoder-cache-gib", type=int, choices=range(0,17), default=0)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    if args.out:
        config = type(config).model_validate(dict(config.model_dump(), output=args.out))
    if args.device:
        config = type(config).model_validate(dict(config.model_dump(), device=args.device))
    from paiec.head_pilot_training import preflight
    report = preflight(ROOT, config, allow_download=args.download_parent)
    write_json(ROOT / config.prepared / (config.device + "_preflight.json"), report)
    print(json.dumps(report, indent=2), flush=True)
    if not args.train:
        print("Preparation verified. No model weights loaded, no inference or training performed.")
        return
    destination = ROOT / config.output
    destination.mkdir(parents=True, exist_ok=False)
    from paiec.head_pilot_training import train_variant
    variants = ["baseline", "enriched"] if args.variant == "both" else [args.variant]
    write_json(destination / "run_plan.json", {"variants": variants, "config": config.model_dump(), "preflight": report,
                                               "encoder_cache_gib": args.encoder_cache_gib})
    summaries = [train_variant(ROOT, config, variant, encoder_cache_gib=args.encoder_cache_gib) for variant in variants]
    write_json(destination / "comparison.json", {"runs": summaries, "constant_brier_alc": .25,
        "note": f"Matched raw heads, same frozen records, {config.max_len}-token limit; two development families only.",
        "enriched_minus_baseline": summaries[1]["brier_alc"] - summaries[0]["brier_alc"] if len(summaries) == 2 else None})
    from report_head_pilot import render
    render(destination)


if __name__ == "__main__":
    main()
