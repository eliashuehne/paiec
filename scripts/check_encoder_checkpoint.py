#!/usr/bin/env python3
"""Verify explicit encoder-experiment checkpoints on training-only input fixtures."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.expanded_data import verify_expanded
from paiec.head_pilot import read_rows, write_json, digest_state
from paiec.head_training import file_hash
from paiec.laya_runtime import local_checkpoint_files
from paiec.predictor_factory import make_laya_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared", default="runs/head_finetuning/enriched_30k_v1")
    parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
    parser.add_argument("--reference", help="same fixtures/checkpoint from another backend")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    target = ROOT / args.out
    if target.exists():
        raise FileExistsError("Keep each runtime check immutable")
    prepared = ROOT / args.prepared
    verify_expanded(ROOT, prepared)
    checkpoint, identity = local_checkpoint_files(args.checkpoint, "english")
    saved = identity["training_provenance"]
    if saved.get("run_type") != "enriched_encoder_experiment":
        raise ValueError("Use an explicit completed encoder-experiment checkpoint")
    if saved["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json"):
        raise ValueError("Checkpoint and prepared inputs differ")
    source = read_rows(prepared / "source/training.jsonl")
    retained = read_rows(prepared / "enriched/training.jsonl")
    # Include short, long and first-window fixtures without consulting outcomes.
    indices = sorted(set([0, 1, 2, min(range(len(retained)), key=lambda i: len(retained[i]["state"])),
                          max(range(len(retained)), key=lambda i: len(retained[i]["state"]))]))
    predictor = make_laya_predictor(backend=args.backend, dtype="float32", batch_size=1,
                                    checkpoint_path=str(checkpoint))
    records = []
    try:
        for index in indices:
            parsed = json.loads(source[index]["state"])
            subject, item = parsed["target_system"], parsed["target_item"]
            rendered = predictor.render_state(subject, item, [])
            if rendered != retained[index]["state"]:
                raise ValueError("Native runtime rendering differs from prepared training")
            probability = predictor(subject, item, [])
            records.append(dict(training_index=index, state_sha256=digest_state(rendered),
                                probability=probability))
            predictor.trim_cache()
        result = dict(status="passed", checkpoint_identity=identity["checkpoint_revision"],
                      runtime=predictor.runtime_metadata, fixtures=records,
                      prepared_manifest_sha256=file_hash(prepared / "manifest.json"),
                      validation_source_sha256=file_hash(Path(__file__)),
                      scope="five or fewer training-only fixtures; exact input reconstruction and native inference",
                      model_training=False, quality_evaluation=False, sealed_outcomes_read=False)
        if args.reference:
            reference = json.loads((ROOT / args.reference).read_text())
            if reference["checkpoint_identity"] != result["checkpoint_identity"]:
                raise ValueError("Cross-backend reference uses different weights")
            if [(r["training_index"], r["state_sha256"]) for r in reference["fixtures"]] != [
                    (r["training_index"], r["state_sha256"]) for r in records]:
                raise ValueError("Cross-backend reference uses different fixtures")
            drift = max(abs(a["probability"]-b["probability"])
                        for a,b in zip(records, reference["fixtures"]))
            result["max_reference_probability_difference"] = drift
            result["cross_backend_max_difference_gate"] = .001
            if drift > .001:
                raise ValueError(f"Cross-backend FP32 drift exceeds .001: {drift}")
        write_json(target, result)
        print(json.dumps(dict(status="passed", backend=args.backend, fixtures=len(records),
            probabilities=[r["probability"] for r in records],
            max_reference_difference=result.get("max_reference_probability_difference"))))
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
