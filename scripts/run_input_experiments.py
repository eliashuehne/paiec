#!/usr/bin/env python
"""Controlled prompt/state experiments; no default predictor changes.

Audit token visibility without inference, then screen questions, then compare
state rendering with one selected question held fixed. All runs use the same
explicit suite/seed, original English weights, MLX FP16 and batch size eight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from paiec.competition import INPUT_CONTRACT, competition_input
from paiec.evaluate import evaluate
from paiec.input_audit import audit_state
from paiec.laya_predictor import LayaPredictor, PROMPT_VERSION, build_state, load_questions
from paiec.laya_runtime import DEFAULT_CHECKPOINT_REVISION, create_runtime
from paiec.split import first_labels, labelled_prefix, make_split
from paiec.tracking import ExperimentTracker, evaluation_split
from run_eval import build_pairs, sample_manifest
from measure_tokens import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def audit(pairs, suite, out):
    tokenizer = load_tokenizer("english")
    question = load_questions()["correct"]
    records = []
    # Shortest, median and longest scored state from the first system per
    # benchmark: reproducible coverage of eight benchmark/task distributions.
    seen = set()
    with (out / "retained_inputs.jsonl").open("w") as handle:
        for pair in pairs:
            if pair.benchmark in seen:
                continue
            seen.add(pair.benchmark)
            split = make_split(pair.item_ids, benchmark=pair.benchmark, subject_id=pair.subject_id,
                               seed=suite["seed"], acquisition_items=suite["acquisition_items"],
                               evaluation_items=suite["evaluation_items"])
            ranked = sorted(split.evaluation, key=lambda item: len(tokenizer(
                build_state(pair.subject, pair.item(item), []))["input_ids"]))
            selected = list(dict.fromkeys([ranked[0], ranked[len(ranked) // 2], ranked[-1]]))
            for item_id in selected:
                for budget in (0, 1, 3, 7, 15, 31):
                    labeled = labelled_prefix(pair, split, budget, first_labels(pair.responses))
                    for state_format in ("json", "text_v1", "context_v1"):
                        state = build_state(pair.subject, pair.item(item_id), labeled,
                                            use_labels=True, state_format=state_format)
                        for max_len in (512, 1024):
                            row = {"benchmark": pair.benchmark, "subject_id": pair.subject_id,
                                   "item_id": item_id, "budget": budget,
                                   "state_format": state_format,
                                   **audit_state(tokenizer, state, question, max_len=max_len,
                                                 state_format=state_format)}
                            if row["revealed_outcomes"] != budget:
                                raise ValueError("Cannot locate revealed outcomes unambiguously in this rendering")
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            records.append({key: value for key, value in row.items()
                                            if key not in {"state", "question", "retained_state",
                                                           "sequence_token_ids"}})
    (out / "audit.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    flattened = []
    for row in records:
        flat = {key: value for key, value in row.items()
                if key not in {"target_system", "target_task", "task_fields"}}
        flat["target_task_complete"] = row["target_task"]["complete"]
        for key, value in row["task_fields"].items():
            flat[f"{key}_complete"] = value["complete"]
        flattened.append(flat)
    frame = pd.DataFrame(flattened)
    frame.to_csv(out / "audit.csv", index=False)
    summary = frame.groupby(["state_format", "max_len", "budget"]).agg(
        states=("truncated", "size"), truncated=("truncated", "sum"),
        complete_tasks=("target_task_complete", "sum"),
        outcomes_supplied=("revealed_outcomes", "sum"),
        outcomes_visible=("revealed_outcomes_used", "sum"))
    summary.to_csv(out / "audit_summary.csv")
    print(summary.to_string(), flush=True)


def run_variant(pairs, suite, variant, out, tracker, runtime, manifest, sample):
    folder = out / variant["id"]
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite {folder}")
    folder.mkdir()
    config = {"suite": suite, "variant": variant, "backend": "mlx", "dtype": "float16",
              "model": "english", "max_len": 512, "batch_size": 8,
              "checkpoint_revision": DEFAULT_CHECKPOINT_REVISION}
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    sample_hash = hashlib.sha256(sample.to_csv(index=False).encode()).hexdigest()
    predictor = LayaPredictor(router=runtime, batch_size=8, dtype="float16",
                              question=variant.get("question"),
                              prompt_version=variant.get("prompt_version"),
                              state_format=variant.get("state_format", "json"),
                              omit_system=variant.get("omit_system", False),
                              use_labels=variant.get("use_labels", False))
    params = {"suite": "input_design", "predictor": "laya", "variant": variant["id"],
              "seed": suite["seed"], "sample_sha256": sample_hash,
              "n_eval_inputs": int(sample.role.eq("evaluation").sum()),
              "input_contract": INPUT_CONTRACT, "laya_backend": "mlx",
              "laya_model": "english", "laya_dtype": "float16", "laya_max_len": 512,
              "laya_batch_size": 8, "laya_use_labels": predictor.use_labels,
              "laya_prompt_version": predictor.prompt_version,
              "state_format": predictor.state_format, "omit_system": predictor.omit_system,
              "dataset_revision": manifest["revision"],
              "question_sha256": hashlib.sha256(json.dumps(predictor.questions, sort_keys=True).encode()).hexdigest()}
    params.update(checkpoint_revision=predictor.revision,
                  checkpoint_weights_sha256=runtime.metadata["checkpoint_sha256"]["model.safetensors"],
                  laya_device_resolved=predictor.device, laya_dtype_resolved=runtime.metadata["compute_dtype"])
    predictions = []
    with tracker.run(f"input_design / {variant['id']} / {len(pairs)} pairs", config, params,
                     {"experiment_stage": "exploratory_input_design"}) as tracked:
        tracked.frame(sample, "sample_manifest.csv")
        tracked.dataset(evaluation_split(sample), "input-design-evaluation-split")
        tracked.json(manifest, "reproducibility/corpus_manifest.json")
        tracked.json(runtime.metadata, "runtime.json")
        tracked.json(predictor.questions, "question_definition.json")
        examples = []

        def progress():
            for index, pair in enumerate(pairs, 1):
                print(f"{variant['id']}: pair {index}/{len(pairs)} {pair.benchmark}", flush=True)
                split = make_split(pair.item_ids, benchmark=pair.benchmark, subject_id=pair.subject_id,
                                   seed=suite["seed"], acquisition_items=suite["acquisition_items"],
                                   evaluation_items=suite["evaluation_items"])
                if len(examples) < 8:
                    item = pair.item(split.evaluation[0])
                    examples.append({"input": competition_input(pair.subject, item), "labeled": [],
                                     "state": predictor.render_state(pair.subject, item, []),
                                     "questions": predictor.questions})
                yield pair
                predictor.trim_cache()

        predictor.synchronize()
        start = time.perf_counter()
        result = evaluate(progress(), predictor, seed=suite["seed"],
                          acquisition_items=suite["acquisition_items"],
                          evaluation_items=suite["evaluation_items"], prediction_records=predictions)
        predictor.synchronize()
        seconds = time.perf_counter() - start
        frame = pd.DataFrame(predictions)
        details = {"run_id": tracked.run_id, "brier_alc": result.brier_alc,
                   "seconds": seconds, "forward_items": predictor.forward_items,
                   "truncated_items": predictor.truncated_items,
                   "state_tokens_dropped": predictor.state_tokens_dropped, **params}
        tracked.result(result, seconds, frame)
        for name in ("forward_items", "truncated_items", "state_tokens_dropped"):
            tracked.client.log_metric(tracked.run_id, name, details[name])
        tracked.json(details, "run_details.json")
        tracked.json(examples, "laya_input_examples.json")
        frame.to_csv(folder / "predictions.csv", index=False)
        result.frame.to_csv(folder / "scores.csv", index=False)
        sample.to_csv(folder / "sample_manifest.csv", index=False)
        (folder / "details.json").write_text(json.dumps(details, indent=2))
        (folder / "inputs.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2))
        (folder / "runtime.json").write_text(json.dumps(runtime.metadata, indent=2))
        print(json.dumps(details), flush=True)
    return details


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["audit", "prompts", "states", "confirm", "evidence"], required=True)
    ap.add_argument("--question", default="baseline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = json.loads((ROOT / "configs/experiments/input_design.json").read_text())
    suite = spec["screen"]
    if args.phase == "confirm":
        suite = dict(suite, evaluation_items=120)
    elif args.phase == "evidence":
        suite = dict(suite, evaluation_items=4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use a new output directory: {out}")
    data = ROOT / "payload/data"
    pairs = build_pairs(data, suite)
    sample = sample_manifest(pairs, suite)
    sample.to_csv(out / "sample_manifest.csv", index=False)
    (out / "suite.json").write_text(json.dumps(suite, indent=2))
    if args.phase == "audit":
        audit(pairs, suite, out)
        return
    questions = {"baseline": {}, **{key: {"question": value, "prompt_version": key}
                                     for key, value in spec["questions"].items()}}
    if args.question not in questions:
        ap.error(f"Unknown question {args.question}")
    selected = questions[args.question]
    if args.phase == "prompts":
        variants = [{"id": key, **value} for key, value in questions.items()]
    elif args.phase == "evidence":
        variants = [{"id": f"{fmt}_labels", **selected, "state_format": fmt, "use_labels": True}
                    for fmt in ("json", "text_v1", "context_v1")]
    else:
        variants = [{"id": "json", **selected},
                    {"id": "text_v1", **selected, "state_format": "text_v1"},
                    {"id": "context_v1", **selected, "state_format": "context_v1"}]
        if args.phase == "states":
            variants.append({"id": "text_v1_no_system", **selected,
                             "state_format": "text_v1", "omit_system": True})
    manifest = json.loads((data / "corpus_manifest.json").read_text())
    tracker = ExperimentTracker(ROOT)
    runtime = create_runtime(backend="mlx", model="english", device="auto", dtype="float16",
                             revision=DEFAULT_CHECKPOINT_REVISION, batch_size=8)
    try:
        details = [run_variant(pairs, suite, variant, out, tracker, runtime, manifest, sample)
                   for variant in variants]
        pd.DataFrame(details).to_csv(out / "summary.csv", index=False)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
