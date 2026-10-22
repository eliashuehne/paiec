#!/usr/bin/env python
"""Score one or more predictors over a named evaluation suite.

    PYTHONPATH=src .venv/bin/python scripts/run_eval.py \
        --predictor constant empirical_mean subject_prior shrunk_mean

The suite is defined in configs/suites.json. All pairs come from the public
training release, so this measures predictors on benchmarks they can see; the
honest cold-start check is the leave-benchmarks-out suite used for the learned
models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from paiec.evaluate import evaluate, format_summary
from paiec.config import EvaluationConfig, LayaConfig, SuiteConfig
from paiec.pairs import choose_pairs, balanced_pairs
from paiec.predictors import Constant, EmpiricalMean, ShrunkMean, SubjectPrior
from paiec.schema import load_benchmark
from paiec.competition import INPUT_CONTRACT, competition_input
from paiec.stats import load_or_build_subject_stats
from paiec.textmodel import TextRidge, load_or_build_text_training
from paiec.tracking import ExperimentTracker, evaluation_split
from paiec.laya_runtime import DEFAULT_CHECKPOINT_REVISION

ROOT = Path(__file__).resolve().parents[1]


def manifest_benchmarks(data_root: Path) -> list[str]:
    path = data_root / "corpus_manifest.json"
    manifest = json.loads(path.read_text())
    return [entry.split("/")[0] for entry in manifest["eligible_dirs"]]


def load_suite(config: Path, name: str) -> dict:
    suites = json.loads(config.read_text())
    if name not in suites:
        raise SystemExit(f"unknown suite {name!r}; have {sorted(suites)}")
    return SuiteConfig.model_validate(suites[name]).model_dump()


def resolve_seed(configured: int | None, override: str | None) -> int:
    if override == "random" or (override is None and configured is None):
        return secrets.randbits(32)
    return configured if override is None else int(override)


def predictor_instance(name, args, data_root, suite, cache):
    if name != "laya":
        return make_predictor(name, data_root, suite, cache, args.k)
    from paiec.laya_predictor import LayaPredictor
    predictor = LayaPredictor(
        model=args.laya_model, device=args.laya_device,
        max_len=args.laya_max_len, batch_size=args.laya_batch_size,
        use_labels=args.laya_use_labels, revision=args.laya_revision,
        backend=args.laya_backend, dtype=args.laya_dtype, checkpoint_path=args.laya_checkpoint,
        state_format=args.laya_state_format,
    )
    print(f"  Laya {predictor.backend}/{predictor.model}, {predictor.device}, max_len={predictor.max_len}, "
          f"revealed examples={predictor.use_labels}, prompt={predictor.prompt_version}", flush=True)
    return predictor


def input_examples(pairs, suite, limit=8):
    from paiec.split import make_split
    examples = []
    for pair in pairs:
        split = make_split(pair.item_ids, benchmark=pair.benchmark, subject_id=pair.subject_id,
                           seed=suite["seed"], acquisition_items=suite["acquisition_items"],
                           evaluation_items=suite["evaluation_items"])
        for item_id in split.evaluation[:4]:
            examples.append({"benchmark": pair.benchmark, "subject_id": pair.subject_id,
                             "item_id": item_id, "subject": competition_input(pair.subject, pair.item(item_id))[0],
                             "item": competition_input(pair.subject, pair.item(item_id))[1]})
            if len(examples) >= limit:
                return examples
    return examples


def build_pairs(data_root: Path, suite: dict):
    pairs = []
    for benchmark in suite["benchmarks"]:
        tables = load_benchmark(data_root, benchmark)
        chosen = choose_pairs(
            tables,
            max_subjects=suite["pairs_per_benchmark"],
            min_items=suite.get("min_items", 80),
            randomize=suite.get("pair_selection") == "random", seed=suite["seed"],
        )
        if not chosen:
            print(f"  {benchmark}: no pair with {suite.get('min_items', 80)}+ items")
        else:
            print(f"  {benchmark}: {len(chosen)} pair(s)")
        pairs.extend(chosen)
    if not pairs:
        raise SystemExit("suite produced no pairs")
    if suite.get("total_pairs") is not None:
        pairs = balanced_pairs(pairs, suite["total_pairs"], seed=suite["seed"])
    return pairs


def sample_manifest(pairs, suite) -> pd.DataFrame:
    """Record all acquisition/evaluation IDs, without copying any outcomes."""
    from paiec.split import make_split
    records = []
    for pair in pairs:
        split = make_split(pair.item_ids, benchmark=pair.benchmark, subject_id=pair.subject_id,
                           seed=suite["seed"], acquisition_items=suite["acquisition_items"],
                           evaluation_items=suite["evaluation_items"])
        for role, ids in (("acquisition", split.acquisition), ("evaluation", split.evaluation)):
            records.extend({"benchmark": pair.benchmark, "subject_id": pair.subject_id,
                            "item_id": item_id, "role": role, "position": position}
                           for position, item_id in enumerate(ids))
    return pd.DataFrame(records)


def evaluation_counts(pairs, suite) -> list[int]:
    """Scored items per pair; the 50/50 split can cap below ``evaluation_items``."""
    from paiec.split import make_split
    return [
        len(make_split(pair.item_ids, benchmark=pair.benchmark, subject_id=pair.subject_id,
                       seed=suite["seed"], acquisition_items=suite["acquisition_items"],
                       evaluation_items=suite["evaluation_items"]).evaluation)
        for pair in pairs
    ]


def make_predictor(name: str, data_root: Path, suite: dict, cache: Path, k: float):
    if name == "constant":
        return Constant(0.5)
    if name == "empirical_mean":
        return EmpiricalMean()
    if name in {"subject_prior", "shrunk_mean", "text_ridge"}:
        train = sorted(b for b in manifest_benchmarks(data_root) if b not in suite["benchmarks"])
        if not train:
            raise SystemExit("no training benchmarks left outside the suite")
        manifest_hash = hashlib.sha256((data_root / "corpus_manifest.json").read_bytes()).hexdigest()
        training_hash = hashlib.sha256(json.dumps({"benchmarks": train, "manifest": manifest_hash,
                                                 "schema": "competition_v1"}, sort_keys=True).encode()).hexdigest()[:16]
    if name in {"subject_prior", "shrunk_mean"}:
        stats = load_or_build_subject_stats(
            data_root, train, cache / f"subject_stats_{training_hash}.parquet"
        )
        prior = SubjectPrior(stats)
        return prior if name == "subject_prior" else ShrunkMean(prior, k=k)
    if name == "text_ridge":
        frame = load_or_build_text_training(
            data_root, train, cache / f"text_train_{suite['seed']}_{training_hash}.parquet",
            per_benchmark=100, seed=suite["seed"],
        )
        print(f"  ridge trained on {len(frame)} items from {len(train)} benchmarks")
        return TextRidge().fit(frame["text"], frame["label"])
    raise SystemExit(f"unknown predictor {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(ROOT / "payload" / "data"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "suites.json"))
    ap.add_argument("--suite", default="default")
    ap.add_argument("--seed", default=None, help="sampling/split seed, or 'random' for a fresh sample")
    ap.add_argument("--predictor", nargs="+", default=["constant"])
    ap.add_argument("--k", type=float, default=5.0, help="prior weight for shrunk_mean")
    ap.add_argument("--cache", default=str(ROOT / "runs" / "cache"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--laya-model", choices=["english", "multilingual"], default="english")
    ap.add_argument("--laya-backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--laya-dtype", choices=["auto", "float16", "float32"], default="auto",
                    help="MLX precision (auto=float16); torch requires auto and retains upstream precision policy")
    ap.add_argument("--laya-device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    ap.add_argument("--laya-max-len", type=int, default=None,
                    help="token ceiling; defaults to the checkpoint's own trained length "
                         "(english 512, multilingual 1024)")
    ap.add_argument("--laya-batch-size", type=int, default=1)
    ap.add_argument("--laya-state-format", choices=["json", "text_v1", "context_v1"], default="json")
    ap.add_argument("--laya-use-labels", action="store_true", help="include revealed examples in Laya's input")
    ap.add_argument("--laya-revision", default=DEFAULT_CHECKPOINT_REVISION, help="pinned upstream checkpoint revision")
    ap.add_argument("--laya-checkpoint", default=None, help="explicit local PAIEC trained checkpoint directory")
    ap.add_argument("--no-tracking", action="store_true", help="disable MLflow logging")
    ap.add_argument("--experiment", default="PAIEC", help="MLflow experiment name")
    ap.add_argument("--tracking-uri", default=None, help="override the project-local SQLite store")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    suite = load_suite(Path(args.config), args.suite)
    suite["seed"] = resolve_seed(suite["seed"], args.seed)
    manifest = json.loads((data_root / "corpus_manifest.json").read_text())
    settings = EvaluationConfig(
        suite=SuiteConfig.model_validate(suite), predictors=args.predictor, prior_weight=args.k,
        laya=LayaConfig(model=args.laya_model, device=args.laya_device,
                        backend=args.laya_backend, dtype=args.laya_dtype, checkpoint_path=args.laya_checkpoint,
                        state_format=args.laya_state_format,
                        max_len=args.laya_max_len, batch_size=args.laya_batch_size,
                        use_labels=args.laya_use_labels, revision=args.laya_revision),
    )
    cache = Path(args.cache)

    print(f"suite {args.suite}: {', '.join(suite['benchmarks'])}")
    pairs = build_pairs(data_root, suite)
    sample = sample_manifest(pairs, suite)
    n_eval_inputs = int(sample.role.eq("evaluation").sum())
    sample_hash = hashlib.sha256(sample.to_csv(index=False).encode()).hexdigest()
    print(f"  {len(pairs)} pairs, {suite['evaluation_items']} eval items "
          f"and {suite['acquisition_items']} acquisition items per pair\n")
    print(f"  seed={suite['seed']}; {n_eval_inputs:,} system-question inputs", flush=True)
    interface_examples = "".join(
        json.dumps({"input": [example["subject"], example["item"]], "labeled": []}, ensure_ascii=False) + "\n"
        for example in input_examples(pairs, suite)
    )

    results = {}
    metadata = {"input_contract": INPUT_CONTRACT, "created_utc": datetime.now(timezone.utc).isoformat(),
                "suite": suite, "sample_sha256": sample_hash, "n_eval_inputs": n_eval_inputs, "predictors": {}}
    suite_hash = hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()
    metadata["dataset_revision"] = manifest.get("revision", "unknown")
    tracker = ExperimentTracker(ROOT, enabled=not args.no_tracking,
                                experiment=args.experiment, tracking_uri=args.tracking_uri)
    for name in args.predictor:
        # Comparison columns only. Suite shape, hashes, and package provenance are
        # artifacts (config.json, suite.json, runtime.json), not table columns.
        params = {"input_contract": INPUT_CONTRACT, "predictor": name, "suite": args.suite,
                  "suite_sha256": suite_hash, "dataset_revision": metadata["dataset_revision"],
                  "sample_sha256": sample_hash, "n_eval_inputs": n_eval_inputs,
                  "seed": suite["seed"]}
        if name == "shrunk_mean":
            params["prior_weight"] = args.k
        title = f"{args.suite} / {name} / {INPUT_CONTRACT}"
        if name == "laya":
            params.update({f"laya_{key}": value for key, value in settings.laya.model_dump().items()
                           if value is not None and key != "revision"})
            # settings.laya.max_len is already resolved, so an omitted flag still
            # labels the run with the concrete ceiling it actually used.
            title += f" / {args.laya_backend} / {args.laya_dtype} / {args.laya_model} / {settings.laya.max_len} / {'with labels' if args.laya_use_labels else 'zero-shot'}"
            from paiec.laya_predictor import PROMPT_VERSION
            params["laya_prompt_version"] = PROMPT_VERSION
            title += f" / {PROMPT_VERSION} / {args.laya_state_format}"
            if args.laya_checkpoint:
                from paiec.laya_runtime import local_checkpoint_files
                _, identity = local_checkpoint_files(args.laya_checkpoint, args.laya_model)
                title += " / " + identity["checkpoint_revision"][:18]
                params["trained_checkpoint_identity"] = identity["checkpoint_revision"]
        with tracker.run(title, settings.model_dump(), params, {}) as tracked:
            if tracked:
                tracked.json(manifest, "reproducibility/corpus_manifest.json")
                tracked.frame(sample, "sample_manifest.csv")
                # One dataset input per run links every run scored on the same
                # evaluation split, so the UI can group and cross-filter them.
                tracked.dataset(evaluation_split(sample),
                                f"{args.suite}-evaluation_split")
                tracked.json(input_examples(pairs, suite), "input_examples.json")
                tracked.client.log_text(tracked.run_id, interface_examples, "competition_input_examples.jsonl")
                if name in {"subject_prior", "shrunk_mean", "text_ridge"}:
                    tracked.client.set_tag(tracked.run_id, "known_limitations", "Public-data local evaluation; see docs/outcome-semantics.md")
            predictor = predictor_instance(name, args, data_root, suite, cache)
            if tracked and name == "laya":
                effective_questions = predictor.questions
                question_hash = hashlib.sha256(json.dumps(effective_questions, sort_keys=True).encode()).hexdigest()
                tracked.json(effective_questions, "question_definition.json")
                tracked.json(predictor.runtime_metadata, "runtime.json")
                tracked.json([
                    {"benchmark": example["benchmark"], "subject_id": example["subject_id"],
                     "item_id": example["item_id"], "budget": 0,
                     "state": predictor.render_state(example["subject"], example["item"], []),
                     "questions": effective_questions}
                    for example in input_examples(pairs, suite)
                ], "laya_input_examples.json")
                tracked.params({"question_sha256": question_hash, "checkpoint_revision": predictor.revision,
                                "laya_device_resolved": predictor.device,
                                "laya_dtype_resolved": predictor.runtime_metadata["compute_dtype"]})
                weights = predictor.runtime_metadata.get("checkpoint_sha256", {}).get("model.safetensors")
                if weights:
                    tracked.params({"checkpoint_weights_sha256": weights})
            counts = evaluation_counts(pairs, suite) if tracked else []
            def progress_pairs():
                completed = 0
                for index, pair in enumerate(pairs, start=1):
                    if name == "laya":
                        print(f"  pair {index}/{len(pairs)}: {pair.benchmark}/{pair.subject_id}", flush=True)
                    yield pair
                    if name == "laya":
                        # MLX caches freed Metal buffers per tensor shape; without this
                        # resident memory climbs with every new item length.
                        predictor.trim_cache()
                    if tracked:
                        completed += counts[index - 1]
                        tracked.client.log_metric(tracked.run_id, "completed_pairs", index)
                        tracked.client.log_metric(tracked.run_id, "completed_eval_inputs", completed)
                        if name == "laya":
                            tracked.client.log_metric(tracked.run_id, "forward_items", predictor.forward_items)
                            tracked.client.log_metric(tracked.run_id, "truncated_items", predictor.truncated_items)
            prediction_records = [] if tracked else None
            if name == "laya":
                predictor.synchronize()
            started = time.perf_counter()
            result = evaluate(
                progress_pairs(), predictor, seed=suite["seed"],
                acquisition_items=suite["acquisition_items"], evaluation_items=suite["evaluation_items"],
                prediction_records=prediction_records,
            )
            if name == "laya":
                predictor.synchronize()
            results[name] = result
            detail = {"seconds": time.perf_counter() - started}
            metadata["predictors"][name] = detail
            if name == "laya":
                detail.update(
                    model=predictor.model, device=predictor.device, max_len=predictor.max_len,
                    batch_size=predictor.batch_size, use_labels=predictor.use_labels,
                    revision=predictor.revision, forward_items=predictor.forward_items,
                    truncated_items=predictor.truncated_items, state_tokens_dropped=predictor.state_tokens_dropped,
                    backend=predictor.backend, runtime=predictor.runtime_metadata,
                    packages=predictor.runtime_metadata["packages"],
                    prompt_version=predictor.prompt_version, state_format=predictor.state_format,
                )
                effective_questions = predictor.questions
                detail["input_contract"] = INPUT_CONTRACT
                detail["question_sha256"] = hashlib.sha256(json.dumps(effective_questions, sort_keys=True).encode()).hexdigest()
            if tracked:
                detail["mlflow_run_id"] = tracked.run_id
                tracked.result(result, detail["seconds"], pd.DataFrame(prediction_records))
                tracked.json(detail, "run_details.json")
                if name == "laya":
                    for key in ("forward_items", "truncated_items", "state_tokens_dropped"):
                        tracked.client.log_metric(tracked.run_id, key, float(detail[key]))
        print(format_summary(result, label=name))
        print()

    summary = pd.DataFrame(
        {
            name: {
                "brier_alc": result.brier_alc,
                "mean_ece": result.ece,
                **{f"brier_{b}": v for b, v in result.per_budget.items()},
            }
            for name, result in results.items()
        }
    ).T.sort_values("brier_alc")

    print("comparison (lower is better)")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    default_out = ROOT / "runs" / f"eval_{args.suite}"
    if "laya" in results:
        mode = "with_labels" if args.laya_use_labels else "zeroshot"
        prompt_version = metadata["predictors"]["laya"]["prompt_version"]
        runtime = metadata["predictors"]["laya"]["runtime"]
        dtype = runtime["compute_dtype"].removeprefix("torch.")
        default_out /= (f"laya_{args.laya_backend}_{dtype}_{INPUT_CONTRACT}_"
                        f"{args.laya_model}_{settings.laya.max_len}_{mode}_{prompt_version}")
    if "laya" in results and args.laya_state_format != "json":
        default_out /= args.laya_state_format
    if "laya" in results and args.laya_checkpoint:
        default_out /= metadata["predictors"]["laya"]["runtime"]["checkpoint_revision"]
    if suite["pair_selection"] == "random":
        default_out /= f"seed_{suite['seed']}"
    out = Path(args.out) if args.out else default_out
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.csv"
    if summary_path.is_file():
        # Keep earlier predictors so a suite can be filled in across runs.
        previous = pd.read_csv(summary_path, index_col=0)
        previous = previous.drop(index=summary.index, errors="ignore")
        summary = pd.concat([previous, summary]).sort_values("brier_alc")
    summary.to_csv(summary_path)
    for name, result in results.items():
        result.frame.to_csv(out / f"scores_{name}.csv", index=False)
    (out / "suite.json").write_text(json.dumps(suite, indent=2))
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    sample.to_csv(out / "sample_manifest.csv", index=False)
    (out / "competition_input_examples.jsonl").write_text(interface_examples)
    if "laya" in results:
        (out / "question_definition.json").write_text(json.dumps(effective_questions, indent=2))
        (out / "runtime.json").write_text(json.dumps(metadata["predictors"]["laya"]["runtime"], indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
