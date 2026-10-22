#!/usr/bin/env python
"""Measure how many Laya state tokens the scored items actually need.

Truncation is decided by the tokenizer, not by character counts, and the
checkpoint tokenizes the rendered state JSON (target system, target item,
optional revealed examples). This script rebuilds exactly the states the
predictor would send for a suite's evaluation split and reports their length,
so a context-limit choice can be attributed to real benchmarks.

    .venv-laya-mlx/bin/python scripts/measure_tokens.py --config runs/configs/sample_1k.json --suite sample_1k

It downloads only the tokenizer files from the pinned checkpoint, not model
weights. Results describe the stored text the adapter can see, not image or
audio modalities, which it cannot read. Run it from the MLX environment, which
is the one that has the tokenizer package installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paiec.laya_predictor import build_state
from paiec.pairs import balanced_pairs, choose_pairs
from paiec.schema import load_benchmark
from paiec.split import make_split

ROOT = Path(__file__).resolve().parents[1]
CEILINGS = (512, 1024, 2048, 4096, 8192)


def load_tokenizer(model: str):
    """Tokenizer files from the pinned checkpoint; model weights are never fetched."""
    from huggingface_hub import snapshot_download
    from laya_mlx.tokenizer import Tokenizer

    from paiec.laya_runtime import DEFAULT_CHECKPOINT_REVISION, MODEL_REPOSITORY

    prefix = "multilingual/" if model == "multilingual" else ""
    snapshot = Path(snapshot_download(
        MODEL_REPOSITORY, revision=DEFAULT_CHECKPOINT_REVISION,
        allow_patterns=[prefix + "tokenizer/*"],
    ))
    return Tokenizer(snapshot / prefix / "tokenizer")


def suite_pairs(data_root: Path, suite: dict):
    pairs = []
    for benchmark in suite["benchmarks"]:
        tables = load_benchmark(data_root, benchmark)
        pairs.extend(choose_pairs(
            tables,
            max_subjects=suite["pairs_per_benchmark"],
            min_items=suite.get("min_items", 80),
            randomize=suite.get("pair_selection") == "random",
            seed=suite["seed"],
        ))
    if suite.get("total_pairs") is not None:
        pairs = balanced_pairs(pairs, suite["total_pairs"], seed=suite["seed"])
    return pairs


def measure(pairs, suite: dict, tokenizer) -> pd.DataFrame:
    """Token length of each scored item's rendered state, one row per item."""
    from laya_mlx.common import encode_text, serialize_state

    records = []
    for pair in pairs:
        split = make_split(pair.item_ids, benchmark=pair.benchmark, subject_id=pair.subject_id,
                           seed=suite["seed"], acquisition_items=suite["acquisition_items"],
                           evaluation_items=suite["evaluation_items"])
        for item_id in split.evaluation:
            # Zero-shot state: the default inference mode. Revealed examples
            # are an opt-in experiment and would need the acquisition labels.
            state = build_state(pair.subject, pair.item(item_id), [])
            tokens = encode_text(
                tokenizer, serialize_state(state).replace(tokenizer.mask_token, " "),
                add_special_tokens=False,
            )["input_ids"]
            records.append({"benchmark": pair.benchmark, "subject_id": pair.subject_id,
                            "item_id": item_id, "tokens": len(tokens)})
    return pd.DataFrame(records)


def report(frame: pd.DataFrame) -> dict:
    by_benchmark = {}
    for benchmark, group in frame.groupby("benchmark", sort=True):
        by_benchmark[benchmark] = {
            "items": int(group.tokens.size),
            "median_tokens": float(group.tokens.median()),
            "max_tokens": int(group.tokens.max()),
            **{f"over_{cap}": int((group.tokens > cap).sum()) for cap in CEILINGS},
        }
    return {
        "items": int(len(frame)),
        "median_tokens": float(frame.tokens.median()),
        "p99_tokens": float(frame.tokens.quantile(0.99)),
        "by_benchmark": by_benchmark,
        "over_ceiling": {
            str(cap): {"items": int((frame.tokens > cap).sum()),
                       "share": float((frame.tokens > cap).mean())}
            for cap in CEILINGS
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "suites.json"))
    ap.add_argument("--suite", default="default")
    ap.add_argument("--data-root", default=str(ROOT / "payload" / "data"))
    ap.add_argument("--seed", type=int, default=None, help="override the suite seed")
    ap.add_argument("--model", choices=["english", "multilingual"], default="english",
                    help="tokenizer to measure with; the checkpoints share a vocabulary")
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--csv", default=None, help="write per-item token counts as CSV")
    args = ap.parse_args()

    suites = json.loads(Path(args.config).read_text())
    if args.suite not in suites:
        raise SystemExit(f"unknown suite {args.suite!r}; have {sorted(suites)}")
    suite = suites[args.suite]
    if args.seed is not None:
        suite = dict(suite, seed=args.seed)

    tokenizer = load_tokenizer(args.model)
    frame = measure(suite_pairs(Path(args.data_root), suite), suite, tokenizer)
    summary = report(frame)

    print(f"suite {args.suite}: {summary['items']:,} scored states, "
          f"median {summary['median_tokens']:.0f} tokens, p99 {summary['p99_tokens']:.0f}")
    print(f"{'benchmark':<12}{'items':>7}{'median':>9}{'max':>8}" +
          "".join(f"{'>' + str(cap):>8}" for cap in CEILINGS))
    for benchmark, row in summary["by_benchmark"].items():
        print(f"{benchmark:<12}{row['items']:>7}{row['median_tokens']:>9.0f}{row['max_tokens']:>8}" +
              "".join(f"{row[f'over_{cap}']:>8}" for cap in CEILINGS))
    print()
    for cap, row in summary["over_ceiling"].items():
        print(f"over {cap:>5}: {row['items']:>5} items ({row['share'] * 100:.1f}%)")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
