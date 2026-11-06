"""Sample text questions with known difficulty from the public benchmarks.

Difficulty = facility: the share of public systems that answered the question
correctly (every trial counts). Keeps questions whose visible text is the real
question (no attached-file stubs) and that at least MIN_SYSTEMS systems answered.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
from paiec.episodes import load_valid_responses  # noqa: E402
from paiec.schema import benchmark_dir, clean, training_benchmarks  # noqa: E402

PER_BENCHMARK, MAX_BENCHMARKS, MIN_SYSTEMS, MAX_CHARS = 150, 40, 8, 6000
data = ROOT / "payload" / "data"
rng = np.random.default_rng(0)
frames = []
for benchmark in training_benchmarks(data):
    responses = load_valid_responses(str(data), benchmark)
    stats = responses.groupby("item_id").agg(systems=("subject_id", "nunique"), facility=("response", "mean"),
                                             rows=("response", "size"))
    stats = stats[stats.systems >= MIN_SYSTEMS]
    if len(stats) < 60 or stats.facility.std() < 0.05:
        continue
    items = pd.read_parquet(benchmark_dir(data, benchmark) / "items.parquet",
                            columns=["item_id", "content", "item_features"]).drop_duplicates("item_id")
    items = items.set_index("item_id").join(stats, how="inner")
    text = items["content"].map(clean)
    stub = text.str.contains("asset_path|multimedia_elements|\"content_type\"", regex=True)
    items = items[~stub & (text.str.len().between(20, MAX_CHARS))]
    if len(items) < 60:
        continue
    chosen = items.sample(min(PER_BENCHMARK, len(items)), random_state=int(rng.integers(1 << 31)))
    chosen = chosen.assign(benchmark=benchmark, content=chosen["content"].map(clean),
                           item_features=chosen["item_features"].map(clean))
    frames.append(chosen.reset_index())
    print(f"{benchmark}: {len(chosen)} questions, facility sd {chosen.facility.std():.2f}", flush=True)

sample = pd.concat(frames, ignore_index=True)
# Keep the benchmarks spread out rather than the alphabetically first ones.
keep = sorted(sample.benchmark.unique())
if len(keep) > MAX_BENCHMARKS:
    keep = list(rng.choice(keep, MAX_BENCHMARKS, replace=False))
sample = sample[sample.benchmark.isin(keep)]
sample.to_parquet(OUT, index=False)
print(f"\n{len(sample)} questions from {sample.benchmark.nunique()} benchmarks -> {OUT}")
