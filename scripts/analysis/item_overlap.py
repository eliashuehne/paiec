"""How often does a public benchmark's question text appear in another public benchmark?"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from paiec.schema import training_benchmarks  # noqa: E402

DATA = ROOT / "payload" / "data"
groups = json.loads((ROOT / "configs/lift/benchmark_groups.json").read_text())["groups"]
family = {}
for name, group in groups.items():
    for member in (group["members"] if isinstance(group, dict) else group):
        family[member] = name


def key(text):
    text = re.sub(r"\s+", " ", str(text)).strip().lower()
    return hashlib.blake2b(text.encode(), digest_size=8).digest() if len(text) >= 20 else None


owners = defaultdict(set)
items_by_bench = {}
for bench in training_benchmarks(DATA):
    items = pd.read_parquet(DATA / bench / "formatted_tables" / "items.parquet", columns=["content"])
    keys = {k for k in map(key, items["content"]) if k is not None}
    items_by_bench[bench] = keys
    for k in keys:
        owners[k].add(bench)

rows = []
for bench, keys in items_by_bench.items():
    fam = family.get(bench, bench)
    shared_any = sum(1 for k in keys if len(owners[k]) > 1)
    shared_other_family = sum(1 for k in keys if any(family.get(b, b) != fam for b in owners[k]))
    rows.append((bench, len(keys), shared_any / max(len(keys), 1), shared_other_family / max(len(keys), 1)))

frame = pd.DataFrame(rows, columns=["benchmark", "items", "shared_any", "shared_outside_family"])
print(f"benchmarks: {len(frame)}, items: {frame['items'].sum()}")
for column in ["shared_any", "shared_outside_family"]:
    print(f"{column}: benchmarks with >=10% shared: {(frame[column] >= 0.1).sum()}, "
          f">=50%: {(frame[column] >= 0.5).sum()}, item-weighted share: "
          f"{(frame[column] * frame['items']).sum() / frame['items'].sum():.3f}")
print(frame.sort_values("shared_any", ascending=False).head(30).to_string(index=False))

