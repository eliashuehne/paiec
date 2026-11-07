"""How much of a question's length is fixed by its benchmark, and how often do texts repeat?

    python scripts/analysis/text_facts.py [paiec root]

Share of the variance of (log) question length between benchmarks, over all questions of
the public benchmarks, and the benchmarks with the most questions per distinct visible text.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from paiec.schema import benchmark_dir, training_benchmarks  # noqa: E402

data = ROOT / "payload" / "data"
sums, repeats = [], []
for benchmark in training_benchmarks(data):
    items = pd.read_parquet(benchmark_dir(data, benchmark) / "items.parquet", columns=["item_id", "content"])
    items = items.drop_duplicates("item_id")
    text = items["content"].map(lambda c: c if isinstance(c, str) else "")
    for name, values in (("chars", text.str.len().to_numpy(float)), ("log_chars", np.log1p(text.str.len().to_numpy(float)))):
        sums.append((name, len(values), values.sum(), (values**2).sum()))
    repeats.append((benchmark, len(text), text.nunique()))

table = pd.DataFrame(sums, columns=["measure", "n", "sum", "squares"])
for measure, rows in table.groupby("measure"):
    total_n, total_sum = rows.n.sum(), rows["sum"].sum()
    total = rows.squares.sum() - total_sum**2 / total_n
    between = (rows["sum"] ** 2 / rows.n).sum() - total_sum**2 / total_n
    print(f"{measure}: {total_n:,} questions, share of variance between benchmarks {between / total:.3f}")

frame = pd.DataFrame(repeats, columns=["benchmark", "questions", "distinct_texts"])
frame["per_text"] = frame.questions / frame.distinct_texts
print(frame[frame.questions >= 200].sort_values("per_text", ascending=False).head(8).to_string(index=False))
