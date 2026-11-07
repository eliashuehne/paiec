"""Are the logistic-prior features broken, or do they just not transfer across benchmarks?

On ~40 training benchmarks: hold out 20% of ITEMS inside each benchmark (by item id).
Fit the same features (pair fixed effects, so only within-pair differences are learned)
and measure within-pair AUC:
  in-sample (training rows)        -> ~0.5 would mean a bug
  held-out items, same benchmarks  -> > 0.5 means features carry within-benchmark signal
  reference: other subjects' mean on the same item (collaborative) on the same held-out rows
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from paiec.features import DENSE, DIM, N_DENSE, item_vector, subject_vector  # noqa: E402
from paiec.schema import benchmark_dir, item_record, load_benchmark, subject_record, training_benchmarks  # noqa: E402

DATA = ROOT / "payload" / "data"
rng = np.random.default_rng(0)
names = [b for b in training_benchmarks(DATA)
         if (benchmark_dir(DATA, b) / "items.parquet").stat().st_size < 150e6]
chosen = sorted(rng.choice(names, size=40, replace=False))

rows_idx, rows_val, labels, pair_ids, held, collab = [], [], [], [], [], []
pid = 0
for benchmark in chosen:
    tables = load_benchmark(DATA, benchmark)
    resp = tables.responses.drop_duplicates(["subject_id", "item_id"])
    resp = resp[resp["subject_id"].isin(tables.subjects.index)]
    item_sum = resp.groupby("item_id")["label"].sum()
    item_cnt = resp.groupby("item_id")["label"].size()
    sizes = resp.groupby("subject_id").size()
    qualifying = sorted(sizes[sizes >= 80].index)
    if len(qualifying) < 2:
        continue
    subjects = rng.choice(qualifying, size=min(20, len(qualifying)), replace=False)
    item_cache = {}
    for subject_id in sorted(subjects):
        s_idx, s_val = subject_vector(subject_record(tables.subjects.loc[subject_id]))
        group = resp[resp["subject_id"] == subject_id]
        group = group.sample(n=min(60, len(group)), random_state=int(rng.integers(2**31)))
        for row in group.itertuples(index=False):
            if row.item_id not in item_cache:
                stored = tables.items.loc[row.item_id]
                item_cache[row.item_id] = item_vector(item_record(
                    {"content": stored["content"], "item_features": stored["item_features"],
                     "interactors": row.interactors}, benchmark))
            i_idx, i_val = item_cache[row.item_id]
            rows_idx.append(np.concatenate([i_idx, s_idx]))
            rows_val.append(np.concatenate([i_val, s_val]))
            labels.append(row.label)
            pair_ids.append(pid)
            held.append(zlib.crc32(f"{benchmark}|{row.item_id}".encode()) % 5 == 0)
            others = item_cnt[row.item_id] - 1
            collab.append((item_sum[row.item_id] - row.label) / others if others > 0 else np.nan)
        pid += 1
    print(f"  {benchmark}: {len(subjects)} pairs", flush=True)

indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows_idx])])
X = csr_matrix((np.concatenate(rows_val), np.concatenate(rows_idx), indptr), shape=(len(labels), DIM))
y, pids, held, collab = np.array(labels, float), np.array(pair_ids), np.array(held), np.array(collab)
print(f"\n{len(y):,} rows, {pid} pairs, {held.mean():.0%} rows on held-out items")


def within_pair_auc(scores, ys, groups):
    aucs = []
    frame = pd.DataFrame({"g": groups, "s": scores, "y": ys}).dropna()
    for _, g in frame.groupby("g"):
        pos, neg = (g.y == 1).sum(), (g.y == 0).sum()
        if pos >= 2 and neg >= 2:
            r = g.s.rank().to_numpy()
            aucs.append((r[g.y.to_numpy() == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return float(np.mean(aucs)), len(aucs)


def fit(matrix, ys, groups, c):
    _, local = np.unique(groups, return_inverse=True)
    onehot = csr_matrix((np.full(len(local), 10.0), (np.arange(len(local)), local)), shape=(len(local), local.max() + 1))
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(hstack([matrix, onehot]).tocsr(), ys)
    return model.coef_.ravel()[: matrix.shape[1]]


train = ~held
print("\nwithin-pair AUC (0.5 = no signal)          in-sample   held-out items")
for label, cols in (("all features", slice(None)), ("22 hand-made only", slice(0, N_DENSE))):
    Xs = X[:, cols]
    for c in (0.01, 0.1, 1.0, 10.0):
        coef = fit(Xs[train], y[train], pids[train], c)
        ins, _ = within_pair_auc(Xs[train] @ coef, y[train], pids[train])
        out, n = within_pair_auc(Xs[held] @ coef, y[held], pids[held])
        print(f"  {label:20s} C={c:<5}                  {ins:.3f}      {out:.3f}  ({n} pairs)")
    if label == "22 hand-made only":
        print("\n  hand-made feature weights (C=1):")
        coef = fit(Xs[train], y[train], pids[train], 1.0)
        for name, w in sorted(zip(DENSE, coef), key=lambda t: -abs(t[1])):
            print(f"    {name:18s} {w:+.3f}")
out, n = within_pair_auc(collab[held], y[held], pids[held])
print(f"\nreference: other subjects' mean on the same item, held-out rows: AUC {out:.3f} ({n} pairs)")
