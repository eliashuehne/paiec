"""Experiment: logistic prior trained on within-pair differences only (fold 0).

Each training pair gets its own (barely regularized) intercept, so the shared
feature weights can only learn which items are harder/easier than their pair's
average. Regularization is chosen on held-out *training* benchmarks by the
within-pair metric (Brier after the best per-pair shift). Then the same
held-out-fold diagnostic as before: within-pair AUC and Brier with a perfect
per-pair level, vs the best constant per pair.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from paiec.features import DIM  # noqa: E402
from paiec.folds import benchmark_folds  # noqa: E402
from paiec.pairs import choose_pairs  # noqa: E402
from paiec.prior import LogisticPrior, training_matrix  # noqa: E402
from paiec.schema import load_benchmark, training_benchmarks  # noqa: E402
from paiec.split import make_split  # noqa: E402

DATA = ROOT / "payload" / "data"
PAIR_SCALE = 10.0  # pair intercept columns scaled up -> their L2 penalty is ~100x weaker
GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
started = time.time()


def pair_ids(weights):
    """benchmark_rows writes each pair's rows contiguously, each with weight 1/size."""
    ids = np.empty(len(weights), dtype=np.int64)
    start, pid = 0, 0
    while start < len(weights):
        size = int(round(1.0 / weights[start]))
        ids[start:start + size] = pid
        start, pid = start + size, pid + 1
    return ids


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def within_pair_scores(logits, labels, pids):
    """(Brier with best per-pair shift, Brier of per-pair mean, mean within-pair AUC); pairs weighted equally."""
    uniq, pids = np.unique(pids, return_inverse=True)
    n = np.bincount(pids).astype(float)
    shift = np.zeros(len(uniq))
    for _ in range(30):
        p = sigmoid(logits + shift[pids])
        shift += np.bincount(pids, labels - p) / (np.bincount(pids, p * (1 - p)) + 1e-6)
        shift = np.clip(shift, -15, 15)
    p = sigmoid(logits + shift[pids])
    shifted = np.mean(np.bincount(pids, (p - labels) ** 2) / n)
    mean = np.bincount(pids, labels) / n
    constant = np.mean(np.bincount(pids, (mean[pids] - labels) ** 2) / n)
    aucs = []
    frame = pd.DataFrame({"pid": pids, "logit": logits, "y": labels})
    for _, g in frame.groupby("pid"):
        pos, neg = (g.y == 1).sum(), (g.y == 0).sum()
        if pos and neg:
            ranks = g.logit.rank().to_numpy()
            aucs.append((ranks[g.y.to_numpy() == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return shifted, constant, float(np.mean(aucs))


def fit_fixed_effects(matrix, labels, weights, pids, regularization):
    _, local = np.unique(pids, return_inverse=True)
    onehot = csr_matrix((np.full(len(local), PAIR_SCALE), (np.arange(len(local)), local)),
                        shape=(len(local), local.max() + 1))
    model = LogisticRegression(C=regularization, max_iter=500)
    model.fit(hstack([matrix, onehot]).tocsr(), labels, sample_weight=weights / weights.mean())
    return model.coef_.ravel()[:DIM]


train, test = benchmark_folds(training_benchmarks(DATA), k=5, seed=0)[0]
matrix, labels, weights, groups = training_matrix(DATA, train)
pids = pair_ids(weights)
assert np.allclose(np.bincount(pids, weights), 1.0)
print(f"training rows {len(labels):,}, pairs {pids.max() + 1:,} ({time.time() - started:.0f}s)", flush=True)

inner_train, inner_val = benchmark_folds(sorted(set(groups)), k=5, seed=1)[0]
fit_rows, val_rows = np.isin(groups, inner_train), np.isin(groups, inner_val)
results = []
for c in GRID:
    coef = fit_fixed_effects(matrix[fit_rows], labels[fit_rows], weights[fit_rows], pids[fit_rows], c)
    scores = within_pair_scores(matrix[val_rows] @ coef, labels[val_rows], pids[val_rows])
    results.append(scores)
    print(f"  C={c}: inner-val shifted Brier {scores[0]:.4f} vs constant {scores[1]:.4f}, AUC {scores[2]:.3f}"
          f" ({time.time() - started:.0f}s)", flush=True)
best = GRID[int(np.argmin([r[0] for r in results]))]
coef = fit_fixed_effects(matrix, labels, weights, pids, best)
print(f"chosen C={best}; refit on all training benchmarks ({time.time() - started:.0f}s)", flush=True)

prior = LogisticPrior()
prior.coef = coef
logits, ys, held_pids, bench = [], [], [], []
pid = 0
for benchmark in test:
    tables = load_benchmark(DATA, benchmark)
    for pair in choose_pairs(tables, max_subjects=20):
        split = make_split(pair.item_ids, benchmark=pair.benchmark, subject_id=pair.subject_id)
        evaluation = list(split.evaluation)[:200]
        resp = pair.responses[pair.responses["item_id"].isin(set(evaluation))]
        by_item = {i: prior.logit(pair.subject, pair.item(i)) for i in evaluation}
        logits.append(resp["item_id"].map(by_item).to_numpy())
        ys.append(resp["label"].to_numpy().astype(float))
        held_pids.append(np.full(len(resp), pid))
        bench += [benchmark] * len(resp)
        pid += 1
logits, ys, held_pids = np.concatenate(logits), np.concatenate(ys), np.concatenate(held_pids)
shifted, constant, auc = within_pair_scores(logits, ys, held_pids)
print(f"\nHELD-OUT fold 0 ({pid} pairs): within-pair AUC {auc:.3f} (v1 prior: 0.526)")
print(f"  Brier, best constant per pair      {constant:.4f}")
print(f"  Brier, prior + perfect pair level  {shifted:.4f}   (v1 prior: 0.1813 vs 0.1817)")
per = pd.DataFrame({"b": bench, "pid": held_pids, "logit": logits, "y": ys})
rows = []
for b, g in per.groupby("b"):
    s, c, a = within_pair_scores(g.logit.to_numpy(), g.y.to_numpy(), g.pid.to_numpy())
    rows.append((b, a, c - s))
by_bench = pd.DataFrame(rows, columns=["benchmark", "auc", "brier_gain"]).sort_values("auc")
print("\nper benchmark:\n" + by_bench.round(3).to_string(index=False))
