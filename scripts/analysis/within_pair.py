"""Does the prior rank items correctly *within* a pair, once the pair's level is known?

Fold 0 only. For held-out pairs: Brier of the oracle pair mean (constant within pair)
vs the prior's logits with the best per-pair shift (fitted on the evaluation items
themselves = oracle offset). The gap is the most the item features can add on top of
a perfect offset update.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from paiec.folds import benchmark_folds  # noqa: E402
from paiec.pairs import choose_pairs  # noqa: E402
from paiec.prior import LogisticPrior  # noqa: E402
from paiec.schema import load_benchmark, training_benchmarks  # noqa: E402
from paiec.split import make_split  # noqa: E402

DATA = ROOT / "payload" / "data"
train, test = benchmark_folds(training_benchmarks(DATA), k=5, seed=0)[0]
prior = LogisticPrior().fit(DATA, train)
print("fitted C", prior.regularization)


def best_shift(logits, labels):
    shift = 0.0
    for _ in range(25):
        p = 1 / (1 + np.exp(-(logits + shift)))
        grad = np.sum(labels - p)
        hess = np.sum(p * (1 - p)) + 1e-6
        shift += grad / hess
        shift = float(np.clip(shift, -15, 15))
    return shift


rows = []
for benchmark in test:
    tables = load_benchmark(DATA, benchmark)
    for pair in choose_pairs(tables, max_subjects=20):
        split = make_split(pair.item_ids, benchmark=pair.benchmark, subject_id=pair.subject_id)
        evaluation = list(split.evaluation)[:200]
        resp = pair.responses[pair.responses["item_id"].isin(set(evaluation))]
        logit_by_item = {i: prior.logit(pair.subject, pair.item(i)) for i in evaluation}
        logits = resp["item_id"].map(logit_by_item).to_numpy()
        labels = resp["label"].to_numpy().astype(float)
        mean = labels.mean()
        shifted = 1 / (1 + np.exp(-(logits + best_shift(logits, labels))))
        order = np.argsort(logits)
        pos, neg = labels[order] == 1, labels[order] == 0
        auc = np.nan
        if pos.any() and neg.any():
            ranks = np.argsort(np.argsort(logits)) + 1
            auc = (ranks[labels == 1].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
        rows.append((benchmark, np.mean((mean - labels) ** 2), np.mean((shifted - labels) ** 2), auc,
                     np.mean((1 / (1 + np.exp(-logits)) - labels) ** 2)))

import pandas as pd  # noqa: E402

f = pd.DataFrame(rows, columns=["benchmark", "oracle_pair_mean", "prior_oracle_shift", "auc", "prior_raw"])
print(f"pairs {len(f)} on {f.benchmark.nunique()} benchmarks")
print(f[["prior_raw", "oracle_pair_mean", "prior_oracle_shift"]].mean().round(4).to_string())
print(f"mean within-pair AUC: {f.auc.mean():.3f}  (0.5 = no ranking signal)")
print("by benchmark (worst/best AUC):")
g = f.groupby("benchmark")[["auc", "oracle_pair_mean", "prior_oracle_shift"]].mean().sort_values("auc")
print(g.head(5).round(3).to_string())
print(g.tail(5).round(3).to_string())
