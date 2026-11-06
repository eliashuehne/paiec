"""Does a question-level signal rank difficulty within unseen benchmarks?

Per signal: mean within-benchmark Spearman correlation with facility and the
share of benchmarks where it has the majority sign. Then leave-one-benchmark-out:
fit logit(facility) - benchmark mean ~ standardized signals on the other
benchmarks, and score the held-out benchmark's expected Brier for a random system
with the benchmark's true level known:
    baseline  p = benchmark mean facility (level oracle)
    model     p = sigmoid(logit(mean) + fitted shift)
    ceiling   p = the question's own facility
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

SLOPE = 0.5  # shrink the fitted slope; ×0.5 was better than ×1 for the 1.5B model
frame = pd.read_parquet(sys.argv[1])
signals = [s for s in ("rating", "entropy", "top_prob", "nll") if s in frame]
eps = 0.02
frame["logit"] = np.log((frame.facility.clip(eps, 1 - eps)) / (1 - frame.facility.clip(eps, 1 - eps)))
frame["centered"] = frame.logit - frame.groupby("benchmark").logit.transform("mean")
for s in signals:
    g = frame.groupby("benchmark")[s]
    frame[f"z_{s}"] = ((frame[s] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0)

print(f"{len(frame)} questions, {frame.benchmark.nunique()} benchmarks\n")
per_bench = frame.groupby("benchmark").apply(
    lambda b: pd.Series({s: spearmanr(b[s], b.facility).statistic for s in signals}))
for s in signals:
    sign = np.sign(per_bench[s].median())
    print(f"{s:9s} within-benchmark Spearman with facility: mean {per_bench[s].mean():+.3f}, "
          f"median {per_bench[s].median():+.3f}, majority sign in {(np.sign(per_bench[s]) == sign).mean():.0%} of benchmarks")


def brier(f, p):
    return float(np.mean(f * (1 - p) ** 2 + (1 - f) * p ** 2))


columns = [f"z_{s}" for s in signals]
rows = []
for held in frame.benchmark.unique():
    train, test = frame[frame.benchmark != held], frame[frame.benchmark == held]
    X = np.c_[train[columns].to_numpy(), np.ones(len(train))]
    coef = np.linalg.lstsq(X, train.centered.to_numpy(), rcond=None)[0][:-1] * SLOPE
    shift = test[columns].to_numpy() @ coef
    f = test.facility.to_numpy()
    # Same mean prediction as the baseline, so only the ranking is compared.
    level = brentq(lambda c: np.mean(1 / (1 + np.exp(-(c + shift)))) - f.mean(), -20, 20)
    rows.append({"benchmark": held,
                 "baseline": brier(f, np.full(len(f), f.mean())),
                 "model": brier(f, 1 / (1 + np.exp(-(level + shift)))),
                 "ceiling": brier(f, f)})
scores = pd.DataFrame(rows)
gain, room = scores.baseline - scores.model, scores.baseline - scores.ceiling
print(f"\nleave-one-benchmark-out, expected Brier for a random system (level known):")
print(f"  level only {scores.baseline.mean():.4f}   + signals {scores.model.mean():.4f}   "
      f"question oracle {scores.ceiling.mean():.4f}")
print(f"  gain {gain.mean():+.4f} of a possible {room.mean():.4f} ({gain.mean() / room.mean():.0%}); "
      f"better on {(gain > 0).mean():.0%} of benchmarks")
print("\nper benchmark:")
print(per_bench.round(2).join(scores.set_index("benchmark")[["baseline", "model"]].round(4)).to_string())
