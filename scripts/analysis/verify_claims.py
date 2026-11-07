"""Check the data facts of Section 2 of the report (run as: python scripts/analysis/verify_claims.py .)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from paiec.schema import canonical_identity, clean, training_benchmarks  # noqa: E402

DATA = ROOT / "payload" / "data"
benches = training_benchmarks(DATA)
resp, subs = [], []
for b in benches:
    d = DATA / b / "formatted_tables"
    r = pd.read_parquet(d / "responses.parquet", columns=["subject_id", "item_id", "trial", "response"])
    r["benchmark"] = b
    resp.append(r)
    s = pd.read_parquet(d / "subjects.parquet")
    s["benchmark"] = b
    subs.append(s)
r = pd.concat(resp, ignore_index=True)
s = pd.concat(subs, ignore_index=True)
for c in ["subject_id", "item_id", "benchmark"]:
    r[c] = r[c].astype("category")
print(f"{len(benches)} public eligible benchmarks; {len(r):,} response rows; NaN responses {r.response.isna().mean():.2%}")
print("distinct raw response values:", sorted(r.response.dropna().unique().tolist())[:10])
r = r.dropna(subset=["response"])  # in memory only, for statistics

# qualifying pairs: >= 80 distinct items with a label; each trial is its own row
g = r.groupby(["benchmark", "subject_id"], observed=True)
pairs = g.agg(n_items=("item_id", "nunique"), acc=("response", "mean"), n_rows=("response", "size")).reset_index()
q = pairs[pairs.n_items >= 80].copy()
print(f"qualifying pairs {len(q):,} on {q.benchmark.nunique()} benchmarks")

print("\n=== CLAIM 1: subject reappearance ===")
bps = r.groupby("subject_id", observed=True).benchmark.nunique()
print(f"subject_ids (with responses): {len(bps):,}; in exactly 1 benchmark: {(bps == 1).mean():.1%}")
qb = q.groupby("subject_id", observed=True).benchmark.nunique()
print(f"subject_ids among qualifying pairs: {len(qb):,}; in exactly 1 benchmark: {(qb == 1).mean():.1%}")
s1 = s.drop_duplicates(["benchmark", "subject_id"])
for f in ["normalized_name", "display_name"]:
    print(f"{f} filled (subjects table rows): {s1[f].map(clean).ne('').mean():.1%}")
s1 = s1.assign(identity=[canonical_identity(clean(a) or clean(b)) for a, b in zip(s1.normalized_name, s1.display_name)])
qi = q.merge(s1[["benchmark", "subject_id", "identity", "normalized_name", "release_date", "reasoning_effort",
                 "provider", "harness"]], left_on=["benchmark", "subject_id"], right_on=["benchmark", "subject_id"], how="left")
print(f"qualifying pairs with normalized_name filled: {qi.normalized_name.map(clean).ne('').mean():.1%}")
idb = pairs.merge(s1[["benchmark", "subject_id", "identity"]], on=["benchmark", "subject_id"], how="left")
idb = idb[idb.identity.fillna("") != ""]
bench_per_id = idb.groupby("identity").benchmark.nunique()
qi["id_other_benchmarks"] = qi.identity.map(bench_per_id).fillna(1) - 1
print(f"qualifying pairs whose canonical identity has data on >=1 OTHER benchmark: {(qi.id_other_benchmarks >= 1).mean():.1%}"
      f"  (>=3 others: {(qi.id_other_benchmarks >= 3).mean():.1%})")

print("\n=== CLAIM 2: release date / reasoning effort ===")
for f in ["release_date", "reasoning_effort", "provider", "harness"]:
    print(f"{f:17s} filled: subjects {s1[f].map(clean).ne('').mean():.1%}   qualifying pairs {qi[f].map(clean).ne('').mean():.1%}")
qi["year"] = pd.to_datetime(qi.release_date.map(clean).replace("", np.nan), errors="coerce").dt.year + \
    pd.to_datetime(qi.release_date.map(clean).replace("", np.nan), errors="coerce").dt.dayofyear / 365
dated = qi.dropna(subset=["year"])
dated = dated[dated.groupby("benchmark").benchmark.transform("size") >= 3]
dm = lambda x: x - dated.groupby("benchmark")[x.name].transform("mean")
corr = np.corrcoef(dm(dated.year), dm(dated.acc))[0, 1]
print(f"within-benchmark corr(release year, pair accuracy): {corr:.2f} over {len(dated)} dated pairs on {dated.benchmark.nunique()} benchmarks")

print("\n=== CLAIM 3: pair-level variation and tau ===")
print("pair accuracy:", q.acc.describe(percentiles=[.1, .5, .9]).round(3).to_dict())
lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
glob = q.acc.mean()
print(f"logit sd of pair acc around global mean: {np.std(lg(q.acc) - lg(glob)):.2f}")
bm = q.groupby("benchmark").acc.transform("mean")
print(f"logit sd around ORACLE benchmark mean (unknowable at test time): {np.std(lg(q.acc) - lg(bm)):.2f}")
# feasible prior: identity's mean accuracy on OTHER benchmarks
tot = idb.assign(k=idb.acc * idb.n_rows).groupby(["identity", "benchmark"])[["k", "n_rows"]].sum().reset_index()
idsum = tot.groupby("identity")[["k", "n_rows"]].sum()
own = tot.set_index(["identity", "benchmark"])
qi2 = qi[qi.id_other_benchmarks >= 1].copy()
key = list(zip(qi2.identity, qi2.benchmark))
kk = idsum.loc[qi2.identity, "k"].values - own.loc[key, "k"].values
nn = idsum.loc[qi2.identity, "n_rows"].values - own.loc[key, "n_rows"].values
qi2["prior"] = kk / nn
print(f"logit sd around identity-on-other-benchmarks prior ({len(qi2)} pairs): {np.std(lg(qi2.acc) - lg(qi2.prior)):.2f}"
      f"  corr(prior, acc) = {np.corrcoef(qi2.prior, qi2.acc)[0, 1]:.2f}")

# pair-weighted Brier with pair-level-only predictors (every trial row is a target)
d = r.merge(q[["benchmark", "subject_id", "acc"]], on=["benchmark", "subject_id"])
def pb(p):
    e = (np.clip(p, 0, 1) - d.response.values) ** 2
    return pd.Series(e).groupby([d.benchmark.values, d.subject_id.values], observed=True).mean().mean()
print("\npair-weighted Brier, predictors that know only pair-level info:")
print(f"  constant 0.5                         {pb(np.full(len(d), .5)):.4f}")
print(f"  global mean {glob:.2f}                    {pb(np.full(len(d), glob)):.4f}")
print(f"  ORACLE benchmark mean                {pb(d.groupby('benchmark', observed=True).response.transform('mean').values):.4f}")
print(f"  ORACLE pair mean (ceiling of offset) {pb(d.acc.values):.4f}")
print(f"  ORACLE item mean across subjects     {pb(d.groupby('item_id', observed=True).response.transform('mean').values):.4f}")
