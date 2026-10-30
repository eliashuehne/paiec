"""Numbers and figures of the technical report, from the saved cross-validation runs.

    python report/make_figures.py                       # tables, bootstrap CIs, learning curves
    python report/make_figures.py --formative PAIRS.csv # plus the formative-pool figure

Each run directory is the output of ``scripts/run_eval.py`` (``runs/<timestamp>/pairs.csv``);
the defaults below are the runs quoted in the report. Paired differences are
compared on the same 1,720 (benchmark, system) pairs, with 95% intervals from a
bootstrap that resamples benchmarks, since pairs of one benchmark are correlated.

``--formative`` takes one row per formative pair draw with the recovered pair
accuracy ``a`` (see Section 6 of the report); it is not redistributed. The
public pair accuracies for that figure are recomputed from ``payload/data``.
Needs matplotlib and scipy in addition to the project dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paiec.metrics import BUDGETS, alc_weights  # noqa: E402

# label: (run directory, predictor, pool) — pool None for runs made before pooling existed.
RUNS = {
    "constant": ("20261022-191510", "constant", None),
    "empirical_mean": ("20261022-191510", "empirical_mean", None),
    "shrunk_mean": ("20261024-150158", "shrunk_mean", None),
    "logreg_prior": ("20261024-175123", "logreg_prior", None),
    "offset": ("20261024-182446", "offset", None),
    "tag_offset": ("20261024-192249", "tag_offset", None),
    "prior_offset": ("20261025-201534", "prior_offset", None),
    "gbdt_offset": ("20261025-213952", "gbdt_offset", None),
    "histogram_mean": ("20261025-221617", "histogram_mean", None),
    "pooled_1": ("20261029-165020", "pooled_offset", "1"),
    "pooled_2": ("20261029-165020", "pooled_offset", "2"),
    "pooled_5": ("20261029-165020", "pooled_offset", "5"),
    "pooled_all": ("20261029-165020", "pooled_offset", "all"),
    "acq_fixed": ("20261030-093545", "prior_offset", "1"),
    "acq_random": ("20261030-093545", "prior_offset", "random"),
    "acq_uncertainty10": ("20261030-093545", "prior_offset", "uncertainty10"),
    "acq_uncertainty15": ("20261030-093545", "prior_offset", "uncertainty15"),
}

# (candidate, reference): candidate minus reference, negative = candidate better.
COMPARISONS = [
    ("empirical_mean", "constant"),
    ("shrunk_mean", "empirical_mean"),
    ("offset", "shrunk_mean"),
    ("histogram_mean", "shrunk_mean"),
    ("tag_offset", "offset"),
    ("prior_offset", "tag_offset"),
    ("gbdt_offset", "prior_offset"),
    ("logreg_prior", "constant"),
    ("pooled_1", "prior_offset"),
    ("pooled_2", "pooled_1"),
    ("pooled_5", "pooled_1"),
    ("pooled_all", "pooled_1"),
    ("acq_random", "acq_fixed"),
    ("acq_uncertainty10", "acq_random"),
    ("acq_uncertainty15", "acq_random"),
]

CURVES = [
    ("constant", "constant 0.5", "0.6", ":"),
    ("empirical_mean", "empirical mean (organizers)", "tab:gray", "--"),
    ("shrunk_mean", "shrunk mean", "tab:brown", "-"),
    ("prior_offset", "prior offset (own labels)", "tab:blue", "-"),
    ("pooled_2", "LfHL, $M=2$", "tab:olive", "-"),
    ("pooled_5", "LfHL, $M=5$", "tab:orange", "-"),
    ("pooled_all", "LfHL, $M\\leq 20$", "tab:red", "-"),
]

# Formative Codabench scores of our own uploads (submissions page, 2026-10-27 to 10-29).
UPLOADS = {
    "prior offset\n(n = 4)": [0.193516, 0.165583, 0.200787, 0.194646],
    "LfHL\n(n = 4)": [0.189107, 0.178399, 0.165440, 0.193250],
    "LfHL-cal\n(n = 20)": [0.153152, 0.147621, 0.172424, 0.185839, 0.169210, 0.182736,
                                    0.176607, 0.176266, 0.191711, 0.166216, 0.182580, 0.194638,
                                    0.174210, 0.195390, 0.195070, 0.176440, 0.178951, 0.204280,
                                    0.202830, 0.171094],
}
TAU = 1.584893192461114
# Embed TrueType fonts (no Type 3) and match the report's Times text.
STYLE = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix"}


def load(runs_dir: Path, label: str) -> pd.DataFrame:
    """Per-pair Brier curve, ALC and fold of one predictor in one run."""
    directory, predictor, pool = RUNS[label]
    frame = pd.read_csv(runs_dir / directory / "pairs.csv", dtype={"pool": str})
    frame = frame[frame["predictor"] == predictor]
    if pool is not None:
        frame = frame[frame["pool"] == pool]
    curves = frame.pivot_table(index=["benchmark", "subject_id"], columns="budget", values="brier")
    curves = curves[list(BUDGETS)]
    curves["alc"] = curves[list(BUDGETS)].to_numpy() @ alc_weights(BUDGETS)
    curves["fold"] = frame.groupby(["benchmark", "subject_id"])["fold"].first()
    curves["ece0"] = frame[frame["budget"] == 0].set_index(["benchmark", "subject_id"])["ece"]
    return curves


def cluster_bootstrap(diff: pd.Series, reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean pair difference, resampling benchmarks."""
    by_benchmark = diff.groupby(level="benchmark").agg(["sum", "size"])
    sums, sizes = by_benchmark["sum"].to_numpy(), by_benchmark["size"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(reps, len(sums)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summary_table(curves: dict) -> dict:
    table = {}
    for label, frame in curves.items():
        folds = frame.groupby("fold")["alc"].mean()
        table[label] = {
            **{f"B{b}": round(float(frame[b].mean()), 4) for b in BUDGETS},
            "ALC": round(float(frame["alc"].mean()), 4),
            "folds": [round(float(v), 4) for v in folds],
            "pairs": int(len(frame)),
        }
    return table


def comparisons(curves: dict) -> dict:
    out = {}
    for candidate, reference in COMPARISONS:
        a, b = curves[candidate], curves[reference]
        common = a.index.intersection(b.index)
        diff = a.loc[common, "alc"] - b.loc[common, "alc"]
        folds = (a.loc[common].groupby("fold")["alc"].mean() - b.loc[common].groupby("fold")["alc"].mean())
        low, high = cluster_bootstrap(diff)
        out[f"{candidate} - {reference}"] = {
            "mean": round(float(diff.mean()), 5), "ci95": [round(low, 5), round(high, 5)],
            "folds_better": int((folds < 0).sum()), "folds": int(len(folds)),
            "pairs_better": round(float((diff < -1e-12).mean()), 3),
            "pairs_worse": round(float((diff > 1e-12).mean()), 3),
            "pairs": int(len(common)),
        }
    return out


def learning_curves(curves: dict, oracle: float, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)
    x = np.log2(np.asarray(BUDGETS) + 1.0)
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for label, name, color, style in CURVES:
        ax.plot(x, [curves[label][b].mean() for b in BUDGETS], style, color=color, marker="o",
                markersize=3, linewidth=1.4, label=name)
    ax.axhline(oracle, color="black", linewidth=0.8, linestyle="-.", label="oracle pair accuracy")
    ax.set_xticks(x, [str(b) for b in BUDGETS])
    ax.set_xlabel("revealed labels per system (budget)")
    ax.set_ylabel("mean Brier score")
    ax.set_ylim(0.165, 0.262)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.annotate(f"empirical mean at budget 1: {curves['empirical_mean'][1].mean():.3f} (off scale)",
                xy=(2.45, 0.2595), fontsize=7, color="tab:gray", va="top")
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def expected_alc(a: np.ndarray, start_probability: float, tau: float = TAU) -> np.ndarray:
    """Expected ALC of the pair-level offset for pairs of true accuracy ``a`` (items exchangeable)."""
    from paiec.update import constant_prior_posterior, predictive

    level = np.log(start_probability / (1 - start_probability))
    total = np.zeros_like(a)
    for weight, budget in zip(alc_weights(BUDGETS), BUDGETS):
        correct = np.arange(budget + 1, dtype=float)
        mean, variance = constant_prior_posterior(level, correct, budget, tau)
        p = predictive(level, mean, variance)
        pmf = np.array([comb(budget, k) * a**k * (1 - a) ** (budget - k) for k in range(budget + 1)])
        total += weight * (pmf * (p[:, None] ** 2 - 2 * p[:, None] * a + a)).sum(axis=0)
    return total


def bayes_optimal_alc(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Expected ALC of the posterior mean under the empirical accuracy distribution.

    Each draw's prior is the accuracies of all draws of *other* pairs (leave one
    pair out), so no draw is predicted with its own accuracy.
    """
    total = np.zeros_like(a)
    others = groups[:, None] != groups[None, :]
    for weight, budget in zip(alc_weights(BUDGETS), BUDGETS):
        brier = np.zeros_like(a)
        for k in range(budget + 1):
            likelihood = a**k * (1 - a) ** (budget - k)
            evidence = others @ likelihood
            p = (others @ (likelihood * a)) / np.maximum(evidence, 1e-300)
            pmf = comb(budget, k) * likelihood
            brier += pmf * (p * p - 2 * p * a + a)
        total += weight * brier
    return total


def public_accuracies(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Evaluation-pool accuracy of every scoreable public pair, weighted as in the harness."""
    from paiec.episodes import training_episodes
    from paiec.schema import training_benchmarks

    episodes = training_episodes(str(data_root), training_benchmarks(data_root))
    return episodes["eval_mean"].to_numpy(), episodes["weight"].to_numpy()


def formative_figure(hidden: np.ndarray, groups: np.ndarray, public: tuple, path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)
    starts = np.round(np.arange(0.20, 0.601, 0.01), 2)
    hidden_alc = np.array([expected_alc(hidden, s).mean() for s in starts])
    public_alc = np.array([np.average(expected_alc(public[0], s), weights=public[1]) for s in starts])

    fig, (left, right) = plt.subplots(1, 2, figsize=(5.5, 2.4), gridspec_kw={"width_ratios": [1.15, 1]})
    left.plot(starts, hidden_alc, color="tab:red", label="formative pool")
    left.plot(starts, public_alc, color="tab:blue", label="public benchmarks")
    for s, style in ((0.5, ":"), (0.35, "--")):
        left.axvline(s, color="0.4", linestyle=style, linewidth=0.8)
    left.set_xlabel("starting probability")
    left.set_ylabel("expected ALC")
    left.legend(fontsize=7, frameon=False, loc="center left")
    left.grid(alpha=0.3, linewidth=0.5)

    rng = np.random.default_rng(0)
    for position, (name, scores) in enumerate(UPLOADS.items()):
        scores = np.asarray(scores)
        right.scatter(position + rng.uniform(-0.12, 0.12, scores.size), scores, s=12, color="tab:gray",
                      edgecolors="none")
        right.hlines(scores.mean(), position - 0.25, position + 0.25, color="black", linewidth=1.5)
    right.set_xticks(range(len(UPLOADS)), list(UPLOADS), fontsize=7)
    right.set_ylabel("formative Brier ALC")
    right.grid(alpha=0.3, linewidth=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    best = lambda values: float(starts[int(np.argmin(values))])  # noqa: E731
    return {
        "hidden_draws": int(hidden.size), "hidden_mean_accuracy": round(float(hidden.mean()), 4),
        "hidden_share_below_half": round(float((hidden < 0.5).mean()), 3),
        "public_pairs": int(public[0].size),
        "public_mean_accuracy": round(float(np.average(public[0], weights=public[1])), 4),
        "hidden_alc_at_0.50": round(float(hidden_alc[starts == 0.5][0]), 4),
        "hidden_alc_at_0.35": round(float(hidden_alc[starts == 0.35][0]), 4),
        "public_alc_at_0.50": round(float(public_alc[starts == 0.5][0]), 4),
        "public_alc_at_0.35": round(float(public_alc[starts == 0.35][0]), 4),
        "hidden_best_start": best(hidden_alc), "public_best_start": best(public_alc),
        "hidden_alc_at_best": round(float(hidden_alc.min()), 4),
        "public_alc_at_best": round(float(public_alc.min()), 4),
        "hidden_bayes_optimal": round(float(bayes_optimal_alc(hidden, groups).mean()), 4),
        "hidden_oracle": round(float(np.mean(hidden * (1 - hidden))), 4),
        "hidden_distinct_pairs": int(np.unique(groups).size),
        "uploads": {" ".join(name.split()): {"n": len(s), "mean": round(float(np.mean(s)), 4),
                                               "sd": round(float(np.std(s, ddof=1)), 4),
                                               "min": round(float(np.min(s)), 4)}
                    for name, s in UPLOADS.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--data", type=Path, default=ROOT / "payload" / "data")
    parser.add_argument("--formative", type=Path, help="CSV of formative pair draws with column 'a'")
    parser.add_argument("--out", type=Path, default=ROOT / "report")
    args = parser.parse_args()

    curves = {label: load(args.runs, label) for label in RUNS}
    # Constant 0.5 puts every prediction in one ECE bin, so ECE = |0.5 - accuracy| per pair.
    oracle = float((0.25 - curves["constant"]["ece0"] ** 2).mean())
    numbers = {
        "table": summary_table(curves),
        "comparisons": comparisons(curves),
        "oracle_pair_accuracy_brier": round(oracle, 4),
        "pairs": int(len(curves["constant"])),
        "benchmarks": int(curves["constant"].index.get_level_values("benchmark").nunique()),
    }
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    learning_curves(curves, oracle, args.out / "figures" / "learning_curves.pdf")
    if args.formative:
        draws = pd.read_csv(args.formative)
        groups = (draws["subject"] + "/" + draws["benchmark"]).to_numpy()
        numbers["formative"] = formative_figure(draws["a"].to_numpy(), groups, public_accuracies(args.data),
                                                args.out / "figures" / "formative.pdf")
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    (args.out / "results" / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
