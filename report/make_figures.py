"""Numbers and figures of the technical report, from the saved cross-validation runs.

    python report/make_figures.py

Each run directory is the output of ``scripts/run_eval.py`` (``runs/<timestamp>/pairs.csv``)
or of ``scripts/run_pooling_ablation.py --merge`` (``runs/ablation/pairs.csv``); the
defaults below are the runs quoted in the report. Paired differences are compared on
the same 1,720 (benchmark, system) pairs, with 95% intervals from a bootstrap that
resamples benchmarks, since pairs of one benchmark are correlated.
Needs matplotlib in addition to the project dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    # Q.G.'s prior variants, run from branch model/subject-features-enhanced (commit 39f93d2).
    "mix_prior_offset": ("20261030-131512", "mix_prior_offset", None),
    "zero_prior_offset": ("20261030-174959", "zero_prior_offset", None),
    "pooled_1": ("20261029-165020", "pooled_offset", "1"),
    "pooled_2": ("20261029-165020", "pooled_offset", "2"),
    "pooled_5": ("20261029-165020", "pooled_offset", "5"),
    "pooled_all": ("20261029-165020", "pooled_offset", "all"),
    "acq_fixed": ("20261030-093545", "prior_offset", "1"),
    "acq_random": ("20261030-093545", "prior_offset", "random"),
    "acq_uncertainty10": ("20261030-093545", "prior_offset", "uncertainty10"),
    "acq_uncertainty15": ("20261030-093545", "prior_offset", "uncertainty15"),
    # scripts/run_pooling_ablation.py: the same fitted models with shared terms switched off.
    "control_2": ("ablation", "ablate_all", "2"),
    **{f"tags_{pool}": ("ablation", "ablate_tags", pool) for pool in ("2", "5", "all")},
    **{f"level_{pool}": ("ablation", "ablate_tags_level", pool) for pool in ("2", "5", "all")},
    **{f"questions_{pool}": ("ablation", "ablate_tags_questions", pool) for pool in ("2", "5", "all")},
    **{f"calibrated_{pool}": ("ablation", "calibrated", pool) for pool in ("1", "2", "5", "all")},
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
    ("mix_prior_offset", "prior_offset"),
    ("zero_prior_offset", "prior_offset"),
    ("logreg_prior", "constant"),
    ("pooled_1", "prior_offset"),
    ("pooled_2", "pooled_1"),
    ("pooled_5", "pooled_1"),
    ("pooled_all", "pooled_1"),
    ("acq_random", "acq_fixed"),
    ("acq_uncertainty10", "acq_random"),
    ("acq_uncertainty15", "acq_random"),
    ("control_2", "pooled_2"),
    *[(f"{part}_{pool}", "pooled_1") for part in ("tags", "level", "questions") for pool in ("2", "5", "all")],
    *[(f"calibrated_{pool}", f"pooled_{pool}") for pool in ("1", "2", "5", "all")],
]

# Colours by what they identify, the same in every figure (checked for colour-blind
# separation against a white page). Neutrals are for reference marks and text only.
OWN_LABELS = ("#86b6ef", "#256abf")  # own-label models, weaker to stronger
OTHER_BENCHMARKS = "#1baf7a"
OTHER_SYSTEMS = ("#ef9169", "#eb6834", "#a93a12")  # LfHL with M = 2, 5, up to 20
LEVEL, QUESTIONS = "#6250d6", "#008300"
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

CURVES = [
    ("constant", "constant 0.5", MUTED, "v"),
    ("empirical_mean", "empirical mean (organizers)", SECONDARY, "s"),
    ("shrunk_mean", "Beta-binomial", OWN_LABELS[0], "o"),
    ("prior_offset", "own labels, tags and text prior", OWN_LABELS[1], "o"),
    ("pooled_2", "LfHL, $M=2$", OTHER_SYSTEMS[0], "o"),
    ("pooled_5", "LfHL, $M=5$", OTHER_SYSTEMS[1], "o"),
    ("pooled_all", "LfHL, $M\\leq 20$", OTHER_SYSTEMS[2], "o"),
]

# Each step adds to the one above: (label, run, colour).
SOURCES = [
    ("Calibrated update on own labels", "shrunk_mean", OWN_LABELS[1]),
    ("+ tag shifts", "tag_offset", OWN_LABELS[1]),
    ("+ text prior", "prior_offset", OTHER_BENCHMARKS),
    ("+ one other system ($M=2$)", "pooled_2", OTHER_SYSTEMS[1]),
    ("+ more systems ($M=5$)", "pooled_5", OTHER_SYSTEMS[1]),
    ("+ more systems ($M\\leq 20$)", "pooled_all", OTHER_SYSTEMS[1]),
]

# Groups of negative results: (label, comparison, reference).
NEGATIVE = [
    [("Logistic text prior alone, no update", "logreg_prior - constant", "constant 0.5"),
     ("LightGBM prior instead of logistic", "gbdt_offset - prior_offset", "+ text prior")],
    [("Nonparametric prior over pair accuracy", "histogram_mean - shrunk_mean", "Beta-binomial"),
     ("Mixture of widths $\\{0.5, 1, 2\\}\\,\\tau$", "mix_prior_offset - prior_offset", "+ text prior")],
    [("Uncertainty acquisition, $|p-0.5|\\leq 0.10$", "acq_uncertainty10 - acq_random", "streamed random"),
     ("Uncertainty acquisition, $|p-0.5|\\leq 0.15$", "acq_uncertainty15 - acq_random", "streamed random")],
]

# Pooling ablation: (label, run prefix, colour, marker).
ABLATION = [
    ("Tags only", "tags", MUTED, "s"),
    ("Tags + level", "level", LEVEL, "o"),
    ("Tags + question shifts", "questions", QUESTIONS, "^"),
    ("All three (LfHL)", "pooled", INK, "D"),
]

# Formative Codabench scores of our own uploads (submissions page). Every upload with a
# score is listed, including two left in status "Scoring"; five uploads that finished
# without a score cannot be included.
UPLOADS = {
    "prior offset": [0.193516, 0.165583, 0.200787, 0.194646],
    "LfHL": [0.189107, 0.178399, 0.165440, 0.193250],
    "LfHL with start 0.35": [
        0.153152, 0.147621, 0.172424, 0.185839, 0.169210, 0.182736, 0.176607, 0.176266, 0.191711, 0.166216,
        0.182580, 0.194638, 0.222347, 0.174210, 0.195390, 0.195070, 0.176440, 0.178951, 0.204280, 0.202830,
        0.214894, 0.171094, 0.176671, 0.182647, 0.171399, 0.190614, 0.193975, 0.193274, 0.173747, 0.201840,
        0.199011, 0.187301, 0.168214, 0.213478, 0.200819, 0.140015, 0.186713, 0.198403, 0.165686, 0.179899,
        0.182864, 0.154155, 0.152993, 0.211372, 0.171697, 0.193064, 0.214446,
    ],
}
# Embed TrueType fonts (no Type 3) and match the report's Times text; recessive axes.
STYLE = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
         "text.color": INK, "axes.labelcolor": SECONDARY, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
         "axes.spines.top": False, "axes.spines.right": False, "xtick.color": AXIS, "ytick.color": AXIS,
         "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY, "xtick.major.width": 0.6,
         "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False,
         "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"}


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


def upload_summary() -> dict:
    """Mean, spread and range of repeated uploads of one unchanged archive."""
    return {name: {"n": len(scores), "mean": round(float(np.mean(scores)), 4),
                   "sd": round(float(np.std(scores, ddof=1)), 4), "min": round(float(np.min(scores)), 4),
                   "max": round(float(np.max(scores)), 4)}
            for name, scores in UPLOADS.items()}


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)
    return plt


def signed(value: float) -> str:
    """Four decimals without the leading zero, as in the report's tables: -.0064, +.0014."""
    return ("−" if value < 0 else "+") + f"{abs(value):.4f}"[1:]


def bare(value: float) -> str:
    return f"{value:.4f}"[1:]


def grid(ax, axis: str = "both"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.5)


def learning_curves(curves: dict, oracle: float, path: Path):
    plt = pyplot()
    x = np.log2(np.asarray(BUDGETS) + 1.0)
    fig, ax = plt.subplots(figsize=(5.5, 2.75))
    for label, name, color, marker in CURVES:
        ax.plot(x, [curves[label][b].mean() for b in BUDGETS], color=color, marker=marker, markersize=3.8,
                markeredgecolor="white", markeredgewidth=0.5, linewidth=1.3, label=name)
    ax.axhline(oracle, color=SECONDARY, linewidth=0.8)
    ax.text(x[0], oracle - 0.0012, f"oracle pair accuracy ({oracle:.3f})", color=SECONDARY, fontsize=7.5, va="top")
    ax.set_xticks(x, [str(b) for b in BUDGETS])
    ax.set_xlabel("revealed labels per system (budget)")
    ax.set_ylabel("mean Brier score")
    ax.set_ylim(0.168, 0.262)
    grid(ax, "y")
    ax.text(x[2] + 0.06, 0.2595, f"empirical mean at budget 1: {curves['empirical_mean'][1].mean():.3f} (off scale)",
            color=SECONDARY, fontsize=7.5, va="top")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.47, -0.2), columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sources_figure(curves: dict, oracle: float, path: Path):
    """Waterfall of ALC: what each information source adds, in the order of Table 1."""
    plt = pyplot()
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(5.5, 1.85))
    y = len(SOURCES)
    # Rounded as in Table 1, so that the printed steps add up to the printed totals.
    previous = round(float(curves["constant"]["alc"].mean()), 4)
    ax.plot([previous, previous], [y - 0.31, y + 0.31], color=INK, linewidth=1.4)
    ax.text(previous - 0.0007, y, f"{bare(previous)}", va="center", ha="left", fontsize=7.5)
    # Parenthesised ALC only where a source ends, so the labels stay sparse.
    totals = {"prior_offset", "pooled_all"}
    for offset, (_, run, color) in enumerate(SOURCES, 1):
        row = y - offset
        current = round(float(curves[run]["alc"].mean()), 4)
        ax.barh(row, previous - current, left=current, height=0.62, color=color, linewidth=0)
        ax.plot([previous, previous], [row + 1 - 0.31, row + 0.31], color=AXIS, linewidth=0.6)
        text = signed(current - previous) + (f" ({bare(current)})" if run in totals else "")
        ax.text(current - 0.0007, row, text, va="center", ha="left", fontsize=7.5, color=INK)
        previous = current
    ax.axvline(oracle, color=SECONDARY, linewidth=0.8, zorder=1)
    ax.text(oracle + 0.0007, y, f"oracle pair accuracy ({bare(oracle)})", color=SECONDARY, fontsize=7.5,
            va="center", ha="right")
    ax.set_yticks([y, *[y - i for i in range(1, y + 1)]], ["Constant 0.5", *[name for name, _, _ in SOURCES]],
                  color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.2515, 0.1715)
    ax.set_ylim(-0.6, y + 0.6)
    ax.set_xticks(np.arange(0.18, 0.2501, 0.01), [f"{v:.2f}" for v in np.arange(0.18, 0.2501, 0.01)])
    ax.set_xlabel("ALC on 1,720 held-out pairs (lower is better)")
    grid(ax, "x")
    ax.legend(handles=[Patch(color=OWN_LABELS[1], label="A system's own labels"),
                       Patch(color=OTHER_BENCHMARKS, label="Knowledge from other benchmarks"),
                       Patch(color=OTHER_SYSTEMS[1], label="Other systems' labels")],
              ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 0.99), columnspacing=1.2, handlelength=1.1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def negative_figure(table: dict, deltas: dict, path: Path):
    """Forest plot of the negative results, with the exact numbers beside each interval."""
    plt = pyplot()
    from matplotlib.transforms import blended_transform_factory

    fig = plt.figure(figsize=(5.5, 1.62))
    ax = fig.add_axes([0.335, 0.235, 0.2, 0.63])
    across = blended_transform_factory(ax.transAxes, ax.transData)
    columns = {"ALC": 1.05, "Δ ALC [95% CI]": 1.36, "Reference": 2.35, "Folds": 3.13}

    def column(name: str, y: float, text: str, **style):
        ax.text(columns[name], y, text, transform=across, va="center", ha="left", clip_on=False, fontsize=7.5, **style)

    y, labels, positions = 0.0, [], []
    for group in reversed(NEGATIVE):
        for name, key, reference in reversed(group):
            delta = deltas[key]
            ax.plot(delta["ci95"], [y, y], color=SECONDARY, linewidth=1.0)
            ax.plot(delta["mean"], y, marker="o", markersize=4.2, color=INK, markeredgecolor="white", markeredgewidth=0.5)
            column("ALC", y, bare(table[key.split(" - ")[0]]["ALC"]))
            column("Δ ALC [95% CI]", y, f"{signed(delta['mean'])} [{signed(delta['ci95'][0])}, {signed(delta['ci95'][1])}]")
            column("Reference", y, reference, color=SECONDARY)
            column("Folds", y, str(delta["folds_better"]))
            labels.append(name)
            positions.append(y)
            y += 1.0
        y += 0.45
    top = y - 0.45
    for name in columns:
        column(name, top + 0.1, name, color=SECONDARY)
    ax.axvline(0, color=SECONDARY, linewidth=0.8, zorder=1)
    ax.set_yticks(positions, labels, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, top + 0.5)
    ax.set_xlim(-0.0075, 0.0105)
    ax.set_xticks([-0.005, 0, 0.005], ["−.005", "0", "+.005"])
    ax.set_xlabel("Δ ALC (negative is better)")
    grid(ax, "x")
    fig.savefig(path)
    plt.close(fig)


def ablation_figure(curves: dict, deltas: dict, path: Path):
    """(a) ALC gain by number of systems, with intervals; (b) Brier gain by budget at M <= 20."""
    plt = pyplot()
    from matplotlib.lines import Line2D

    fig, (left, right) = plt.subplots(1, 2, figsize=(5.5, 2.2))
    pools = ["2", "5", "all"]
    for index, (name, part, color, marker) in enumerate(ABLATION):
        # A gain is a negated difference, so its interval runs from -high to -low.
        gains, below, above = [], [], []
        for pool in pools:
            delta = deltas[f"{part}_{pool} - pooled_1"]
            gains.append(-delta["mean"])
            below.append(delta["ci95"][1] - delta["mean"])
            above.append(delta["mean"] - delta["ci95"][0])
        x = np.arange(len(pools)) + (index - 1.5) * 0.07
        left.errorbar(x, gains, yerr=[below, above], color=color, marker=marker, markersize=3.8,
                      markeredgecolor="white", markeredgewidth=0.5, linewidth=1.2, elinewidth=0.8, capsize=0, label=name)
        budgets = [b for b in BUDGETS if b > 0]
        base = curves["pooled_1"]
        series = [float(base[b].mean() - curves[f"{part}_all"][b].mean()) for b in budgets]
        right.plot(np.log2(np.asarray(budgets) + 1.0), series, color=color, marker=marker, markersize=3.8,
                   markeredgecolor="white", markeredgewidth=0.5, linewidth=1.2)
    left.set_xticks(range(len(pools)), ["2", "5", "$\\leq$ 20"])
    left.set_xlabel("systems sharing a benchmark ($M$)")
    left.set_ylabel("reduction in ALC")
    left.set_title("(a) Against one system, by $M$", loc="left", fontsize=8, color=INK)
    budgets = [b for b in BUDGETS if b > 0]
    right.set_xticks(np.log2(np.asarray(budgets) + 1.0), [str(b) for b in budgets])
    right.set_xlabel("revealed labels per system (budget)")
    right.set_ylabel("reduction in Brier score")
    right.set_title("(b) Against one system, by budget, $M \\leq 20$", loc="left", fontsize=8, color=INK)
    for ax in (left, right):
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.3f}"[1:] if value else "0")
        grid(ax, "y")
    handles = [Line2D([], [], color=color, marker=marker, markersize=3.8, markeredgecolor="white",
                      markeredgewidth=0.5, linewidth=1.2, label=name) for name, _, color, marker in ABLATION]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07), columnspacing=1.4, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def uploads_figure(path: Path):
    """Every scored upload of three unchanged archives, with mean, spread and range beside it."""
    plt = pyplot()
    from matplotlib.transforms import blended_transform_factory

    fig = plt.figure(figsize=(5.5, 1.2))
    ax = fig.add_axes([0.2, 0.31, 0.42, 0.67])
    across = blended_transform_factory(ax.transAxes, ax.transData)
    rows = [("LfHL with start 0.35", "LfHL with start 0.35", MUTED), ("LfHL (selected)", "LfHL", OTHER_SYSTEMS[1]),
            ("Prior offset", "prior offset", MUTED)]
    rng = np.random.default_rng(0)
    for y, (_, key, color) in enumerate(rows):
        scores = np.asarray(UPLOADS[key])
        ax.scatter(scores, y + rng.uniform(-0.2, 0.2, scores.size), s=13, color=color, edgecolors="white",
                   linewidths=0.4, zorder=3)
        ax.plot([scores.mean()] * 2, [y - 0.36, y + 0.36], color=INK, linewidth=1.4, zorder=4)
        ax.text(1.03, y, f"$n$ = {scores.size}, mean {bare(scores.mean())}, SD {bare(scores.std(ddof=1))}, "
                         f"{bare(scores.min())}–{bare(scores.max())}",
                transform=across, va="center", ha="left", fontsize=7.5, clip_on=False)
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(0.135, 0.227)
    ax.set_xticks([0.14, 0.16, 0.18, 0.20, 0.22], [".14", ".16", ".18", ".20", ".22"])
    ax.set_xlabel("formative Brier ALC of one upload")
    grid(ax, "x")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
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
        "uploads": upload_summary(),
    }
    # The figures sit next to the report's source, so the folder compiles as it is (also on Overleaf).
    learning_curves(curves, oracle, args.out / "learning_curves.pdf")
    sources_figure(curves, oracle, args.out / "sources.pdf")
    negative_figure(numbers["table"], numbers["comparisons"], args.out / "negative_results.pdf")
    ablation_figure(curves, numbers["comparisons"], args.out / "ablation.pdf")
    uploads_figure(args.out / "uploads.pdf")
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    (args.out / "results" / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
