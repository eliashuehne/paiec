#!/usr/bin/env python3
"""Audit saved-head discrimination, diagnostic oracles and untuned evidence.

No model inference, gradient training, sealed-set access or deployment changes.
Oracles deliberately use development outcomes and must never become features.
"""

import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from paiec.competition import competition_input
from paiec.evidence import EvidencePredictor
from paiec.evaluate import evaluate
from paiec.head_training import file_hash
from paiec.pairs import iter_pairs
from paiec.schema import benchmark_dir, load_benchmark
from paiec.split import make_split


def signature(subject, item):
    return json.dumps(competition_input(subject, item), sort_keys=True, ensure_ascii=False)


def auc(group):
    labels, scores = [], []
    for row in group.itertuples():
        positives, total = int(row.n_outcomes_one), int(row.n_responses)
        labels.extend([1] * positives + [0] * (total - positives))
        scores.extend([row.probability] * total)
    y = np.asarray(labels)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if not n1 or not n0:
        return None
    ranks = rankdata(scores)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-output", default="runs/head_finetuning/compact_pilot_v3_modal_enriched/output")
    parser.add_argument("--out", default="runs/head_finetuning/compact_strategy_audit_v1")
    args = parser.parse_args()
    head, out = ROOT / args.head_output, ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    prediction_path = head / "enriched/best_predictions.csv"
    saved = pd.read_csv(prediction_path)
    zero = saved[saved.budget.eq(0)].copy()
    keys = ["benchmark", "subject_id"]
    n_pairs = len(zero[keys].drop_duplicates())
    zero["weight"] = zero.n_responses / zero.groupby(keys).n_responses.transform("sum") / n_pairs
    y = (zero.n_outcomes_one / zero.n_responses).to_numpy()
    p, w = zero.probability.to_numpy(), zero.weight.to_numpy()

    def score(probabilities):
        return float(np.sum(w * (y * (1 - probabilities) ** 2 + (1 - y) * probabilities ** 2)))

    training_report = json.loads((head / "enriched/report.json").read_text())
    assert abs(score(p) - training_report["brier_alc"]) < 1e-12
    assert all(np.allclose(g.probability, g.probability.iloc[0], atol=1e-12)
               for _, g in saved.groupby(keys + ["item_id"]))
    pairs_table = []
    for (benchmark, subject), group in zero.groupby(keys):
        rate = float(group.n_outcomes_one.sum() / group.n_responses.sum())
        pairs_table.append(dict(benchmark=benchmark, subject_id=subject,
            n_items=len(group), n_responses=int(group.n_responses.sum()),
            outcome_rate=rate, mean_probability=float(np.average(group.probability, weights=group.n_responses)),
            within_pair_auc=auc(group),
            diagnostic_pair_constant_oracle_brier=rate * (1 - rate)))
    pair_frame = pd.DataFrame(pairs_table)
    pair_frame.to_csv(out / "per_pair.csv", index=False)
    stretch = pd.DataFrame([dict(factor=a, brier=score(np.clip(.5 + a * (p - .5), 0, 1)))
                            for a in [0, 1, 2, 4, 8]])
    stretch.to_csv(out / "sharpness_diagnostic.csv", index=False)
    deviation = p - .5
    oracle_factor = float(np.sum(w * deviation * (y - .5)) / np.sum(w * deviation ** 2))

    split_path = ROOT / "configs/head_finetuning/frozen_split_v1.json"
    split = json.loads(split_path.read_text())
    if set(zero.benchmark) - set(split["roles"]["development"]):
        raise ValueError("Saved predictions include a non-development family")
    original = ROOT / split["artifact_directory"] / "development.jsonl"
    if file_hash(original) != split["artifact_fingerprints"]["development.jsonl"]:
        raise ValueError("Frozen development inputs changed")
    lookup, pairs, sources, profiles = {}, [], [prediction_path, split_path, original], []
    for benchmark, group in zero.groupby("benchmark"):
        tables = load_benchmark(ROOT / "payload/data", benchmark)
        for name in ["subjects", "items", "response"]:
            sources.append(benchmark_dir(ROOT / "payload/data", benchmark) / (name + ".parquet"))
        for pair in iter_pairs(tables):
            if pair.subject_id not in set(group.subject_id):
                continue
            rows = group[group.subject_id.eq(pair.subject_id)]
            frozen = make_split(pair.item_ids, benchmark=benchmark, subject_id=pair.subject_id,
                                seed=split["seed"], acquisition_items=40, evaluation_items=120)
            if set(rows.item_id) != set(frozen.evaluation):
                raise ValueError("Evaluation IDs differ from frozen saved predictions")
            for row in rows.itertuples():
                key = signature(pair.subject, pair.item(row.item_id))
                if key in lookup and abs(lookup[key] - row.probability) > 1e-10:
                    raise ValueError("Identical allowed inputs have inconsistent saved probabilities")
                lookup[key] = row.probability
            profiles.append(dict(benchmark=benchmark, subject_id=pair.subject_id,
                visible_profile=json.dumps(competition_input(pair.subject, {})[0], sort_keys=True)))
            pairs.append(pair)

    class SavedHead:
        use_labels = False
        runtime_metadata = {"source": "saved compact head; no model inference"}

        def predict_many(self, subject, items, labeled):
            return [lookup[signature(subject, item)] for item in items]

    class Half:
        use_labels = False
        runtime_metadata = {"source": "constant 0.5"}

        def predict_many(self, subject, items, labeled):
            return [.5] * len(items)

    evidence_rows = []
    for name, predictor in [("saved_head", SavedHead()),
                            ("saved_head_beta5", EvidencePredictor(SavedHead(), 5.)),
                            ("half_beta5", EvidencePredictor(Half(), 5.))]:
        records = []
        result = evaluate(pairs, predictor, seed=split["seed"], acquisition_items=40,
                          evaluation_items=120, shared_evidence=True, prediction_records=records)
        if name == "saved_head":
            assert abs(result.brier_alc - training_report["brier_alc"]) < 1e-12
        result.frame.to_csv(out / (name + "_scores.csv"), index=False)
        pd.DataFrame(records).to_csv(out / (name + "_predictions.csv"), index=False)
        evidence_rows.append(dict(predictor=name, brier_alc=result.brier_alc,
            **{f"brier_{int(b)}": float(value) for b, value in result.per_budget.items()}))
    comparison = pd.DataFrame(evidence_rows)
    comparison.to_csv(out / "evidence_comparison.csv", index=False)
    profiles = pd.DataFrame(profiles)
    profile_rates = pair_frame.merge(profiles, on=keys)
    ambiguities = []
    for (benchmark, _), group in profile_rates.groupby(["benchmark", "visible_profile"]):
        if len(group) > 1:
            ambiguities.append(dict(benchmark=benchmark, indistinguishable_systems=len(group),
                outcome_rate_min=float(group.outcome_rate.min()), outcome_rate_max=float(group.outcome_rate.max())))
    summary = dict(raw_brier_alc=score(p), n_pairs=n_pairs,
        n_items=len(zero), n_responses=int(zero.n_responses.sum()),
        within_pair_auc_mean=float(pair_frame.within_pair_auc.mean()), pooled_auc=auc(zero),
        probability_min=float(p.min()), probability_max=float(p.max()),
        diagnostic_pair_constant_oracle_brier=float(pair_frame.diagnostic_pair_constant_oracle_brier.mean()),
        diagnostic_development_fitted_unclipped_stretch_factor=oracle_factor,
        diagnostic_development_fitted_stretch_brier=score(.5 + oracle_factor * deviation),
        identical_visible_profiles=ambiguities,
        numeric_evidence_strength=5., evidence_scope="all sampled pairs; allowed visible-profile matching only",
        model_inference=False, model_training=False, calibration_or_confirmation_opened=False,
        diagnostic_oracles_use_development_labels=True, oracles_are_valid_predictors=False,
        note="Oracles/sharpness diagnostics are never fitted deployment transformations. Beta strength 5 is fixed in advance.",
        input_sha256={str(path.relative_to(ROOT)): file_hash(path) for path in sources})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    axes[0].scatter(pair_frame.mean_probability, pair_frame.outcome_rate)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
                ylabel="Observed outcome rate", title="System–benchmark averages")
    axes[1].bar(range(n_pairs), pair_frame.within_pair_auc)
    axes[1].axhline(.5, color="gray", linestyle="--")
    axes[1].set(ylim=(0, 1), xlabel="Development pair (audit index)", ylabel="AUROC",
                title="Ranking within each pair")
    for row in evidence_rows:
        axes[2].plot(range(6), [row[f"brier_{b}"] for b in [0, 1, 3, 7, 15, 31]],
                     marker="o", label=row["predictor"])
    axes[2].axhline(.15, color="gray", linestyle=":", label="Goal 0.15")
    axes[2].set(xticks=range(6), xticklabels=[0, 1, 3, 7, 15, 31], xlabel="Revealed-label budget",
                ylabel="Brier", title="Untuned evidence controls")
    axes[2].legend(fontsize=8)
    fig.savefig(out / "diagnostics.png", dpi=150)
    plt.close(fig)
    html = '<!doctype html><meta charset="utf-8"><title>Compact head strategy audit</title>'
    html += '<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px;line-height:1.5}td,th{padding:8px}img{width:100%}pre{white-space:pre-wrap}</style>'
    html += '<h1>Compact head strategy audit</h1><p>Saved predictions and original development/acquisition pools only. No new model inference, gradient training, or sealed-set access.</p>'
    html += '<img src="diagnostics.png" alt="Pair averages, within-pair ranking and evidence budget curves">'
    html += '<h2>Measured diagnostics</h2>' + pair_frame.round(6).to_html(index=False)
    html += '<h2>Confidence stretching (diagnostic only)</h2><p>p_new = clip(0.5 + factor × (p_old − 0.5)). No transformation is deployed.</p>' + stretch.round(6).to_html(index=False)
    html += '<h2>Untuned numeric evidence</h2><p>Fixed k=5; random frozen acquisition trajectory, nested budgets, shared evidence and visible-profile matching. Evaluation items are excluded from evidence.</p>' + comparison.round(6).to_html(index=False)
    html += '<h2>Limits</h2><p>The pair-constant oracle uses evaluation outcomes and private scorer-side pair IDs. It is the best constant-per-pair fit on this sample, not a valid predictor or an irreducible-noise estimate. The fitted stretch is also an evaluation-label oracle. Eight pairs from two development families do not establish generalization to unseen competition benchmarks.</p>'
    html += '<pre>' + escape(json.dumps(summary, indent=2)) + '</pre>'
    (out / "report.html").write_text(html + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
