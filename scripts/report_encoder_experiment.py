#!/usr/bin/env python3
"""Explain completed matched encoder/evidence results; no fitting or inference."""
import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from paiec.head_pilot import write_json
from paiec.head_training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    training, evidence, out = [ROOT / getattr(args, name) for name in ["training_output", "evidence_output", "out"]]
    out.mkdir(parents=True, exist_ok=False)
    comparison = pd.read_csv(evidence / "comparison.csv")
    diagnostics, reports, histories, sources = [], {}, {}, [evidence / "comparison.csv"]
    for arm in comparison.arm.unique():
        path = training / arm
        report = json.loads((path / "report.json").read_text())
        if report["status"] != "finished":
            raise ValueError("Only completed arms may be reported")
        reports[arm] = report
        histories[arm] = pd.read_json(path / "history.jsonl", lines=True)
        prediction_path = path / "predictions.csv"
        sources += [prediction_path, path / "report.json", path / "history.jsonl"]
        frame = pd.read_csv(prediction_path)
        frame = frame[frame.budget.eq(0)]
        for (benchmark, subject), group in frame.groupby(["benchmark", "subject_id"], sort=True):
            labels, probabilities = [], []
            for row in group.itertuples():
                n, s = int(row.n_responses), int(row.n_outcomes_one)
                labels.extend([1]*s+[0]*(n-s))
                probabilities.extend([row.probability]*n)
            labels, p = np.array(labels), np.array(probabilities)
            diagnostics.append(dict(arm=arm, benchmark=benchmark, subject_id=subject,
                n_responses=len(labels), outcome_rate=float(labels.mean()), mean_probability=float(p.mean()),
                probability_min=float(p.min()), probability_max=float(p.max()),
                within_pair_auc=float(roc_auc_score(labels, p)) if len(set(labels)) > 1 else None,
                brier=float(np.mean((p-labels)**2))))
    pairs = pd.DataFrame(diagnostics)
    pairs.to_csv(out / "per_pair.csv", index=False)
    summary = pairs.groupby("arm").agg(brier_alc=("brier", "mean"), mean_within_pair_auc=("within_pair_auc", "mean"),
        probability_min=("probability_min", "min"), probability_max=("probability_max", "max"))
    for arm, report in reports.items():
        if abs(summary.loc[arm, "brier_alc"]-report["brier_alc"]) > 1e-12:
            raise ValueError("Diagnostic score differs from completed run")
    summary.to_csv(out / "raw_summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for arm, history in histories.items():
        axes[0].plot(history["update"], history.loss.rolling(20, min_periods=1).mean(), label=arm)
    axes[0].set(xlabel="Optimizer update", ylabel="Weighted cross-entropy", title="Training loss (20-update average)")
    axes[0].legend()
    pivot = pairs.pivot(index=["benchmark", "subject_id"], columns="arm", values="within_pair_auc")
    x = np.arange(len(pivot))
    for i, arm in enumerate(pivot.columns):
        axes[1].bar(x+(i-.5)*.35, pivot[arm], width=.35, label=arm)
    axes[1].axhline(.5, color="gray", linestyle="--")
    axes[1].set(ylim=(0, 1), xlabel="Original development pair", ylabel="AUROC", title="Question ranking within pairs",
                xticks=x, xticklabels=[f"{b} {i+1}" for i,(b,_) in enumerate(pivot.index)])
    axes[1].tick_params(axis='x', labelrotation=55)
    axes[1].legend()
    budgets = [0, 1, 3, 7, 15, 31]
    for row in comparison.itertuples():
        if row.method in {"raw", "learned"}:
            axes[2].plot(range(6), [getattr(row, f"brier_{b}") for b in budgets], marker="o", label=f"{row.arm} / {row.method}")
    axes[2].axhline(.25, color="gray", linestyle="--", label="Constant 0.5")
    axes[2].axhline(.15, color="gray", linestyle=":", label="Research target")
    axes[2].set(xlabel="Revealed-label budget", ylabel="Brier", title="Separate evidence adaptation",
                xticks=range(6), xticklabels=budgets)
    axes[2].legend(fontsize=8)
    fig.savefig(out / "diagnostics.png", dpi=150)
    plt.close(fig)
    runtime = pd.DataFrame([dict(arm=arm, peak_cuda_gib=report["peak_cuda_allocated_bytes"]/1024**3,
        peak_process_rss_gib=report["peak_host_rss_kib"]/1024**2,
        median_update_seconds=float(histories[arm].seconds.median()),
        phase_minutes=report["training_development_export_seconds"]/60,
        updates=report["updates"], presentations=report["presentations"]) for arm, report in reports.items()])
    runtime.to_csv(out / "runtime.csv", index=False)
    write_json(out / "summary.json", dict(raw_heads=summary.reset_index().to_dict("records"),
        evidence=comparison.to_dict("records"), runtime=runtime.to_dict("records"),
        development_pairs=8, development_families=2, main_fitting_records=30000,
        separate_evidence_fitting_queries=1000, calibration_opened=False, confirmation_opened=False,
        fitting=False, model_inference=False,
        input_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sources}))
    html = '<!doctype html><meta charset="utf-8"><title>Enriched 30k experiment review</title><style>body{font:16px system-ui;max-width:1450px;margin:32px auto;padding:16px;line-height:1.5}td,th{padding:8px}img{width:100%}pre{white-space:pre-wrap}</style>'
    html += '<h1>Enriched 30k experiment review</h1><p>Same enriched fitting data and original development for both heads. Upper_2 trains the final two encoder blocks plus final normalization and decision head; frozen trains the decision head only. Constant 50% gives Brier ALC 0.25; lower is better. Results cover eight pairs in two families.</p>'
    html += '<h2>Raw trained heads</h2>'+summary.round(6).to_html()+'<img src="diagnostics.png" alt="Training loss, within-pair ranking and revealed-evidence curves">'
    html += '<h2>Separate evidence comparison</h2><p>Corrections use 1,000 additional training-role monitor queries and training-role support outcomes. They never fit original development labels. Episode rows repeat queries; budget zero stays unchanged. Fixed beta strength 5 is untuned. No calibration/confirmation outcomes were opened.</p>'+comparison.round(6).to_html(index=False)
    html += '<h2>Measured resources</h2>'+runtime.round(3).to_html(index=False)+'<p>Phase time includes training, development scoring and export/reload; model loading, local preparation, cloud setup and container tail are additional. Peak RSS is process RAM, not GPU memory or a billing statement. Raw-head budgets repeat the same probabilities.</p>'
    html += '<details><summary>Per-pair diagnostics</summary>'+pairs.round(6).to_html(index=False)+'</details><pre>'+escape((out/'summary.json').read_text())+'</pre>'
    (out / "report.html").write_text(html)
    print(summary.to_string())
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
