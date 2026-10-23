#!/usr/bin/env python3
"""Audit existing head predictions and allowed input features; no fitting/inference."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.head_pilot import load_config, verify_prepared, read_rows, pair_weights


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def cluster_interval(frame, repeats=4000, seed=20261022):
    """Resample item clusters jointly across subjects within each fixed benchmark.

    Keeps trials together, recomputes each pair's response-weighted score, then
    averages pairs equally. Conditional on these selected benchmarks/subjects;
    cannot establish performance on unseen benchmark families.
    """
    rng = np.random.default_rng(seed)
    pair_deltas = []
    for _, bench in frame.groupby("benchmark", sort=True):
        ids = sorted(bench.item_id.unique())
        weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=repeats)
        positions = {item: i for i, item in enumerate(ids)}
        for _, group in bench.groupby("subject_id", sort=True):
            losses = np.zeros(len(ids))
            counts = np.zeros(len(ids))
            for item, rows in group.groupby("item_id"):
                pos = positions[item]
                losses[pos] = (((rows.probability-rows.label)**2)-.25).sum()
                counts[pos] = len(rows)
            denominator = weights @ counts
            if np.any(denominator == 0):
                raise ValueError("Bootstrap draw has no observations for a pair")
            pair_deltas.append((weights @ losses) / denominator)
    delta = np.mean(pair_deltas, axis=0)
    return dict(repetitions=repeats, seed=seed,
        delta_brier_minus_half_interval_95=np.quantile(delta,[.025,.975]).tolist(),
        scope="Item-cluster uncertainty conditional on the eight selected pairs/two benchmarks; not a new-family confidence interval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/head_finetuning/approach_audit_v1")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs/head_finetuning/enrichment_pilot_v1.json")
    prepared, prep = verify_prepared(ROOT, config)
    run = ROOT / "runs/head_finetuning/enrichment_pilot_v1_modal_cached/output/enriched"
    report = json.loads((run / "report.json").read_text())
    meta = pd.read_parquet(prepared / "enriched/provenance.parquet")
    source = ROOT / "runs/head_finetuning/data_v2"
    frames, coverage, collisions, input_examples = {}, [], [], []
    inputs = [Path(__file__).resolve(), ROOT / "configs/head_finetuning/enrichment_pilot_v1.json",
              prepared / "manifest.json", prepared / "enriched/provenance.parquet",
              run / "best_predictions.csv", run / "report.json", run / "epochs.json", run / "history.json"]
    for role in ["training", "development"]:
        rows = read_rows(prepared / "enriched" / (role+".jsonl"))
        originals = read_rows(source / (role+".jsonl"))
        f = meta[meta.role.eq(role)].sort_values("jsonl_line").copy()
        assert f.jsonl_line.tolist() == list(range(len(rows)))
        f["label"] = [int(row["expected"]["correct"]) for row in rows]
        assert [row["expected"] for row in originals] == [row["expected"] for row in rows]
        f["state"] = [row["state"] for row in rows]
        f["model_properties"] = ["Published model properties" in row["state"] for row in rows]
        f["harness_capabilities"] = ["Published harness capabilities" in row["state"] for row in rows]
        parsed = [json.loads(row["state"]) for row in originals]
        f["subject_signature"] = [json.dumps(row["target_system"],sort_keys=True,ensure_ascii=False) for row in parsed]
        f["cjk_task"] = [any('\u4e00' <= c <= '\u9fff' for c in row["target_item"].get("item_content", "")) for row in parsed]
        f["reasoning_effort_present"] = [bool(row["target_system"].get("reasoning_effort")) for row in parsed]
        f["task_chars"] = [len(row["target_item"].get("item_content", "")) for row in parsed]
        frames[role] = f
        inputs += [prepared / "enriched" / (role+".jsonl"), source / (role+".jsonl")]
        for bench,g in f.groupby("benchmark",sort=True):
            coverage.append(dict(role=role,benchmark=bench,records=len(g),pairs=g.subject_id.nunique(),
                outcome_rate=g.label.mean(),truncated=int(g.truncated.sum()),
                model_property_rows=int(g.model_properties.sum()),harness_capability_rows=int(g.harness_capabilities.sum()),
                unique_tasks=g.task_sha256.nunique(),unique_effective_inputs=g.effective_input_sha256.nunique(),
                cjk_task_rows=int(g.cjk_task.sum()),reasoning_effort_rows=int(g.reasoning_effort_present.sum())))
        grouped = f.groupby("effective_input_sha256").agg(records=("row_id","size"),tasks=("task_sha256","nunique"),labels=("label","nunique"))
        bad = grouped[grouped.tasks.gt(1)]
        collision_rows = f[f.effective_input_sha256.isin(bad.index)]
        collisions.append(dict(role=role,unique_effective_inputs=len(grouped),
            multitask_collision_groups=len(bad),multitask_collision_rows=int(bad.records.sum()),
            conflicting_label_groups=int(grouped.labels.gt(1).sum()),
            conflicting_label_rows=int(grouped.loc[grouped.labels.gt(1),"records"].sum()),
            collision_rows_by_benchmark=collision_rows.groupby("benchmark").size().to_dict()))
        if role == "training":
            largest = bad.sort_values("records", ascending=False).head(3)
            for key in largest.index:
                group = collision_rows[collision_rows.effective_input_sha256.eq(key)]
                distinct = group.drop_duplicates("task_sha256").head(2)
                input_examples.append(dict(effective_input_sha256=key,benchmark=group.benchmark.iloc[0],
                    n_records=len(group),n_distinct_tasks=group.task_sha256.nunique(),
                    row_indices=distinct.jsonl_line.tolist(),
                    state_character_counts=distinct.state.str.len().tolist(),
                    state_prefixes=[s[:1350] for s in distinct.state],
                    state_tails=[s[-650:] for s in distinct.state],
                    note="Different full tasks share the same prepared retained-token hash; full text shown here is not all retained by Laya."))
    train, dev = frames["training"], frames["development"]
    assert not set(train.effective_input_sha256) & set(dev.effective_input_sha256)
    pred_all = pd.read_csv(run / "best_predictions.csv")
    assert pred_all.groupby(["benchmark","subject_id","item_id"]).probability.nunique().max() == 1
    pred = pred_all[pred_all.budget.eq(0)]
    dev = dev.merge(pred, on=["benchmark","subject_id","item_id"],validate="many_to_one")
    assert dev.probability.notna().all()
    outcomes = dev.groupby(["benchmark","subject_id","item_id"]).label.agg(["sum","size"]).reset_index()
    checked = outcomes.merge(pred,on=["benchmark","subject_id","item_id"],validate="one_to_one")
    assert np.array_equal(checked["sum"],checked.n_outcomes_one)
    assert np.array_equal(checked["size"],checked.n_responses)
    prior = float(np.average(train.label,weights=pair_weights(train)))
    subjects = {}
    for benchmark in sorted(dev.benchmark.unique()):
        path = ROOT / "payload/data" / benchmark / "formatted_tables/subjects.parquet"
        table = pd.read_parquet(path)
        inputs.append(path)
        for row in table[table.subject_id.isin(dev.subject_id)].to_dict("records"):
            subjects[(benchmark,row["subject_id"])] = row.get("display_name") or row["subject_id"]
    metrics = []
    for (benchmark,subject),g in dev.groupby(["benchmark","subject_id"],sort=True):
        y,p = g.label.to_numpy(),g.probability.to_numpy()
        rate, mean = float(y.mean()),float(p.mean())
        covariance = float(np.mean((p-mean)*(y-rate)))
        metrics.append(dict(benchmark=benchmark,subject_id=subject,audit_name=subjects[(benchmark,subject)],
            records=len(g),items=g.item_id.nunique(),outcome_rate=rate,mean_prediction=mean,
            prediction_std=float(p.std()),prediction_min=float(p.min()),prediction_max=float(p.max()),
            brier=float(np.mean((p-y)**2)),brier_half=.25,
            brier_train_only_prior=float(np.mean((y-prior)**2)),
            brier_constant_at_own_mean_prediction=float(np.mean((y-mean)**2)),
            item_variation_penalty=float(p.var()-2*covariance),
            brier_oracle_pair_rate_DIAGNOSTIC=float(rate*(1-rate)),
            auc=float(roc_auc_score(y,p)) if np.unique(y).size==2 else None,
            missing_model_name=bool(g.missing_model_name.iloc[0]),model_properties=bool(g.model_properties.iloc[0])))
    pairs = pd.DataFrame(metrics)
    assert abs(pairs.brier.mean()-report["brier_alc"]) < 1e-12
    assert np.allclose(pairs.brier-pairs.brier_constant_at_own_mean_prediction,pairs.item_variation_penalty)
    bootstrap = cluster_interval(dev)
    grouped = []
    for low,high in zip(np.arange(0,1,.05), np.arange(.05,1.05,.05)):
        g=dev[dev.probability.ge(low)&dev.probability.lt(high)]
        if len(g):
            grouped.append(dict(low=float(low),high=float(high),n=len(g),prediction=float(g.probability.mean()),rate=float(g.label.mean())))
    summary = dict(status="completed analysis of existing artifacts",new_model_training=False,new_inference=False,
        sealed_calibration_or_confirmation_opened=False,
        predictor_fields_changed=False,development_used_for_analysis_and_existing_checkpoint_selection=True,
        training_records=len(train),training_pairs=train.groupby(["benchmark","subject_id"]).ngroups,
        training_unique_tasks=train.task_sha256.nunique(),training_truncated_rows=int(train.truncated.sum()),
        trainable_parameters=report["parameter_counts"]["trainable"],
        training_pair_weighted_outcome_rate=prior,training_unique_system_profiles=train.subject_signature.nunique(),
        development_responses=len(dev),development_items=len(pred),development_pairs=len(pairs),
        development_unique_system_profiles=dev.subject_signature.nunique(),development_benchmarks=dev.benchmark.nunique(),
        macro_brier=float(pairs.brier.mean()),macro_auc=float(pairs.auc.mean()),
        macro_brier_train_only_prior=float(pairs.brier_train_only_prior.mean()),
        macro_brier_constant_at_own_mean_prediction=float(pairs.brier_constant_at_own_mean_prediction.mean()),
        macro_item_variation_penalty=float(pairs.item_variation_penalty.mean()),
        hindsight_pair_rate_brier_DIAGNOSTIC=float(pairs.brier_oracle_pair_rate_DIAGNOSTIC.mean()),
        probabilities_quantiles=dev.probability.quantile([0,.05,.25,.5,.75,.95,1]).to_dict(),
        uncertainty=bootstrap,
        scoring_note="Equal subject-benchmark weights; repeated trials retained; six budgets reuse identical predictions. AUROC 0.5 is chance. Oracle pair rates use dev labels and are NOT valid baselines or deployment features.")
    pd.DataFrame(coverage).to_csv(out / "coverage.csv",index=False)
    pairs.to_csv(out / "per_pair.csv",index=False)
    pd.DataFrame(grouped).to_csv(out / "calibration_bins.csv",index=False)
    write_json(out / "summary.json",summary)
    write_json(out / "input_collisions.json",collisions)
    write_json(out / "collision_examples.json",input_examples)
    errors = dev.copy()
    errors["squared_error"]=(errors.probability-errors.label)**2
    errors["gain_over_half"] = .25-errors.squared_error
    selected=[]
    for benchmark,g in errors.groupby("benchmark"):
        for title,subset in [("most_helpful",g.nlargest(3,"gain_over_half")),("most_harmful",g.nsmallest(3,"gain_over_half"))]:
            for row in subset.to_dict("records"):
                selected.append(dict(kind=title,benchmark=benchmark,subject_id=row["subject_id"],
                    audit_name=subjects[(benchmark,row["subject_id"])],item_id=row["item_id"],
                    outcome=row["label"],probability=row["probability"],gain_over_half=row["gain_over_half"],
                    state_excerpt=row["state"][:1600]))
    write_json(out / "error_examples.json",selected)
    write_json(out / "provenance.json",dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        report_source="best epoch 1, cached enriched pilot", audit_display_names="Reporting only; never prediction features"))
    plot(out,dev,pairs,train,run)
    html = ['<!doctype html><meta charset="utf-8"><title>Head approach audit</title>',
        '<style>body{font:16px system-ui;max-width:1240px;margin:40px auto;padding:0 24px;line-height:1.5}table{border-collapse:collapse;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd}img{width:100%}pre{white-space:pre-wrap}a{color:#1763a0}</style>',
        '<h1>Head approach audit</h1><p>Existing predictions and training inputs only. No training, inference, calibration fitting or confirmation-label access.</p>',
        '<p>Prediction range: 46.8–55.6%. Per-pair item discrimination is weak. Many training tasks lose distinguishing content through truncation. See the accompanying engineering review for prioritized actions.</p>',
        '<p><a href="../../../docs/head-approach-review.md">Detailed engineering review and experiment plan</a></p>',
        '<img src="diagnostics.png" alt="Prediction, ranking, learning and truncation diagnostics">',
        '<h2>What needs attention</h2><ul>'
        '<li>2,042 of 5,000 training records are truncated; 654 belong to groups where different full tasks become identical retained inputs.</li>'
        '<li>Three C-Eval systems have identical supplied profiles. Source-table display names are unavailable to the predictor.</li>'
        '<li>The 26.2M-parameter head improves training loss while later development scores worsen.</li>'
        '<li>Revealed labels are ignored, leaving the same predictions at every budget.</li>'
        '<li>A matched dataset-only training control is still needed before crediting web enrichment.</li></ul>',
        '<h2>Proposed order of work — not executed</h2><ol>'
        '<li>Preserve task-specific fields and inspect retained question/options/context.</li>'
        '<li>Compare small regularized text/structured-feature controls; use Unicode-aware features for Chinese.</li>'
        '<li>Test smoothed updates from the permitted revealed outcomes.</li>'
        '<li>Run matched enrichment ablations, then assess capacity and language/encoder changes.</li></ol>',
        '<h2>Per-system development performance</h2><p>Names below are audit labels from source tables, not supplied predictor features. Rates refer to each system’s sampled tasks.</p>',
        pairs[["benchmark","audit_name","outcome_rate","mean_prediction","brier","auc","item_variation_penalty"]].round(5).to_html(index=False),
        '<h2>Input coverage</h2>',pd.DataFrame(coverage).round(4).to_html(index=False),
        '<h2>Uncertainty and scope</h2><p>The item-cluster interval includes no gain over 50%. It is conditional on these selected pairs and does not correct checkpoint-selection optimism or measure unseen-family generalization.</p><pre>'+escape(json.dumps(summary,indent=2))+'</pre>',
        '<h2>Truncation collisions</h2><pre>'+escape(json.dumps(collisions,indent=2))+'</pre>',
        '<p>Positive item_variation_penalty means item-varying predictions performed worse than a constant equal to that pair’s own mean prediction. This comparison fits no evaluation labels.</p>',
        '<p><a href="per_pair.csv">Pair statistics</a> · <a href="error_examples.json">Real error examples</a> · <a href="collision_examples.json">Collision examples</a> · <a href="provenance.json">Input hashes</a></p>']
    (out / "report.html").write_text('\n'.join(html))
    print(json.dumps(summary,indent=2))


def plot(out,dev,pairs,train,run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for bench,g in dev.groupby("benchmark"):
        axes[0,0].hist(g.probability,bins=np.linspace(0,1,41),alpha=.65,label=bench)
    axes[0,0].axvline(.5,color="black",ls="--");axes[0,0].set(xlim=(0,1),title="Predictions crowd around 50%",xlabel="Predicted success probability",ylabel="Recorded outcomes");axes[0,0].legend()
    y=np.arange(len(pairs))
    axes[0,1].scatter(pairs.outcome_rate,y,label="Observed outcome rate",marker="o")
    axes[0,1].scatter(pairs.mean_prediction,y,label="Mean prediction",marker="x")
    axes[0,1].set(yticks=y,yticklabels=pairs.audit_name,xlim=(0,1),title="Different systems receive similar predictions",xlabel="Probability / observed rate");axes[0,1].tick_params(axis='y',labelsize=8);axes[0,1].legend(fontsize=8)
    epochs=json.loads((run/"epochs.json").read_text())
    axes[1,0].plot([e['epoch'] for e in epochs],[e['brier_alc'] for e in epochs],marker='o',label="Development Brier ALC")
    axes[1,0].axhline(.25,color='black',ls='--',label='Constant 50%')
    axes[1,0].set(xticks=[1,2,3,4],title="Later epochs generalize worse",xlabel="Epoch",ylabel="Brier ALC (lower is better)");axes[1,0].legend(fontsize=8)
    rates=train.groupby('benchmark').truncated.mean().sort_values()
    axes[1,1].barh(rates.index,rates.values*100)
    axes[1,1].set(xlim=(0,100),title="Training input truncation",xlabel="Records truncated (%)");axes[1,1].tick_params(axis='y',labelsize=8)
    fig.savefig(out/"diagnostics.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
