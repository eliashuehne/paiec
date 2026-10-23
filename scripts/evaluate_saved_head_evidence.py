#!/usr/bin/env python3
"""One untuned evidence control on saved head probabilities; no model inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
from paiec.competition import competition_input
from paiec.evidence import EvidencePredictor
from paiec.evaluate import evaluate
from paiec.head_training import file_hash
from paiec.pairs import iter_pairs
from paiec.schema import load_benchmark
from paiec.split import make_split


def finalize(out):
    """Render/log already computed scores; no inference or score recomputation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paiec.tracking import ExperimentTracker
    from paiec.schema import benchmark_dir
    frame=pd.read_csv(out/"comparison.csv")
    budgets=[0,1,3,7,15,31]
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    for row in frame.to_dict("records"):
        ax.plot(range(6),[row[f"brier_{n}"] for n in budgets],marker="o",label=row["predictor"])
    ax.axhline(.25,color="gray",linestyle="--",label="Constant 50%")
    ax.set(xticks=range(6),xticklabels=budgets,xlabel="Revealed outcomes per pair (log-spaced budgets)",
           ylabel="Brier score (lower is better)",title="Untuned evidence update on saved development probabilities")
    ax.legend(fontsize=8)
    fig.savefig(out/"budgets.png",dpi=150)
    plt.close(fig)
    html='<!doctype html><meta charset="utf-8"><title>Saved-head evidence control</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;line-height:1.5}td,th{padding:8px}img{width:100%}</style>'
    html+='<h1>Saved-head evidence control</h1><p>Existing enriched checkpoint predictions plus an untuned numeric update, strength k=5. No new model inference, training or calibration fitting. Eight previously used development pairs; confirmation untouched. This does not evaluate the new task-v2 inputs.</p>'
    html+='<img src="budgets.png" alt="Brier curves across revealed-label budgets">'+frame.round(6).to_html(index=False)
    html+='<p>The update worsens small budgets on this acquisition trajectory and improves later ones. Tune strength only on training-family folds. Evidence is pooled across all eight sampled pairs and matched using visible profiles and anonymous benchmark IDs, never private subject IDs.</p>'
    (out/"report.html").write_text(html)
    files=[Path(__file__).resolve(),ROOT/"src/paiec/evidence.py",ROOT/"src/paiec/evaluate.py"]
    for benchmark in ("ceval","erbench"):
        folder=benchmark_dir(ROOT/"payload/data",benchmark)
        files.extend(folder/name for name in ("subjects.parquet","items.parquet","response.parquet"))
    provenance={str(p.relative_to(ROOT)):file_hash(p) for p in files}
    (out/"audit_sources.json").write_text(json.dumps(provenance,indent=2)+"\n")
    tracker=ExperimentTracker(ROOT,experiment="PAIEC Feature Controls")
    with tracker.run("saved head / numeric evidence k=5 / no new inference",json.loads((out/"manifest.json").read_text()),
        {"run_type":"saved_prediction_postprocessing","evidence_strength":5.,"new_model_inference":False},
        {"confirmation_scored":"false","evidence_scope":"all_sampled_pairs"}) as tracked:
        if tracked:
            for row in frame.to_dict("records"):
                for key,value in row.items():
                    if key!="predictor":tracked.client.log_metric(tracked.run_id,row["predictor"]+"/"+key,float(value))
            for name in ("manifest.json","comparison.csv","report.html","budgets.png","audit_sources.json"):
                tracked.client.log_artifact(tracked.run_id,str(out/name))
            (out/"mlflow_receipt.json").write_text(json.dumps({"run_id":tracked.run_id})+"\n")


def signature(subject,item):
    return json.dumps(competition_input(subject,item),sort_keys=True,ensure_ascii=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/head_finetuning/evidence_control_v1")
    args=parser.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=False)
    split_path=ROOT/"configs/head_finetuning/frozen_split_v1.json"
    split=json.loads(split_path.read_text())
    dev_path=ROOT/split["artifact_directory"]/"development.jsonl"
    if file_hash(dev_path)!=split["artifact_fingerprints"]["development.jsonl"]:
        raise ValueError("Frozen development export changed")
    pred_path=ROOT/"runs/head_finetuning/enrichment_pilot_v1_modal_cached/output/enriched/best_predictions.csv"
    pred=pd.read_csv(pred_path)
    pred=pred[pred.budget.eq(0)]
    pairs=[]; lookup={}
    for benchmark,group in pred.groupby("benchmark"):
        tables=load_benchmark(ROOT/"payload/data",benchmark)
        selected=[p for p in iter_pairs(tables) if p.subject_id in set(group.subject_id)]
        for pair in selected:
            saved=group[group.subject_id.eq(pair.subject_id)]
            frozen=make_split(pair.item_ids,benchmark=benchmark,subject_id=pair.subject_id,
                seed=split["seed"],acquisition_items=40,evaluation_items=120)
            if set(saved.item_id)!=set(frozen.evaluation):
                raise ValueError("Evaluation items differ from the saved experiment")
            for row in saved.to_dict("records"):
                key=signature(pair.subject,pair.item(row["item_id"]))
                if key in lookup and abs(lookup[key]-row["probability"])>1e-10:
                    raise ValueError("Identical visible inputs have different saved predictions")
                lookup[key]=row["probability"]
        pairs.extend(selected)
    class SavedHead:
        use_labels=False
        runtime_metadata={"source":"saved epoch-one predictions; no inference"}
        def predict_many(self,subject,items,labeled):
            return [lookup[signature(subject,item)] for item in items]
        def __call__(self,subject,item,labeled):
            return self.predict_many(subject,[item],[])[0]
    class Half:
        use_labels=False
        runtime_metadata={"source":"constant 0.5"}
        def predict_many(self,subject,items,labeled):return [.5]*len(items)
    rows=[]
    settings=dict(seed=split["seed"],acquisition_items=40,evaluation_items=120,shared_evidence=True)
    for name,predictor in [("saved_head",SavedHead()),("saved_head_beta5",EvidencePredictor(SavedHead(),5.)),
                           ("half_beta5",EvidencePredictor(Half(),5.))]:
        predictions=[]
        result=evaluate(pairs,predictor,prediction_records=predictions,**settings)
        if name=="saved_head" and abs(result.brier_alc-.24908013244879984)>1e-12:
            raise ValueError("Saved score was not reproduced")
        result.frame.to_csv(out/(name+"_scores.csv"),index=False)
        pd.DataFrame(predictions).to_csv(out/(name+"_predictions.csv"),index=False)
        rows.append(dict(predictor=name,brier_alc=result.brier_alc,
            **{f"brier_{b}":float(v) for b,v in result.per_budget.items()}))
    pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(
        training=False,model_inference=False,calibration_fitted=False,confirmation_opened=False,
        evidence_strength=5.,strength_selected_using_development=False,
        evidence_scope="all selected pairs at each nested budget; visible-profile matching only",
        development_already_used_for_checkpoint_selection=True,
        input_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in (split_path,dev_path,pred_path)},
        results=rows),indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False))
    finalize(out)


if __name__=="__main__":main()
