#!/usr/bin/env python3
"""Fit training-only episodic evidence correction; evaluate saved heads separately."""
import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from paiec.competition import competition_input
from paiec.encoder_training import EncoderTrainingConfig
from paiec.expanded_data import verify_expanded
from paiec.head_pilot import write_json
from paiec.learned_evidence import LearnedEvidencePredictor, fit_from_training_monitor
from paiec.evidence import EvidencePredictor
from paiec.evaluate import evaluate
from paiec.development_evidence import original_development_pairs
from paiec.tracking import ExperimentTracker


def signature(subject, item):
    return json.dumps(competition_input(subject, item), sort_keys=True, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_finetuning/enriched_30k_encoder_v1.json")
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = EncoderTrainingConfig.model_validate_json((ROOT / args.config).read_text())
    prepared = ROOT / config.prepared
    manifest = verify_expanded(ROOT, prepared)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    split = json.loads((ROOT / manifest["data_config"]["split"]).read_text())
    frame = pd.read_parquet(prepared / "enriched/provenance.parquet")
    development = frame[frame.role.eq("development")].sort_values("jsonl_line").reset_index(drop=True)
    pairs = original_development_pairs(ROOT, manifest, development, split["seed"])
    summaries = []
    tracker = ExperimentTracker(ROOT, experiment="PAIEC Evidence Adaptation")
    for arm_name in ["frozen", f"upper_{config.upper_layers}"]:
        destination = out / arm_name
        destination.mkdir()
        arm = ROOT / args.training_output / arm_name
        fitted = fit_from_training_monitor(prepared, arm, destination)
        saved = pd.read_csv(arm / "predictions.csv")
        zero = saved[saved.budget.eq(0)]
        lookup = {}
        for pair in pairs:
            for row in zero[zero.benchmark.eq(pair.benchmark) & zero.subject_id.eq(pair.subject_id)].itertuples():
                key = signature(pair.subject, pair.item(row.item_id))
                if key in lookup and abs(lookup[key]-row.probability) > 1e-5:
                    raise ValueError("Identical allowed inputs have inconsistent predictions")
                lookup[key] = row.probability

        class Saved:
            use_labels = False
            runtime_metadata = {"source": "saved trained head; no model inference", "checkpoint_revision":fitted["checkpoint_identity"]}
            def predict_many(self, subject, items, labeled):
                return [lookup[signature(subject, item)] for item in items]

        for method, predictor in [("raw", Saved()), ("fixed_beta5", EvidencePredictor(Saved(), 5.)),
                                  ("learned", LearnedEvidencePredictor(Saved(), fitted))]:
            with tracker.run(f"30k / {arm_name} / {method}", {"model": fitted, "arm":arm_name, "method":method},
                             {"arm":arm_name, "method":method, "query_training_records":1000},
                             {"run_type":"evidence_adaptation", "evidence_correction_fitted":str(method == "learned").lower(),
                              "sealed_calibration_opened":"false", "confirmation_scored":"false"}) as tracked:
                records = []
                result = evaluate(pairs, predictor, seed=split["seed"], acquisition_items=40,
                                  evaluation_items=120, shared_evidence=True, prediction_records=records)
                if method == "raw" and abs(result.brier_alc-json.loads((arm / "report.json").read_text())["brier_alc"]) > 1e-10:
                    raise ValueError("Saved raw score changed")
                result.frame.to_csv(destination / (method+"_scores.csv"), index=False)
                prediction_frame = pd.DataFrame(records)
                prediction_frame.to_csv(destination / (method+"_predictions.csv"), index=False)
                summary = dict(arm=arm_name, method=method, brier_alc=result.brier_alc,
                    **{f"brier_{int(b)}":float(v) for b,v in result.per_budget.items()})
                summaries.append(summary)
                if tracked:
                    tracked.result(result, 0., prediction_frame)
                    tracked.json(fitted, "learned_evidence.json")
    table = pd.DataFrame(summaries)
    table.to_csv(out / "comparison.csv", index=False)
    write_json(out / "report.json", dict(runs=summaries, training_role_queries=1000,
        main_heads_training_records=30000, model_inference=False, laya_gradient_training=False, evidence_model_fitted=True,
        development_fit=False, calibration_opened=False, confirmation_opened=False,
        acquisition="original nested random trajectory; shared matching allowed profiles and anonymous IDs"))
    html = '<!doctype html><meta charset="utf-8"><title>Separate learned evidence experiment</title><style>body{font:16px system-ui;margin:40px;line-height:1.5}td,th{padding:8px}pre{white-space:pre-wrap}</style>'
    html += '<h1>Separate learned evidence experiment</h1><p>Each head uses 30,000 fitting records. The correction learns from 1,000 additional training-role monitor queries, never original development labels. Episode rows repeat these queries and are not independent new questions. Budget zero stays unchanged. Original acquisition/evaluation exclusion, trials and pair weights are preserved. No new Laya inference or cloud work.</p>'
    html += table.round(6).to_html(index=False)+'<pre>'+escape((out/'report.json').read_text())+'</pre>'
    (out / "report.html").write_text(html)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
