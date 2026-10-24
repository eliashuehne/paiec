"""Original frozen-development inference and scoring, separate from fitting.

Only published subject/item fields leave the local scorer. Native GPU inference
loads the completed checkpoint already stored in the declared training job.
"""
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .competition_fields import normalize_fields
from .feature_preparation import read_jsonl, write_json, write_jsonl, local_path
from .grouped_checkpoint import verify_grouped_checkpoint
from .grouped_data import GroupedTrainingConfig, checked_artifacts
from .head_training import file_hash
from .head_evaluation import score_frozen_development
from .laya_predictor import load_questions


class GroupedEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["grouped_frozen_development_v1"] = "grouped_frozen_development_v1"
    prepared: str = "runs/grouped_development/prepared_v1"
    training_job_id: str = Field(pattern=r"^job_[0-9a-f]{32}$")
    training_output: str = "runs/workflows/grouped_enriched_training_v1/job"
    checkpoint_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: Literal[0] = 0

    @model_validator(mode="after")
    def paths(self):
        for name in ["prepared", "training_output"]:
            p = Path(getattr(self, name))
            if p.is_absolute() or ".." in p.parts or p.parts[:1] != ("runs",):
                raise ValueError("Use repository-relative evaluation/training outputs under runs/")
        return self


def prepare_development(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    training = GroupedTrainingConfig()
    inputs = root / training.inputs
    if file_hash(inputs / "input_manifest.json") != training.input_manifest_sha256:
        raise ValueError("Collected input manifest changed")
    artifacts = json.loads((inputs / "input_manifest.json").read_text())["artifacts"]
    checked_artifacts(inputs, {n: artifacts[n] for n in ["collected/development.jsonl", "row_links.jsonl"]})
    split_path = root / "configs/head_finetuning/frozen_split_v1.json"
    split = json.loads(split_path.read_text())
    source = root / split["artifact_directory"]
    for name in ["development.jsonl", "provenance.parquet"]:
        if file_hash(source / name) != split["artifact_fingerprints"][name]:
            raise ValueError("Original frozen development artifact changed")
    frame = pd.read_parquet(source / "provenance.parquet", columns=["role", "pool", "exported", "jsonl_line", "benchmark", "subject_id", "item_id"],
                            filters=[("role", "==", "development"), ("pool", "==", "evaluation"), ("exported", "==", True)])
    frame = frame.sort_values("jsonl_line").reset_index(drop=True)
    rows = list(read_jsonl(inputs / "collected/development.jsonl"))
    originals = list(read_jsonl(source / "development.jsonl"))
    if len(rows) != 963 or len(originals) != len(rows) or len(frame) != len(rows) or frame.jsonl_line.tolist() != list(range(len(rows))):
        raise ValueError("Original development response alignment differs")
    if not set(frame.benchmark) <= set(split["roles"]["development"]):
        raise ValueError("Scoring inputs escaped original development families")
    if len(frame[["benchmark", "subject_id", "item_id"]].drop_duplicates()) != 960 or len(frame[["benchmark", "subject_id"]].drop_duplicates()) != 8:
        raise ValueError("Original development scored-item universe differs")
    requests, scoring = [], []
    for i, (row, original, meta) in enumerate(zip(rows, originals, frame.to_dict("records"), strict=True)):
        if set(row) != {"state", "questions"} or row["questions"] != load_questions() or original["questions"] != load_questions():
            raise ValueError("Development input contains targets or changed questions")
        state, native = row["state"], json.loads(original["state"])
        subject, item = normalize_fields(state["target_system_recorded"], dict(state["target_task"], benchmark_id=""))
        old_subject, old_item = normalize_fields(native["target_system"], native["target_item"])
        if subject != old_subject or any(item[k] != old_item[k] for k in ["item_content", "item_features", "interactors"]):
            raise ValueError("Collected inputs differ from original frozen subject/task")
        label = original["expected"]["correct"]
        if type(label) not in {int, bool} or label not in (0, 1):
            raise ValueError("Original native development label is not binary")
        requests.append({"subject": subject, "item": item})
        scoring.append(dict(meta, row=i, label=int(label)))
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "requests.jsonl", requests)
    write_jsonl(out / "scoring.jsonl", scoring)
    write_json(out / "inference_manifest.json", {"version": "grouped_development_requests_v1", "responses": len(rows),
        "input_manifest_sha256": training.input_manifest_sha256, "contains_labels": False,
        "artifacts": {"requests.jsonl": file_hash(out / "requests.jsonl")}})
    receipt = {"version": "grouped_development_scoring_v1", "n_responses": len(rows), "n_items": 960, "n_pairs": 8,
        "frozen_split_sha256": file_hash(split_path), "development_source_sha256": file_hash(source / "development.jsonl"),
        "frozen_provenance_sha256": file_hash(source / "provenance.parquet"), "calibration_opened": False,
        "confirmation_opened": False, "supplemental_development_outcomes_used": False,
        "inference_inputs_have_labels": False, "native_labels_preserved": True,
        "artifacts": {n: file_hash(out / n) for n in ["requests.jsonl", "scoring.jsonl", "inference_manifest.json"]}}
    write_json(out / "scoring_manifest.json", receipt)
    return receipt


def verify_requests(root, config):
    path = local_path(root, config.prepared)
    manifest = json.loads((path / "inference_manifest.json").read_text())
    if manifest.get("version") != "grouped_development_requests_v1" or manifest.get("responses") != 963 or manifest.get("contains_labels") is not False:
        raise ValueError("Use the outcome-free original development request bundle")
    if set(manifest["artifacts"]) != {"requests.jsonl"}:
        raise ValueError("Only subject/item requests belong in the inference bundle")
    checked_artifacts(path, manifest["artifacts"])
    count = 0
    for row in read_jsonl(path / "requests.jsonl"):
        if set(row) != {"subject", "item"}:
            raise ValueError("Predictor requests may contain only published subject/item fields")
        normalize_fields(row["subject"], row["item"])
        count += 1
    if count != manifest["responses"]:
        raise ValueError("Development request count differs")
    return path, manifest


def predict_development(root, config, out, *, storage=Path("/jobs")):
    import torch
    from .grouped_predictor import make_grouped_laya_predictor
    config = GroupedEvaluationConfig.model_validate(config)
    prepared, request_manifest = verify_requests(root, config)
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen-development inference uses explicit qualified Torch/CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    base = Path(storage).resolve() / config.training_job_id / "work" / config.training_output
    if not base.resolve().is_relative_to(Path(storage).resolve() / config.training_job_id):
        raise ValueError("Checkpoint path escaped the declared training job")
    report = json.loads((base / "report.json").read_text())
    if report.get("status") != "finished" or report.get("updates") != 543 or report.get("original_examples") != 34718 or not report.get("training_inference_token_parity"):
        raise ValueError("Declared grouped training and reload verification are incomplete")
    checkpoint = base / "final"
    if file_hash(checkpoint / "paiec_training.json") != config.checkpoint_manifest_sha256:
        raise ValueError("Declared fitted checkpoint manifest changed")
    verify_grouped_checkpoint(checkpoint)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    predictor = make_grouped_laya_predictor(checkpoint_path=checkpoint, device="cuda", dtype="float16")
    count = 0
    try:
        with (out / "predictions.partial.jsonl").open("x") as stream:
            for i, row in enumerate(read_jsonl(prepared / "requests.jsonl")):
                if set(row) != {"subject", "item"}:
                    raise ValueError("Evaluation predictor input contains extra metadata or outcomes")
                p = predictor(row["subject"], row["item"])
                stream.write(json.dumps({"row": i, "probability": p}) + "\n")
                count += 1
                if count % 100 == 0:
                    print(json.dumps({"predicted_responses": count, "forward_groups": predictor.forward_groups}), flush=True)
    finally:
        predictor.close()
    if count != request_manifest["responses"]:
        raise ValueError("Development inference response count differs")
    (out / "predictions.partial.jsonl").rename(out / "predictions.jsonl")
    result = {"version": "grouped_development_inference_v1", "status": "finished", "responses": count,
        "forward_groups": predictor.forward_groups, "forward_views": predictor.forward_items,
        "checkpoint_manifest_sha256": config.checkpoint_manifest_sha256,
        "request_manifest_sha256": file_hash(prepared / "inference_manifest.json"), "labels_available_to_inference": False,
        "calibration_opened": False, "confirmation_opened": False, "probability": "raw_grouped_softmax"}
    write_json(out / "report.json", result)
    return result


def ranking_and_calibration(frame):
    # Mann–Whitney AUROC with average ranks handles ties without another dependency.
    values = []
    reliability = []
    for (benchmark, subject), group in frame.groupby(["benchmark", "subject_id"], sort=True):
        y = group.label.to_numpy(dtype=int)
        p = group.probability.to_numpy(dtype=float)
        positive, negative = int(y.sum()), int((1 - y).sum())
        ranks = pd.Series(p).rank(method="average").to_numpy()
        auc = float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)) if positive and negative else None
        values.append({"benchmark": benchmark, "subject_id": subject, "auroc": auc,
            "responses": len(group), "mean_probability": float(p.mean()), "observed_rate": float(y.mean())})
    bins = np.minimum((frame.probability.to_numpy() * 10).astype(int), 9)
    for b in sorted(set(bins)):
        g = frame.loc[bins == b]
        reliability.append({"bin": int(b), "responses": len(g), "mean_probability": float(g.probability.mean()),
            "observed_rate": float(g.label.mean())})
    aucs = [r["auroc"] for r in values if r["auroc"] is not None]
    return {"mean_within_pair_auroc": float(np.mean(aucs)) if aucs else None, "pair_diagnostics": values,
        "reliability_bins_pooled_diagnostic": reliability, "calibration_fitted": False}


def score_development(prepared, prediction_dir, out, *, checkpoint_manifest_sha256):
    prepared, prediction_dir, out = Path(prepared), Path(prediction_dir), Path(out)
    manifest = json.loads((prepared / "scoring_manifest.json").read_text())
    checked_artifacts(prepared, manifest["artifacts"])
    inference = json.loads((prediction_dir / "report.json").read_text())
    if (inference.get("status") != "finished" or inference["checkpoint_manifest_sha256"] != checkpoint_manifest_sha256 or
            inference["request_manifest_sha256"] != file_hash(prepared / "inference_manifest.json") or inference["labels_available_to_inference"]):
        raise ValueError("Predictions differ from the declared checkpoint/outcome-free request bundle")
    records = list(read_jsonl(prepared / "scoring.jsonl"))
    predictions = list(read_jsonl(prediction_dir / "predictions.jsonl"))
    if len(records) != 963 or len(predictions) != len(records) or [r["row"] for r in predictions] != list(range(len(records))):
        raise ValueError("Prediction/response order or counts differ")
    frame = pd.DataFrame(records)
    if frame.row.tolist() != list(range(len(frame))):
        raise ValueError("Frozen scoring response order differs")
    probabilities = [r["probability"] for r in predictions]
    result, saved_predictions = score_frozen_development(frame, frame.label.tolist(), probabilities)
    constant, _ = score_frozen_development(frame, frame.label.tolist(), [.5] * len(frame))
    frame["probability"] = probabilities
    diagnostics = ranking_and_calibration(frame)
    out.mkdir(parents=True, exist_ok=False)
    result.frame.to_csv(out / "scores.csv", index=False)
    saved_predictions.to_csv(out / "predictions.csv", index=False)
    report = {"version": "grouped_frozen_development_result_v1", "status": "finished", "brier_alc": result.brier_alc,
        "pair_weighted_ece": result.ece, "constant_brier_alc": constant.brier_alc,
        "brier_alc_delta_vs_constant": result.brier_alc - constant.brier_alc, **diagnostics,
        "n_responses": len(frame), "n_eval_inputs": 960, "n_pairs": 8, "inference": inference,
        "predictions_sha256": file_hash(prediction_dir / "predictions.jsonl"), "scoring_manifest_sha256": file_hash(prepared / "scoring_manifest.json"),
        "revealed_evidence_used": False, "predictions_shared_across_budgets": True, "calibration_opened": False,
        "confirmation_opened": False, "supplemental_development_outcomes_used": False,
        "enrichment_effect_isolated": False, "source_policy": "research_only"}
    write_json(out / "report.json", report)
    return report
