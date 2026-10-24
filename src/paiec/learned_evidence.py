"""Small, explicitly fitted evidence correction; no native model dependency."""
import math

import numpy as np

from .evidence import evidence_counts

FEATURE_NAMES = ("query_logit", "evidence_logit", "reliability", "query_x_reliability",
                 "evidence_x_reliability", "log_count")


def features(probability, count, successes):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid query probability")
    if count < 0 or not 0 <= successes <= count:
        raise ValueError("Invalid revealed evidence")
    p = min(1-1e-6, max(1e-6, probability))
    query = math.log(p/(1-p))
    evidence = math.log((successes+1)/(count-successes+1))
    reliability = count/(count+5)
    return [query, evidence, reliability, query*reliability, evidence*reliability,
            min(1., math.log1p(count)/math.log(32))]


def corrected_probability(probability, count, successes, model):
    # Budget zero stays the exact trained-head probability, isolating adaptation.
    values = features(probability, count, successes)
    if not count:
        return probability
    if model["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Learned evidence feature contract differs")
    logit = float(np.dot(model["coefficients"], values) + model["intercept"])
    return float(1/(1+math.exp(-min(50., max(-50., logit)))))


class LearnedEvidencePredictor:
    def __init__(self, base, fitted):
        if base.use_labels:
            raise ValueError("Learned correction requires a base without text evidence")
        expected = fitted.get("checkpoint_identity")
        if expected and base.runtime_metadata.get("checkpoint_revision") != expected:
            raise ValueError("Learned evidence correction belongs to a different checkpoint")
        self.base, self.fitted = base, fitted
        self.runtime_metadata = {**base.runtime_metadata, "evidence_update": {
            "mode": "learned_logistic_v1", "scope": "matching allowed visible profile and anonymous benchmark",
            "fitting_role": "training-only monitor; main encoder did not train on these queries",
            "zero_budget_unchanged": True}}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def predict_many(self, subject, items, labeled):
        p = self.base.predict_many(subject, items, [])
        return [corrected_probability(value, *evidence_counts(subject, item, labeled), self.fitted)
                for value, item in zip(p, items)]

    def __call__(self, subject, item, labeled):
        return self.predict_many(subject, [item], labeled)[0]

    def predict(self, input, labeled):
        return self(*input, labeled)


def fit_from_training_monitor(prepared, arm, out, seed=20261023):
    """Fit only query-held-out head predictions within the original training role."""
    from collections import defaultdict
    import json
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from .competition import competition_input
    from .head_pilot import read_rows, write_json
    from .head_training import file_hash
    from .metrics import BUDGETS, alc_weights
    def scope(row):
        parsed = json.loads(row["state"])
        subject, item = competition_input(parsed["target_system"], parsed["target_item"])
        return json.dumps([subject, item["benchmark_id"]], sort_keys=True, ensure_ascii=False)
    source = {role: read_rows(prepared / "source" / (role + ".jsonl")) for role in ["training", "monitor"]}
    frame = pd.read_parquet(prepared / "enriched/provenance.parquet")
    monitor = frame[frame.role.eq("monitor")].sort_values("jsonl_line").reset_index(drop=True)
    prediction_path = arm / "monitor_predictions.csv"
    training_report = json.loads((arm / "report.json").read_text())
    if training_report["status"] != "finished":
        raise ValueError("Evidence fitting requires a completed base head")
    prediction = pd.read_csv(prediction_path)
    if prediction.row_id.tolist() != monitor.row_id.tolist():
        raise ValueError("Training-monitor prediction ordering changed")
    supports = defaultdict(list)
    for row in source["training"]:
        supports[scope(row)].append(int(row["expected"]["correct"]))
    if set(frame[frame.role.eq("training")].task_sha256) & set(monitor.task_sha256):
        raise ValueError("Episode query/support task overlap")
    rng = np.random.default_rng(seed)
    x, y, weights = [], [], []
    budget_weights = dict(zip(BUDGETS, alc_weights()))
    episodes = 20
    for row, record, p in zip(source["monitor"], monitor.to_dict("records"), prediction.probability):
        pool = supports[scope(row)]
        for _ in range(episodes):
            sampled = rng.permutation(pool)[:31]
            for budget in BUDGETS[1:]:
                n = min(budget, len(sampled))
                if not n:
                    continue
                x.append(features(float(p), n, int(sampled[:n].sum())))
                y.append(int(row["expected"]["correct"]))
                weights.append(record["sample_weight"]*budget_weights[budget]/episodes)
    if len(set(y)) < 2:
        raise ValueError("Evidence fitting needs both observed classes")
    # Fixed regularization: no tuning against original development labels.
    estimator = LogisticRegression(C=1., max_iter=1000, solver="lbfgs")
    estimator.fit(x, y, sample_weight=weights)
    model = dict(version="learned_evidence_v1", feature_names=list(FEATURE_NAMES),
        coefficients=estimator.coef_[0].tolist(), intercept=float(estimator.intercept_[0]),
        episodes_per_query=episodes, unique_query_records=len(monitor), expanded_episode_rows=len(y),
        regularization_C=1., source_role="training-only monitor", zero_budget_unchanged=True,
        checkpoint_identity=training_report["checkpoint_identity"],
        prepared_manifest_sha256=file_hash(prepared / "manifest.json"),
        monitor_predictions_sha256=file_hash(prediction_path),
        support_trial_selection="same one recorded trial per selected training system/item",
        note="Head query predictions are out of its fitting sample. Support counts use training outcomes. No development fitting.")
    write_json(out / "learned_evidence.json", model)
    return model
