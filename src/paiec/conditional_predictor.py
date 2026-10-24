"""Portable conditional predictor and independent budget-limited adaptation.

Runtime imports only the Python standard library and the supplied field encoder.
No fitting, downloads, lookup catalog, native models or shared label state.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from .conditional_features import VERSION, DIMS, checked_fields, feature_vector
from .competition_fields import matching_evidence


def logistic(value):
    return 1. / (1. + math.exp(-max(-50., min(50., value))))


def logit(value):
    value = max(.001, min(.999, value))
    return math.log(value / (1. - value))


def evidence_logit_offset(support_probabilities, labels, regularization=1.):
    if len(labels) != len(support_probabilities) or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Invalid offset update inputs")
    if any(type(y) not in {int, bool} or y not in (0, 1) for y in labels):
        raise ValueError("Revealed labels must be binary")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in support_probabilities):
        raise ValueError("Invalid base probability")
    if not labels:
        return 0.
    offsets = [logit(p) for p in support_probabilities]
    delta = 0.
    for _ in range(30):
        values = [logistic(p + delta) for p in offsets]
        gradient = sum(p - y for p, y in zip(values, labels)) + regularization * delta
        hessian = sum(p * (1 - p) for p in values) + regularization
        updated = max(-8., min(8., delta - gradient / hessian))
        if abs(updated - delta) < 1e-10:
            delta = updated
            break
        delta = updated
    return delta


def logit_shift(probability, support_probabilities, labels, regularization=1.):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid base probability")
    delta = evidence_logit_offset(support_probabilities, labels, regularization)
    return logistic(logit(probability) + delta) if labels else probability


class ConditionalPredictor:
    name = "conditional_ridge"
    use_labels = False

    def __init__(self, state, *, evidence_mode="none", prior_strength=5., offset_regularization=1.):
        if (state.get("version") != "conditional_ridge_v1" or state.get("feature_version") != VERSION or
            state.get("source_policy") != "provided_dataset_only" or state.get("dimensions") != DIMS or
            type(state.get("interactions")) is not bool):
            raise ValueError("Fitted state uses another feature or source contract")
        if any(not isinstance(state.get(k), list) or len(state[k]) != DIMS or not all(type(v) in {int, float} and math.isfinite(v) for v in state[k])
               for k in ["center", "weights"]):
            raise ValueError("Invalid fitted vector")
        if type(state.get("intercept")) not in {int, float} or not math.isfinite(state["intercept"]):
            raise ValueError("Invalid fitted intercept")
        if evidence_mode not in {"none", "beta_mean", "logit_shift"}:
            raise ValueError("Unknown evidence mode")
        if not all(math.isfinite(v) and v > 0 for v in [prior_strength, offset_regularization]):
            raise ValueError("Evidence regularization must be positive and finite")
        self.state = json.loads(json.dumps(state))
        self.evidence_mode, self.prior_strength = evidence_mode, prior_strength
        self.offset_regularization = offset_regularization
        self.runtime_metadata = {"feature_version": VERSION, "source_policy": "provided_dataset_only",
                                 "evidence_mode": evidence_mode, "interactions": state["interactions"]}
        self._base_cached = lru_cache(maxsize=512)(self._base_text)

    def _base_text(self, subject_text, item_text):
        subject, item = json.loads(subject_text), json.loads(item_text)
        vector = feature_vector(subject, item, interactions=self.state["interactions"])
        value = self.state["intercept"] + sum((x - c) * w for x, c, w in zip(vector, self.state["center"], self.state["weights"]))
        return float(max(.001, min(.999, value)))

    def base_probability(self, subject, item):
        checked_fields(subject, item)
        # Anonymous IDs scope evidence but do not enter base features or cache keys.
        item = dict(item, benchmark_id="")
        return self._base_cached(json.dumps(subject, sort_keys=True, ensure_ascii=False),
                                 json.dumps(item, sort_keys=True, ensure_ascii=False))

    def __call__(self, subject, item, labeled):
        probability = self.base_probability(subject, item)
        if self.evidence_mode == "none":
            return probability
        evidence = matching_evidence(subject, item, labeled)
        labels = [label for _, label in evidence]
        if not labels:
            return probability
        if self.evidence_mode == "beta_mean":
            return float((self.prior_strength * probability + sum(labels)) / (self.prior_strength + len(labels)))
        support_probabilities = [self.base_probability(subject, observed) for observed, _ in evidence]
        return logit_shift(probability, support_probabilities, labels, self.offset_regularization)

    def predict(self, input, labeled):
        return self(*input, labeled)

    def predict_many(self, subject, items, labeled):
        return [self(subject, item, labeled) for item in items]

    @classmethod
    def load(cls, path, *, expected_sha256, **kwargs):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("Fitted artifact hash changed")
        return cls(json.loads(data), **kwargs)
