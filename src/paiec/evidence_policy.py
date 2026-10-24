"""Portable, stateless evidence policy with explicit fitted hyperparameters."""
import hashlib
import json
import math
from pathlib import Path

from .competition_fields import matching_evidence


def posterior_mean(labels, strength, prior=.5):
    if (type(strength) not in {float, int} or not math.isfinite(strength) or strength < 0 or
        type(prior) not in {float, int} or not math.isfinite(prior) or not 0 <= prior <= 1 or
        any(type(y) not in {bool, int} or y not in (0, 1) for y in labels)):
        raise ValueError("Invalid native evidence or prior")
    return float((sum(labels) + strength * prior) / (len(labels) + strength)) if labels else float(prior)


class EvidencePolicy:
    """Return a probability using only this call's compatible revealed outcomes."""
    name = "evidence_policy"
    use_labels = False

    def __init__(self, state):
        if (state.get("version") != "evidence_policy_v1" or state.get("source_policy") != "provided_dataset_only" or
            set(state) - {"version", "source_policy", "prior", "strength", "training_provenance"}):
            raise ValueError("Unknown evidence policy/source contract")
        posterior_mean([], state.get("strength"), state.get("prior"))
        self.strength, self.prior = float(state["strength"]), float(state["prior"])
        self.runtime_metadata = {"mode": "stateless_beta_mean", "source_policy": "provided_dataset_only",
                                 "strength": self.strength, "prior": self.prior}

    def __call__(self, subject, item, labeled):
        return posterior_mean([label for _, label in matching_evidence(subject, item, labeled)], self.strength, self.prior)

    def predict(self, input, labeled=None):
        return self(*input, labeled)

    @classmethod
    def load(cls, path, *, expected_sha256):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("Evidence policy artifact changed")
        return cls(json.loads(data))
