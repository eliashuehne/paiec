"""Explicit, prediction-time adaptation from budget-permitted observations."""
import math

from .competition import competition_input, competition_labels


def evidence_counts(subject, item, labeled):
    subject, item = competition_input(subject, item)
    if not item["benchmark_id"]:
        return 0, 0
    matches = [label for (observed_subject, observed_item), label in competition_labels(labeled)
               if observed_subject == subject and observed_item["benchmark_id"] == item["benchmark_id"]
               and observed_item != item]
    return len(matches), sum(matches)


def update_probability(probability, count, successes, strength):
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("Evidence prior strength must be finite and positive")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid base probability")
    if count < 0 or not 0 <= successes <= count:
        raise ValueError("Invalid evidence counts")
    return (strength * probability + successes) / (strength + count)


class EvidencePredictor:
    """Leave checkpoint input/cache intact; apply an explicitly logged update.

    Identical visible profiles cannot be disambiguated. No private subject IDs
    are available here. Source evidence multiplicity is retained, not deduped
    by question text, since repeated trials are legitimate observations.
    """
    def __init__(self, base, strength=5.0):
        update_probability(.5, 0, 0, strength)
        if base.use_labels:
            raise ValueError("Use numeric evidence or text examples, not both")
        self.base, self.strength = base, strength
        self.runtime_metadata = {**base.runtime_metadata, "evidence_update": {
            "mode": "beta_mean", "prior_strength": strength, "fitted": False,
            "scope": "matching visible subject profile and anonymous benchmark"}}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def predict_many(self, subject, items, labeled):
        probabilities = self.base.predict_many(subject, items, [])
        return [update_probability(p, *evidence_counts(subject, item, labeled), self.strength)
                for p, item in zip(probabilities, items)]

    def __call__(self, subject, item, labeled):
        return self.predict_many(subject, [item], labeled)[0]

    def predict(self, input, labeled):
        return self(*input, labeled)
