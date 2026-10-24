import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from paiec.conditional_features import SUBJECT_FIELDS, ITEM_FIELDS
from paiec.evidence_policy import EvidencePolicy, posterior_mean
from paiec.evidence_policy_analysis import select_strength, STRENGTHS


class EvidencePolicyTests(unittest.TestCase):
    def fixture(self):
        subject = {k: "" for k in SUBJECT_FIELDS}
        item = {k: "" for k in ITEM_FIELDS}
        item.update(item_content="Target", benchmark_id="anonymous")
        state = {"version": "evidence_policy_v1", "source_policy": "provided_dataset_only", "prior": .5, "strength": 1.}
        return subject, item, state

    def test_policy_keeps_native_trials_and_ignores_wrong_scope_and_query(self):
        subject, item, state = self.fixture()
        observed = dict(item, item_content="Observed")
        labels = [[[subject, observed], 1], [[subject, observed], 1], [[subject, item], 0],
                  [[dict(subject, harness="OTHER"), observed], 0], [[subject, dict(observed, benchmark_id="other")], 0]]
        policy = EvidencePolicy(state)
        self.assertEqual(policy.predict([subject, item], labels), 2.5 / 3)
        self.assertEqual(policy.predict([subject, item], []), .5)
        self.assertEqual(policy.predict([subject, dict(item, benchmark_id="")], labels), .5)

    def test_budget_calls_are_independent_and_zero_strength_is_defined(self):
        subject, item, state = self.fixture()
        observed = dict(item, item_content="Observed")
        policy = EvidencePolicy(dict(state, strength=0.))
        for budget in [31, 0, 7, 1, 0]:
            labels = [[[subject, observed], 0]] * budget
            self.assertEqual(policy.predict([subject, item], labels), 0. if budget else .5)
        for labels, strength in [(["1"], 1.), ([2], 1.), ([1], -1.), ([1], float("nan"))]:
            with self.assertRaises(ValueError):
                posterior_mean(labels, strength)

    def test_held_family_scores_cannot_choose_its_own_strength(self):
        scores = []
        for family in ["A", "B", "C"]:
            for strength in STRENGTHS:
                scores.append({"family": family, "scope_key": family, "strength": strength,
                               "brier_alc": .1 if strength == 1. else .2})
        first, _ = select_strength(scores, excluded_family="A")
        for row in scores:
            if row["family"] == "A":
                row["brier_alc"] = .0 if row["strength"] == 20 else 1.
        second, _ = select_strength(scores, excluded_family="A")
        self.assertEqual(first, second)
        self.assertEqual(first, 1.)

    def test_serialized_policy_is_bound_to_hash_and_source_contract(self):
        subject, item, state = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(json.dumps(state))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(EvidencePolicy.load(path, expected_sha256=sha).predict([subject, item]), .5)
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed"):
                EvidencePolicy.load(path, expected_sha256=sha)
        with self.assertRaises(ValueError):
            EvidencePolicy(dict(state, source_policy="web_enriched"))
