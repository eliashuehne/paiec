import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paiec.conditional_features import (SUBJECT_FIELDS, ITEM_FIELDS, VERSION, DIMS, NUMERIC_NAMES,
    feature_vector, numeric_features, task_features)
from paiec.conditional_predictor import ConditionalPredictor, matching_evidence, logit_shift, logit
from paiec.conditional_training import fit_conditional_ridge, predict_matrix
from paiec.schema import SUBJECT_FIELDS as OFFICIAL_SUBJECT, ITEM_FIELDS as OFFICIAL_ITEM
from paiec.conditional_analysis import family_training_mask
from paiec.conditional_gate import assess_conditional_controls


class ConditionalPredictorTests(unittest.TestCase):
    def fields(self, system="A", text="Question 猫"):
        return ({key: system if key == "normalized_name" else "" for key in SUBJECT_FIELDS},
                {key: text if key == "item_content" else "anonymous" if key == "benchmark_id" else "" for key in ITEM_FIELDS})

    def state(self):
        return {"version": "conditional_ridge_v1", "feature_version": VERSION, "source_policy": "provided_dataset_only",
                "dimensions": DIMS, "interactions": True, "center": [0.] * DIMS,
                "weights": [0.] * DIMS, "intercept": .5}

    def test_exact_fields_unicode_tail_and_anonymous_id_independence(self):
        self.assertEqual(SUBJECT_FIELDS, OFFICIAL_SUBJECT)
        self.assertEqual(ITEM_FIELDS, OFFICIAL_ITEM)
        subject, item = self.fields(text=" long prefix" * 2000 + "尾巴甲")
        self.assertEqual(feature_vector(subject, item), feature_vector(subject, dict(item, benchmark_id="different")))
        self.assertNotEqual(task_features(item), task_features(dict(item, item_content=item["item_content"][:-1] + "乙")))
        self.assertNotEqual(task_features(dict(item, item_content="None")), task_features(dict(item, item_content="")))
        with self.assertRaises(ValueError):
            feature_vector(dict(subject, display_name="PRIVATE"), item)

    def test_interactions_can_express_system_task_compatibility_beyond_additive_features(self):
        inputs, labels = [], []
        for system in ["A", "B"]:
            for task in ["algebra", "language"]:
                inputs.extend([self.fields(system, task)] * 10)
                labels.extend([int((system == "A") == (task == "algebra"))] * 10)
        matrix = np.asarray([feature_vector(*pair) for pair in inputs])
        additive = fit_conditional_ridge(matrix, labels, np.ones(len(labels)), interactions=False, l2=.001)
        conditional = fit_conditional_ridge(matrix, labels, np.ones(len(labels)), interactions=True, l2=.001)
        self.assertGreater(np.mean((predict_matrix(additive, matrix) - labels) ** 2), .2)
        self.assertLess(np.mean((predict_matrix(conditional, matrix) - labels) ** 2), .01)
        runtime = ConditionalPredictor(conditional)
        actual = [runtime(*pair, []) for pair in inputs]
        np.testing.assert_allclose(actual, predict_matrix(conditional, matrix), rtol=0, atol=1e-12)

    def test_scope_query_exclusion_and_repeated_trial_evidence(self):
        subject, item = self.fields()
        other_item = dict(item, item_content="Different observed question")
        evidence = [[[subject, other_item], 1], [[subject, other_item], 1],
                    [[subject, item], 0], [[dict(subject, harness="other"), other_item], 0],
                    [[subject, dict(other_item, benchmark_id="other")], 0]]
        self.assertEqual(len(matching_evidence(subject, item, evidence)), 2)
        predictor = ConditionalPredictor(self.state(), evidence_mode="beta_mean")
        self.assertEqual(predictor(subject, item, evidence), 4.5 / 7)
        self.assertEqual(predictor(subject, item, []), .5)
        self.assertEqual(predictor(subject, dict(item, benchmark_id=""), evidence), .5)

    def test_budget_order_and_unrelated_calls_do_not_leak_label_state(self):
        subject, item = self.fields()
        observed = dict(item, item_content="Observed")
        predictor = ConditionalPredictor(self.state(), evidence_mode="logit_shift")
        one = [[[subject, observed], 1]]
        first = predictor(subject, item, one)
        predictor(subject, item, [[[subject, observed], 0]] * 31)
        self.assertEqual(predictor(subject, item, []), .5)
        self.assertEqual(first, predictor(subject, item, one))

    def test_offset_preserves_item_discrimination_and_zero_budget(self):
        self.assertEqual(logit_shift(.8, [], []), .8)
        a = logit_shift(.2, [.5] * 7, [1] * 7)
        b = logit_shift(.8, [.5] * 7, [1] * 7)
        self.assertTrue(0 < a < b < 1)
        self.assertAlmostEqual(logit(b) - logit(a), logit(.8) - logit(.2), places=9)
        self.assertLess(logit_shift(.8, [.5], [0]), .8)

    def test_serialization_hash_source_contract_and_invalid_values(self):
        subject, item = self.fields()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(json.dumps(self.state()))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(ConditionalPredictor.load(path, expected_sha256=sha).predict([subject, item], []), .5)
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash"):
                ConditionalPredictor.load(path, expected_sha256=sha)
        for changed in [dict(self.state(), source_policy="web_enriched"), dict(self.state(), intercept=math.nan),
                        dict(self.state(), weights=[0.])]:
            with self.assertRaises(ValueError):
                ConditionalPredictor(changed)

    def test_nominal_capacity_does_not_invent_moe_active_parameters(self):
        subject, item = self.fields("Model 8x7B")
        values = dict(zip(NUMERIC_NAMES, numeric_features(subject, item)))
        self.assertEqual(values["nominal_dense_parameters_known"], 0.)
        huge, item = self.fields("Model " + "9" * 1000 + "B")
        self.assertTrue(all(math.isfinite(v) for v in feature_vector(huge, item)))

    def test_whole_family_and_shared_query_tasks_are_excluded_from_fitting(self):
        train, held = family_training_mask(["A", "A", "B", "B", "C"], ["q1", "q2", "q1", "q3", "q4"], "A")
        self.assertEqual(np.flatnonzero(held).tolist(), [0, 1])
        self.assertEqual(np.flatnonzero(train).tolist(), [3, 4])

    def test_failed_quality_or_incomplete_budget_coverage_prevents_release(self):
        report = {"source_policy": "provided_dataset_only", "training_families": 3,
            "original_development_outcomes_used": False, "monitor_outcomes_used": False, "sealed_roles_opened": False,
            "zero_budget_brier_all_held_rows_equal_scope": {"constant": .25, "conditional": .3},
            "brier_alc_on_common_eligible_scopes": {"constant": .25, "beta_k5": .21, "conditional/logit_shift": .24},
            "visible_scopes": 10, "full_budget_eligible_scopes": 5, "excluded_small_scopes": 5}
        assessment = assess_conditional_controls(report)
        self.assertFalse(assessment["any_candidate_passed_initial_checks"])
        self.assertEqual(len(assessment["candidates"]["conditional/logit_shift"]["reasons"]), 3)
        report["zero_budget_brier_all_held_rows_equal_scope"]["conditional"] = .23
        report["brier_alc_on_common_eligible_scopes"]["conditional/logit_shift"] = .2
        self.assertFalse(assess_conditional_controls(report)["any_candidate_passed_initial_checks"])
        report.update(full_budget_eligible_scopes=10, excluded_small_scopes=0)
        self.assertTrue(assess_conditional_controls(report)["any_candidate_passed_initial_checks"])
        with self.assertRaises(ValueError):
            assess_conditional_controls(dict(report, monitor_outcomes_used=True))
