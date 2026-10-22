"""Adapter checks use a fake router; no download or torch is needed."""

import json
import unittest
from unittest.mock import patch

from paiec.laya_predictor import LayaPredictor, build_state

SUBJECT = {"normalized_name": "test-model", "provider": "test-provider"}
ITEM = {"item_content": "What is 2 + 2?", "benchmark_id": "private-name"}


class FakeRouter:
    def __init__(self, value=0.7):
        self.requests = []
        self.value = value

    def predict_batch(self, requests, batch_size=None):
        self.requests.extend(requests)
        return [{"answers": {"correct": {"noul": self.value}}, "usage": {"truncated": True, "state_tokens_dropped": 2}} for _ in requests]


class LayaTests(unittest.TestCase):
    def test_state_has_profile_and_item_but_no_benchmark_name(self):
        state = build_state(SUBJECT, ITEM, [])
        self.assertIn("test-model", state)
        self.assertIn("What is 2 + 2?", state)
        self.assertNotIn("private-name", state)

    def test_zero_shot_ignores_labels_and_reuses_predictions_across_budgets(self):
        router = FakeRouter()
        predictor = LayaPredictor(router=router)
        self.assertEqual(predictor(SUBJECT, ITEM, []), 0.7)
        evidence = [[[SUBJECT, dict(ITEM, item_content="Another question")], 1]]
        self.assertEqual(predictor(SUBJECT, ITEM, evidence), 0.7)
        self.assertEqual(len(router.requests), 1)
        self.assertEqual(predictor.truncated_items, 1)
        self.assertEqual(predictor.state_tokens_dropped, 2)

    def test_optional_evidence_uses_only_matching_pair(self):
        evidence = [
            [[SUBJECT, dict(ITEM, item_content="matching")], 1],
            [[SUBJECT, dict(ITEM, benchmark_id="other", item_content="wrong benchmark")], 0],
            [[{"normalized_name": "other"}, dict(ITEM, item_content="wrong subject")], 0],
        ]
        state = build_state(SUBJECT, ITEM, evidence, use_labels=True)
        parsed = json.loads(state)
        self.assertEqual(parsed["revealed_examples"][0]["outcome"], 1)
        self.assertEqual(parsed["revealed_examples"][0]["item"]["item_content"], "matching")
        self.assertNotIn("wrong benchmark", state)
        self.assertNotIn("wrong subject", state)

    def test_batch_preserves_order_and_deduplicates_identical_states(self):
        router = FakeRouter()
        predictor = LayaPredictor(router=router, model="multilingual", max_len=1024, batch_size=2)
        other = dict(ITEM, item_content="other item")
        self.assertEqual(predictor.predict_many(SUBJECT, [ITEM, other, ITEM], []), [0.7] * 3)
        self.assertEqual(len(router.requests), 2)
        self.assertEqual(router.requests[0]["model"], "multilingual")
        self.assertEqual(router.requests[0]["max_len"], 1024)

    def test_invalid_probabilities_fail_instead_of_silently_becoming_scores(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LayaPredictor(router=FakeRouter(value))(SUBJECT, ITEM, [])

    def test_unknown_benchmarks_use_same_question_without_registry_access(self):
        router = FakeRouter()
        with patch("paiec.benchmark_registry.load_benchmark_definition", side_effect=AssertionError("registry read")):
            predictor = LayaPredictor(router=router)
            predictor.predict([SUBJECT, ITEM], [])
            predictor.predict([SUBJECT, dict(ITEM, benchmark_id="never-seen-benchmark")], [])
        self.assertEqual(len(router.requests), 1)
        self.assertNotIn("criteria", router.requests[0]["questions"]["correct"])
        self.assertNotIn("benchmark_id", router.requests[0]["state"])
        self.assertNotIn("private-name", router.requests[0]["state"])
        self.assertNotIn("never-seen-benchmark", router.requests[0]["state"])

    def test_hidden_fields_and_display_name_never_reach_router(self):
        router = FakeRouter()
        predictor = LayaPredictor(router=router, use_labels=True)
        subject = {"display_name": "SECRET_NAME", "normalized_name": None}
        item = dict(ITEM, grading_criterion="SECRET_GRADER", reference_answer="SECRET_ANSWER", response=1)
        evidence = [[[subject, dict(item, item_content="observed", item_features="visible features", interactors="visible tool")], 0]]
        predictor.predict([subject, item], evidence)
        state = json.loads(router.requests[0]["state"])
        self.assertEqual(state["target_system"], {})
        self.assertEqual(state["revealed_examples"][0]["item"]["item_features"], "visible features")
        self.assertEqual(state["revealed_examples"][0]["item"]["interactors"], "visible tool")
        for secret in ("SECRET_NAME", "SECRET_GRADER", "SECRET_ANSWER", '"response"'):
            self.assertNotIn(secret, router.requests[0]["state"])
        self.assertNotIn("benchmark_id", router.requests[0]["state"])
        self.assertNotIn("subject", state["revealed_examples"][0])

    def test_changed_revealed_evidence_changes_input_and_cache(self):
        router = FakeRouter()
        predictor = LayaPredictor(router=router, use_labels=True)
        predictor(SUBJECT, ITEM, [])
        predictor(SUBJECT, ITEM, [[[SUBJECT, dict(ITEM, item_content="revealed")], 1]])
        self.assertEqual(len(router.requests), 2)

    def test_state_omits_empty_fields_but_retains_available_task_information(self):
        state = json.loads(build_state(dict(SUBJECT, harness="test-harness", reasoning_effort=""),
                                       dict(ITEM, item_features="options A/B", interactors="tool"), []))
        self.assertEqual(state["target_system"]["harness"], "test-harness")
        self.assertNotIn("reasoning_effort", state["target_system"])
        self.assertEqual(state["target_item"], {"item_content": ITEM["item_content"],
                                              "item_features": "options A/B", "interactors": "tool"})

    def test_relabeling_anonymous_ids_preserves_text_and_matching_evidence(self):
        def state(benchmark):
            target = dict(ITEM, benchmark_id=benchmark)
            observed = dict(target, item_content="observed task")
            return build_state(SUBJECT, target, [[[SUBJECT, observed], 1]], use_labels=True)
        self.assertEqual(state("benchmark_123"), state("benchmark_456"))
        self.assertIn("observed task", state("benchmark_123"))

    def test_zero_budget_state_is_identical_in_both_evidence_modes(self):
        self.assertEqual(build_state(SUBJECT, ITEM, [], use_labels=True),
                         build_state(SUBJECT, ITEM, [], use_labels=False))


if __name__ == "__main__":
    unittest.main()
