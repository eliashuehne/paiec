import unittest
from paiec.feature_collection import TaskAssessment


class AssessmentTests(unittest.TestCase):
    def test_run_recovery_uses_visible_identity_and_rejects_conflicts(self):
        from types import SimpleNamespace
        from paiec.feature_collection import recover_recorded_run_features
        from paiec.feature_preparation import stable_hash
        from paiec.competition import competition_input
        from paiec.system_profiles import Fact, Evidence
        subject = competition_input({"harness": "test"}, {})[0]
        key = stable_hash(subject)
        def profile(count, raw=subject):
            fact = Fact[int](value=count, status="recorded", evidence=[Evidence(
                source_id="fixture", locator="recorded field", excerpt=str(count))])
            return SimpleNamespace(entity_type="recorded_run", specifications=SimpleNamespace(
                raw_supplied_subject=raw, few_shot_count=fact))
        result = recover_recorded_run_features({key: {"recorded": subject}}, [profile(5)], ["few_shot_count"])
        self.assertEqual(result[0]["features"]["few_shot_count"]["value"], 5)
        result = recover_recorded_run_features({key: {"recorded": subject}}, [profile(5), profile(3)], ["few_shot_count"])
        self.assertIsNone(result[0]["features"]["few_shot_count"]["value"])
        result = recover_recorded_run_features({key: {"recorded": subject}}, [profile(5, {"harness": "different"})], ["few_shot_count"])
        self.assertEqual(result[0]["features"]["few_shot_count"]["status"], "unknown")

    def candidate(self):
        return {"task_key": "a" * 64, "claims": {"skills": {"values": ["addition"],
                "evidence": [{"start": 0, "end": 5, "quote": "2 + 2"}]}}}

    def test_evidence_is_bound_to_exact_task_and_span(self):
        assessment = TaskAssessment.model_validate(self.candidate())
        assessment.validate_against({"task_key": "a" * 64, "recorded": {"item_content": "2 + 2?"}})
        for key, text in [("b" * 64, "2 + 2?"), ("a" * 64, "3 + 3?")]:
            with self.assertRaises(ValueError):
                assessment.validate_against({"task_key": key, "recorded": {"item_content": text}})

    def test_difficulty_labels_and_review_claims_cannot_be_silently_added(self):
        for extra in [{"difficulty": .7}, {"expected": True}, {"review_status": "reviewed"}]:
            with self.assertRaises(ValueError):
                TaskAssessment.model_validate(dict(self.candidate(), **extra))
        assessment = TaskAssessment.model_validate(self.candidate())
        self.assertEqual(assessment.review_status, "candidate")
        self.assertIsNone(assessment.difficulty)
