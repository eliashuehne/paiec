import unittest
from paiec.feature_collection import TaskAssessment


class AssessmentTests(unittest.TestCase):
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
