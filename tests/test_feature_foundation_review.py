import importlib
import unittest


class ReviewTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("paiec.feature_foundation.review")
        except ModuleNotFoundError:
            self.fail("Review ledger validation is missing")

    def test_review_rejects_foreign_hash_unapproved_status_and_false_human_claim(self):
        m = self.module()
        case = {"case_id": "a", "input_hash": "hash", "kind": "task", "split": "training"}
        for change in [{"input_hash": "other"}, {"status": "pending"}, {"method": "human", "reviewer": "implementation_agent"}]:
            row = {**case, "status": "reviewed", "method": "agent_structural_and_taxonomy_audit", "reviewer": "implementation_agent",
                   "checks": ["raw_preservation"], "disposition": "accepted_observables", **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_reviews([case], [row])

    def test_nontraining_cases_and_review_duplicates_are_rejected(self):
        m = self.module()
        case = {"case_id": "a", "input_hash": "hash", "kind": "task", "split": "test"}
        with self.assertRaises(ValueError):
            m.validate_reviews([case], [])
