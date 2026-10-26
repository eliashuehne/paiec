import importlib
import unittest


class EnrichmentTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("paiec.feature_foundation.enrichment")
        except ModuleNotFoundError:
            self.fail("Full enrichment projection is missing")

    def fact(self, **kwargs):
        return dict(value=["en"], status="documented", evidence=[{"source_id": "source", "locator": "line1", "excerpt": "English"}], **kwargs)

    def test_known_functional_fact_keeps_research_and_historical_qualifiers(self):
        r = self.module().project_fact("documented_languages", self.fact(), "normalized_name", {"source": {"eligibility": "unreviewed"}})
        self.assertEqual(r["value_json"], '["en"]')
        self.assertEqual(r["eligibility"], "research_only")
        self.assertEqual(r["historical_applicability"], "unverified")

    def test_base_family_does_not_certify_target_languages(self):
        r = self.module().project_fact("documented_languages", self.fact(), "base_model", {"source": {"eligibility": "unreviewed"}})
        self.assertIsNone(r["value_json"])
        self.assertEqual(r["reason"], "base_only_nonarchitecture")

    def test_review_unknown_and_denied_source_are_withheld(self):
        m = self.module()
        r = m.project_fact("documented_languages", {"value": None, "status": "unknown", "evidence": [], "review_disposition": "identity_ambiguous"}, "normalized_name", {})
        self.assertEqual(r["status"], "unknown")
        denied = m.project_fact("documented_languages", self.fact(), "explicit", {"source": {"eligibility": "not_permitted"}})
        self.assertIsNone(denied["value_json"])
        with self.assertRaisesRegex(ValueError, "source"):
            m.project_fact("documented_languages", self.fact(), "explicit", {})

    def test_supplied_only_does_not_depend_on_match_signals(self):
        m = self.module()
        self.assertEqual(m.feature_projection([{"field": "language", "value_json": '["en"]'}], "supplied_only"), [])

