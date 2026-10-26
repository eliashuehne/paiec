"""Tests for the documented profile feature block and its release gate."""
import json
from pathlib import Path
import tempfile
import unittest

from paiec.catboost_features import PROFILE_COLUMNS, feature_names, feature_record
from paiec.catboost_gate import assess_research_only_catboost_base
from paiec.system_profile_features import ProfileFacts

SUBJECT = {"normalized_name": "Test A", "provider": "TestCo", "release_date": "2024-01-01",
           "access_date": "", "harness": "TestH", "harness_version": "", "reasoning_effort": "",
           "subject_features_extra": ""}
ITEM = {"item_content": "What is 2 + 2?", "item_features": "", "interactors": "",
        "benchmark_id": "benchmark_ffff"}


def fact(value, source="src-ok"):
    return {"value": value, "status": "documented",
            "evidence": [{"source_id": source, "locator": "json:field", "excerpt": "{...}"}],
            "note": "Published configuration."}


def write_catalog(root):
    """A two-profile catalog: one permitted architecture fact, one barred fact."""
    root.mkdir(parents=True)
    sources = [
        {"source_id": "src-ok", "source_type": "model_config", "location": "https://example.test/a",
         "retrieved_on": "2026-10-01", "eligibility": "unreviewed", "eligibility_note": "External."},
        {"source_id": "src-barred", "source_type": "model_card", "location": "https://example.test/b",
         "retrieved_on": "2026-10-01", "eligibility": "not_permitted", "eligibility_note": "Terms."}]
    model = {"schema_version": "1.0", "profile_id": "model.test-a", "entity_type": "model",
             "identity": {}, "specifications": {"architecture": {
                 "family": fact("llama"), "total_parameters": fact(7_000_000_000),
                 "layers": fact(32), "hidden_size": fact(4096, "src-barred")},
                 "model_type": fact("instruction-tuned"),
                 "input_modalities": fact(["text"])}}
    harness = {"schema_version": "1.0", "profile_id": "harness.test-h", "entity_type": "harness",
               "identity": {}, "specifications": {"workflow": fact(["few-shot inference"])}}
    (root / "sources.jsonl").write_text("\n".join(json.dumps(item) for item in sources) + "\n")
    (root / "profiles.jsonl").write_text("\n".join(json.dumps(item) for item in (model, harness)) + "\n")
    policy = root / "policy.json"
    policy.write_text(json.dumps({
        "version": "enriched_v1", "usage": "research_only", "catalog": str(root),
        "max_added_tokens": 96, "selected_model_fields": ["architecture.family"],
        "models": [{"profile_id": "model.test-a", "identifiers": ["Test A"]}],
        "harnesses": [{"profile_id": "harness.test-h", "identifiers": ["TestH"]}]}))
    return policy


class ProfileFactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "catalog"
        self.policy = write_catalog(self.root)
        self.facts = ProfileFacts(self.policy, root=Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_record_matches_the_declared_columns(self):
        record = self.facts.record(SUBJECT)
        self.assertEqual(tuple(record), PROFILE_COLUMNS)
        self.assertEqual(record["profile_model_found"], 1.0)
        self.assertEqual(record["profile_harness_found"], 1.0)
        self.assertEqual(record["profile_layers"], 32.0)
        self.assertEqual(record["profile_family"], "llama")
        self.assertEqual(record["profile_model_type"], "instruction-tuned")
        self.assertEqual(record["profile_input_modalities"], "text")
        self.assertEqual(record["profile_workflow"], "few-shot inference")
        self.assertGreater(record["profile_total_parameters_log"], 0.0)

    def test_barred_evidence_is_treated_as_unknown(self):
        record = self.facts.record(SUBJECT)
        self.assertEqual(record["profile_hidden_size_log"], 0.0)
        # total, layers and active/expert/head facts: only two are documented here.
        self.assertEqual(record["profile_documented_numeric_facts"], 2.0)

    def test_unresolved_harness_reports_explicit_missingness(self):
        record = self.facts.record(dict(SUBJECT, harness="Unknown Harness"))
        self.assertEqual(record["profile_harness_found"], 0.0)
        self.assertEqual(record["profile_workflow"], "__missing__")
        self.assertEqual(record["profile_model_found"], 1.0)

    def test_conflicting_identifiers_fail_closed(self):
        subject = dict(SUBJECT, normalized_name="Test A", subject_features_extra="model_identifier=other/model")
        record = self.facts.record(subject)
        self.assertEqual(record["profile_model_found"], 0.0)
        self.assertEqual(record["profile_family"], "__missing__")

    def test_columns_appear_only_in_the_declared_group(self):
        self.assertNotIn("profile_family", feature_names(("task_structure",)))
        self.assertIn("profile_family", feature_names(("task_structure", "system_profile")))
        record = feature_record(SUBJECT, ITEM, self.facts.record(SUBJECT))
        self.assertEqual(record["profile_family"], "llama")
        with self.assertRaises(ValueError):
            feature_record(SUBJECT, ITEM, {"profile_family": "llama"})


class ResearchGateTests(unittest.TestCase):
    def report(self, cold, alc):
        return {"source_policy": "provided_plus_research_profiles", "training_families": 7,
                "original_development_outcomes_used": False, "monitor_outcomes_used": False,
                "sealed_roles_opened": False, "visible_scopes": 170,
                "full_budget_eligible_scopes": 170, "excluded_small_scopes": 0,
                "zero_budget_brier_all_held_rows_equal_scope": {"constant": .25, "catboost": cold},
                "brier_alc_on_common_eligible_scopes": {"constant": .25, "beta_k5": .21,
                                                        "catboost/none": cold, "catboost/logit_shift": alc}}

    def test_external_profiles_bar_release_but_keep_the_measurement(self):
        strong = assess_research_only_catboost_base(self.report(.20, .15))
        self.assertTrue(strong["base_predictor_out_of_family_cold_start_beats_constant"])
        self.assertFalse(strong["release_eligible"])
        self.assertFalse(strong["any_adaptation_candidate_passed_initial_checks"])
        candidate = strong["candidates"]["catboost/logit_shift"]
        self.assertTrue(candidate["predictive_checks_passed"])
        self.assertEqual(candidate["reasons"], ["external_research_profiles_not_competition_eligible"])
        weak = assess_research_only_catboost_base(self.report(.26, .22))
        self.assertFalse(weak["candidates"]["catboost/logit_shift"]["predictive_checks_passed"])

    def test_dataset_only_policy_is_not_assessed_as_research(self):
        report = self.report(.20, .15)
        report["source_policy"] = "provided_dataset_only"
        with self.assertRaises(ValueError):
            assess_research_only_catboost_base(report)


if __name__ == "__main__":
    unittest.main()
