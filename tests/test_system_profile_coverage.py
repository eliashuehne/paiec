import importlib.util
import json
from pathlib import Path
import unittest

from paiec.profile_enrichment import ProfileEnricher
from paiec.schema import SUBJECT_FIELDS
from paiec.system_profiles import load_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "research/system_profiles/collection_v2"
POLICY = ROOT / "configs/head_finetuning/enrichment_catalog_v2.json"
spec = importlib.util.spec_from_file_location("coverage_audit", ROOT / "scripts/audit_system_profile_coverage.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def enricher():
    return ProfileEnricher(POLICY, root=ROOT, token_count=lambda text: len(text.split()), state_room=10000)


class ExpandedCollectionTests(unittest.TestCase):
    def test_exact_approved_scope_and_truthful_coverage(self):
        rows = audit.read_jsonl(CATALOG / "subjects.jsonl")
        self.assertEqual(len(rows), 178)
        self.assertEqual(sum(row["role"] == "training" for row in rows), 170)
        self.assertEqual(sum(row["role"] == "development" for row in rows), 8)
        self.assertEqual(len({(row["benchmark"], row["subject_id"]) for row in rows}), 178)
        for row in rows:
            self.assertEqual(set(row["subject"]), set(SUBJECT_FIELDS))
            self.assertNotIn(row["benchmark"], {"afrieval", "causal_axioms", "bertaqa", "evouna"})
            self.assertFalse(row["exact_evaluated_checkpoint_verified"])
        coverage = audit.subject_coverage(rows, enricher())
        counts = audit.totals(coverage)
        self.assertEqual(counts["model_facts"], 169)
        self.assertEqual(counts["harness_facts"], 171)
        self.assertEqual(counts["both_model_and_harness_facts"], 162)
        self.assertEqual(counts["any_sourced_facts"], 178)
        gaps = audit.missing_facts(rows, coverage)
        self.assertEqual(sum(gap["kind"] == "model" for gap in gaps), 9)
        self.assertEqual(sum(gap["kind"] == "harness" for gap in gaps), 7)

    def test_missing_identity_and_private_checkpoint_are_not_reidentified(self):
        e = enricher()
        for row in audit.read_jsonl(CATALOG / "subjects.jsonl"):
            if row["model_match_status"] == "unresolved_model":
                subject = dict(row["subject"], display_name="Alibaba Qwen 7B",
                               subject_id=row["subject_id"], benchmark=row["benchmark"])
                self.assertIsNone(e.resolve(subject)["model_profile_id"])
            if not row["subject"]["harness"]:
                self.assertIsNone(e.resolve(row["subject"])["harness_profile_id"])
        self.assertIsNone(e.resolve({"normalized_name": "OpenAI GPT-4o",
                                     "subject_features_extra": "source_model=unknown-private-model"})["model_profile_id"])

    def test_family_profiles_do_not_claim_specific_variant_settings(self):
        profiles, _ = load_catalog(CATALOG)
        profiles = {p.profile_id: p for p in profiles}
        for name in ["metamath-llama-family", "metamath-mistral-family"]:
            spec = profiles["model." + name].specifications
            self.assertIsNone(spec.architecture.total_parameters.value)
            self.assertIsNone(spec.architecture.layers.value)
            self.assertIsNone(spec.max_combined_context_tokens.value)
        llama = profiles["model.llama3-8b-family"].specifications
        self.assertIsNone(llama.model_type.value)
        historical = profiles["model.gpt3.5-turbo-family"].specifications
        self.assertIsNone(historical.max_combined_context_tokens.value)
        self.assertIsNone(historical.max_output_tokens.value)
        service = profiles["model.bedrock-mistral7b-family"].specifications
        self.assertIsNone(service.architecture.attention_types.value)
        self.assertIsNone(service.max_combined_context_tokens.value)

    def test_mistral_versions_remain_distinct_and_mirrors_are_excluded(self):
        e = enricher()
        matches = [e.resolve({"normalized_name": "Mistral 7B", "subject_features_extra": "source_model=" + value})["model_profile_id"]
                   for value in ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "mistral.mistral-7b-instruct-v0:2"]]
        self.assertEqual(len(set(matches)), 3)
        for profile in e.profiles.values():
            def facts(value):
                if isinstance(value, dict):
                    if "status" in value and "evidence" in value:
                        yield value
                    else:
                        for child in value.values():
                            yield from facts(child)
            for fact in facts(profile.model_dump()):
                for evidence in fact["evidence"]:
                    self.assertNotEqual(e.sources[evidence["source_id"]].eligibility, "not_permitted")
        for name in ["gpt5.4", "claude-opus4.7", "claude-sonnet4.6", "gemini3.1-pro-preview"]:
            self.assertIsNone(e.profiles["model." + name].specifications.architecture.total_parameters.value)


if __name__ == "__main__":
    unittest.main()
