import json
from pathlib import Path
import tempfile
import shutil
import unittest

from paiec.input_contract import resolve_input_contract
from paiec.head_training import file_hash
from paiec.laya_predictor import build_state, LayaPredictor
from paiec.profile_enrichment import ProfileEnricher, catalog_fingerprint

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/head_finetuning/enrichment_v1.json"
SUBJECT = {"normalized_name": "Mistral 7B", "subject_features_extra":
           "source_model=mistralai/Mistral-7B-Instruct-v0.1"}
ITEM = {"item_content": "Which option is correct? A: one B: two", "benchmark_id": "anonymous-id"}


def enrich(room=10000):
    return ProfileEnricher(POLICY, root=ROOT, token_count=lambda s: len(s.split()), state_room=room)


class EnrichmentTests(unittest.TestCase):
    def test_documented_allowlist_and_original_text_are_preserved(self):
        e = enrich()
        base = build_state(SUBJECT, ITEM, [], state_format="context_v1")
        state, audit = e.enrich(base, SUBJECT)
        self.assertTrue(state.startswith(base))
        self.assertIn("layers: 32", state)
        self.assertIn("sliding_window_attention", state)
        self.assertNotIn("anonymous-id", state)
        self.assertNotIn("https://", state)
        self.assertLessEqual(audit["added_tokens"], 96)
        self.assertTrue(audit["selected_facts"])
        self.assertTrue(all(fact["evidence"] for fact in audit["selected_facts"]))

    def test_missing_ambiguous_or_conflicting_models_are_not_guessed(self):
        for subject in [
            {"display_name": "Mistral 7B", "benchmark": "bird_sql", "subject_id": "known"},
            {"normalized_name": "Meta Llama 3 8B"},
            {"normalized_name": "OpenAI GPT-4o", "subject_features_extra": "model_identifier=unknown-private-model"},
            {"normalized_name": "OpenAI GPT-4o", "subject_features_extra": "source_model=mistralai/Mistral-7B-Instruct-v0.1"},
            {"subject_features_extra": "model_identifier=MetaMath-Mistral"},
            {"subject_features_extra": "source_model=gemini"},
        ]:
            state, audit = enrich().enrich("original task", subject)
            self.assertEqual(state, "original task")
            self.assertIsNone(audit["model_profile_id"])

    def test_no_task_tokens_are_displaced_and_facts_are_whole(self):
        e = enrich(room=3)
        state, audit = e.enrich("three original words", SUBJECT)
        self.assertEqual(state, "three original words")
        self.assertEqual(audit["added_tokens"], 0)
        e = enrich(room=25)
        state, audit = e.enrich("original task", SUBJECT)
        self.assertLessEqual(len(state.split()), 25)
        self.assertTrue(state.startswith("original task"))
        for fact in audit["selected_facts"]:
            self.assertIn(fact["line"], state)

    def test_current_api_defaults_and_unsupported_sources_do_not_enter_state(self):
        e = enrich()
        state, _ = e.enrich("task", {"normalized_name": "OpenAI GPT-4o"})
        self.assertIn("input_modalities: text, image", state)
        self.assertNotIn("max_output_tokens", state)
        self.assertNotIn("128000", state)
        self.assertNotIn("total_parameters", state)
        match = e.resolve(SUBJECT)["model_profile_id"]
        fact = e.profiles[match].specifications.architecture.layers
        for evidence in fact.evidence:
            e.sources[evidence.source_id].eligibility = "not_permitted"
        self.assertIsNone(e._fact_line(match, "architecture.layers", "layers"))

    def test_capabilities_are_distinguished_from_enabled_tools(self):
        state, _ = enrich().enrich("task", {"harness": "ERBench"})
        self.assertIn("not confirmed enabled", state)
        self.assertIn("optional retrieval", state)
        self.assertNotIn("enabled_tools", state)

    def test_system_ablation_does_not_reintroduce_profile_facts(self):
        state = build_state(SUBJECT, ITEM, [], state_format="enriched_v1", enricher=enrich(), omit_system=True)
        self.assertNotIn("Published model properties", state)
        self.assertNotIn("Mistral", state)
        self.assertIn(ITEM["item_content"], state)

    def test_base_model_architecture_does_not_become_actual_finetuned_model_type(self):
        subject = {"subject_features_extra": "base_model=Llama-3.1-8B-Instruct;method=TTT_10"}
        state, audit = enrich().enrich("task", subject)
        self.assertEqual(audit["model_identity_basis"], "base_model")
        self.assertIn("backbone: transformer", state)
        self.assertNotIn("model_type:", state)


class InputContractTests(unittest.TestCase):
    def bundle(self, root):
        directory = root / "profile_bundle"
        (directory / "catalog").mkdir(parents=True)
        for name in ["profiles.jsonl", "sources.jsonl"]:
            shutil.copyfile(ROOT / "research/system_profiles/collection_v1" / name, directory / "catalog" / name)
        policy = json.loads(POLICY.read_text())
        policy["catalog"] = "profile_bundle/catalog"
        (directory / "policy.json").write_text(json.dumps(policy))
        contract = dict(state_format="enriched_v1", max_len=1024, head_max_len=192,
                        question_sha256=file_hash(ROOT / "configs/laya_question.json"),
                        policy_file="profile_bundle/policy.json", policy_sha256=file_hash(directory / "policy.json"),
                        catalog_sha256=catalog_fingerprint(directory / "catalog"))
        (root / "paiec_training.json").write_text(json.dumps(dict(input_contract=contract)))

    def test_checkpoint_automatically_restores_format_and_context_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.bundle(root)
            result = resolve_input_contract(folder)
            self.assertEqual(result["state_format"], "enriched_v1")
            self.assertEqual(result["max_len"], 1024)
            for override in [dict(state_format="json"), dict(max_len=512), dict(enrichment_policy=str(POLICY))]:
                with self.assertRaises(ValueError):
                    resolve_input_contract(folder, **override)
            policy = root / "profile_bundle/policy.json"
            policy.write_text(policy.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "policy changed"):
                resolve_input_contract(folder)

    def test_catalog_and_question_changes_are_rejected(self):
        for mutation in ["catalog", "question"]:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.bundle(root)
                if mutation == "catalog":
                    path = root / "profile_bundle/catalog/profiles.jsonl"
                    path.write_text(path.read_text() + "\n")
                else:
                    manifest = json.loads((root / "paiec_training.json").read_text())
                    manifest["input_contract"]["question_sha256"] = "different"
                    (root / "paiec_training.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    resolve_input_contract(folder)

    def test_old_checkpoints_keep_historical_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "paiec_training.json").write_text("{}")
            result = resolve_input_contract(folder)
            self.assertEqual(result["state_format"], "json")
            self.assertIsNone(result["max_len"])

    def test_shared_predictor_uses_enriched_builder_without_changing_question(self):
        class Router:
            def predict_batch(self, requests, batch_size):
                self.requests = requests
                return [{"answers": {"correct": {"noul": .6}}} for _ in requests]
        router = Router()
        predictor = LayaPredictor(router=router, state_format="enriched_v1",
                                  enrichment_policy=str(POLICY), enricher=enrich())
        self.assertEqual(predictor.predict([SUBJECT, ITEM], []), .6)
        self.assertIn("layers: 32", router.requests[0]["state"])
        self.assertEqual(set(router.requests[0]["questions"]), {"correct"})
        self.assertEqual(router.requests[0]["questions"]["correct"]["type"], "noul")
