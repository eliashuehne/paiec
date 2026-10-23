import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paiec.compact_inputs import CompactStateBuilder, InputCapacityError
from paiec.compact_profiles import CompactProfiles
from paiec.config import LayaConfig
from paiec.head_pilot import HeadPilotConfig, input_contract, snapshot_profiles
from paiec.predictor_factory import make_laya_predictor

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/head_finetuning/compact_profiles_v3.json"
SUBJECT = {"harness": "Petavue NL2SQL-Benchmark", "normalized_name": "Mistral 7B",
           "reasoning_effort": "high", "subject_features_extra": "source_model=mistralai/Mistral-7B-Instruct-v0.1;shot_size=0;max_tokens=8000"}


class WordTokenizer:
    mask_token = "[MASK]"
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(re.findall(r"\S+", text))))}


class Runtime:
    def __init__(self):
        self.agent = SimpleNamespace(tok=WordTokenizer())
        self.metadata = {"backend": "mlx", "device": "cpu"}
        self.calls = []
    def predict_batch(self, requests, batch_size):
        self.calls.append(requests)
        return [{"answers": {"correct": {"noul": .42}}, "usage": {"truncated": False}} for _ in requests]


class CompactPipelineTests(unittest.TestCase):
    def setUp(self):
        self.profiles = CompactProfiles(POLICY, root=ROOT)
        self.builder = CompactStateBuilder(WordTokenizer(), 900, self.profiles)

    def test_scoped_aliases_never_become_global_or_override_contradictions(self):
        for alias in ["claude", "gemini", "gpt35", "gpt4", "llama", "mistral", "gpt_finetune", "gpt_finetune_2"]:
            subject = {"harness": "ERBench", "subject_features_extra": "source_model=" + alias}
            self.assertIsNotNone(self.profiles.resolve(subject)["model_profile_id"])
            for extra in [{"harness": "other"}, {"harness_version": "future-revision"},
                          {"normalized_name": "unrecognized conflicting model"},
                          {"subject_features_extra": "source_model=" + alias + ";model_identifier=private"}]:
                self.assertIsNone(self.profiles.resolve(dict(subject, **extra))["model_profile_id"])
        self.assertIsNone(self.profiles.resolve({"display_name": "Alibaba Qwen 7B", "subject_id": "known"})["model_profile_id"])

    def test_private_finetunes_are_explicitly_base_family_context(self):
        text, details = self.profiles.block({"harness": "ERBench", "subject_features_extra": "source_model=gpt_finetune_2"}, self.builder.count)
        self.assertEqual(details["model_identity_basis"], "documented_harness_scoped_base_family")
        self.assertIn("Base model:", text)
        self.assertFalse(details["exact_evaluated_checkpoint_verified"])
        self.assertNotIn("Soccer", text)
        self.assertNotIn("success", text)
        run = self.profiles.profiles["run.erbench.9c243d02353e1108"]
        self.assertIsNone(run.specifications.model_profile_id)

    def test_metadata_has_no_effect_on_the_retained_task_and_options(self):
        options = [f"{letter}: exact choice {letter} 中文" for letter in "ABCD"]
        content = "Long shared context. " * 1500 + "Which choice is valid?\n" + "\n".join(options)
        pair = self.builder.build(SUBJECT, {"item_content": content})
        self.assertTrue(pair.visibility["task_omitted"])
        self.assertTrue(pair.visibility["matched_task_payload"])
        self.assertIn(pair.task_text, pair.baseline)
        self.assertIn(pair.task_text, pair.enriched)
        self.assertIn("Which choice is valid?", pair.task_text)
        for option in options:
            self.assertIn(option, pair.task_text)
        self.assertEqual(pair.visibility["options_detected"], 4)
        self.assertEqual(pair.visibility["options_omitted"], 0)
        self.assertEqual(pair.enrichment["available"], pair.enrichment["delivered"])
        self.assertIn("reasoning=high", pair.enriched)
        self.assertIn("max_tokens=8000", pair.enriched)
        self.assertLessEqual(self.builder.count(pair.enriched), 900)

    def test_structured_question_and_choices_take_priority_over_schema(self):
        item = {"item_content": json.dumps({"schema": "CREATE TABLE context. " * 3000,
                "question": "Which school has free meals?", "choices": ["first choice", "second choice"]})}
        pair = self.builder.build(SUBJECT, item)
        self.assertIn("Which school has free meals?", pair.task_text)
        self.assertIn("first choice", pair.task_text)
        self.assertIn("second choice", pair.task_text)
        self.assertEqual(pair.visibility["options_detected"], 2)
        self.assertFalse(pair.visibility["question_or_user_omitted"])

    def test_marked_question_is_separate_from_a_large_retrieved_passage(self):
        question = "Which city contains this attraction?"
        content = json.dumps({"system": "Answer with one name.", "user":
            "Question: " + question + "\nContext: " + "Long retrieved passage. " * 5000 + "\nAnswer: "})
        pair = self.builder.build(SUBJECT, {"item_content": content})
        self.assertIn(question, pair.task_text)
        self.assertIn("Answer with one name.", pair.task_text)
        self.assertFalse(pair.visibility["question_or_user_omitted"])
        self.assertTrue(pair.visibility["task_omitted"])

    def test_oversized_options_fail_explicitly_instead_of_being_shortened(self):
        content = "Choose.\nA: " + "large " * 1000 + "\nB: " + "large " * 1000
        with self.assertRaises(InputCapacityError):
            self.builder.build(SUBJECT, {"item_content": content})

    def test_unavailable_facts_are_unknown_and_labels_cannot_enter_rendering(self):
        first = self.builder.build({}, {"item_content": "What is 2+2?", "benchmark_id": "SECRET", "reference_answer": "SECRET"})
        second = self.builder.build({"outcome": 1, "display_name": "SECRET"},
                                    {"item_content": "What is 2+2?", "label": 0})
        self.assertEqual(first.enriched, second.enriched)
        self.assertNotIn("SECRET", first.enriched)
        self.assertEqual(first.enrichment["available"], {"model": False, "harness": False})

    def test_runtime_uses_the_same_builder_and_cache_without_counting_previews(self):
        runtime = Runtime()
        count = lambda text: len(runtime.agent.tok(text)["input_ids"])
        with patch("paiec.compact_predictor.native_budget", return_value=(count, 900)):
            predictor = make_laya_predictor(state_format="compact_enriched_v3", enrichment_policy=str(POLICY), router=runtime)
        item = {"item_content": "Question.\nA: first\nB: second"}
        expected = self.builder.build(SUBJECT, item).enriched
        self.assertEqual(predictor.render_state(SUBJECT, item, []), expected)
        self.assertEqual(predictor.visibility_counts["model_facts_available"], 0)
        self.assertEqual(predictor.predict([SUBJECT, item], []), .42)
        self.assertEqual(predictor.predict([SUBJECT, item], []), .42)
        self.assertEqual(predictor.forward_items, 1)
        self.assertEqual(predictor.visibility_counts["model_facts_delivered"], 2)
        self.assertEqual(runtime.calls[0][0]["state"], expected)

    def test_compact_baseline_checkpoint_pins_and_bundles_its_reservation_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot_profiles(POLICY, root, ROOT)
            config = HeadPilotConfig(input_pipeline="compact_v3", max_len=2048, policy=str(POLICY))
            contract = input_contract(root, "baseline", config)
            self.assertEqual(contract["state_format"], "compact_v3")
            self.assertIn("policy_sha256", contract)
            (root / "paiec_training.json").write_text(json.dumps({"input_contract": contract}))
            runtime = Runtime()
            with patch("paiec.compact_predictor.native_budget", return_value=(len, 900)):
                predictor = make_laya_predictor(checkpoint_path=str(root), router=runtime)
            self.assertEqual(predictor.state_format, "compact_v3")
            (root / "profile_bundle/policy.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "policy changed"):
                make_laya_predictor(checkpoint_path=str(root), router=runtime)

    def test_default_length_and_legacy_constraints_remain_distinct(self):
        c = LayaConfig(state_format="compact_enriched_v3", enrichment_policy=str(POLICY))
        self.assertEqual(c.max_len, 2048)
        self.assertEqual(LayaConfig().max_len, 512)
        with self.assertRaises(ValueError):
            HeadPilotConfig(input_pipeline="task_v2", max_len=2048)


if __name__ == "__main__":
    unittest.main()
