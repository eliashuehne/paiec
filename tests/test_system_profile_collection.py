import importlib.util
import json
from pathlib import Path
import unittest

from paiec.schema import SUBJECT_FIELDS
from paiec.system_profiles import load_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "research/system_profiles/collection_v1"
spec = importlib.util.spec_from_file_location(
    "build_profiles", ROOT / "scripts/build_system_profile_collection.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ProfileCollectionTests(unittest.TestCase):
    def test_nested_dimensions_are_not_runtime_context_limits(self):
        config = {
            "num_hidden_layers": 999,
            "text_config": {
                "num_hidden_layers": 40,
                "num_key_value_heads": 2,
                "max_position_embeddings": 262144,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "layer_types": ["linear_attention", "full_attention", "linear_attention"],
            },
        }
        facts = {name: value for name, value, _, _ in builder.configured_facts(config)}
        self.assertEqual(facts["layers"], 40)
        self.assertEqual(facts["experts_active_per_token"], 8)
        self.assertEqual(facts["attention_types"], ["linear_attention", "full_attention"])
        self.assertNotIn("max_combined_context_tokens", facts)
        self.assertNotIn("max_input_tokens", facts)

    def test_collected_catalog_keeps_source_and_identity_boundaries(self):
        profiles, sources = load_catalog(CATALOG)
        profiles = {profile.profile_id: profile for profile in profiles}
        self.assertEqual(len(profiles), 58)
        external = [source for source in sources if source.source_type != "benchmark_table"]
        self.assertTrue(external)
        self.assertTrue(all(source.eligibility == "unreviewed" for source in external))
        self.assertTrue(all(source.sha256 for source in sources))
        for name in ["gpt-4o", "gpt-4o-mini", "gpt-5.4-mini", "claude3-sonnet", "gemini1.5-flash"]:
            self.assertIsNone(profiles["model." + name].specifications.architecture.total_parameters.value)
        # A learned positional limit from the config is deliberately not a
        # documented supported context window.
        self.assertIsNone(profiles["model.deepseek-r1"].specifications.max_combined_context_tokens.value)
        self.assertEqual(profiles["model.qwen3.6-35b-a3b"].specifications.architecture.layers.value, 40)
        self.assertFalse(any("huggingface.co/codellama/" in source.location for source in sources))

    def test_inventory_does_not_resolve_missing_names_or_private_checkpoints(self):
        subjects = [json.loads(line) for line in (CATALOG / "subjects.jsonl").read_text().splitlines()]
        self.assertEqual(len(subjects), 32)
        self.assertEqual({row["role"] for row in subjects}, {"training", "development"})
        for row in subjects:
            self.assertEqual(set(row["subject"]), set(SUBJECT_FIELDS))
            if row["benchmark"] in {"adaptivestep", "erbench"}:
                self.assertIsNone(row["model_profile_id"])
            if row["benchmark"] == "ceval" and not row["subject"]["normalized_name"]:
                self.assertIsNone(row["model_profile_id"])
            if row["model_match_status"].startswith("unresolved"):
                self.assertIsNone(row["model_profile_id"])


if __name__ == "__main__":
    unittest.main()
