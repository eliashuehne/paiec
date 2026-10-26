import importlib
import unittest


class SettingsTests(unittest.TestCase):
    def parse(self, extra, **fields):
        try:
            module = importlib.import_module("paiec.feature_foundation.settings")
        except ModuleNotFoundError:
            self.fail("Typed settings extraction is missing")
        return module.parse_run_settings(dict(subject_features_extra=extra, **fields))

    def test_known_zero_and_empty_tool_list_remain_known(self):
        result = self.parse('n_shots=5;temperature=0;max_retries=0;enabled_tools=[]')
        values = {r["field"]: r for r in result["resolved"]}
        self.assertEqual(values["few_shot_count"]["value_json"], "5")
        self.assertEqual(values["temperature"]["value_json"], "0.0")
        self.assertEqual(values["max_retries"]["status"], "known")
        self.assertEqual(values["enabled_tools"]["value_json"], "[]")
        self.assertIsNone(values["max_turns"]["value_json"])

    def test_quoted_nested_settings_preserve_values_and_duplicate_conflicts(self):
        result = self.parse('system_prompt="a;b";n_shots=5;n_shots=2;configuration={"llm":{"top_p":0.8}}')
        self.assertEqual(next(r for r in result["resolved"] if r["field"] == "few_shot_count")["status"], "conflict")
        self.assertEqual(next(r for r in result["resolved"] if r["field"] == "top_p")["value_json"], "0.8")
        self.assertIn('"a;b"', [r["raw_value"] for r in result["observations"]])
        self.assertEqual(sum(r["raw_key"] == "n_shots" for r in result["observations"]), 2)

    def test_dedicated_field_conflicts_with_extra_and_no_method_inference(self):
        result = self.parse('reasoning_effort=high;method=ICL_10;max_tokens=4096;reported_task_generation_size=512', reasoning_effort="low")
        values = {r["field"]: r for r in result["resolved"]}
        self.assertEqual(values["reasoning_effort"]["status"], "conflict")
        self.assertEqual(values["few_shot_count"]["status"], "unknown")
        self.assertEqual(values["max_output_tokens"]["status"], "unknown")

    def test_explicit_url_container_decodes_but_arbitrary_literal_does_not(self):
        result = self.parse('agent_configuration=%7B%22temperature%22%3A0%7D;note=%7B%22max_turns%22%3A5%7D')
        values = {r["field"]: r for r in result["resolved"]}
        self.assertEqual(values["temperature"]["status"], "known")
        self.assertEqual(values["max_turns"]["status"], "unknown")

    def test_malformed_duplicate_json_and_nonfinite_values_abstain(self):
        for extra in ['configuration={"temperature":0,"temperature":1}', 'temperature=NaN', 'n_shots=5;unparseable remainder']:
            with self.subTest(extra=extra):
                result = self.parse(extra)
                if extra.startswith('configuration='):
                    self.assertEqual(sum(r['raw_key']=='temperature' for r in result['observations']), 2)
                    self.assertEqual(next(r for r in result['resolved'] if r['field']=='temperature')['status'], 'conflict')
                else:
                    self.assertTrue(any(r["status"] in {"invalid", "literal"} for r in result["observations"]))
                self.assertTrue(all(r["status"] != "known" for r in result["resolved"]))

    def test_bad_numeric_types_and_repeated_identical_claims(self):
        result = self.parse('n_shots=true;top_p=2;max_turns=-1;temperature=0;temperature=0.0')
        values = {r["field"]: r for r in result["resolved"]}
        self.assertEqual(values["few_shot_count"]["status"], "invalid")
        self.assertEqual(values["temperature"]["status"], "known")
        self.assertEqual(len(values["temperature"]["observation_ids"]), 2)

    def test_released_configuration_preserves_sampling_false_and_qualified_limits(self):
        result = self.parse('released_configuration={"generation_parameters":{"do_sample":false,"max_new_tokens":20,"temperature":0.0}};selection=minimum_perplexity;method=ICL_10')
        values = {r["field"]: r for r in result["resolved"]}
        self.assertEqual(values["temperature"]["status"], "known")
        self.assertEqual(values["max_output_tokens"]["value_json"], "20")
        self.assertEqual(values.get("sampling_enabled", {}).get("value_json"), "false")
        self.assertEqual(values["max_output_tokens"].get("source_scope"), "released_configuration")
        self.assertEqual(values["selection_method"]["value_json"], '"minimum_perplexity"')
        self.assertEqual(values["selection_method"].get("source_scope"), "selection_scope_unspecified")
        self.assertEqual(values["reported_method"]["value_json"], '"ICL_10"')
        self.assertIsNone(values["few_shot_count"]["value_json"])

    def test_nested_duplicates_quoted_containers_and_unresolved_containers_block_false_certainty(self):
        for nested in ['{"n_shots":2,"n_shots":3}', '"{\\"n_shots\\":2}"', '{broken}']:
            with self.subTest(nested=nested):
                result = self.parse('n_shots=5;configuration='+nested)
                value = next(r for r in result['resolved'] if r['field']=='few_shot_count')
                self.assertNotEqual(value['status'], 'known')
                self.assertIsNone(value['value_json'])
        result = self.parse('configuration={"n_shots":2,"n_shots":3}')
        self.assertEqual(sum(r['raw_key']=='n_shots' for r in result['observations']), 2)

    def test_aliases_under_examples_defaults_and_other_components_do_not_establish_run_settings(self):
        for branch in ['examples','defaults','other_model']:
            with self.subTest(branch=branch):
                result = self.parse('configuration={"'+branch+'":{"n_shots":2},"temperature":0}')
                values = {r['field']:r for r in result['resolved']}
                self.assertEqual(values['few_shot_count']['status'], 'unknown')
                self.assertEqual(values['temperature']['status'], 'known')
                self.assertTrue(any(r['raw_key']=='n_shots' for r in result['observations']))

    def test_typed_setting_with_object_value_is_invalid_even_when_an_outer_claim_is_valid(self):
        result = self.parse('n_shots=5;configuration={"n_shots":{}}')
        value = next(r for r in result['resolved'] if r['field']=='few_shot_count')
        self.assertEqual(value['status'], 'invalid')
        self.assertIsNone(value['value_json'])
