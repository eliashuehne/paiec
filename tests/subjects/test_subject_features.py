import math
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.lift_report import is_label_free
from paiec.subject_features import (
    COLUMNS,
    build_subject_features,
    extra_identity,
    family_of,
    identity_key,
    load_families,
    parse_extra,
    parse_params,
    parse_release,
    subject_row,
    tier_of,
)

FAMILIES_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "subjects" / "model_families.json"
FAMILIES = load_families(FAMILIES_CONFIG)


def fields(**overrides):
    row = {
        "subject_id": "s1",
        "normalized_name": "Meta Llama 3 8B",
        "provider": "Meta",
        "release_date": "2024-04-18",
        "access_date": np.nan,
        "harness": "HELM",
        "harness_version": np.nan,
        "reasoning_effort": np.nan,
        "subject_features_extra": np.nan,
    }
    return pd.DataFrame([{**row, **overrides}])


class ParserTests(unittest.TestCase):
    def test_parameter_counts(self):
        cases = {
            "Meta Llama 3 8B": (8, 8),
            "TinyLlama 1.1B CPT OpenWebMath": (1.1, 1.1),
            "Llama-2-70b-chat-hf": (70, 70),
            "paligemma-3b-mix-448": (3, 3),
            "Mistral Mixtral 8x7B": (56, 56),
            "Alibaba Qwen3 30B A3B": (30, 3),
            "BERT 350M": (0.35, 0.35),
            "AllenAI Molmo 7B-O": (7, 7),
        }
        for name, expected in cases.items():
            self.assertEqual(parse_params(name), expected, name)

    def test_names_without_a_size(self):
        for name in ("OpenAI GPT-4o", "gpt-35-turbo-230613-5shot", "phi-4-5shot", "Alibaba Qwen2.5 Plus", None):
            self.assertTrue(math.isnan(parse_params(name)[0]), name)

    def test_release_dates(self):
        self.assertEqual(parse_release("2020-01-11"), (10.0, "day"))
        self.assertEqual(parse_release("2020-02"), (31.0, "month"))
        self.assertTrue(math.isnan(parse_release("soon")[0]))
        self.assertEqual(parse_release(np.nan)[1], None)

    def test_extra_pairs(self):
        self.assertEqual(
            parse_extra("source_model_label=Yi-6B;note=a=b; subject_kind = human"),
            {"source_model_label": "Yi-6B", "note": "a=b", "subject_kind": "human"},
        )
        self.assertEqual(parse_extra(np.nan), {})

    def test_identity_key_drops_provider_and_org_path(self):
        self.assertEqual(identity_key("Alibaba Qwen2-VL 7B", "Alibaba"), "qwen2vl7b")
        self.assertEqual(identity_key("meta-llama/llama-3-8b-chat-hf", None), "llama38bchathf")
        self.assertIsNone(identity_key(np.nan, "Meta"))

    def test_tiers(self):
        cases = {
            "OpenAI GPT-4o mini": "small", "Google Gemini 2.0 Flash-Lite": "small", "Anthropic Claude 3 Haiku": "small",
            "Google Gemini 1.5 Pro": "large", "Anthropic Claude 3 Opus": "large",
            "Google Gemini 2.5": None, "MiniMax abab7": None, "MiniGPT-4 7B": None, "Meta Llama 3 8B": None,
        }
        for name, expected in cases.items():
            self.assertEqual(tier_of(name), expected, name)


class ExtraIdentityTests(unittest.TestCase):
    def test_any_key_naming_a_model_counts(self):
        cases = {
            "source_model=Qwen/Qwen2.5-72B-Instruct": ("Qwen/Qwen2.5-72B-Instruct", "source_model"),
            "recorded_model_label=gpt-5.4;settings_status=not logged": ("gpt-5.4", "recorded_model_label"),
            "declared_model=BioMistral/BioMistral-7B-DARE": ("BioMistral/BioMistral-7B-DARE", "declared_model"),
            'model_identifiers=["glm-5.1"]': ("glm-5.1", "model_identifiers"),
        }
        for text, expected in cases.items():
            self.assertEqual(extra_identity(parse_extra(text)), expected, text)

    def test_model_name_inside_a_configuration(self):
        found = extra_identity(parse_extra("source_configuration={'model_name': 'EleutherAI/pythia-160m', 'dtype': 'f16'}"))
        self.assertEqual(found, ("EleutherAI/pythia-160m", "source_configuration.model_name"))

    def test_keys_about_something_else_are_ignored(self):
        for text in (
            "historical_model_revision=abc123",        # a revision
            "model_run=models/adacos/s005-rope-s42",   # a run path
            "input_mode=single_image",                 # "mode" is not "model"
            "source_model=not_recorded",               # empty value
            "source_model=The model weights were never released by the authors",  # a sentence
            "lalm=qwen3omni",                          # key does not say "model": left out on purpose
        ):
            self.assertEqual(extra_identity(parse_extra(text)), (None, None), text)

    def test_priority_name_then_label_then_extra(self):
        extra = "source_model_label=Yi-6B;source_model=other-model"
        named = subject_row(fields(subject_features_extra=extra).iloc[0], FAMILIES)
        self.assertEqual((named["identity"], named["identity_source"]), ("Meta Llama 3 8B", "name"))
        labelled = subject_row(fields(normalized_name=np.nan, subject_features_extra=extra).iloc[0], FAMILIES)
        self.assertEqual((labelled["identity"], labelled["identity_source"]), ("Yi-6B", "label"))
        found = subject_row(fields(normalized_name=np.nan, subject_features_extra="source_model=Pixtral-12B").iloc[0], FAMILIES)
        self.assertEqual((found["identity"], found["identity_source"], found["identity_extra_key"]),
                         ("Pixtral-12B", "extra", "source_model"))
        self.assertEqual(found["family"], "mistral")


class FamilyTableTests(unittest.TestCase):
    def test_every_example_matches_its_own_family_first(self):
        for entry in FAMILIES:
            for example in entry["examples"]:
                self.assertEqual(family_of(example, FAMILIES)["family"], entry["family"], example)

    def test_families_are_unique_and_patterns_compile(self):
        names = [entry["family"] for entry in FAMILIES]
        self.assertEqual(len(names), len(set(names)))
        for entry in FAMILIES:
            re.compile(entry["pattern"])

    def test_near_misses(self):
        self.assertIsNone(family_of("MiniGPT-4 7B", FAMILIES))  # not OpenAI
        self.assertIsNone(family_of("Ovis2-34B", FAMILIES))  # "o" + digit inside a word is not openai_o
        self.assertEqual(family_of("DeepSeek R1 Distill Qwen 1.5B", FAMILIES)["family"], "deepseek")
        self.assertEqual(family_of("LLaVA-OneVision Qwen2 7B", FAMILIES)["family"], "llava")


class SubjectRowTests(unittest.TestCase):
    def row(self, **overrides):
        return subject_row(fields(**overrides).iloc[0], FAMILIES)

    def test_named_subject(self):
        row = self.row()
        self.assertEqual((row["identity"], row["identity_source"], row["family"]), ("Meta Llama 3 8B", "name", "llama"))
        self.assertAlmostEqual(row["log_params_b"], math.log10(8))
        self.assertEqual(row["release_precision"], "day")
        self.assertEqual(row["release_year"], 2024.0)

    def test_release_year_is_calendar_year(self):
        for date, year in (("2019-12", 2019.0), ("2020-01-01", 2020.0), ("2025-12-31", 2025.0)):
            self.assertEqual(self.row(release_date=date)["release_year"], year, date)
        self.assertTrue(math.isnan(self.row(release_date=np.nan)["release_year"]))

    def test_label_is_used_only_when_the_name_is_empty(self):
        extra = "source_model_label=Qwen2-VL-72B-Instruct"
        named = self.row(subject_features_extra=extra)
        self.assertEqual(named["identity"], "Meta Llama 3 8B")
        unnamed = self.row(normalized_name=np.nan, provider=np.nan, subject_features_extra=extra)
        self.assertEqual((unnamed["identity"], unnamed["identity_source"]), ("Qwen2-VL-72B-Instruct", "label"))
        self.assertEqual(unnamed["provider_filled"], "Alibaba")  # from the family table
        self.assertEqual((unnamed["is_instruct"], unnamed["is_vision"]), (1, 1))

    def test_placeholder_provider_is_replaced_and_agents_flagged(self):
        row = self.row(normalized_name="Agent: MDAgents", provider="(agent scaffold)")
        self.assertEqual(row["is_agent"], 1)
        self.assertTrue(row["provider_filled"] is None)
        self.assertTrue(self.row(normalized_name="Mistral 7B", provider="Unknown")["provider_filled"] == "Mistral")

    def test_subject_kind_and_no_identity(self):
        row = self.row(normalized_name=np.nan, subject_features_extra="subject_kind=human;age=30")
        self.assertEqual(row["subject_kind"], "human")
        self.assertTrue(row["identity"] is None and row["family"] is None and row["identity_source"] is None)


class TableTests(unittest.TestCase):
    def table(self):
        frame = pd.concat([
            fields(),
            fields(subject_id="s2", normalized_name=np.nan, subject_features_extra="source_model_label=gpt-3.5"),
            fields(subject_id="s0", normalized_name="Anthropic Claude 3 Haiku", provider="Anthropic"),
        ], ignore_index=True)
        return frame, build_subject_features(frame, FAMILIES)

    def test_one_row_per_subject_with_the_contract_columns(self):
        _, table = self.table()
        self.assertEqual(list(table.columns), COLUMNS)
        self.assertEqual(table["subject_id"].tolist(), ["s0", "s1", "s2"])
        self.assertNotIn("display_name", table.columns)

    def test_duplicate_subjects_raise(self):
        with self.assertRaises(ValueError):
            build_subject_features(pd.concat([fields(), fields()]), FAMILIES)

    def test_display_name_is_never_read(self):
        frame, table = self.table()
        with_display = build_subject_features(frame.assign(display_name="Secret Model 999B"), FAMILIES)
        pd.testing.assert_frame_equal(table, with_display)

    def test_builder_is_label_free(self):
        frame, _ = self.table()
        frame = frame.assign(label=[0, 1, 1])
        self.assertTrue(is_label_free(lambda f: build_subject_features(f.drop(columns="label"), FAMILIES), frame))


if __name__ == "__main__":
    unittest.main()
