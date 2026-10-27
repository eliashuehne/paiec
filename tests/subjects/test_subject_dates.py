import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.lift_report import is_label_free
from paiec.subject_dates import (
    References,
    accepted_sources,
    add_version_columns,
    backtest,
    fill_release_dates,
    name_date,
    parse_line,
    parse_version,
    summarise_backtest,
)
from paiec.subject_features import family_of, load_families

FAMILIES = load_families(Path(__file__).resolve().parents[2] / "configs" / "subjects" / "model_families.json")


def version_and_line(name):
    entry = family_of(name, FAMILIES)
    return parse_version(name, entry), parse_line(name, entry)


def timeline(rows):
    """Subjects as (subject_id, identity, family, line, version, log_params_b, release_days, provider)."""
    frame = pd.DataFrame(rows, columns=["subject_id", "identity_key", "family", "line", "version",
                                        "log_params_b", "release_days", "provider_filled"])
    return frame.assign(name_date_days=math.nan)


CLAUDE = [
    ("a", "claude3sonnet", "claude", "sonnet", 3.0, math.nan, 100.0, "Anthropic"),
    ("b", "claude35sonnet", "claude", "sonnet", 3.5, math.nan, 300.0, "Anthropic"),
    ("c", "claude4sonnet", "claude", "sonnet", 4.0, math.nan, 700.0, "Anthropic"),
]


class VersionParsingTests(unittest.TestCase):
    def test_versions_and_lines_from_real_names(self):
        cases = {
            "Anthropic Claude 3.5 Sonnet": (3.5, "sonnet"),
            "Claude3-5V_Sonnet_20241022": (3.5, "sonnet"),
            "claude-haiku-4-5-2025-12-05": (4.5, "haiku"),
            "claude-3-opus-20240229": (3.0, "opus"),
            "gpt-35-turbo-230613-5shot": (3.5, "turbo"),
            "OpenAI GPT-4o mini": (4.0, "omni"),
            "GeminiPro1-5-002": (1.5, "pro"),
            "Alibaba Qwen2.5 72B": (2.5, None),
            "Llama-2-70b-chat-hf": (2.0, None),
            "llama-3_3-70b-instruct-turbomind": (3.3, None),
            "InternVL2_5-78B": (2.5, None),
            "phi-3-5-mini-instruct": (3.5, "mini"),
            "DeepSeek R1 Distill Qwen 1.5B": (1.0, "r"),
        }
        for name, expected in cases.items():
            self.assertEqual(version_and_line(name), expected, name)

    def test_sizes_are_not_versions(self):
        for name in ("Qwen-7B", "Meta Code Llama 34B", "gemma-7b-it", "deepseek-llm-67b-chat", "Yi-6B-Chat"):
            self.assertTrue(math.isnan(version_and_line(name)[0]), name)

    def test_turbomind_is_not_a_product_line(self):
        self.assertIsNone(version_and_line("qwen2.5-7b-instruct-turbomind")[1])

    def test_dates_in_names(self):
        self.assertEqual(name_date("claude-3-opus-20240229"), float((pd.Timestamp("2024-02-29") - pd.Timestamp("2020-01-01")).days))
        self.assertEqual(name_date("gpt-4o-mini-240718"), float((pd.Timestamp("2024-07-18") - pd.Timestamp("2020-01-01")).days))
        self.assertTrue(math.isnan(name_date("gpt-35-turbo-0125")))
        self.assertTrue(math.isnan(name_date("model-20241399")))  # no 13th month


class EstimateTests(unittest.TestCase):
    def probe(self, **values):
        row = dict(zip(["subject_id", "identity_key", "family", "line", "version", "log_params_b",
                        "release_days", "provider_filled"], ("z", "new", "claude", "sonnet", 3.7, math.nan,
                                                             math.nan, "Anthropic")))
        row.update(values)
        return timeline([tuple(row.values())]).iloc[0]

    def test_interpolation_brackets_between_neighbouring_versions(self):
        estimate, low, high, source = References(timeline(CLAUDE)).estimate(self.probe())
        self.assertEqual((source, low, high, estimate), ("interpolated", 300.0, 700.0, 500.0))

    def test_same_version_beats_interpolation(self):
        self.assertEqual(References(timeline(CLAUDE)).estimate(self.probe(version=3.5))[3], "same_version")

    def test_one_sided_is_capped_by_the_data(self):
        estimate, low, high, source = References(timeline(CLAUDE)).estimate(self.probe(version=5.0))
        self.assertEqual((source, low, high), ("one_sided", 700.0, 700.0))
        self.assertEqual(estimate, 700.0)

    def test_missing_line_is_a_valid_key(self):
        rows = [(s, k, "qwen", None, v, math.nan, d, "Alibaba") for s, k, v, d in
                (("a", "qwen2", 2.0, 100.0), ("b", "qwen3", 3.0, 500.0))]
        probe = timeline([("z", "x", "qwen", None, 2.5, math.nan, math.nan, "Alibaba")]).iloc[0]
        self.assertEqual(References(timeline(rows)).estimate(probe)[3], "interpolated")

    def test_no_identity_gives_no_estimate(self):
        probe = timeline([("z", None, None, None, math.nan, math.nan, math.nan, None)]).iloc[0]
        self.assertIsNone(References(timeline(CLAUDE)).estimate(probe)[3])


class FillTests(unittest.TestCase):
    def frame(self):
        undated = [("u", "claude37sonnet", "claude", "sonnet", 3.7, math.nan, math.nan, "Anthropic")]
        return timeline(CLAUDE + undated)

    def test_real_dates_always_win(self):
        table = fill_release_dates(self.frame(), {})
        real = table.set_index("subject_id").loc[["a", "b", "c"]]
        self.assertTrue((real["release_source"] == "field").all())
        np.testing.assert_array_equal(real["release_days_filled"], [100.0, 300.0, 700.0])
        self.assertEqual(table.set_index("subject_id").loc["u", "release_filled"], 1)

    def test_references_from_the_subjects_own_fold_are_not_used(self):
        folds = {"a": frozenset({"0"}), "b": frozenset({"1"}), "c": frozenset({"1"}), "u": frozenset({"1"})}
        row = fill_release_dates(self.frame(), folds).set_index("subject_id").loc["u"]
        self.assertEqual(row["release_source"], "one_sided")  # only version 3 (fold 0) is usable
        self.assertEqual(row["release_lo"], 100.0)

    def test_sealed_subjects_are_never_references(self):
        folds = {"a": frozenset({"sealed"}), "b": frozenset({"sealed"}), "c": frozenset({"sealed"}),
                 "u": frozenset({"0"})}
        self.assertTrue(pd.isna(fill_release_dates(self.frame(), folds).set_index("subject_id").loc["u", "release_source"]))

    def test_unaccepted_sources_are_dropped(self):
        table = fill_release_dates(self.frame(), {}, accepted=["same_version"]).set_index("subject_id")
        self.assertTrue(math.isnan(table.loc["u", "release_days_filled"]))
        self.assertEqual(table.loc["u", "release_filled"], 0)

    def test_filled_year_is_the_calendar_year(self):
        table = fill_release_dates(self.frame(), {}).set_index("subject_id")
        self.assertEqual(table.loc["a", "release_year_filled"], 2020.0)

    def test_label_free(self):
        frame = self.frame().assign(label=[0, 1, 1, 0])
        self.assertTrue(is_label_free(lambda f: fill_release_dates(f.drop(columns="label"), {}), frame))


class BacktestTests(unittest.TestCase):
    def test_backtest_and_acceptance(self):
        rows = CLAUDE + [("d", "claude45sonnet", "claude", "sonnet", 4.5, math.nan, 900.0, "Anthropic")]
        results = backtest(timeline(rows), "identity")
        hidden = results.set_index("identity_key").loc["claude35sonnet"]
        self.assertEqual((hidden["source"], hidden["estimate"]), ("interpolated", 400.0))
        summary = summarise_backtest(results)
        self.assertIn("interpolated", accepted_sources(summary))

    def test_acceptance_needs_both_rules(self):
        summary = pd.DataFrame({"source": ["good", "clustered", "far"], "median_abs_days": [40, 119, 200],
                                "within_180": [0.9, 0.62, 0.9]})
        self.assertEqual(accepted_sources(summary), ["good"])

    def test_version_columns_on_a_feature_table(self):
        features = pd.DataFrame({"identity": ["Anthropic Claude 3.5 Sonnet", None],
                                 "family": ["claude", np.nan]})
        out = add_version_columns(features, FAMILIES)
        self.assertEqual(out["version"].iloc[0], 3.5)
        self.assertTrue(math.isnan(out["version"].iloc[1]))


if __name__ == "__main__":
    unittest.main()
