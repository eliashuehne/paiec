import unittest

from pydantic import ValidationError

from paiec.config import EvaluationConfig, LayaConfig, SuiteConfig


def suite(**overrides):
    return SuiteConfig.model_validate({"benchmarks": ["bbq"], "pairs_per_benchmark": 1,
                                      "evaluation_items": 8, "acquisition_items": 40, **overrides})


class ConfigTests(unittest.TestCase):
    def test_existing_suite_settings_are_preserved(self):
        self.assertEqual(suite().min_items, 80)
        self.assertEqual(suite().benchmarks, ["bbq"])

    def test_duplicate_benchmarks_cannot_silently_change_pair_weights(self):
        with self.assertRaises(ValidationError):
            suite(benchmarks=["bbq", "bbq"])

    def test_invalid_pool_sizes_and_typo_fail_early(self):
        for values in ({"acquisition_items": 30}, {"evaluation_items": 0},
                       {"min_items": 79}, {"evaluation_item": 8}, {"seed": "0"}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                suite(**values)

    def test_invalid_predictor_or_prior_weight_is_rejected(self):
        for values in ({"predictors": ["unknown"]}, {"predictors": ["laya", "laya"]},
                       {"prior_weight": float("nan")}, {"prior_weight": 0.0}):
            settings = {"suite": suite(), "predictors": ["constant"], **values}
            with self.subTest(values=values), self.assertRaises(ValidationError):
                EvaluationConfig.model_validate(settings)

    def test_invalid_model_settings_are_rejected(self):
        for values in ({"max_len": 8193}, {"max_len": 0}, {"batch_size": 0}, {"model": "unknown"},
                       {"backend": "unknown"}, {"backend": "mlx", "device": "cuda"},
                       {"backend": "torch", "dtype": "float16"}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                LayaConfig.model_validate(values)

    def test_omitted_max_len_resolves_to_the_checkpoints_trained_length(self):
        self.assertEqual(LayaConfig().max_len, 512)
        self.assertEqual(LayaConfig(model="multilingual").max_len, 1024)
        self.assertEqual(LayaConfig(model="multilingual", max_len=2048).max_len, 2048)
