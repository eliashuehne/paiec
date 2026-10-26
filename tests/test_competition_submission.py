import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

from paiec.competition_submission import build, checked_state
from paiec.rate_model import ITEM_FEATURES, item_feature_frame

GATE = {"verdict": "clears_matched_adaptive_control"}
REJECTED = {"verdict": "no_gain_over_fixed_controls"}


def state(**overrides):
    base = {"mu": 0.0, "ability": {"openai gpt-4o": 0.5}, "intercept": 0.1,
            "feature_mean": [0.0] * len(ITEM_FEATURES), "feature_scale": [1.0] * len(ITEM_FEATURES),
            "coefficients": [0.0] * len(ITEM_FEATURES), "prior_strength": 5.0,
            "benchmark_weight": 0.5}
    base.update(overrides)
    return base


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUBJECT = {"normalized_name": "OpenAI GPT-4o", "provider": "OpenAI", "release_date": "",
           "access_date": "", "harness": "", "harness_version": "", "reasoning_effort": "",
           "subject_features_extra": ""}
ITEM = {"item_content": "Which is correct?\nA: one\nB: two", "item_features": "",
        "interactors": "", "benchmark_id": "benchmark_x"}


class PackagerTests(unittest.TestCase):
    def test_a_rejected_gate_produces_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "gate rejects"):
                build(state(), Path(temporary) / "package", gate=REJECTED)
            self.assertFalse((Path(temporary) / "package").exists())

    def test_package_is_written_and_self_contained(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "package"
            result = build(state(), package, gate=GATE)
            self.assertEqual(result["files"], ["model.py", "labeling.py", "state.json"])
            self.assertEqual(json.loads((package / "state.json").read_text())["mu"], 0.0)
            self.assertFalse((package / "requirements.txt").exists())
            with self.assertRaisesRegex(ValueError, "fresh package"):
                build(state(), package, gate=GATE)

    def test_state_shape_is_enforced(self):
        checked_state(state())
        for broken in [state(feature_mean=[0.0]), state(coefficients="none"),
                       state(feature_scale=[0.0] * len(ITEM_FEATURES)), state(mu=None),
                       {key: value for key, value in state().items() if key != "mu"}]:
            with self.assertRaises(ValueError):
                checked_state(broken)


class ModelTemplateTests(unittest.TestCase):
    def package(self, temporary, **overrides):
        out = Path(temporary) / "package"
        build(state(**overrides), out, gate=GATE)
        return load_module(out / "model.py", "submission_model_" + Path(temporary).name)

    def test_task_features_match_the_fitted_encoder(self):
        samples = ["", "What is 2+2?", "Which is correct?\nA: one\nB: two",
                   '{"prompt": "Write SQL", "k": 3}', "中文问题 \u0436 \u0627 42 ?", "x | y ```code```"]
        frame = item_feature_frame(pd.DataFrame({
            "benchmark": ["b"] * len(samples), "item_id": [str(i) for i in range(len(samples))],
            "content": samples}))
        with tempfile.TemporaryDirectory() as temporary:
            module = self.package(temporary)
            for index, text in enumerate(samples):
                produced = module.task_features(text)
                expected = [float(frame.iloc[index][name]) for name in ITEM_FEATURES]
                for value, reference in zip(produced, expected):
                    self.assertAlmostEqual(value, reference, places=4)

    def test_prediction_uses_ability_and_only_matching_benchmark_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = self.package(temporary, coefficients=[0.0] * len(ITEM_FEATURES))
            self.assertAlmostEqual(module.predict([SUBJECT, ITEM], []),
                                   module.sigmoid(module.STATE["intercept"]
                                                  + module.STATE["ability"]["openai gpt-4o"]))
            own = [[[SUBJECT, dict(ITEM, item_content="other task")], 1]]
            raised = module.predict([SUBJECT, ITEM], own)
            self.assertGreater(raised, module.predict([SUBJECT, ITEM], []))
            other_subject = dict(SUBJECT, normalized_name="Different Model")
            shared = [[[other_subject, dict(ITEM, item_content="other task")], 1]]
            self.assertGreater(module.predict([SUBJECT, ITEM], shared),
                               module.predict([SUBJECT, ITEM], []))
            elsewhere = [[[other_subject, dict(ITEM, benchmark_id="benchmark_y")], 1]]
            self.assertAlmostEqual(module.predict([SUBJECT, ITEM], elsewhere),
                                   module.predict([SUBJECT, ITEM], []))
            for value in (raised, module.predict([SUBJECT, ITEM], shared)):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_canonical_names_collapse_benchmark_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = self.package(temporary)
            self.assertEqual(module.canonical_name("CMMLU / gpt-4o"), module.canonical_name("GPT-4o"))

    def test_labeling_is_neutral_while_budget_remains(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "package"
            build(state(), package, gate=GATE)
            labeling = load_module(package / "labeling.py", "submission_labeling")
            self.assertTrue(labeling.acquisition_function(None, None, None, {"labels_remaining": 3,
                                                                             "items_remaining": 5}))
            self.assertFalse(labeling.acquisition_function(None, None, None, {"labels_remaining": 3,
                                                                              "items_remaining": 0}))
            self.assertFalse(labeling.acquisition_function(None, None, None, {"labels_remaining": 0,
                                                                              "items_remaining": 5}))
            self.assertTrue(labeling.acquisition_function(None, None, None, None))


if __name__ == "__main__":
    unittest.main()
