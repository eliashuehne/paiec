import unittest
from unittest.mock import patch

from paiec.laya_predictor import load_questions
from paiec.config import LayaConfig, LayaQuestion
from pydantic import ValidationError


class LayaQuestionTests(unittest.TestCase):
    def test_shared_question_is_only_the_correctness_question(self):
        self.assertEqual(load_questions(), {"correct": {
            "type": "noul",
            "instructions": "Will the target AI system, using the supplied settings, solve this task correctly on a single attempt?",
        }})

    def test_shared_question_does_not_read_benchmark_definitions(self):
        with patch("paiec.benchmark_registry.load_benchmark_definition", side_effect=AssertionError("registry read")):
            self.assertEqual(set(load_questions()["correct"]), {"type", "instructions"})

    def test_question_rejects_empty_text_and_extra_criteria(self):
        for question in ({"type": "noul", "instructions": " "},
                         {"type": "noul", "instructions": "Will it succeed?", "criteria": {"true": "1", "false": "0"}}):
            with self.subTest(question=question), self.assertRaises(ValidationError):
                LayaQuestion.model_validate(question)

    def test_retired_question_modes_are_rejected(self):
        for mode in ("with_criteria", "without_criteria", "legacy", "benchmark_specific"):
            with self.subTest(mode=mode), self.assertRaises(ValidationError):
                LayaConfig(question_mode=mode)


if __name__ == "__main__":
    unittest.main()
