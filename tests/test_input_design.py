import json
import unittest
from importlib.util import find_spec
from types import SimpleNamespace

from paiec.laya_predictor import LayaPredictor, build_state
from paiec.state_rendering import readable
from test_laya_predictor import FakeRouter, SUBJECT, ITEM


class InputDesignTests(unittest.TestCase):
    def test_default_rendering_remains_byte_identical(self):
        expected = {"target_system": SUBJECT,
                    "target_item": {"item_content": ITEM["item_content"]}}
        self.assertEqual(build_state(SUBJECT, ITEM, []), json.dumps(expected, indent=2, ensure_ascii=False))

    def test_readable_nested_task_preserves_options_unicode_and_false_values(self):
        content = json.dumps({"question": "你好?", "options": ["A\nsecond line", "B"],
                              "count": 0, "flag": False, "missing": None, "empty": []})
        state = build_state(SUBJECT, dict(ITEM, item_content=content), [], state_format="text_v1")
        for value in ("你好?", "A\nsecond line", "B", "count: 0", "flag: false", "missing: null", "empty: []"):
            self.assertIn(value, state)
        self.assertNotIn('\\"question\\"', state)
        self.assertEqual(readable("not JSON"), "not JSON")
        # A task may ask about a literal JSON string: do not recursively reinterpret it.
        self.assertIn('question: {"literal": 1}', readable('{"question": "{\\"literal\\": 1}"}'))

    def test_projection_and_evidence_matching_precede_system_ablation(self):
        target = dict(ITEM, reference_answer="SECRET", display_name="SECRET")
        labels = [[[SUBJECT, dict(ITEM, item_content="matched")], 1],
                  [[dict(SUBJECT, normalized_name="other"), dict(ITEM, item_content="wrong")], 0]]
        state = build_state(SUBJECT, target, labels, use_labels=True,
                            state_format="text_v1", omit_system=True)
        self.assertIn("matched", state)
        self.assertIn("outcome: 1", state)
        for value in ("SECRET", "test-model", "private-name", "wrong"):
            self.assertNotIn(value, state)

    def test_custom_questions_require_distinct_version_and_no_extra_fields(self):
        question = {"type": "noul", "instructions": "Will the target system succeed?"}
        with self.assertRaises(ValueError):
            LayaPredictor(router=FakeRouter(), question=question)
        with self.assertRaises(ValueError):
            LayaPredictor(router=FakeRouter(), question=dict(question, criteria="hidden"), prompt_version="test")
        router = FakeRouter()
        predictor = LayaPredictor(router=router, question=question, prompt_version="test",
                                  state_format="text_v1", omit_system=True)
        predictor(SUBJECT, ITEM, [])
        self.assertEqual(router.requests[0]["questions"]["correct"], question)
        self.assertNotIn("test-model", router.requests[0]["state"])

    def test_text_evidence_keeps_numeric_zero_and_anonymous_ids_out_of_text(self):
        labels = [[[SUBJECT, dict(ITEM, item_content="observed")], 0]]
        state = build_state(SUBJECT, ITEM, labels, use_labels=True, state_format="text_v1")
        self.assertIn("outcome: 0", state)
        self.assertNotIn("benchmark_", state)

    def test_unknown_renderer_fails_before_runtime_loading(self):
        with self.assertRaises(ValueError):
            LayaPredictor(state_format="typo")


class CharacterTokenizer:
    cls_token_id, sep_token_id, mask_token_id = 1, 2, 3
    mask_token = "[MASK]"

    def __init__(self):
        self.backend = self

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(c) + 10 for c in text],
                               offsets=[(i, i + 1) for i in range(len(text))])

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text).ids}

    def decode(self, ids):
        return "".join(chr(i - 10) for i in ids if i >= 10)


@unittest.skipUnless(find_spec("laya_mlx"), "Native MLX token audit")
class TokenAuditTests(unittest.TestCase):
    def test_audit_matches_native_sequence_and_tracks_lost_outcomes(self):
        from laya_mlx.common import build_sequence
        from paiec.input_audit import audit_state
        question = {"type": "noul", "instructions": "Will the target succeed?"}
        internal = {"t": "noul", "ins": question["instructions"], "crit": None}
        tok = CharacterTokenizer()
        for fmt in ("json", "text_v1", "context_v1"):
            for length in (5, 900):
                with self.subTest(fmt=fmt, length=length):
                    item = dict(ITEM, item_content="中" * length)
                    labels = [[[SUBJECT, dict(ITEM, item_content="observed")], 0]]
                    state = build_state(SUBJECT, item, labels, use_labels=True, state_format=fmt)
                    row = audit_state(tok, state, question, state_format=fmt)
                    sequence = build_sequence(tok, state, internal, 512, 192,
                                              return_stats=True, return_truncation_stats=True)
                    self.assertEqual(row["sequence_token_ids"], sequence[0])
                    self.assertEqual(row["state_tokens_used"], sequence[3]["state_tokens_used"])
                    self.assertEqual(row["revealed_outcomes"], 1)
                    self.assertEqual(row["revealed_outcomes_used"], int(length == 5))
                    self.assertEqual(row["task_fields"]["item_content"]["complete"], length == 5)
