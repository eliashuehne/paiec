"""Inspect the exact MLX input sequence without loading model weights."""

import json
import re

from .config import LayaQuestion


def audit_state(tokenizer, state, question, *, max_len=512, head_max_len=192,
                state_format="json"):
    from laya_mlx.common import build_head, finish_sequence

    q = LayaQuestion.model_validate(question)
    prefix, markers, _ = build_head(tokenizer, {"t": q.type, "ins": q.instructions,
                                               "crit": None}, head_max_len)
    # This replacement is also performed by the native runtime before encoding.
    text = state.replace(tokenizer.mask_token, " ")
    encoded = tokenizer.backend.encode(text, add_special_tokens=False)
    ids, _, stats = finish_sequence(tokenizer, prefix, markers, encoded.ids, max_len)
    offsets = encoded.offsets[:stats["state_tokens_used"]]

    def coverage(start, end):
        relevant = [i for i, (a, b) in enumerate(encoded.offsets) if b > start and a < end]
        used = sum(i < len(offsets) for i in relevant)
        return {"tokens": len(relevant), "tokens_used": used,
                "complete": bool(relevant) and used == len(relevant)}

    fields = {}
    if state_format == "json":
        target_start = text.index('  "target_item":')
        evidence_start = text.find('  "revealed_examples":')
        target_end = evidence_start if evidence_start >= 0 else len(text)
        for match in re.finditer(r'(?m)^    "(item_content|item_features|interactors)": ', text):
            if target_start < match.start() < target_end:
                _, length = json.JSONDecoder().raw_decode(text[match.end():])
                fields[match.group(1)] = coverage(match.end(), match.end() + length)
        system_end = target_start
        outcome_pattern = r'(?m)^      "outcome": [01]'
    else:
        target_start = text.index("Target task\n")
        evidence_start = text.find("\n\nRevealed example ")
        target_end = evidence_start if evidence_start >= 0 else len(text)
        field_matches = list(re.finditer(r'(?m)^(item_content|item_features|interactors): ',
                                         text[target_start:target_end]))
        for i, match in enumerate(field_matches):
            end = (target_start + field_matches[i + 1].start()
                   if i + 1 < len(field_matches) else target_end)
            fields[match.group(1)] = coverage(target_start + match.end(), end)
        system_end = target_start
        outcome_pattern = r'(?m)^outcome: [01]'
    outcomes = [match for match in re.finditer(outcome_pattern, text)
                if evidence_start >= 0 and match.start() > evidence_start]
    return {**stats, "max_len": max_len, "head_tokens": len(prefix),
            "target_system": coverage(0, system_end),
            "target_task": coverage(target_start, target_end), "task_fields": fields,
            "revealed_outcomes": len(outcomes),
            "revealed_outcomes_used": sum(coverage(m.end() - 1, m.end())["complete"] for m in outcomes),
            "retained_state": tokenizer.decode(encoded.ids[:len(offsets)]),
            "sequence_token_ids": ids, "state": state, "question": question}
