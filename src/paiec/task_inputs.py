"""Versioned, label-blind task features and native-token-bounded Laya states.

Only projected competition fields enter this module. No data/source lookup,
model-name guessing, reference answers, or generated summaries.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import re

from .competition import competition_input
from .state_rendering import split_settings, setting_value

VERSION = "task_v2"


def structured(text):
    """Decode containers without silently losing duplicate keys or nonfinite values."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        return value if isinstance(value, (dict, list)) else text
    except (ValueError, TypeError, RecursionError):
        return text


def task_sections(text):
    """Keep paths/roles; prioritize the explicit question and last user turn.

    Unknown structures remain data, never instructions to this program. Role
    ordering is deliberately versioned; no benchmark-specific parsing is used.
    """
    value = structured(text)
    sections = []
    def visit(node, path="item_content", priority=2):
        if isinstance(node, dict):
            if "role" in node and "content" in node and isinstance(node["role"], str):
                role = node["role"]
                visit(node["content"], path + "." + role,
                      0 if role == "user" else 4 if role in {"system", "developer"} else 3)
                for key, child in node.items():
                    if key not in {"role", "content"}:
                        visit(child, path + "." + key, priority)
            else:
                for key, child in node.items():
                    p = (0 if key in {"question", "query", "user", "task"} else
                         1 if key in {"options", "choices", "answers"} else
                         4 if key in {"system", "instructions"} else priority)
                    visit(child, path + "." + str(key), p)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]", priority)
        else:
            literal = node if isinstance(node, str) else json.dumps(node, ensure_ascii=False)
            sections.append((priority, path, literal))
    try:
        visit(value)
    except RecursionError:
        sections = [(2,"item_content (deep structure kept literal)",text)]
    # Within an equally important conversation, the most recent user turn wins.
    return sorted(enumerate(sections), key=lambda x: (x[1][0], -x[0] if x[1][1].endswith('.user') else x[0]))


def extract_features(subject, item):
    subject, item = competition_input(subject, item)
    text = item["item_content"]
    settings = split_settings(subject["subject_features_extra"]) or []
    recorded = [{"key": key, "value": setting_value(key, value)} for key, value in settings]
    identifiers = [v for k, v in settings if k in {
        "model_identifier", "model_id", "model_name", "source_model", "base_model", "checkpoint"}]
    cjk = sum('\u4e00' <= c <= '\u9fff' for c in text)
    latin = sum(c.isalpha() and c.isascii() for c in text)
    options = re.findall(r"(?m)^\s*([A-H])[.:：)]\s*", text)
    parts = task_sections(text)
    return {"version": VERSION,
        "system": {"canonical_identity_present": bool(subject["normalized_name"]),
                   "recorded_identifiers": identifiers,
                   "identity_unknown": not bool(subject["normalized_name"] or identifiers),
                   "recorded_settings": recorded,
                   "missing_fields": [k for k, v in subject.items() if not v]},
        "task": {"characters": len(text), "cjk_characters": cjk,
                 "script_hint": "cjk" if cjk > latin else "latin" if latin else "other_or_unknown",
                 "option_count_hint": len(set(options)),
                 "structured": not isinstance(structured(text), str),
                 "section_count": len(parts), "interactors_recorded": bool(item["interactors"])}}


@dataclass
class BuiltState:
    state: str
    features: dict
    visibility: dict

    def audit(self):
        return asdict(self)


class TaskStateBuilder:
    """Budget actual tokenizer tokens, with explicit middle omissions.

    All full source fields stay in the exported source/audit. A single bounded
    state is lossy for long tasks: tail retention prevents common-prefix
    collapse, but does not claim all relevant evidence fits.
    """
    def __init__(self, tokenizer, state_room):
        if state_room < 160:
            raise ValueError("task_v2 needs at least 160 state tokens")
        self.tokenizer, self.room = tokenizer, state_room

    def ids(self, text):
        return self.tokenizer(text.replace(self.tokenizer.mask_token, " "),
                              add_special_tokens=False)["input_ids"]

    def count(self, text):
        return len(self.ids(text))

    def clip(self, text, budget):
        if self.count(text) <= budget:
            return text, False
        # Use character boundaries: decoding a token slice can split a Unicode
        # codepoint. Verify the final native token length after every edit.
        marker = "\n[... middle omitted ...]\n"
        low, high = 0, len(text)
        while low < high:
            n = (low + high + 1) // 2
            left = n // 3
            candidate = text[:left] + marker + (text[-(n-left):] if n-left else "")
            if self.count(candidate) <= budget:
                low = n
            else:
                high = n - 1
        n = low
        candidate = text[:n//3] + marker + (text[-(n-n//3):] if n-n//3 else "")
        if self.count(candidate) > budget:
            candidate = "[omitted]" if self.count("[omitted]") <= budget else ""
        return candidate, True

    def build(self, subject, item):
        subject, item = competition_input(subject, item)
        features = extract_features(subject, item)
        system_parts = []
        for key, value in subject.items():
            if not value:
                continue
            settings = split_settings(value) if key == "subject_features_extra" else None
            system_parts.append(("recorded settings: " + "; ".join(f"{k}={setting_value(k,v)}" for k,v in settings))
                                if settings else f"{key}: {value}")
        system = "\n".join(system_parts)
        system, system_omitted = self.clip(system, max(48, self.room // 5))
        if features["system"]["identity_unknown"]:
            system = "model identity: unknown\n" + system
        metadata = "\n".join(f"{key}: {item[key]}" for key in ("item_features", "interactors") if item[key])
        metadata, metadata_omitted = self.clip(metadata, max(24, self.room // 12))
        task = features["task"]
        hints = f"script_hint={task['script_hint']}; option_count_hint={task['option_count_hint']}"
        header = f"Target AI system\n{system}\n\nDerived task hints: {hints}\n\nTarget task\n"
        footer = f"\n\nRecorded task attributes\n{metadata}" if metadata else ""
        sections = task_sections(item["item_content"])
        structure_fallback = sum(self.count(s[1])+5 for _,s in sections) > self.room // 4
        if structure_fallback:
            # A schema with hundreds of tiny leaves can spend the entire
            # budget on path names. Preserve the literal source as one field.
            sections = [(0,(2,"item_content (large structure kept literal)",item["item_content"]))]
        # Reserve bounded space for low-priority instructions/context, while
        # task-specific content gets the majority. Give every section a share.
        room = self.room - self.count(header + footer) - sum(self.count(s[1])+5 for _,s in sections) - 12
        if room < 0:
            raise ValueError("Too many structured sections for the task_v2 token budget")
        weights = [4 if section[0] <= 1 else 2 if section[0] == 2 else 1 for _, section in sections]
        costs = [self.count(s[2]) for _,s in sections]
        budgets = [0] * len(sections)
        active = list(range(len(sections)))
        while room > 0 and active:
            total = sum(weights[i] for i in active)
            spent = 0
            for i in active:
                amount = min(costs[i]-budgets[i], max(1, room*weights[i]//total), room-spent)
                budgets[i] += amount
                spent += amount
            room -= spent
            active = [i for i in active if budgets[i] < costs[i]]
        rendered, retained = [], []
        for (_, (priority, path, text)), budget in zip(sections, budgets):
            label = path + ":\n"
            content, omitted = self.clip(text, budget)
            rendered.append(label + content)
            retained.append({"path": path, "priority": priority, "original_tokens": self.count(text),
                             "retained_text_tokens": self.count(content), "omitted": omitted})
        state = header + "\n\n".join(rendered) + footer
        # Token counts are not additive across section boundaries. Explicitly
        # budget the full serialization; never rely on the native prefix clamp.
        if self.count(state) > self.room:
            raise ValueError("Too many structured sections for the task_v2 token budget")
        features["task"]["source_tokens"] = self.count(item["item_content"])
        visibility = {"state_tokens": self.count(state), "state_room": self.room,
                      "system_omitted": system_omitted, "metadata_omitted": metadata_omitted,
                      "task_omitted": any(s["omitted"] for s in retained), "sections": retained,
                      "structure_fallback": structure_fallback,
                      "retained_task_sha256": hashlib.sha256(json.dumps(rendered,ensure_ascii=False).encode()).hexdigest(),
                      "source_task_content_sha256": hashlib.sha256(item["item_content"].encode()).hexdigest()}
        return BuiltState(state, features, visibility)
