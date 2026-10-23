"""Matched compact control/enriched inputs; task payload is shared exactly."""
from dataclasses import dataclass
import hashlib
import json
import re

from .competition import competition_input
from .state_rendering import split_settings, setting_value
from .task_inputs import TaskStateBuilder, extract_features, task_sections

VERSION = "compact_v3"
OMISSION = "\n[... middle omitted ...]\n"
OPTION = re.compile(r"(?m)^[ \t]*(?:Option\s+)?([A-H])[.:：)．、][ \t]*")
TEXT_FIELD = re.compile(r"(?im)^[ \t]*(?:#{1,4}[ \t]*)?(question|context|passage|evidence|frage|sachverhalt|kontext|问题|上下文)[ \t]*[:：][ \t]*")
QUESTION_FIELDS = {"question", "frage", "问题"}


class InputCapacityError(ValueError):
    """Explicit options or required system/fact text cannot fit safely."""


@dataclass
class CompactPair:
    baseline: str
    enriched: str
    task_text: str
    features: dict
    visibility: dict
    enrichment: dict


def sections_for_task(text):
    sections = []
    expanded = []
    for _, (priority, path, content) in task_sections(text):
        fields = list(TEXT_FIELD.finditer(content))
        # Only parse a leading marked template with a question/context pair.
        # Do not interpret arbitrary labels deep inside retrieved source text.
        marked_pair = (len(fields) >= 2 and not content[:fields[0].start()].strip()
                       and ((fields[0][1].casefold() in QUESTION_FIELDS)
                            != (fields[1][1].casefold() in QUESTION_FIELDS)))
        if marked_pair:
            first, second = fields[:2]
            expanded.extend([(0 if first[1].casefold() in QUESTION_FIELDS else 2,
                              path + "." + first[1].casefold(), content[first.end():second.start()]),
                             (0 if second[1].casefold() in QUESTION_FIELDS else 2,
                              path + "." + second[1].casefold(), content[second.end():])])
        else:
            expanded.append((priority, path, content))
    for priority, path, content in expanded:
        short_path = path.removeprefix("item_content.")
        matches = list(OPTION.finditer(content))
        is_option_path = bool(re.search(r"(?:^|\.)(?:options|choices|answers)(?:\[|\.|$)", short_path))
        # Detect contiguous A/B/... list entries, not arbitrary scattered letters.
        contiguous = len(matches) >= 2 and [m[1] for m in matches] == list("ABCDEFGH"[:len(matches)])
        if contiguous:
            prefix = content[:matches[0].start()]
            if prefix.strip():
                sections.append(dict(path=short_path, text=prefix, priority=priority, option=False))
            for i, match in enumerate(matches):
                end = matches[i+1].start() if i+1 < len(matches) else len(content)
                sections.append(dict(path=short_path + ".option_" + match[1],
                                     text=content[match.start():end], priority=1, option=True))
        else:
            sections.append(dict(path=short_path, text=content, priority=priority, option=is_option_path))
    return sections


class CompactStateBuilder(TaskStateBuilder):
    """Allocate facts before packing a common task for both experimental arms.

    Context loss is explicit. Entire detected options are mandatory. An oversized
    question/user message can still need middle omission, identically in both
    arms; that is reported separately, never called lossless compression.
    """
    def __init__(self, tokenizer, state_room, profiles):
        super().__init__(tokenizer, state_room)
        self.profiles = profiles

    def system_text(self, subject):
        names = {"normalized_name": "model", "harness": "harness", "provider": "provider",
                 "release_date": "release", "access_date": "access", "harness_version": "harness version",
                 "reasoning_effort": "reasoning"}
        essential = {"normalized_name", "harness", "reasoning_effort", "harness_version"}
        lines = [names[k] + "=" + subject[k] for k in names if subject[k] and k in essential]
        other = [names[k] + "=" + subject[k] for k in names if subject[k] and k not in essential]
        extra = subject["subject_features_extra"]
        settings = split_settings(extra)
        if settings:
            high = {"model_identifier", "model_id", "model_name", "source_model", "base_model", "checkpoint",
                    "max_tokens", "max_output_tokens", "n_shots", "shot_size", "reasoning_effort",
                    "temperature", "top_p", "tools", "enabled_tools"}
            settings = sorted(enumerate(settings), key=lambda x: (x[1][0] not in high, x[0]))
            for _, (key, value) in settings:
                (lines if key in high else other).append(f"{key}={setting_value(key, value)}")
        elif extra:
            other.append("extra=" + extra)
        if not subject["normalized_name"] and not any(k in extra for k in ["model_identifier=", "source_model=", "base_model="]):
            lines.insert(0, "model identity not recorded")
        budget = min(384, max(96, self.room // 5))
        text = "\n".join(lines)
        if self.count(text) > budget:
            raise InputCapacityError("Recorded identity/reasoning/tools/budgets do not fit system allocation")
        omitted = False
        if other:
            extra, omitted = self.clip("\n".join(other), max(0, budget - self.count(text) - 4))
            text += "\n" + extra
        return text, omitted

    def build(self, subject, item):
        subject, item = competition_input(subject, item)
        features = extract_features(subject, item)
        features["version"] = VERSION
        system, system_omitted = self.system_text(subject)
        facts, enrichment = self.profiles.block(subject, self.count)
        attributes = "\n".join(k + "=" + item[k] for k in ["item_features", "interactors"] if item[k])
        attributes, attributes_omitted = self.clip(attributes, min(96, max(24, self.room // 16)))
        header = "Target system (recorded)\n" + system + "\n\n"
        footer = "\n\nRecorded item attributes\n" + attributes if attributes else ""
        fact_header = facts + "\n\n" if facts else ""
        task_header = "Target task\n"
        sections = sections_for_task(item["item_content"])
        labels = [s["path"] + ":\n" for s in sections]
        fallback = sum(self.count(label) + 4 for label in labels) > self.room // 4
        if fallback:
            # Preserve explicit questions/options; collapse excessive low-priority
            # tree paths into a literal context block with an audit flag.
            core = [s for s in sections if s["option"] or s["priority"] == 0]
            sections = core + [dict(path="other supplied context (literal)", text=item["item_content"], priority=2, option=False)]
            labels = [s["path"] + ":\n" for s in sections]
        # Fixed reservation in both arms, including unresolved systems. The
        # baseline task allocation must not encode the lengths of web facts.
        fixed = header + task_header + footer
        room = (self.room - self.count(fixed) - self.profiles.policy.max_added_tokens - 4
                - sum(self.count(label) + 4 for label in labels) - 24)
        if room < 32:
            raise InputCapacityError("No safe task space after system/fact fields")
        costs = [self.count(s["text"]) for s in sections]
        budgets = [0] * len(sections)
        options = [i for i, s in enumerate(sections) if s["option"]]
        if sum(costs[i] for i in options) > room:
            raise InputCapacityError("All answer options do not fit; increase context rather than drop an option")
        for i in options:
            budgets[i] = costs[i]
            room -= costs[i]
        # Core questions take priority over surrounding instructions/schema.
        groups = [[i for i, s in enumerate(sections) if not s["option"] and s["priority"] == 0],
                  [i for i, s in enumerate(sections) if not s["option"] and s["priority"] != 0]]
        reserve = min(192, room // 8) if groups[1] and sum(costs[i] for i in groups[0]) > room else 0
        room -= reserve
        for n, group in enumerate(groups):
            if n == 1:
                room += reserve
            active = list(group)
            while room > 0 and active:
                share = max(1, room // len(active))
                for i in active:
                    amount = min(costs[i] - budgets[i], share, room)
                    budgets[i] += amount
                    room -= amount
                active = [i for i in active if budgets[i] < costs[i]]
        parts, retained = [], []
        for section, label, budget in zip(sections, labels, budgets):
            content, omitted = self.clip(section["text"], budget)
            if section["option"] and omitted:
                raise InputCapacityError("Answer option was shortened")
            parts.append(label + content)
            retained.append(dict(path=section["path"], priority=section["priority"], option=section["option"],
                                 original_tokens=self.count(section["text"]), retained_text_tokens=self.count(content),
                                 omitted=omitted))
        task_text = "\n\n".join(parts)
        baseline = header + task_header + task_text + footer
        enriched = header + fact_header + task_header + task_text + footer
        if max(self.count(baseline), self.count(enriched)) > self.room:
            raise InputCapacityError("Final native token serialization exceeds budget")
        visibility = {"state_room": self.room, "state_tokens": self.count(enriched),
                      "baseline_tokens": self.count(baseline), "system_omitted": system_omitted,
                      "metadata_omitted": attributes_omitted, "task_omitted": any(s["omitted"] for s in retained),
                      "question_or_user_omitted": any(s["omitted"] and s["priority"] == 0 for s in retained),
                      "options_detected": len(options), "options_omitted": 0,
                      "structure_fallback": fallback, "sections": retained,
                      "retained_task_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
                      "source_task_content_sha256": hashlib.sha256(item["item_content"].encode()).hexdigest(),
                      "matched_task_payload": True}
        return CompactPair(baseline, enriched, task_text, features, visibility, enrichment)
