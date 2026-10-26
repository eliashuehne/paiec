"""Full-input observables; scripts and syntax markers are explicitly heuristics."""
import hashlib
import json
import re

from ..head_data import task_key as legacy_task_key
from ..split import stable_seed

PATTERNS = {
    "numeric_spans": r"\d+(?:\.\d+)?", "cjk_characters": r"[\u3400-\u9fff]",
    "cyrillic_characters": r"[\u0400-\u04ff]", "arabic_characters": r"[\u0600-\u06ff]",
    "option_marker_count_hint": r"(?m)^\s*[A-Ha-h][).:]\s+",
}


def task_observables(item):
    if set(item) != {"item_content", "item_features", "interactors"} or any(type(v) is not str for v in item.values()):
        raise ValueError("Expected only the three visible task string fields")
    text = item["item_content"]
    result = {"content_characters": len(text), "whitespace_words": len(text.split()),
              "lines": text.count("\n") + bool(text),
              "paragraphs": len(re.split(r"\n\s*\n", text)) if text else 0,
              **{name: len(re.findall(pattern, text)) for name, pattern in PATTERNS.items()},
              "has_code_fence": "```" in text, "has_math_marker_hint": bool(re.search(r"\$|\\frac|\\sum|\\sqrt", text)),
              "has_table_pipe_hint": "|" in text, "has_supplied_item_features": bool(item["item_features"]),
              "has_supplied_interactors": bool(item["interactors"]), "structure_hint": "literal",
              "structured_option_count_hint": 0, "section_count_hint": 0, "message_count_hint": 0,
              "has_visible_asset_reference_hint": bool(re.search(
                  r'asset_path|input_files|sensor_files|\.(?:csv|parquet|sqlite|png|jpg|jpeg|mp4|wav|mp3|txt)\b',
                  text + "\n" + item["item_features"], re.IGNORECASE))}
    if text.lstrip().startswith(("{", "[")):
        try:
            value = json.loads(text)
            result["structure_hint"] = "json_object" if isinstance(value, dict) else "message_list_hint" if isinstance(value, list) else "json_scalar"
            if isinstance(value, (dict, list)):
                result["section_count_hint"] = len(value)
            if isinstance(value, dict):
                options = value.get("options", value.get("choices"))
                if isinstance(options, (dict, list)):
                    result["structured_option_count_hint"] = len(options)
            messages = value.get("messages", []) if isinstance(value, dict) else value
            if isinstance(messages, list):
                result["message_count_hint"] = sum(isinstance(m, dict) and "role" in m and "content" in m for m in messages)
        except (ValueError, RecursionError):
            result["structure_hint"] = "invalid_json_hint"
    return result


def content_group(text):
    normalized = legacy_task_key(text)
    return hashlib.sha256((normalized or text).encode()).hexdigest()


def effective_split(content, population_row, legacy_monitor_seed):
    if population_row["split"] == "training" and population_row.get("frozen_role") == "training" and legacy_monitor_seed is not None:
        key = hashlib.sha256(legacy_task_key(content).encode()).hexdigest()
        if stable_seed(legacy_monitor_seed, "training-only-monitor", key) % 20 == 0:
            return "validation"
    return population_row["split"]
