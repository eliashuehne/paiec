"""Sparse features of the subject and item dictionaries that ``predict`` receives.

Pure numpy and standard library, so the same code runs offline for training and
inside the submission. Each vector is ``(indices, values)`` in one space of
``DIM`` slots: the first ``N_DENSE`` slots hold hand-made numeric features, the
rest hold hashed tokens. Item and subject vectors are separate so their logit
contributions can be cached independently.
"""

from __future__ import annotations

import math
import re
import zlib

import numpy as np

FEATURE_VERSION = 1
DIM = 2**16

ITEM_DENSE = (
    "log_chars", "log_words", "log_lines", "digit_share", "non_ascii_share", "has_code",
    "has_math", "has_options", "chance_rate", "has_question", "is_json", "empty_content",
    "has_interactors", "has_item_features",
)
SUBJECT_DENSE = (
    "release_offset", "release_missing", "effort_level", "effort_missing", "log_params",
    "params_missing", "name_missing", "provider_missing",
)
DENSE = ITEM_DENSE + SUBJECT_DENSE
N_DENSE = len(DENSE)
SLOT = {name: index for index, name in enumerate(DENSE)}

CONTENT_CHARS = 2000
VALUE_CHARS = 40
TOKEN = re.compile(r"\w+")
OPTION = re.compile(r"(?m)^\s*\(?([A-Ha-h])[).:]\s+\S")
CODE = re.compile(r"```|\bdef |\bimport |\breturn\b|#include|\bfunction\b|;\s*$|\{\s*$", re.M)
MATH = re.compile(r"\\frac|\\sum|\\int|\$[^$]+\$|\\\(|\^\{|\\sqrt")
PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*([bm])\b", re.I)
YEAR = re.compile(r"(19|20)\d{2}")
EFFORT = {"none": 0.0, "minimal": 0.5, "low": 1.0, "medium": 2.0, "high": 3.0, "xhigh": 4.0}


def _slot(token: str) -> int:
    return N_DENSE + zlib.crc32(token.encode("utf-8")) % (DIM - N_DENSE)


def _hashed(tokens: list[str]) -> dict[int, float]:
    """Unique tokens, each 1/sqrt(n) so long texts don't get larger weights."""
    unique = set(tokens)
    if not unique:
        return {}
    value = 1.0 / math.sqrt(len(unique))
    slots: dict[int, float] = {}
    for token in unique:
        slot = _slot(token)
        slots[slot] = slots.get(slot, 0.0) + value
    return slots


def _key_values(text: str, prefix: str) -> list[str]:
    tokens = []
    for part in text.split(";"):
        key, _, value = part.partition("=")
        key = key.strip().lower()
        if key:
            tokens.append(f"{prefix}k|{key}")
            if value:
                tokens.append(f"{prefix}v|{key}={value.strip().lower()[:VALUE_CHARS]}")
    return tokens


def _to_arrays(dense: dict[str, float], hashed: dict[int, float]):
    indices = [SLOT[name] for name, value in dense.items() if value] + list(hashed)
    values = [value for value in dense.values() if value] + list(hashed.values())
    return np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.float64)


def item_vector(item: dict):
    """Sparse features of one item dictionary."""
    content = item.get("item_content") or ""
    features = item.get("item_features") or ""
    interactors = item.get("interactors") or ""
    head = content[:CONTENT_CHARS]
    words = TOKEN.findall(head.lower())
    options = set(OPTION.findall(head))
    n_chars = max(len(head), 1)

    dense = {
        "log_chars": math.log1p(len(content)) / 10,
        "log_words": math.log1p(len(content.split())) / 10,
        "log_lines": math.log1p(content.count("\n")) / 5,
        "digit_share": sum(c.isdigit() for c in head) / n_chars,
        "non_ascii_share": sum(ord(c) > 127 for c in head) / n_chars,
        "has_code": float(bool(CODE.search(head))),
        "has_math": float(bool(MATH.search(head))),
        "has_options": float(len(options) >= 2),
        "chance_rate": 1.0 / len(options) if len(options) >= 2 else 0.0,
        "has_question": float("?" in head),
        "is_json": float(head.lstrip()[:1] in ("{", "[")),
        "empty_content": float(not content.strip()),
        "has_interactors": float(bool(interactors.strip())),
        "has_item_features": float(bool(features.strip())),
    }
    tokens = [f"w|{w}" for w in words]
    tokens += [f"b|{a} {b}" for a, b in zip(words, words[1:])]
    tokens += _key_values(features, "f")
    tokens += [f"i|{w}" for w in TOKEN.findall(interactors[:200].lower())]
    return _to_arrays(dense, _hashed(tokens))


def _params_billion(text: str) -> float | None:
    for number, unit in PARAMS.findall(text):
        value = float(number) * (1.0 if unit.lower() == "b" else 0.001)
        if 0.01 <= value <= 5000:
            return value
    return None


def subject_vector(subject: dict):
    """Sparse features of one subject dictionary."""
    name = subject.get("normalized_name") or ""
    provider = subject.get("provider") or ""
    harness = subject.get("harness") or ""
    version = subject.get("harness_version") or ""
    effort = (subject.get("reasoning_effort") or "").strip().lower()
    extra = subject.get("subject_features_extra") or ""
    year = YEAR.search(subject.get("release_date") or "")
    params = _params_billion(f"{name} {extra[:300]}")

    dense = {
        "release_offset": (int(year.group(0)) - 2024) / 2 if year else 0.0,
        "release_missing": float(year is None),
        "effort_level": EFFORT.get(effort, 0.0) / 3,
        "effort_missing": float(effort not in EFFORT),
        "log_params": math.log10(params) / 2 if params else 0.0,
        "params_missing": float(params is None),
        "name_missing": float(not name.strip()),
        "provider_missing": float(not provider.strip()),
    }
    tokens = [f"p|{provider.strip().lower()}", f"h|{harness.strip().lower()}",
              f"hv|{harness.strip().lower()} {version.strip().lower()}", f"e|{effort}"]
    tokens += [f"n|{w}" for w in TOKEN.findall(name.lower())]
    tokens += _key_values(extra[:600], "x")
    return _to_arrays(dense, _hashed(tokens))


__all__ = ["DENSE", "DIM", "FEATURE_VERSION", "N_DENSE", "item_vector", "subject_vector"]
