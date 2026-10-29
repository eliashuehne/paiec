"""Tag groups inside one benchmark, decided from the revealed items alone.

``item_features`` holds ``key=value;`` tags (``task=qa3``, ``context_length=2k``).
Some keys split a benchmark into groups that behave differently; others are
per-item identifiers. A key counts as a grouping key when its values repeat
among the revealed items.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

MAX_VALUE_CHARS = 40
MIN_REVEALED = 4
MAX_DISTINCT_SHARE = 0.75
HASHLIKE = re.compile(r"^[0-9a-f]{16,}$|^\d{6,}$")


def parse_tags(item_features: str) -> dict[str, str]:
    """``key=value`` pairs, skipping long or hash-like values."""
    tags = {}
    for part in (item_features or "").split(";"):
        key, sep, value = part.partition("=")
        key, value = key.strip().lower(), value.strip().lower()
        if sep and key and value and len(value) <= MAX_VALUE_CHARS and not HASHLIKE.match(value):
            tags[key] = value
    return tags


def grouping_keys(revealed_tags: list[dict[str, str]]) -> list[str]:
    """Keys with enough observations, multiple values, and some repetition."""
    values: dict[str, list[str]] = {}
    for tags in revealed_tags:
        for key, value in tags.items():
            values.setdefault(key, []).append(value)
    keys = []
    for key, seen in values.items():
        distinct = len(set(seen))
        if len(seen) >= MIN_REVEALED and 2 <= distinct <= MAX_DISTINCT_SHARE * len(seen):
            keys.append(key)
    return sorted(keys)


def tag_values(tags: dict[str, str], keys: list[str]) -> list[str]:
    """The ``key=value`` group memberships of one item under the chosen keys."""
    return [f"{key}={tags[key]}" for key in keys if key in tags]


@dataclass
class TagDesign:
    """The grouping choices and matrix learned from one set of revealed items.

    Column zero is the pair shift. Each remaining column is a tag value, with
    a one for every revealed item that belongs to that group.
    """

    keys: list[str]
    column_of: dict[str, int]
    matrix: np.ndarray

    @classmethod
    def from_revealed(cls, revealed_tags: list[dict[str, str]]) -> TagDesign:
        keys = grouping_keys(revealed_tags)
        memberships = [tag_values(tags, keys) for tags in revealed_tags]
        column_of: dict[str, int] = {}
        for values in memberships:
            for value in values:
                column_of.setdefault(value, len(column_of) + 1)

        matrix = np.zeros((len(revealed_tags), len(column_of) + 1))
        matrix[:, 0] = 1.0
        for row, values in enumerate(memberships):
            columns = [column_of[value] for value in values]
            matrix[row, columns] = 1.0
        return cls(keys, column_of, matrix)

    def target_columns(self, tags: dict[str, str]) -> tuple[list[int], int]:
        """Return this item's known columns and count of unseen tag values."""
        return target_columns(tags, self.keys, self.column_of)


def target_columns(tags: dict[str, str], keys: list[str], column_of: dict[str, int]):
    """Seen tag columns of a target item, and how many of its tag values were never revealed."""
    values = tag_values(tags, keys)
    seen = [column_of[value] for value in values if value in column_of]
    return seen, len(values) - len(seen)


__all__ = ["TagDesign", "grouping_keys", "parse_tags", "tag_values", "target_columns"]
