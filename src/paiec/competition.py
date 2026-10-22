"""The information boundary between local scoring data and predictors.

Real benchmark names remain scorer-side. The local anonymous IDs are synthetic,
not an attempt to identify private Codabench IDs. Supplied text is preserved:
we do not remove names that happen to occur inside official input fields.
"""

import hashlib
import re

from .schema import ITEM_FIELDS, SUBJECT_FIELDS, clean

INPUT_CONTRACT = "codabench_v1"


def anonymous_benchmark_id(value: str) -> str:
    value = clean(value)
    if not value or re.fullmatch(r"benchmark_[0-9a-f]+", value):
        return value
    digest = hashlib.sha256(("paiec-local-id-v1:" + value).encode()).hexdigest()[:16]
    return "benchmark_" + digest


def competition_input(subject: dict, item: dict) -> list[dict]:
    """Project onto the exact published keys, with string values and empty nulls."""
    subject = {field: clean(subject.get(field)) for field in SUBJECT_FIELDS}
    item = {field: clean(item.get(field)) for field in ITEM_FIELDS}
    item["benchmark_id"] = anonymous_benchmark_id(item["benchmark_id"])
    return [subject, item]


def competition_labels(labeled: list) -> list:
    """Project only already-revealed evidence; never read outcomes from tables."""
    result = []
    for (subject, item), label in labeled or []:
        if label not in (0, 1):
            raise ValueError("Revealed outcomes must be binary")
        result.append([competition_input(subject, item), int(label)])
    return result
