"""Portable, full-input features; standard library only, with no outside data."""
from datetime import date
import math
import re
from zlib import crc32
from .competition_fields import SUBJECT_FIELDS, ITEM_FIELDS, checked_fields

VERSION = "conditional_fields_v1"
TASK_DIMS, SYSTEM_DIMS, INTERACTION_DIMS = 256, 64, 128
NUMERIC_NAMES = ("log_task_characters", "log_item_feature_characters", "log_interactor_characters",
                 "task_digit_fraction", "task_nonascii_fraction", "release_year", "release_date_known",
                 "access_year", "access_date_known", "log_nominal_dense_parameters", "nominal_dense_parameters_known",
                 *["missing_" + key for key in SUBJECT_FIELDS])
DIMS = TASK_DIMS + SYSTEM_DIMS + len(NUMERIC_NAMES) + INTERACTION_DIMS
_TOKENS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_DENSE_SIZE = re.compile(r"(?<![\w.xX])\d+(?:\.\d+)?\s*[bBmM](?!\w)")


def normalize(values):
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm else list(values)


def bag_features(record, keys, dims):
    """Consume all supplied tokens and Unicode bigrams, without text clipping."""
    counts = [0.] * dims
    def add(token):
        hashed = crc32(token.encode("utf-8"))
        counts[hashed % dims] += 1. if hashed & 0x80000000 else -1.
    for key in keys:
        previous = None
        for match in _TOKENS.finditer(record[key].casefold()):
            token = match.group()
            add(key + ":u:" + token)
            if previous is not None:
                add(key + ":b:" + previous + "\x1f" + token)
            # Unicode scripts without whitespace still get local character features.
            if any(ord(c) > 127 for c in token):
                for i in range(len(token) - 1):
                    add(key + ":c:" + token[i:i + 2])
            previous = token
    return normalize([math.copysign(math.log1p(abs(v)), v) for v in counts])


def task_features(item):
    return bag_features(item, ("item_content", "item_features", "interactors"), TASK_DIMS)


def system_features(subject):
    return bag_features(subject, SUBJECT_FIELDS, SYSTEM_DIMS)


def numeric_features(subject, item):
    text = item["item_content"]
    values = [math.log1p(len(item[k])) / 10 for k in ("item_content", "item_features", "interactors")]
    values += [sum(c.isdigit() for c in text) / max(1, len(text)),
               sum(ord(c) > 127 for c in text) / max(1, len(text))]
    for key in ("release_date", "access_date"):
        try:
            year = date.fromisoformat(subject[key]).year
            values += [(year - 2000) / 50, 1.]
        except ValueError:
            values += [0., 0.]
    # A supplied simple 8B label is nominal, not a verified exact model size.
    # Do not multiply MoE labels, or extract 7B from the tail of 8x7B.
    sizes = _DENSE_SIZE.findall(subject["normalized_name"])
    if len(sizes) == 1:
        nominal = sizes[0].replace(" ", "")
        number = float(nominal[:-1]) * (1e9 if nominal[-1].casefold() == "b" else 1e6)
        values += [math.log1p(number) / 30, 1.] if math.isfinite(number) else [0., 0.]
    else:
        values += [0., 0.]
    values += [float(not subject[k]) for k in SUBJECT_FIELDS]
    return values


def feature_vector(subject, item, *, interactions=True, task_vector=None, system_vector=None):
    checked_fields(subject, item)
    task = task_features(item) if task_vector is None else task_vector
    system = system_features(subject) if system_vector is None else system_vector
    if len(task) != TASK_DIMS or len(system) != SYSTEM_DIMS:
        raise ValueError("Cached feature dimensions differ")
    task_fold = normalize([sum(task[i::16]) for i in range(16)])
    system_fold = normalize([sum(system[i::8]) for i in range(8)])
    cross = [t * s for t in task_fold for s in system_fold] if interactions else [0.] * INTERACTION_DIMS
    return [*task, *system, *numeric_features(subject, item), *cross]
