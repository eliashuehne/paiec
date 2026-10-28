"""PAIEC submission: logistic prior + Bayesian pair offset with tag shifts (numpy only).

For a target (subject, item): a per-item starting logit from the logistic prior,
scaled and centred, plus a pair shift and per-tag shifts learned from this
subject's revealed labels on the same anonymous benchmark (see paiec_runtime).

It also writes a short diagnostic to stderr describing what ``labeled`` contains
(entries, distinct subjects, other subjects on the same benchmark). Predictions
use only the target subject's own labels on the target benchmark.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paiec_runtime.features import item_vector, subject_vector  # noqa: E402
from paiec_runtime.groups import TagDesign, parse_tags  # noqa: E402
from paiec_runtime.update import tag_offset_posterior, tag_predictive  # noqa: E402

with np.load(HERE / "artifacts" / "prior_offset.npz") as _stored:
    COEF = _stored["coef"].astype(float)
    INTERCEPT, LEVEL, SHIFT, SCALE, CENTER, TAU, SIGMA = (
        float(_stored[name]) for name in ("intercept", "level", "shift", "scale", "center", "tau", "sigma")
    )

CACHE_LIMIT = 20000
DIAGNOSTIC_LIMIT = 400
_item_parts: dict = {}
_subject_parts: dict = {}
_posteriors: dict = {}
_diagnostics_seen: set = set()


def _key(record: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in record.items()))


def _prior_logit(subject: dict, item: dict) -> float:
    item_key = (item.get("item_content"), item.get("item_features"), item.get("interactors"))
    if item_key not in _item_parts:
        if len(_item_parts) >= CACHE_LIMIT:
            _item_parts.clear()
        indices, values = item_vector(item)
        _item_parts[item_key] = float(COEF[indices] @ values)
    subject_key = _key(subject)
    if subject_key not in _subject_parts:
        indices, values = subject_vector(subject)
        _subject_parts[subject_key] = float(COEF[indices] @ values)
    return INTERCEPT + _item_parts[item_key] + _subject_parts[subject_key]


def _start(subject: dict, item: dict) -> float:
    return LEVEL + SHIFT + SCALE * (_prior_logit(subject, item) - CENTER)


def _diagnose(subject: dict, item: dict, labeled) -> None:
    benchmark = item.get("benchmark_id")
    entries = list(labeled or [])
    marker = (benchmark, len(entries))
    if marker in _diagnostics_seen or len(_diagnostics_seen) >= DIAGNOSTIC_LIMIT:
        return
    _diagnostics_seen.add(marker)
    subjects = {_key(s) for (s, _), _ in entries}
    same_bench = [(s, i) for (s, i), _ in entries if i.get("benchmark_id") == benchmark]
    others_same_bench = {_key(s) for s, _ in same_bench if s != subject}
    print(
        "PAIEC-DIAG"
        f" benchmark={benchmark} labeled={len(entries)} distinct_subjects={len(subjects)}"
        f" benchmarks={len({i.get('benchmark_id') for (_, i), _ in entries})}"
        f" same_benchmark={len(same_bench)} own={sum(1 for s, _ in same_bench if s == subject)}"
        f" other_subjects_same_benchmark={len(others_same_bench)}",
        file=sys.stderr,
        flush=True,
    )


def predict(input, labeled=None) -> float:
    subject, item = input
    _diagnose(subject, item, labeled)
    benchmark = item.get("benchmark_id")
    examples = [
        (observed_item, int(label))
        for (observed_subject, observed_item), label in labeled or []
        if observed_subject == subject and observed_item.get("benchmark_id") == benchmark
    ]
    key = (_key(subject), benchmark, tuple((e.get("item_content"), e.get("item_features"), y) for e, y in examples))
    if key not in _posteriors:
        if len(_posteriors) >= CACHE_LIMIT:
            _posteriors.clear()
        design = TagDesign.from_revealed([parse_tags(e.get("item_features", "")) for e, _ in examples])
        revealed = np.array([_start(subject, e) for e, _ in examples], dtype=float)
        mean, covariance = tag_offset_posterior(revealed, design.matrix, [y for _, y in examples], TAU, SIGMA)
        _posteriors[key] = (design, mean, covariance)
    design, mean, covariance = _posteriors[key]
    columns, unseen = design.target_columns(parse_tags(item.get("item_features", "")))
    probability = tag_predictive(_start(subject, item), mean, covariance, columns, unseen, SIGMA)
    return float(min(max(probability, 1e-4), 1 - 1e-4))
