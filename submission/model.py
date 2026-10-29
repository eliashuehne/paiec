"""PAIEC submission: logistic prior + pooled Bayesian offset (numpy only).

For a target (subject, item): a per-item starting logit from the logistic prior,
scaled and centred, plus shifts fitted on every revealed label of the same
anonymous benchmark in ``labeled``: a benchmark level and tag shifts shared by
all systems, the target system's own shift, and a shift for a question other
systems have answered (see paiec_runtime.pooling).

It also writes a short diagnostic to stderr describing what ``labeled`` contains
(entries, distinct subjects, other subjects on the same benchmark).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paiec_runtime.features import item_vector, subject_vector  # noqa: E402
from paiec_runtime.groups import parse_tags  # noqa: E402
from paiec_runtime.pooling import PoolPosterior, PoolPrior  # noqa: E402

with np.load(HERE / "artifacts" / "pooled_offset.npz") as _stored:
    COEF = _stored["coef"].astype(float)
    INTERCEPT, LEVEL, SHIFT, SCALE, CENTER, TAU, SIGMA, SHARE, ITEM_SIGMA = (
        float(_stored[name]) for name in
        ("intercept", "level", "shift", "scale", "center", "tau", "sigma", "share", "item_sigma")
    )
PRIOR = PoolPrior(TAU, SHARE, SIGMA, ITEM_SIGMA)

CACHE_LIMIT = 20000
DIAGNOSTIC_LIMIT = 400
_item_parts: dict = {}
_subject_parts: dict = {}
_posteriors: dict = {}
_current = {"labeled": None, "size": -1, "pools": {}}
_diagnostics_seen: set = set()


def _key(record: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in record.items()))


def _item_key(item: dict) -> tuple:
    return item.get("item_content"), item.get("item_features")


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


def _pool(benchmark, labeled) -> PoolPosterior:
    # Targets of one checkpoint share the same ``labeled`` list; fit each benchmark once.
    size = len(labeled or ())
    if labeled is not _current["labeled"] or size != _current["size"]:
        _current.update(labeled=labeled, size=size, pools={})
    pools = _current["pools"]
    if benchmark not in pools:
        examples = [(s, e, int(y)) for (s, e), y in labeled or () if e.get("benchmark_id") == benchmark]
        key = (benchmark, tuple((_key(s), _item_key(e), y) for s, e, y in examples))
        if key not in _posteriors:
            if len(_posteriors) >= CACHE_LIMIT:
                _posteriors.clear()
            _posteriors[key] = PoolPosterior.fit(
                [_start(s, e) for s, e, _ in examples],
                [_key(s) for s, _, _ in examples],
                [parse_tags(e.get("item_features", "")) for _, e, _ in examples],
                [_item_key(e) for _, e, _ in examples],
                [y for _, _, y in examples],
                PRIOR,
            )
        pools[benchmark] = _posteriors[key]
    return pools[benchmark]


def predict(input, labeled=None) -> float:
    subject, item = input
    _diagnose(subject, item, labeled)
    posterior = _pool(item.get("benchmark_id"), labeled)
    probability = posterior.predict(_start(subject, item), _key(subject),
                                    parse_tags(item.get("item_features", "")), _item_key(item))
    return float(min(max(probability, 1e-4), 1 - 1e-4))
