"""Reference predictors, ordered from trivial to useful.

Every predictor is a callable ``(subject, item, labeled) -> float`` so the
scoring loop does not care which one it runs. The empirical mean is imported
from the baseline submodule rather than reimplemented, so the reference number
is the organizers' predictor and not a copy of it.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

from .schema import canonical_identity, clean

_MIN, _MAX = 0.02, 0.98


def baseline_path() -> Path:
    """Locate the baseline submodule, allowing an override for odd checkouts."""
    override = os.environ.get("PAIEC_BASELINE")
    if override:
        return Path(override) / "empirical_mean" / "model.py"
    return Path(__file__).resolve().parents[2] / "baseline" / "empirical_mean" / "model.py"


def _load_baseline_mean():
    path = baseline_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"baseline submodule missing at {path}; run `git submodule update --init`"
        )
    spec = importlib.util.spec_from_file_location("paiec_baseline_mean", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict


class Constant:
    """The 0.5 predictor the competition uses to explain Brier 0.25."""

    name = "constant"

    def __init__(self, value: float = 0.5):
        self.value = float(value)

    def __call__(self, subject: dict, item: dict, labeled: list) -> float:
        return self.value


class EmpiricalMean:
    """The organizers' predictor: mean acquired label for the pair, else 0.5."""

    name = "empirical_mean"

    def __init__(self):
        self._predict = _load_baseline_mean()

    def __call__(self, subject: dict, item: dict, labeled: list) -> float:
        return float(self._predict([subject, item], labeled))


class SubjectPrior:
    """Cross-benchmark pass rate for the subject, excluding the current benchmark.

    This is the information a cold start actually has: what the system did on
    other benchmarks, with no labels from the one being predicted.
    """

    name = "subject_prior"

    def __init__(self, stats: pd.DataFrame):
        self.stats = stats
        self._by_identity: dict[str, dict[str, tuple[int, int]]] = {}
        for row in stats.itertuples(index=False):
            self._by_identity.setdefault(row.identity, {})[row.benchmark] = (
                int(row.n),
                int(row.successes),
            )
        self._global = (
            int(stats["successes"].sum()),
            int(stats["n"].sum()),
        )

    def prior(self, identity: str, benchmark: str) -> tuple[float, int]:
        """Smoothed mean and count over every benchmark except this one."""
        successes = total = 0
        for name, (n, hits) in self._by_identity.get(identity, {}).items():
            if name == benchmark:
                continue
            successes += hits
            total += n
        if total == 0:
            successes, total = self._global
        return (successes + 1.0) / (total + 2.0), total

    def __call__(self, subject: dict, item: dict, labeled: list) -> float:
        identity = canonical_identity(subject.get("normalized_name"))
        value, _ = self.prior(identity, clean(item.get("benchmark_id")))
        return min(max(value, _MIN), _MAX)


class ShrunkMean:
    """Acquired pair labels pulled toward the subject's cross-benchmark rate.

    ``k`` is the number of prior observations the cross-benchmark rate is worth.
    At budget 0 this reduces to SubjectPrior; with many labels it approaches the
    empirical mean.
    """

    name = "shrunk_mean"

    def __init__(self, prior: SubjectPrior, k: float = 5.0):
        self.prior = prior
        self.k = float(k)

    def __call__(self, subject: dict, item: dict, labeled: list) -> float:
        benchmark = clean(item.get("benchmark_id"))
        successes = count = 0
        for (observed_subject, observed_item), label in labeled or []:
            if observed_subject != subject:
                continue
            if clean(observed_item.get("benchmark_id")) != benchmark:
                continue
            successes += int(label)
            count += 1
        base, _ = self.prior.prior(canonical_identity(subject.get("normalized_name")), benchmark)
        value = (successes + self.k * base) / (count + self.k)
        return min(max(value, _MIN), _MAX)


__all__ = [
    "Constant",
    "EmpiricalMean",
    "SubjectPrior",
    "ShrunkMean",
    "baseline_path",
]
