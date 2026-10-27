"""Estimate missing release dates from family, product line and version.

Within a family, version numbers order models in time, so dated versions
around an undated one bracket its release. Most undated subjects are other
spellings of dated systems ("Claude3-5V_Sonnet" is "Claude 3.5 Sonnet"), so a
signature match (family, line, version, size) comes before interpolation.

A date inside a name is only an upper bound: names such as
"gpt-4o-2025-12-11" carry the evaluation snapshot, not the release.

References come from other folds only. At evaluation the predictor can look
dates up only in public data, so a subject never uses a date from a benchmark
in its own fold. The release date is a visible field, not a label.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .lift_sample import SEALED

SOURCES = (
    "field",
    "same_identity",
    "same_signature",
    "same_version",
    "interpolated",
    "one_sided",
    "name_date",
    "family",
    "provider",
)
# Typical gap between consecutive versions, used when only one side is dated.
ONE_SIDED_GAP_DAYS = 180.0
DATE_ORIGIN = pd.Timestamp("2020-01-01")
_NAME_DATES = (
    re.compile(r"(?<!\d)(20[12]\d)-(\d{2})-(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20[12]\d)(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(2[3-6])(\d{2})(\d{2})(?!\d)"),
)


def _normalized(identity: str) -> str:
    return identity.lower().replace("_", "-")


def parse_version(identity, entry: Mapping | None) -> float:
    """Major.minor version from the family's pattern; NaN if absent or a parameter count."""
    if not isinstance(identity, str) or not entry or not entry.get("version_pattern"):
        return math.nan
    text = _normalized(identity)
    for match in re.finditer(entry["version_pattern"], text):
        after = text[match.end():match.end() + 2]
        if re.match(r"[bm](?![a-z])", after):
            continue  # "7b" is a size
        minor = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        if minor is None and text[match.end(1):match.end(1) + 1].isdigit():
            continue  # "llama 34b": the 3 is part of a size, not a version
        return float(match.group(1)) + (float(minor) / 10 if minor else 0.0)
    return math.nan


def parse_line(identity, entry: Mapping | None) -> str | None:
    """The first product line whose pattern matches (sonnet, flash, vl, ...)."""
    if not isinstance(identity, str) or not entry:
        return None
    text = _normalized(identity)
    return next((line for line, pattern in entry.get("lines", {}).items() if re.search(pattern, text)), None)


def name_date(identity) -> float:
    """Days after 2020-01-01 of a valid date written in the name, or NaN."""
    if not isinstance(identity, str):
        return math.nan
    for pattern in _NAME_DATES:
        for match in pattern.finditer(identity):
            year, month, day = match.groups()
            year = int(year) + (2000 if len(year) == 2 else 0)
            try:
                return float((pd.Timestamp(year=year, month=int(month), day=int(day)) - DATE_ORIGIN).days)
            except ValueError:
                continue
    return math.nan


def add_version_columns(features: pd.DataFrame, families: list[dict]) -> pd.DataFrame:
    """Add ``line``, ``version`` and ``name_date_days`` from each subject's identity and family."""
    by_family = {entry["family"]: entry for entry in families}
    entries = [by_family.get(family) if isinstance(family, str) else None for family in features["family"]]
    return features.assign(
        line=[parse_line(i, e) for i, e in zip(features["identity"], entries)],
        version=[parse_version(i, e) for i, e in zip(features["identity"], entries)],
        name_date_days=features["identity"].map(name_date),
    )


def _keys(row) -> dict:
    """Lookup keys for one subject; a key is None when a part it needs is missing.

    A missing product line is a valid value (""), a missing family, version or
    size is not.
    """
    family = row.family if isinstance(row.family, str) else None
    line = row.line if isinstance(row.line, str) else ""
    version = None if _missing(row.version) else float(row.version)
    size = None if _missing(row.log_params_b) else round(float(row.log_params_b), 2)
    provider = row.provider_filled if isinstance(row.provider_filled, str) else None
    identity = row.identity_key if isinstance(row.identity_key, str) else None
    versioned = family is not None and version is not None
    return {
        "same_identity": ("identity", identity) if identity else None,
        "same_signature": ("signature", family, line, version, size) if versioned and size is not None else None,
        "same_version": ("version", family, line, version) if versioned else None,
        "family_versions": (family, line, version) if versioned else None,
        "family": ("family", family) if family else None,
        "provider": ("provider", provider) if provider else None,
    }


class References:
    """Dated subjects indexed for each estimation tier."""

    def __init__(self, dated: pd.DataFrame):
        self.dates = dated["release_days"].to_numpy(float)
        self.first, self.last = (float(self.dates.min()), float(self.dates.max())) if len(dated) else (math.nan,) * 2
        self.by = defaultdict(list)
        for row in dated.itertuples(index=False):
            date = float(row.release_days)
            keys = _keys(row)
            for tier in ("same_identity", "same_signature", "same_version", "family", "provider"):
                if keys[tier]:
                    self.by[keys[tier]].append(date)
            if keys["family_versions"]:
                family, line, version = keys["family_versions"]
                self.by["family_versions", family, line].append((version, date))
                self.by["family_versions", family, "*"].append((version, date))

    def estimate(self, row) -> tuple[float, float, float, str | None]:
        """(estimate, earliest, latest, source) for one undated subject."""
        keys = _keys(row)
        for source in ("same_identity", "same_signature", "same_version"):
            dates = self.by.get(keys[source], []) if keys[source] else []
            if dates:
                return float(np.median(dates)), min(dates), max(dates), source
        if keys["family_versions"]:
            family, line, version = keys["family_versions"]
            found = self._between(family, line, version) or self._between(family, "*", version)
            if found:
                return found
        for source in ("family", "provider"):
            dates = self.by.get(keys[source], []) if keys[source] else []
            if dates:
                return float(np.median(dates)), min(dates), max(dates), source
        return math.nan, math.nan, math.nan, None

    def _between(self, family, line, version):
        """Interpolate between the nearest dated versions below and above."""
        pairs = self.by.get(("family_versions", family, line), [])
        lower = [(v, d) for v, d in pairs if v < version]
        higher = [(v, d) for v, d in pairs if v > version]
        if lower and higher:
            below = np.median([d for v, d in lower if v == max(v for v, _ in lower)])
            above = np.median([d for v, d in higher if v == min(v for v, _ in higher)])
            low, high = sorted((float(below), float(above)))
            return (low + high) / 2, low, high, "interpolated"
        if lower:
            below = float(np.median([d for v, d in lower if v == max(v for v, _ in lower)]))
            return min(below + ONE_SIDED_GAP_DAYS, self.last), below, self.last, "one_sided"
        if higher:
            above = float(np.median([d for v, d in higher if v == min(v for v, _ in higher)]))
            return max(above - ONE_SIDED_GAP_DAYS, self.first), self.first, above, "one_sided"
        return None


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _apply_name_date(estimate, low, high, source, row, first):
    """A name date caps the estimate; with no other estimate it becomes one."""
    cap = row.name_date_days
    if _missing(cap):
        return estimate, low, high, source
    if source is None:
        return cap, first, cap, "name_date"
    return min(estimate, cap), min(low, cap), min(high, cap), source


def subject_folds(sample: pd.DataFrame) -> dict[str, frozenset]:
    """Folds of the benchmarks each subject appears in."""
    return {s: frozenset(f) for s, f in sample.groupby("subject_id")["fold"].unique().items()}


def fill_release_dates(
    features: pd.DataFrame,
    folds_of: Mapping[str, frozenset],
    accepted: Iterable[str] | None = SOURCES,
    excluded_folds: Iterable[str] = (),
) -> pd.DataFrame:
    """Release estimates per subject, using references from other folds only.

    ``features`` needs the version columns (``add_version_columns``). Subjects
    with a real date keep it (source ``field``). Estimates from sources not in
    ``accepted`` are dropped. Subjects outside the sample are references for
    everyone; sealed subjects are references for no one. With ``accepted=None``,
    methods are selected by back-testing that same permitted reference set.
    ``excluded_folds`` additionally excludes the outer validation fold when
    rebuilding training features inside the lift loop.
    """
    fixed_sources = set(accepted) | {"field"} if accepted is not None else None
    excluded_folds = set(excluded_folds) | {SEALED}
    dated = features[features["release_days"].notna()]
    dated_folds = dated["subject_id"].map(lambda s: folds_of.get(s, frozenset()))
    result = {}
    undated = features[features["release_days"].isna()]
    for fold_set, group in undated.groupby(undated["subject_id"].map(lambda s: folds_of.get(s, frozenset()))):
        usable = dated_folds.map(lambda f: not (f & (fold_set | excluded_folds)))
        permitted = dated.loc[usable.to_numpy(dtype=bool)]
        # Select methods using exactly the reference set available to this group.
        sources = fixed_sources
        if sources is None:
            sources = set(accepted_sources(summarise_backtest(backtest(permitted)))) | {"field"}
        references = References(permitted)
        for row in group.itertuples(index=False):
            estimate = _apply_name_date(*references.estimate(row), row, references.first)
            result[row.subject_id] = estimate if estimate[3] in sources else (math.nan,) * 3 + (None,)
    rows = []
    for row in features.itertuples(index=False):
        if not _missing(row.release_days):
            rows.append((row.subject_id, row.release_days, row.release_days, row.release_days, "field"))
        else:
            rows.append((row.subject_id, *result[row.subject_id]))
    table = pd.DataFrame(rows, columns=["subject_id", "release_days_filled", "release_lo", "release_hi", "release_source"])
    table["release_filled"] = (table["release_source"].notna() & (table["release_source"] != "field")).astype(int)
    table["release_year_filled"] = [
        float(DATE_ORIGIN.year + math.floor(d / 365.25)) if not _missing(d) else math.nan
        for d in table["release_days_filled"]
    ]
    return table


def backtest(features: pd.DataFrame, leave_out: str = "identity") -> pd.DataFrame:
    """Hide each dated identity in turn and estimate its date from the rest.

    ``leave_out="identity"`` removes only that identity's subjects, so other
    spellings of the same system stay (the realistic case for undated
    subjects). ``"signature"`` also removes every subject with the same family,
    line and version, which tests interpolation.
    """
    dated = features[features["release_days"].notna() & features["identity_key"].notna()]
    rows = []
    for key, group in dated.groupby("identity_key"):
        first = group.iloc[0]
        hide = dated["identity_key"] == key
        if leave_out == "signature" and isinstance(first.family, str) and not _missing(first.version):
            line = first.line if isinstance(first.line, str) else ""
            hide |= (dated["family"] == first.family) & (dated["version"] == first.version) & (
                dated["line"].fillna("") == line)
        references = References(dated[~hide.to_numpy()])
        probe = first.copy()
        estimate, low, high, source = _apply_name_date(*references.estimate(probe), probe, references.first)
        truth = float(np.median(group["release_days"]))
        rows.append({"identity_key": key, "truth": truth, "estimate": estimate, "low": low, "high": high,
                     "source": source})
    return pd.DataFrame(rows, columns=["identity_key", "truth", "estimate", "low", "high", "source"])


def summarise_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Per source: count, median absolute error in days, share within 90 and 180 days, bracket coverage."""
    scored = results.dropna(subset=["estimate"]).assign(error=lambda d: (d["estimate"] - d["truth"]).abs())
    scored["inside"] = (scored["truth"] >= scored["low"]) & (scored["truth"] <= scored["high"])
    table = scored.groupby("source").agg(
        n=("error", "size"),
        median_abs_days=("error", "median"),
        within_90=("error", lambda e: float((e <= 90).mean())),
        within_180=("error", lambda e: float((e <= 180).mean())),
        bracket_coverage=("inside", "mean"),
    )
    order = [s for s in SOURCES if s in table.index]
    return table.loc[order].reset_index()


def accepted_sources(
    summary: pd.DataFrame, max_median_days: float = 180.0, min_within_180: float = 0.75
) -> list[str]:
    """Sources with a median back-test error below ``max_median_days`` and enough estimates within 180 days.

    The second rule drops crude tiers whose median looks fine only because
    many systems cluster around the same recent dates (provider: 119 days
    median, but 62% within 180 days).
    """
    keep = (summary["median_abs_days"] < max_median_days) & (summary["within_180"] >= min_within_180)
    return summary.loc[keep, "source"].tolist()


__all__ = [
    "ONE_SIDED_GAP_DAYS",
    "SOURCES",
    "References",
    "accepted_sources",
    "add_version_columns",
    "backtest",
    "fill_release_dates",
    "name_date",
    "parse_line",
    "parse_version",
    "subject_folds",
    "summarise_backtest",
]
