"""Turn the visible subject fields into typed features, one row per subject.

Everything here reads only the fields a predictor receives (see
``schema.SUBJECT_FIELDS``) and works on one subject at a time, so the table
carries no label information and needs no fitting. Identity falls back from
``normalized_name`` to ``source_model_label`` inside ``subject_features_extra``,
which raises identity coverage from 53% to 67% of sample rows.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Placeholder providers that say nothing about the system.
NO_PROVIDER = {"unknown", "(agent scaffold)"}
DATE_ORIGIN = pd.Timestamp("2020-01-01")

_MIXTURE = re.compile(r"(?<![\w.])(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b(?![a-z])", re.IGNORECASE)
_SIZE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", re.IGNORECASE)
_ACTIVE = re.compile(r"(?<![a-z0-9])a(\d+(?:\.\d+)?)b(?![a-z])", re.IGNORECASE)

FLAGS = {
    "is_instruct": re.compile(r"instruct|chat|(?<![a-z])it(?![a-z])|sft|rlhf|dpo", re.IGNORECASE),
    "is_vision": re.compile(
        r"(?<![a-z])vl(?![a-z])|vision|llava|internvl|pixtral|minicpm-v|paligemma|idefics|molmo|kosmos"
        r"|cogvlm|(?<![a-z])vila(?![a-z])|minigpt-4|ovis|bunny|cambrian",
        re.IGNORECASE,
    ),
    "is_reasoning": re.compile(
        r"(?<![a-z0-9])o[1-4](?![a-z0-9])|(?<![a-z0-9])r1(?![a-z0-9])|reason|thinking|qwq", re.IGNORECASE
    ),
    "is_moe": re.compile(r"\d+\s*x\s*\d+(?:\.\d+)?\s*b|(?<![a-z0-9])a\d+(?:\.\d+)?b(?![a-z])|moe", re.IGNORECASE),
}
# Size tier for systems without a parameter count (most closed models).
# "mini" must not match inside Gemini, MiniMax or MiniGPT.
_SMALL_TIER = re.compile(r"(?<!ge)mini(?!max|gpt)|nano|lite|flash|haiku|small|tiny|micro", re.IGNORECASE)
_LARGE_TIER = re.compile(r"(?<![a-z])(pro|opus|ultra|max|large|plus)(?![a-z])", re.IGNORECASE)

COLUMNS = [
    "subject_id",
    "identity",
    "identity_source",
    "identity_key",
    "release_days",
    "release_year",
    "release_precision",
    "log_params_b",
    "log_active_params_b",
    "family",
    "provider_filled",
    "tier",
    *FLAGS,
    "is_agent",
    "subject_kind",
]


def parse_extra(text) -> dict[str, str]:
    """``k=v;k=v`` into a dict; values keep any further ``=``."""
    if not isinstance(text, str):
        return {}
    pairs = (part.split("=", 1) for part in text.split(";") if "=" in part)
    return {key.strip(): value.strip() for key, value in pairs}


def parse_params(identity) -> tuple[float, float]:
    """Total and active parameters in billions from a name, or NaN.

    "8x7B" is a mixture of 8 experts (56 total); "30B A3B" has 30 total and 3
    active. Without an active count, active equals total.
    """
    if not isinstance(identity, str):
        return math.nan, math.nan
    mixture = _MIXTURE.search(identity)
    if mixture:
        total = float(mixture.group(1)) * float(mixture.group(2))
    else:
        size = _SIZE.search(identity)
        if not size:
            return math.nan, math.nan
        total = float(size.group(1)) / (1000.0 if size.group(2).lower() == "m" else 1.0)
    active = _ACTIVE.search(identity)
    return total, float(active.group(1)) if active else total


def parse_release(text) -> tuple[float, str | None]:
    """Days after 2020-01-01 and the precision (``day`` or ``month``); a bare month is its 1st."""
    if not isinstance(text, str):
        return math.nan, None
    for pattern, precision in (("%Y-%m-%d", "day"), ("%Y-%m", "month")):
        try:
            return float((pd.to_datetime(text, format=pattern) - DATE_ORIGIN).days), precision
        except ValueError:
            continue
    return math.nan, None


def load_families(path: str | Path) -> list[dict]:
    """Read the family table and compile each pattern."""
    families = json.loads(Path(path).read_text())["families"]
    return [{**entry, "regex": re.compile(entry["pattern"], re.IGNORECASE)} for entry in families]


def family_of(identity, families: list[dict]) -> dict | None:
    """The first family whose pattern matches the identity."""
    if not isinstance(identity, str):
        return None
    lowered = identity.lower()
    return next((entry for entry in families if entry["regex"].search(lowered)), None)


def identity_key(identity, provider) -> str | None:
    """A comparable key: no org path, no leading provider name, only lowercase letters and digits."""
    if not isinstance(identity, str):
        return None
    text = identity.split("/")[-1].strip()
    if isinstance(provider, str) and text.lower().startswith(provider.lower() + " "):
        text = text[len(provider) + 1:]
    return re.sub(r"[^a-z0-9]", "", text.lower()) or None


def tier_of(identity) -> str | None:
    if not isinstance(identity, str):
        return None
    if _SMALL_TIER.search(identity):
        return "small"
    if _LARGE_TIER.search(identity):
        return "large"
    return None


def subject_row(row: pd.Series, families: list[dict]) -> dict:
    """All features for one subject from its visible fields."""
    extra = parse_extra(row.get("subject_features_extra"))
    name, label = row.get("normalized_name"), extra.get("source_model_label") or None
    identity = name if isinstance(name, str) else label
    provider = row.get("provider")
    provider = provider if isinstance(provider, str) and provider.lower() not in NO_PROVIDER else None
    family = family_of(identity, families)
    total, active = parse_params(identity)
    release_days, precision = parse_release(row.get("release_date"))
    text = identity or ""
    return {
        "subject_id": row["subject_id"],
        "identity": identity,
        "identity_source": "name" if isinstance(name, str) else ("label" if label else None),
        "identity_key": identity_key(identity, row.get("provider")),
        "release_days": release_days,
        # Coarse on purpose: exact days are nearly unique per system, and trees
        # used them to fit single training pairs instead of the trend.
        "release_year": float(DATE_ORIGIN.year + math.floor(release_days / 365.25))
        if not math.isnan(release_days) else math.nan,
        "release_precision": precision,
        "log_params_b": math.log10(total) if total > 0 else math.nan,
        "log_active_params_b": math.log10(active) if active > 0 else math.nan,
        "family": family["family"] if family else None,
        "provider_filled": provider or (family["provider"] if family else None),
        "tier": tier_of(identity),
        **{flag: int(bool(regex.search(text))) for flag, regex in FLAGS.items()},
        "is_agent": int(text.lower().startswith("agent:") or row.get("provider") == "(agent scaffold)"),
        "subject_kind": extra.get("subject_kind") or None,
    }


def build_subject_features(fields: pd.DataFrame, families: list[dict]) -> pd.DataFrame:
    """One feature row per subject_id from the table ``subject_fields`` writes."""
    if fields["subject_id"].duplicated().any():
        raise ValueError("subject fields must have one row per subject_id")
    rows = [subject_row(row, families) for _, row in fields.iterrows()]
    table = pd.DataFrame(rows, columns=COLUMNS)
    return table.replace({None: np.nan}).sort_values("subject_id").reset_index(drop=True)


__all__ = [
    "COLUMNS",
    "FLAGS",
    "build_subject_features",
    "family_of",
    "identity_key",
    "load_families",
    "parse_extra",
    "parse_params",
    "parse_release",
    "subject_row",
    "tier_of",
]
