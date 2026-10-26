"""Collect the predictor-visible subject fields into one table keyed by subject_id.

The same ``subject_id`` can appear in several benchmarks; its fields must agree
everywhere, or the table would depend on loading order. ``display_name`` is not
a predictor field, so unlike ``schema.subject_record`` there is no fallback to
it: an empty ``normalized_name`` stays missing.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .schema import SUBJECT_FIELDS, clean

COLUMNS = ["subject_id", *SUBJECT_FIELDS]


def subject_fields(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """One row per subject_id with the visible fields; empty values become NaN."""
    parts = [table.reindex(columns=COLUMNS) for table in tables]
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    fields = pd.concat(parts, ignore_index=True)
    for column in SUBJECT_FIELDS:
        fields[column] = fields[column].map(clean).replace("", np.nan)
    fields = fields.drop_duplicates()
    conflicts = sorted(fields.loc[fields.duplicated("subject_id"), "subject_id"].unique())
    if conflicts:
        raise ValueError(f"subjects with different fields across benchmarks: {conflicts[:5]}")
    return fields.sort_values("subject_id").reset_index(drop=True)


__all__ = ["COLUMNS", "subject_fields"]
