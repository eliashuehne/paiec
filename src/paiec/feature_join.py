"""Checked many-to-one joins shared by lift scoring and feature preparation."""


def safe_left_join(base, features, keys):
    """Preserve base rows; refuse missing/duplicate keys and column collisions."""
    if not keys or any(key not in base or key not in features for key in keys):
        raise ValueError("Explicit shared keys are required")
    if features[list(keys)].isna().any().any() or features.duplicated(list(keys)).any():
        raise ValueError("Feature keys must be non-null and unique")
    if (set(base) & set(features)) - set(keys):
        raise ValueError("Feature columns collide with existing columns")
    return base.merge(features, on=list(keys), how="left", validate="many_to_one", sort=False)
