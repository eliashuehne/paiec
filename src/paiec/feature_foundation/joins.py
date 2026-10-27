"""Many-to-one table joins, adapted from upstream lift.join_features.

Source: origin/main b202afd; no LightGBM dependency or implicit key guessing.
"""


def safe_left_join(base, features, keys):
    if not keys or any(key not in base or key not in features for key in keys):
        raise ValueError("Explicit shared keys are required")
    if features[list(keys)].isna().any().any() or features.duplicated(list(keys)).any():
        raise ValueError("Feature keys must be non-null and unique")
    if (set(base) & set(features)) - set(keys):
        raise ValueError("Feature columns collide with existing columns")
    return base.merge(features, on=list(keys), how="left", validate="many_to_one", sort=False)
