"""CPU ridge fitting for explicit, dataset-only conditional controls."""
import math
import numpy as np

from .conditional_features import VERSION, DIMS


def fit_conditional_ridge(matrix, labels, sample_weight, *, interactions, l2=100.):
    x, y, weights = np.asarray(matrix, dtype=np.float64), np.asarray(labels, dtype=float), np.asarray(sample_weight, dtype=float)
    if (x.ndim != 2 or x.shape[1] != DIMS or len(x) < 2 or y.shape != (len(x),) or weights.shape != y.shape or
        not np.isfinite(x).all() or not np.isin(y, [0, 1]).all() or not np.isfinite(weights).all() or
        (weights <= 0).any() or not math.isfinite(l2) or l2 <= 0 or type(interactions) is not bool):
        raise ValueError("Fitting requires aligned finite features, binary targets and positive weights/regularization")
    x = x.copy()
    if not interactions:
        x[:, -128:] = 0.
    weights = weights / weights.mean()
    center = np.average(x, axis=0, weights=weights)
    intercept = float(np.average(y, weights=weights))
    centered = x - center
    gram = centered.T @ (weights[:, None] * centered) + l2 * np.eye(DIMS)
    coefficients = np.linalg.solve(gram, centered.T @ (weights * (y - intercept)))
    return {"version": "conditional_ridge_v1", "feature_version": VERSION, "source_policy": "provided_dataset_only",
            "dimensions": DIMS, "interactions": interactions, "l2": l2, "center": center.tolist(),
            "intercept": intercept, "weights": coefficients.tolist()}


def predict_matrix(state, matrix):
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != DIMS or not np.isfinite(x).all():
        raise ValueError("Prediction matrix violates the feature contract")
    if not state["interactions"]:
        x = x.copy()
        x[:, -128:] = 0.
    return np.clip(state["intercept"] + (x - np.asarray(state["center"])) @ np.asarray(state["weights"]), .001, .999)
