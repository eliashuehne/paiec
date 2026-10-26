"""CPU CatBoost fitting with training-family-only configuration selection.

The model is a binary log-loss classifier over the structured supplied-field
features. Sample weights give every visible system/benchmark scope equal total
weight; classes are not rebalanced. Depth, learning rate and L2 come from a
small predeclared grid; the winner, its iteration count and early stopping are
chosen from inner training-family folds, never from the held outer family.
"""
import numpy as np

try:  # CatBoost is imported only where the CPU fitting environment provides it.
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - native package absent from model environments
    CatBoostClassifier = None

from .conditional_analysis import family_training_mask

FIT_KEYS = ("depth", "learning_rate", "l2_leaf_reg")


def checked_configs(configs):
    """Validate the predeclared grid; no unbounded search is accepted."""
    if not isinstance(configs, list) or not configs:
        raise ValueError("A nonempty predeclared configuration grid is required")
    checked = []
    for config in configs:
        if not isinstance(config, dict) or set(config) != set(FIT_KEYS):
            raise ValueError("Each fit configuration needs exactly depth, learning_rate and l2_leaf_reg")
        depth, rate, l2 = config["depth"], config["learning_rate"], config["l2_leaf_reg"]
        if type(depth) is not int or not 2 <= depth <= 8:
            raise ValueError("Depth must be an integer in [2, 8]")
        if type(rate) not in {int, float} or not 0 < rate <= .5:
            raise ValueError("Learning rate must be in (0, .5]")
        if type(l2) not in {int, float} or not l2 > 0:
            raise ValueError("L2 regularization must be positive")
        checked.append({"depth": depth, "learning_rate": float(rate), "l2_leaf_reg": float(l2)})
    if len({tuple(sorted(config.items())) for config in checked}) != len(checked):
        raise ValueError("Duplicate fit configuration")
    return checked


def scope_weights(scopes):
    """Inverse scope frequency normalised to mean one; no class rebalancing."""
    scopes = np.asarray(scopes)
    if scopes.ndim != 1 or not len(scopes):
        raise ValueError("Scope weights need a nonempty one-dimensional scope array")
    _, inverse, counts = np.unique(scopes, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    return weights / weights.mean()


def log_loss(labels, probabilities, *, epsilon=1e-12):
    labels, probabilities = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not len(labels):
        raise ValueError("Log loss needs aligned nonempty arrays")
    if ((labels < 0) | (labels > 1)).any() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Log loss needs probabilities and binary targets")
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def fit_model(frame, labels, weights, cat_features, config, *, iterations, seed, threads,
              evaluation=None, early_stopping_rounds=None):
    """Fit one model; an inner evaluation set enables early stopping only."""
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost is not installed in this environment")
    if set(config) != set(FIT_KEYS):
        raise ValueError("Unexpected fit configuration")
    options = {"iterations": int(iterations), "loss_function": "Logloss", "eval_metric": "Logloss",
               "random_seed": int(seed), "thread_count": int(threads), "verbose": False,
               "allow_writing_files": False, "cat_features": list(cat_features),
               **{key: config[key] for key in FIT_KEYS}}
    if evaluation is not None:
        if not early_stopping_rounds:
            raise ValueError("Early stopping needs a positive patience")
        options.update(od_type="Iter", early_stopping_rounds=int(early_stopping_rounds))
    model = CatBoostClassifier(**options)
    if evaluation is None:
        model.fit(frame, labels, sample_weight=weights)
    else:
        model.fit(frame, labels, sample_weight=weights, eval_set=evaluation)
    return model


def probabilities(model, frame):
    values = np.asarray(model.predict_proba(frame), dtype=float)[:, 1]
    if not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("CatBoost returned invalid probabilities")
    return np.clip(values, 1e-6, 1 - 1e-6)


def select_configuration(frame, labels, scopes, families, groups, outer_train, configs, *, cat_features,
                         seed, threads, max_iterations, early_stopping_rounds, min_iterations=50):
    """Choose a configuration using inner training-family folds only.

    ``outer_train`` is the fitting side of the outer family fold. Every inner
    fold removes one inner family and its shared query tasks, so the outer held
    family never contributes to an inner fit or an early-stopping decision.
    """
    outer_train = np.asarray(outer_train, dtype=bool)
    scopes, families, groups = np.asarray(scopes), np.asarray(families), np.asarray(groups)
    inner_families = sorted(set(families[outer_train]))
    if len(inner_families) < 2:
        raise ValueError("Inner selection needs at least two training families")
    rows = []
    for config in configs:
        fold_rows = []
        for inner_family in inner_families:
            inner_train, inner_held = family_training_mask(families, groups, inner_family)
            inner_train &= outer_train
            inner_held &= outer_train
            if not inner_train.any() or not inner_held.any():
                raise ValueError("Inner fold has an empty fitting or validation side")
            model = fit_model(frame.loc[inner_train], labels[inner_train],
                              scope_weights(scopes[inner_train]), cat_features, config,
                              iterations=max_iterations, seed=seed, threads=threads,
                              evaluation=(frame.loc[inner_held], labels[inner_held]),
                              early_stopping_rounds=early_stopping_rounds)
            held_probabilities = probabilities(model, frame.loc[inner_held])
            fold_rows.append({"inner_held_family": str(inner_family),
                              "logloss": log_loss(labels[inner_held], held_probabilities),
                              "brier": float(np.mean((held_probabilities - labels[inner_held]) ** 2)),
                              "best_iteration": int(model.get_best_iteration()) + 1,
                              "held_rows": int(inner_held.sum())})
        total = sum(row["held_rows"] for row in fold_rows)
        rows.append({"config": dict(config),
                     "mean_inner_logloss": float(sum(row["logloss"] * row["held_rows"] for row in fold_rows) / total),
                     "inner_folds": fold_rows})
    chosen = min(range(len(rows)), key=lambda index: rows[index]["mean_inner_logloss"])
    iterations = int(round(np.median([row["best_iteration"] for row in rows[chosen]["inner_folds"]])))
    return {"selected_config": rows[chosen]["config"], "selected_index": chosen,
            "iterations": max(int(min_iterations), iterations), "candidates": rows}
