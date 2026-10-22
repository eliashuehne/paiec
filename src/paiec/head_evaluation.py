"""Score frozen response exports without redrawing or flattening recorded trials."""
import numpy as np
import pandas as pd
from .compare import pair_curves, alc_from_curves
from .evaluate import EvalResult
from .metrics import BUDGETS, brier, ece


def score_frozen_development(manifest, labels, probabilities):
    frame = manifest.copy().reset_index(drop=True)
    if not (frame.role.eq('development') & frame.pool.eq('evaluation') & frame.exported).all():
        raise ValueError('Only exported development evaluation responses may be scored')
    y, p = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if len(frame) != len(y) or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Response labels and probabilities must align')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid probabilities')
    frame['label'], frame['probability'] = y, p
    keys = ['benchmark', 'subject_id', 'item_id']
    if (frame.groupby(keys).probability.nunique() != 1).any():
        raise ValueError('Repeated trials of the same input have inconsistent predictions')
    grouped = frame.groupby(keys, sort=True).agg(probability=('probability', 'first'),
              n_responses=('label', 'size'), n_outcomes_one=('label', 'sum')).reset_index()
    predictions = pd.concat([grouped.assign(budget=budget) for budget in BUDGETS], ignore_index=True)
    records = []
    for (benchmark, subject), group in frame.groupby(keys[:2], sort=True):
        for budget in BUDGETS:
            records.append(dict(benchmark=benchmark, subject_id=subject, budget=budget,
                                brier=brier(group.label, group.probability),
                                ece=ece(group.label, group.probability), n_responses=len(group),
                                n_items=group.item_id.nunique()))
    scores = pd.DataFrame(records)
    curves = alc_from_curves(pair_curves(predictions))
    curves.index.names = ['benchmark', 'subject_id']
    return EvalResult(scores, scores.groupby('budget').brier.mean(), curves,
                      float(curves.mean()), float(scores.ece.mean())), predictions
