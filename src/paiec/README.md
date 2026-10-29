# Reading the package

Start with `predictors.py`: `build(name, data_root, train_benchmarks)` creates a
fresh predictor and fits it on the supplied training benchmarks. The evaluator
then calls that predictor with `(subject, item, labeled)` to get a probability.
Each revealed example in `labeled` is `[[subject, item], label]`.

| Responsibility | Modules |
| --- | --- |
| Load tables and construct subject/benchmark pairs | `schema.py`, `pairs.py` |
| Choose held-out benchmarks and split a pair's items | `folds.py`, `split.py` |
| Build training examples using those item splits | `episodes.py` |
| Fit and call predictors | `shrinkage.py`, `prior.py`, `offset.py` |
| Encode text features and revealed tag groups | `features.py`, `groups.py` |
| Compute posterior updates and evaluation scores | `update.py`, `metrics.py`, `evaluate.py` |

`contracts.py` describes two interfaces: a callable `Predictor`, and a
`TrainablePredictor` that also has `fit`. They are Python protocols, so a simple
function or a callable class can satisfy them without inheriting implementation
from a shared base class.

Dataclasses group state that belongs together: model parameters, a subject's
episode, aligned logistic training rows, and the tag design and posterior.
`TagOffset` uses these objects to share a fitted posterior across target items
with the same revealed evidence. The numerical formulas remain small functions
in `update.py`.

To add a predictor, implement `__call__` and, if needed, `fit`, then register its
factory in `predictors.REGISTRY`. Keep learned state on the instance so each
cross-validation fold gets an independent model. Run `python -m pytest` from the
repository root to check the package.
