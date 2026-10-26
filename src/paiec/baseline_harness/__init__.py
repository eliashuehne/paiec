"""The benchmark-fold baseline harness, kept intact from ``main`` under one package.

``main`` and this branch each wrote a harness at ``paiec.schema`` / ``pairs`` /
``evaluate`` / ``predictors`` and ``scripts/run_eval.py``. They are different
programs with the same paths: this package scores the reference predictors
(constant, organizers' empirical mean) with benchmark-level cross-validation,
while the top-level modules carry the submission harness and the structured
models. Both are kept; nothing here is imported by the modelling code.
"""
