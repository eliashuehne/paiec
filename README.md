# PAIEC

Our entry to the NeurIPS Predictive AI Evaluation Competition: estimate the
probability that an AI system answers an item correctly, given the system's
attributes, the item, and a handful of labels revealed during evaluation.

`main` holds what we share with everyone: the organizers' baseline as a
submodule, the local scoring harness (`src/paiec/`, `scripts/run_eval.py`), the
feature lift loop below, and the links. Experimental predictors live on
their own branches (e.g. `laya`).

## Links

| | |
| --- | --- |
| Competition page | https://aims.stanford.edu/competition |
| Submissions (Codabench) | https://www.codabench.org/competitions/17828/ |
| Codabench ids | competition `17828`, phase `29785` |
| Public training data | https://huggingface.co/datasets/aims-foundations/measurement-db |
| Organizers' baselines | https://github.com/aims-foundations/paiec_baseline |

## Baseline

`baseline/` is a submodule of the organizers' repository, pinned at commit
`cc88a95` (2026-10-15). A fresh clone needs it initialized before anything in
it can run:

```bash
git submodule update --init
```

Their `submit_api.py` handles Codabench authentication and upload. It needs an
approved competition account and the `requests` package, and it stores a token
at `~/.config/paiec/codabench-token` rather than in the archive:

```bash
python baseline/submit_api.py --login
python baseline/submit_api.py dist/<archive>.zip
python baseline/submit_api.py --status <submission-id>
```

Submission archives are built by their tools and validated locally before
upload:

```bash
python baseline/tools/build_mean_zip.py
python baseline/check_submission_zip.py dist/empirical_mean.zip
```

See `baseline/README.md` for the layout of the archive and the rules their
predictors follow.

## Feature lift loop

A fast check of whether a feature helps on benchmarks the model has never seen,
before running the full harness. Usage, guards and current baselines are in
[docs/lift.md](docs/lift.md).

```bash
python scripts/lift/build_lift_sample.py
python scripts/lift/run_lift.py --features runs/features/items.parquet --columns n_chars
```

Subject features (identity, release date, size, family, provider) and the
release-date estimates, with their results, are in [docs/subjects.md](docs/subjects.md).

## Feature preparation

[Feature foundation](docs/feature-foundation.md) builds 14 linked tables over the
full downloaded release: preserved inputs, recorded settings, task measurements
and separate research facts. Its plan-first command and standalone validator
require no model or experiment-tracking service. Generated data stays in `runs/`.
Its supplied-only export adds recorded settings and task measurements to the
[existing lift loop](docs/feature-foundation.md#use-with-the-lift-loop).
