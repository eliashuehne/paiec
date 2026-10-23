# Contributing

Use Python 3.12 and initialize the `baseline/` submodule. See the
[README](README.md) for a fresh checkout and [requirements](requirements/README.md)
for separate MLX, Torch and Modal environments.

## Before sharing a change

From the repo root:

```bash
make check PYTHON=.venv-laya-mlx/bin/python
```

`check` runs unit tests, validates all four checked-in profile catalogs, and
checks diff whitespace. For runtime or training changes, also run the suite in
`.venv-laya`; each environment skips checks for its absent native framework.
Core CI uses Python 3.12 with optional tracking/SciPy dependencies, recursive
submodule checkout and synthetic fixtures. It performs no training or model
inference. Hardware qualification remains a separate, explicitly scoped task.

Tests must work with only tracked files and the baseline submodule. Build small
synthetic fixtures in temporary directories; never read local `runs/` or
`payload/` files from a unit test. Test scientific boundaries, integrity and
behavior rather than duplicating implementation details.

## Where changes belong

- Keep scientific code in `src/paiec/` and command orchestration in `scripts/`.
  Preserve native-runtime imports behind their selected backend and cloud imports
  behind the execution adapter. Avoid importing CLI modules into library code.
- Keep recipes, policies, splits and JSON Schemas in `configs/`; use the
  [config index](configs/README.md) to distinguish current and historical files.
- Keep researched facts and provenance in `research/system_profiles/`. Full web
  snapshots, data exports, checkpoints and reports belong under ignored `runs/`
  or the existing external caches.
- Keep guides and findings in `docs/`, indexed by topic. Dated session notes are
  historical material in `docs/history/`, not the current setup instructions.
- Keep dependencies in `requirements/`. Do not check in virtual environments,
  credentials, downloaded datasets, model tensors, local tracking databases or
  generated Python/build/editor caches.

## Naming and versions

Use `snake_case.py` for Python modules and scripts, `kebab-case.md` for guides,
and descriptive `snake_case_vN.json` for scientific recipe revisions. Stable
entrypoint names do not need a version suffix. Dated history notes use
`YYYY-MM-DD.md`; the package version remains in `pyproject.toml`.

Schema versions, catalog collection versions, input format versions and training
recipes describe different things. Collecting more facts does not change the
profile schema. An input-format change needs a new format/recipe and paired
validation; do not rename a frozen v1 artifact to look current.

Saved preparations and checkpoints pin renderer paths and hashes. Keep those
modules byte-for-byte stable during unrelated cleanup. New behavior belongs in
a new builder/adapter selected by the predictor factory. Preserve immutable
configurations, split roles, old catalogs and recorded metrics; a newer
experiment must not replace historical evidence.

## Scientific constraints

Read [AGENTS.md](AGENTS.md) before work involving training or evaluation.
Maintain benchmark holdouts, acquisition exclusion, raw labels, the published
predictor input boundary and separate sealed calibration/confirmation roles.
Use matching samples and checkpoint/input contracts for comparisons. Record
actual device, precision and provenance. Never imply that a smoke check, a
prepared recipe or an interrupted job is a completed quality result.
