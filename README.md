# LfHL — Learning from a Handful of Labels

LfHL is our entry to the NeurIPS Predictive AI Evaluation Competition. It
predicts whether an AI system will answer an item correctly and adapts using
revealed labels from systems on the same benchmark.

- [Technical report](report/LfHL.pdf)
- [Codabench competition](https://www.codabench.org/competitions/17828/) and selected submission **#1019997**
- [Competition rules and predictor interface](https://aims.stanford.edu/competition)

## Build the selected submission

From a fresh checkout, use Python 3.10 and initialize the organizers' baseline
submodule at its pinned commit (`cc88a95`). The build uses the checked-in
`artifacts/pooled_offset.npz` weights and needs no training data.

```bash
git clone https://github.com/eliashuehne/paiec.git
cd paiec
git submodule update --init
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python scripts/build_submission.py
```

The build writes `dist/submission.zip` and runs the organizers' ZIP checker.
The archive contains `model.py` at its root, the NumPy runtime modules, and the
fitted weights. At scoring time it runs offline and uses Codabench's default
random label acquisition.

## Reproduce training and evaluation

The downloader pins public [Measurement DB](https://huggingface.co/datasets/aims-foundations/measurement-db)
revision `54074453`. Request dataset access through Hugging Face and authenticate
before downloading. Refit to a separate path to preserve the released weights.

```bash
hf auth login
python scripts/download_data.py
python scripts/train_submission.py --out dist/retrained/pooled_offset.npz
python scripts/build_submission.py --artifacts dist/retrained/pooled_offset.npz --zip dist/retrained/submission.zip
python scripts/run_eval.py --predictors pooled_offset --pool-sizes 2
```

The last command reruns the two-system result. Appendix A of the
[report](report/LfHL.pdf) gives the full evaluation and figure commands,
including the other pool sizes, acquisition experiments, and pooling ablations.

## Baseline and exploratory work

The [organizers' baseline](https://github.com/aims-foundations/paiec_baseline/blob/cc88a95/README.md) documents the submission
format, baseline predictors, and Codabench tooling. The feature-screening
instructions below are for exploratory work.

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
