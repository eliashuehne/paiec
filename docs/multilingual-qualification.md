# Multilingual recipe and bounded qualification

2026-10-24. The user authorized starting the proposed input/learning/hardware
qualification. This is a new bounded job, not a restart of the stopped English
30k or upper-encoder runs, and it cannot launch a full pilot automatically.

## Recipe and inputs

Scientific recipe: `configs/head_finetuning/multilingual_qualification_v1.json`.
ZenML launch recipe: `configs/pipelines/multilingual_qualification_workflow_v1.json`.
The parent is the original `multilingual/` checkpoint at revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`, with its actual tokenizer,
8,192 total tokens and a 256-token head cap. The shared correctness question
and native binary labels are unchanged. Inputs are research-only.

Local preparation is `runs/multilingual_qualification/prepared_v1/`. It retains
all 34,718 training, 1,000 monitor and 963 original development input records.
No nontraining outcomes are opened. The earlier 521 overflowing training
states are handled explicitly: seven become one view after repacking, while
514 use multiple views. Seventeen monitor states also use multiple views.
The maximum is six views. Every supplied task token is covered; no record is
silently shortened or removed.

An ordinary fitting state keeps its original native encoding. An oversized
state retains the complete shared system/feature context in each view and
splits its full original task into token windows with 128-token overlap.
Weights sum to one, proportional to newly covered task tokens. The prediction
uses the weighted mean of the two native view logits, then softmax. The loss
is computed once per original example, not once per fragment. This is a new
grouped-input hypothesis: averaging predictions does not guarantee reasoning
across distant windows. A later recorded/enriched comparison must share its
task partitions to avoid confusing a partition change with a feature gain.

Multi-view backward uses a graph-free first pass, followed by each view's
chain-rule contribution. It replays CPU/CUDA dropout RNG states and restores
the final state. A tiny Torch regression verifies the gradients and RNG match
the ordinary joint graph. This keeps graph memory bounded by one view. The
single-view path uses ordinary native cross-entropy backward.

## Qualification gates

The probe uses L4, two CPU cores, 12 GiB requested host memory and a 1,200-second
job timeout. It tests frozen-encoder and final-two-encoder-block policies.
Compute is FP16 autocast with an FP32 scorer, gradient scaling, physical batch
one, and requested head/encoder gradient checkpointing. No throughput claim
is made before measurement.

- Short real-training fixtures compare FP16 against FP32: maximum probability
  difference below .01, sampled-gradient cosine above .98, nonzero decision-head
  gradients, and encoder gradients only when adaptation is selected.
- Near-4,096, near-8,192 and the longest grouped training fixture run forward,
  backward and zero-LR optimizer work. Peak allocated CUDA memory must remain
  below 85% of the device. Zero-LR weights must remain numerically unchanged;
  signed-zero normalization is the same explicit disposable-probe exception
  used by the earlier qualification.
- Fresh original-parent models run isolated synthetic English/German/Chinese
  YES/NO-signal controls (at most 80 updates). Held synthetic Brier must be
  <=.08 and accuracy >=95%.
- Separately reset models fit 16 short distinct training examples, eight per
  native class (at most 60 updates). In-sample Brier must be <=.10 and improve
  at least 40% from its initial value. This deliberately balanced diagnostic
  is not a production training distribution or a generalization result.
- Frozen parameters must remain byte-identical during learning controls.
  Native export/reload must preserve fixture probabilities within .005 and
  retain the 8,192/256 context configuration. Disposable weights are deleted.

A failed gate cannot authorize a pilot. Calibration/confirmation and original
development outcomes are unused. Tiny training-fit scores and synthetic-held
scores are machinery checks, never competition or unseen-family results.

## Execution and inspection

```bash
# Planning only
.venv-pipeline/bin/python scripts/run_pipeline.py training \
  --config configs/pipelines/multilingual_qualification_workflow_v1.json \
  --out runs/workflows/NEW_MULTILINGUAL_QUALIFICATION
```

Actual execution additionally requires `--execute` under an authorized scope.
ZenML remains framework-free; it dispatches through the authenticated
`.venv-modal` client, and the existing Modal worker owns CUDA dependencies.
Only six verified input files (about 5.75 MB) are uploaded, including fixture
states and a full input-layout audit. No corpus, trace table, sealed outcome
file or model weight is uploaded. The worker fetches only the pinned parent.

The authorized run is `runs/workflows/multilingual_qualification_v1/`, ZenML
`935ae3ea-5745-4fe8-b076-d12c023c1234`, Modal
`job_6f3ff7c62b2e4891a1ea37e37c66ab46`, detached call
`fc-01M5QJ4XYVAN0FD7BNQJ8ZNYTP`. Inspect its receipt and logs before any further
launch; do not duplicate it. Final results belong in `job/output/qualification.json`.
Live running status is not a completed qualification. No matched pilot has
been launched by this workflow.

## Completed result

The run completed successfully: **both policies passed every declared gate**.
Worker return code is zero; GPU qualification time was **318.47 seconds**
(about 5m18s), versus about 6m24s for the local orchestration including setup and
result recovery. Parent MLflow `88fc2c14313c4062938ecea27b668879` is FINISHED.
Results and diagnostic metrics are logged under its `qualification/` artifacts.

| Measured check | Frozen encoder | Last two blocks adapted |
| --- | ---: | ---: |
| Peak allocated CUDA memory across capacity fixtures | 8.183 GiB | 8.073 GiB |
| Near-8k forward/backward/zero-LR optimizer, one example | 0.522 s | 1.221 s |
| Six-view longest example, same operations | 4.235 s | 7.536 s |
| FP16 vs FP32 maximum fixture probability difference | 2.38e-7 | 2.38e-7 |
| Sampled gradient cosine | 0.99999976 | 0.99999976 |
| Synthetic-held Brier after 20 control updates | 0.034839 | 0.026417 |
| Real 16-example in-sample Brier after 50 control updates | 0.013189 | 0.036423 |
| FP16 native export/reload maximum probability difference | 0.000271 | 0.000313 |

These timings are individual measured fixture operations, not full-epoch
throughput estimates. The real canary started at Brier 0.499787. Its fit scores
do not select an encoder policy or demonstrate unseen-task quality. All learning
controls were discarded; the next model must start from the pinned parent.
The zero-LR probes preserved numerical weights and learning controls preserved
all frozen parameters. The export/reload probability checks use identical
encoded inputs and raw grouped softmax probabilities; no calibration was fitted.
An independent local tokenizer save/reload check reproduced the full native
group encodings for all 19 real/stress fixture groups, including six views.

Peak host RSS was 7.573 GiB. Runtime versions were read from the same completed
worker container: Torch 2.14.1, Transformers 5.20.2, Tokenizers 0.23.3, Laya
0.4.13 and Safetensors 0.8.0. The app is stopped, has zero tasks, and final GPU
container inventory is empty. The client warning about waiting for final app
logs did not represent a failed job: the saved result, worker return code and
ZenML run all confirm successful completion.

Both native environments and a clean source-only export passed **333 tests**
(13 MLX-environment skips; one Torch skip); four catalogs and whitespace checks
passed. The source-only export includes no datasets or weights.

The user subsequently selected **enriched-only** training. Its grouped trainer,
checkpoint-bound inference wrapper and complete offline corpus are now
[implemented and run](grouped-enriched-training.md). The full pilot scored
Brier ALC 0.282439 versus constant 0.25; Laya work is now on hold.
The old English trainer is not a drop-in replacement. This qualification
supplies tested preparation, loss and resource settings, not a completed full
pilot or a generalization result.
