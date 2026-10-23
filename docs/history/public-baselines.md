# Historical public baseline results

Archived from the earlier root README during cleanup on 2026-10-23. These are
exploratory results from before the corrected predictor boundary and holdouts;
they are not comparisons with the current compact-v3 recipe.

`default` suite: 32 pairs across bbq, editbench, devbench, cmmlu, afrimedqa,
cruxeval, ai2d_test, and helm_afr. Lower is better.

| predictor | Brier ALC | Brier@0 | Brier@1 | Brier@31 | mean ECE |
| --- | --- | --- | --- | --- | --- |
| shrunk_mean | 0.2349 | 0.2466 | 0.2491 | 0.2188 | 0.134 |
| subject_prior | 0.2466 | 0.2466 | 0.2466 | 0.2466 | 0.170 |
| constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.180 |
| text_ridge | 0.2706 | 0.2706 | 0.2706 | 0.2706 | 0.217 |
| empirical_mean (baseline) | 0.2864 | 0.2500 | 0.4545 | 0.2194 | 0.199 |

Three things are worth reading off this table. The organizers' empirical mean
is the worst predictor on the metric, because an unsmoothed single label is a
0-or-1 prediction and Brier ALC weights budget 1 heavily. Shrinking that label
toward a prior fixes most of the damage. And item text alone, without any
subject information, does worse than a constant, so a text model has to carry
the system profile as well.

The subject prior is weak because the public tables name systems
inconsistently across benchmarks, and most systems appear in only one. Identity
lookups are therefore a poor adaptation strategy; the state Laya sees has to
make the profile informative, not just name the model.

These numbers are historical exploratory public-data results, predating the
`codabench_v1` input boundary. Earlier subject-prior fallback counts included
the evaluation benchmark, and text-training cache names omitted the benchmark
selection. The runner now excludes the entire selected evaluation suite from
prior/text training and keys those caches by training benchmarks and data
manifest. Earlier model-name fallbacks and benchmark-specific Laya questions
also used information unavailable in the published interface. Old results are
retained for reference, not presented as results of the corrected setup.
Laya pretraining overlap with public benchmarks remains unknown; future
fine-tuning still requires benchmark-level holdouts.
