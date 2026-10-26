# Subject features

Written in Simplified Technical English (ASD-STE100).

## 1. Purpose

A subject is one AI system or one human group that answers items. This
document describes the subject features and the estimates of missing release
dates. It also gives the results.

The features use only the eight fields that the predictor receives. The lift
loop measures each feature ([docs/lift.md](lift.md)). The loop uses benchmarks
that the model did not see, at all six official budgets.

## 2. Summary of the result

- The feature table is correct and you can use it again.
- The release-date estimates are accurate.
- No subject feature makes the result better than baseline KM. KM uses the
  revealed labels.
- Newer systems get higher scores in each benchmark. But this effect is
  small. Its maximum value is approximately 0.0005 to 0.0006 Brier ALC. The
  lift loop cannot measure an effect of this size.

## 3. Procedure: build the table

1. Make the lift sample. Do this step one time:
   `python scripts/lift/build_lift_sample.py`
2. Make the subject fields. Do this step one time:
   `python scripts/lift/build_subject_fields.py`
3. Make the subject features:
   `python scripts/subjects/build_subject_features.py`

NOTE: Step 3 takes approximately 30 seconds. It includes the back-test of the
release dates (section 6).

The procedure writes two files:
- `runs/features/subjects.parquet`: one row for each `subject_id`.
- `runs/features/subjects_manifest.json`: the input hashes, the fill rates,
  the back-test and the accepted date sources.

To use the table in the lift loop, add `--features runs/features/subjects.parquet`.

## 4. Columns

The fill rate is the percentage of lift-sample rows that have a value
(revision `54074453`).

| Column | Meaning | Rows |
| --- | --- | ---: |
| `identity` | `normalized_name`. If it is empty: `source_model_label` from `subject_features_extra` | 65.1% |
| `identity_source` | `name` or `label` | 65.1% |
| `identity_key` | lowercase letters and digits. The provider prefix and the organization path are removed | 65.1% |
| `release_days`, `release_year` | real `release_date`, as days after 2020-01-01 and as calendar year | 52.0% |
| `release_precision` | `day` or `month` | 52.0% |
| `log_params_b`, `log_active_params_b` | log10 of the parameters in billions, from the name. "8x7B" gives 56. "30B A3B" gives 30 total and 3 active | 23.8% |
| `family` | model family, from name patterns in `configs/subjects/model_families.json` | 58.5% |
| `provider_filled` | `provider`, without placeholder values. If it is empty: the provider of the family | 60.4% |
| `tier` | `small` (mini, flash, haiku, lite) or `large` (pro, opus, max). For closed models that have no parameter count | 16.6% |
| `is_instruct`, `is_vision`, `is_reasoning`, `is_moe`, `is_agent` | flags from the name | 0.1 to 4.0% set |
| `subject_kind` | `human`, `llm_configuration` and other values, from `subject_features_extra` | 1.5% |
| `line`, `version` | product line and version from the name. "Claude3-5V_Sonnet" gives sonnet and 3.5 | 26.6%, 47.7% |
| `name_date_days` | a date in the name. This date is only a latest limit (section 5) | 2.0% |
| `release_days_filled`, `release_year_filled` | real date. If there is no real date: an accepted estimate | 57.4% |
| `release_lo`, `release_hi` | earliest and latest possible release date | 57.4% |
| `release_source` | `field`, or the source of the estimate | 57.4% |
| `release_filled` | 1 if the date is an estimate | 5.4% set |

WARNING: DO NOT USE `display_name`, LABELS OR DATA FROM OUTSIDE THE RELEASE.
THE PREDICTOR DOES NOT RECEIVE THEM.

Rules for the table:
1. Each row uses the fields of one subject only.
2. Exception: a date estimate uses the visible release dates of other
   subjects. These subjects must be in other folds (section 5).
3. The tests make sure that the builders do not read `display_name`.
4. The tests make sure that the builders do not use labels.

## 5. Estimate missing release dates

The code is in `src/paiec/subject_dates.py`.

In a family, the version numbers show the order of the models in time. Thus,
dated versions below and above an undated version give its earliest and
latest date.

Many undated systems are dated systems with a different spelling. For
example, "Llama-2-70b-chat-hf" is "Meta Llama 2 70B". Thus, these matches
come first.

The estimate uses the first source that applies:

| Source | Rule | Earliest and latest |
| --- | --- | --- |
| `field` | real `release_date` | the same date |
| `same_identity` | the same `identity_key` has a date | the same date |
| `same_signature` | the same family, line, version and size have dates | the range of these dates |
| `same_version` | the same family, line and version have dates | the range of these dates |
| `interpolated` | dated versions are below and above | the nearest dated version below and above |
| `one_sided` | dated versions are only below or only above | one side is the limit of the data |
| `family`, `provider` | only the family or the provider is known | their range of dates. Not accepted (section 6) |

CAUTION: A DATE IN A NAME IS NOT A RELEASE DATE. NAMES SUCH AS
`gpt-4o-2025-12-11` CONTAIN THE DATE OF THE EVALUATION. GPT-4o WAS RELEASED IN
2024. THUS, A DATE IN A NAME IS ONLY A LATEST LIMIT.

WARNING: USE REFERENCE DATES FROM OTHER FOLDS ONLY. AT EVALUATION, THE
PREDICTOR CAN GET DATES ONLY FROM PUBLIC DATA. THUS, A SUBJECT DOES NOT USE A
DATE FROM A BENCHMARK IN ITS OWN FOLD. SEALED SUBJECTS ARE NOT REFERENCES.

## 6. Back-test of the estimates

The back-test removes the date of each dated system. Then it estimates the
date again from the other systems.

There are two modes:
- **Remove identity**: remove only this system. Other spellings of the same
  system stay.
- **Remove signature**: also remove all systems with the same family, line
  and version. This mode tests the interpolation.

A source is accepted only if:
1. Its median error is less than 180 days, and
2. 75% or more of its estimates are within 180 days.

NOTE: The provider source has a good median error (119 days). But only 62% of
its estimates are within 180 days. Many systems have dates in the same recent
months. Thus, the median looks good, but the source is not accurate.

Values in each cell: median error in days; percentage within 180 days;
percentage of true dates between the earliest and latest date.

| Source | Remove identity | Remove signature | Accepted |
| --- | --- | --- | --- |
| `same_signature` | 85.5; 83%; 42% | not applicable | yes |
| `same_version` | 0; 96%; 81% | not applicable | yes |
| `interpolated` | 41; 96%; 98% | **45; 88%; 91%** | yes |
| `one_sided` | 61; 91%; 93% | 60; 84%; 91% | yes |
| `family` | 189.5; 47%; 87% | 203.5; 46%; 87% | no |
| `provider` | 119; 62%; 70% | 119; 62%; 70% | no |

Effect on the fill rate of sample rows:
- Real dates: 52.0%.
- Real dates and accepted estimates: **57.4%**.
- The increase comes from: same version 2.9%, same signature 1.6%, one-sided
  0.6%, interpolated 0.3%.
- 43% of rows have no estimate. These rows have no identity, or no version
  and no accepted source.

## 7. Results in the lift loop

Conditions for all runs:
- Base: KM, the baseline with revealed labels (ALC 0.2034).
- Budgets: all six. The result is the ALC with the official weights.
- A positive change is worse.
- The interval is a 95% bootstrap interval over the 125 held-out benchmarks.
- The runs are in `runs/lift/history.jsonl` or `runs/lift/dev/`.
- For comparison, a column of random noise costs +0.0003 [+0.0002, +0.0004].

### 7.1 Each column alone

No column gets a change of 0.001 or more in 4 of 5 folds.

| Column | ALC change | Folds better | Budget 1 | Fingerprint | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `release_days` | +0.0025 [+0.0013, +0.0038] | 0/5 | +0.0035 | 0.75 | HURTS |
| `log_params_b` | +0.0008 [+0.0003, +0.0013] | 1/5 | +0.0007 | 0.58 | no clear effect |
| `provider_filled` | +0.0007 | 2/5 | +0.0009 | 0.02 | no clear effect |
| `family` | +0.0004 | 1/5 | +0.0007 | 0.00 | no clear effect |
| `is_instruct` | +0.0003 | 1/5 | +0.0006 | 0.42 | no clear effect |
| `tier`, `is_vision`, `is_reasoning`, `subject_kind` | +0.0001 | 1-2/5 | up to +0.0003 | 0.00 to 1.00 | no clear effect |
| all columns above together | +0.0040 [+0.0023, +0.0057] | 0/5 | +0.0052 | | HURTS |
| subject ability (`normalized_name`) | +0.0001 [-0.0003, +0.0004] | 2/5 | +0.0002 | | no clear effect |

### 7.2 Forms of the release date

| Variant | ALC change | Folds better | Verdict |
| --- | ---: | ---: | --- |
| `release_year` (calendar year) | +0.0003 | 1/5 | no clear effect |
| `release_year`, monotone (`--increasing`) | +0.0002 [-0.0003, +0.0007] | 2/5 | no clear effect |
| `release_days`, monotone | +0.0002 | 1/5 | no clear effect |

### 7.3 Filled dates against real dates

Command: `--base-columns release_year --columns release_year_filled`.

| Comparison | ALC change | Folds better | Verdict |
| --- | ---: | ---: | --- |
| filled against real only | -0.0002 [-0.0004, +0.0001] | 3/5 | no clear effect |
| filled with the `release_filled` flag against real only | +0.0004 [+0.0001, +0.0007] | 0/5 | no clear effect |
| filled against KM | +0.0002 [-0.0003, +0.0007] | 2/5 | no clear effect |

## 8. Why subject features do not help

1. **The revealed labels show the strength of the system.** K pulls the rate
   of each pair toward the rate of its benchmark. From budget 1, the labels
   of the pair show the strength of the system on this benchmark.
2. **The real effect is small.** In one benchmark, a system that is one year
   newer has a score that is 7.1 points higher. But the release years of the
   systems in one benchmark have a standard deviation of only 0.45 years. If
   the trend is known exactly, the maximum gain is 0.0010 Brier on dated rows.
   On all rows, the maximum gain is 0.0005 to 0.0006. K already takes a part
   of this gain.
3. **Exact values let the trees memorize.** The value of `release_days` is
   almost unique for each system (fingerprint 0.75). Thus, the trees fit
   individual training pairs. A coarse or monotone form stops this problem.
   But it does not give a gain.
4. **At budget 0, the level of the benchmark is the main unknown.** Benchmarks
   that test newer systems are more difficult (correlation -0.11). Thus,
   subject features give incorrect information before the first label.

## 9. Limits

- The results are predictive relations on public benchmarks. They are not
  causes.
- The hidden benchmarks can contain systems from many years. In this case,
  the maximum gain in section 8 is higher. The result can then be different.
- The result applies to KM: LightGBM that corrects K. One method is not
  tested: subject features that change the prior of each pair.
- Parameter counts are available for 23.8% of rows only. Closed models have
  no parameter count. `tier` gives an approximate size for these models.
- The limits of 180 days and 75% are decisions, not measurements. The manifest
  keeps the back-test that supports them.

## 10. Files

| Path | Function |
| --- | --- |
| `src/paiec/subject_features.py` | parsers and the feature table |
| `src/paiec/subject_dates.py` | version parsing, references from other folds, estimates, back-test, acceptance |
| `configs/subjects/model_families.json` | family patterns, providers, version patterns and product lines (v2), with examples that the tests check |
| `scripts/subjects/build_subject_features.py` | makes the table and the manifest |
| `tests/subjects/` | parser cases, table contract, fold rule, label-free checks |
