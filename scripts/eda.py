#!/usr/bin/env python
"""Audit the local public payload; write aggregate tables, charts, and a report.

Read every non-trace table. Trace row counts/sizes/schemas come from Parquet
metadata, without scanning the 23 GB of trace strings. No network is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from paiec.schema import benchmark_dir, canonical_identity

ROOT = Path(__file__).resolve().parents[1]


def missing_count(column):
    mask = pc.is_null(column)
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        blank = pc.fill_null(pc.equal(pc.utf8_trim_whitespace(column), ""), False)
        mask = pc.or_(mask, blank)
    return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)


def add_missing(table, kind, benchmark, records):
    for name in table.column_names:
        records.append({"benchmark": benchmark, "table": kind, "column": name,
                        "rows": len(table), "missing": missing_count(table[name])})


def fmt_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def run(data_root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_root / "corpus_manifest.json").read_text())
    inventory, missing, benchmarks, pairs, profiles, samples = [], [], [], [], [], []
    schemas = defaultdict(dict)
    lengths_all = []
    for index, entry in enumerate(manifest["eligible_dirs"], 1):
        name = entry.split("/")[0]
        directory = benchmark_dir(data_root, name)
        print(f"[{index}/{len(manifest['eligible_dirs'])}] {name}", flush=True)
        for path in sorted(directory.glob("*.parquet")):
            pf = pq.ParquetFile(path)
            inventory.append({"benchmark": name, "table": path.stem, "rows": pf.metadata.num_rows,
                              "bytes": path.stat().st_size, "path": str(path.relative_to(data_root))})
            schema = [{"column": field.name, "type": str(field.type), "nullable": field.nullable}
                      for field in pf.schema_arrow]
            schemas[path.stem][json.dumps(schema)] = schema

        meta = pq.read_table(directory / "benchmarks.parquet")
        add_missing(meta, "benchmarks", name, missing)
        info = meta.to_pylist()[0]
        subjects_arrow = pq.read_table(directory / "subjects.parquet")
        add_missing(subjects_arrow, "subjects", name, missing)
        subjects = subjects_arrow.to_pandas()
        subjects["benchmark"] = name
        normalized = subjects["normalized_name"].fillna("").str.strip()
        subjects["identity_name"] = normalized.where(normalized.ne(""), subjects["display_name"])
        subjects["identity"] = subjects["identity_name"].map(canonical_identity)
        profiles.append(subjects)

        # Keep small IDs/hashes for referential checks; stream large prompt text.
        item_keys = pq.read_table(directory / "items.parquet", columns=["item_id", "content_hash"])
        item_ids = item_keys["item_id"]
        content_lengths = []
        n_assets = n_jsonlike = n_reference = n_blank_content = 0
        sample = None
        for batch in pq.ParquetFile(directory / "items.parquet").iter_batches(batch_size=4096):
            table = pa.Table.from_batches([batch])
            add_missing(table, "items", name, missing)
            content = pc.fill_null(table["content"], "")
            content_lengths.append(pc.utf8_length(content).to_numpy())
            n_blank_content += missing_count(table["content"])
            n_assets += len(table) - missing_count(table["asset_manifest"])
            n_reference += int(pc.sum(pc.cast(pc.match_substring(table["grading_criterion"], '"reference_answer"'), pa.int64())).as_py() or 0)
            n_jsonlike += int(pc.sum(pc.cast(pc.match_substring_regex(content, r"^\s*[\{\[]"), pa.int64())).as_py() or 0)
            if sample is None:
                first = table.slice(0, 1).to_pylist()[0]
                sample = {"benchmark": name, "subject": subjects_arrow.slice(0, 1).to_pylist()[0],
                          "item_id": first["item_id"], "content_excerpt": (first["content"] or "")[:1200],
                          "item_features": first["item_features"], "asset_manifest": first["asset_manifest"],
                          "grading_criterion_excerpt": first["grading_criterion"][:500]}
        lengths = np.concatenate(content_lengths).astype(np.int64)
        lengths_all.append(lengths)
        responses_arrow = pq.read_table(directory / "response.parquet")
        add_missing(responses_arrow, "response", name, missing)
        # Arrow-backed columns avoid expanding millions of repeated ID strings.
        responses = responses_arrow.to_pandas(types_mapper=pd.ArrowDtype)
        binary = responses["response"].isin([0, 1])
        linked = responses["item_id"].isin(item_ids.to_pylist()) & responses["subject_id"].isin(subjects["subject_id"])
        valid = responses.loc[binary & linked]
        by_pair = valid.groupby("subject_id", sort=False).agg(
            n_items=("item_id", "nunique"), n_responses=("response", "size"), success_rate=("response", "mean")
        ).reset_index()
        by_pair["benchmark"] = name
        by_pair["eligible_80_items"] = by_pair["n_items"] >= 80
        pairs.append(by_pair)
        repeat = valid.groupby(["subject_id", "item_id"], sort=False).agg(
            n_trials=("response", "size"), n_outcomes=("response", "nunique")
        )
        repeated = repeat["n_trials"] > 1
        hash_values = item_keys["content_hash"].to_pandas().dropna()
        hash_values = hash_values[hash_values.str.strip().ne("")]
        n_null = int(responses["response"].isna().sum())
        row = {
            "benchmark": name, "name": info["name"], "domain": ", ".join(info["domain"]),
            "modality": ", ".join(info["modality"]), "license": info["license"],
            "description": info["description"], "response_scale": info["response_scale"],
            "subjects": len(subjects), "items": len(item_ids), "responses_raw": len(responses),
            "responses_binary": int(binary.sum()), "responses_binary_linked": len(valid),
            "outcomes_null": n_null, "outcomes_other": int((~binary).sum()) - n_null,
            "outcomes_zero": int((responses["response"] == 0).sum()),
            "outcomes_one": int((responses["response"] == 1).sum()),
            "success_rate_binary": float(responses.loc[binary, "response"].mean()) if binary.any() else None,
            "null_outcome_pct": 100 * n_null / len(responses),
            "eligible_pairs": int(by_pair["eligible_80_items"].sum()),
            "eligible_pairs_including_null_outcomes": int((responses.groupby("subject_id")["item_id"].nunique() >= 80).sum()),
            "subjects_with_binary_linked_responses": len(by_pair),
            "unique_subject_item_links": len(repeat), "repeated_subject_item_links": int(repeated.sum()),
            "links_with_both_outcomes": int((repeat["n_outcomes"] > 1).sum()),
            "duplicate_response_ids": int(responses["response_id"].duplicated().sum()),
            "duplicate_subject_ids": int(subjects["subject_id"].duplicated().sum()),
            "duplicate_item_ids": len(item_ids) - len(pc.unique(item_ids)),
            "duplicate_nonempty_content_hashes": len(hash_values) - hash_values.nunique(),
            "responses_missing_item": int((~responses["item_id"].isin(item_ids.to_pylist())).sum()),
            "responses_missing_subject": int((~responses["subject_id"].isin(subjects["subject_id"])).sum()),
            "items_without_any_response": int((~pd.Series(item_ids.to_pylist()).isin(responses["item_id"])).sum()),
            "items_without_binary_response": int((~pd.Series(item_ids.to_pylist()).isin(valid["item_id"])).sum()),
            "blank_content": n_blank_content, "items_with_asset_manifest": n_assets,
            "content_starts_jsonlike": n_jsonlike, "grading_contains_reference_answer_key": n_reference,
            "content_chars_p50": float(np.quantile(lengths, .5)),
            "content_chars_p90": float(np.quantile(lengths, .9)),
            "content_chars_p99": float(np.quantile(lengths, .99)), "content_chars_max": int(lengths.max()),
            "declared_n_items": info["n_items"], "declared_n_subjects": info["n_subjects"],
            "declared_n_responses": info["n_responses"],
        }
        benchmarks.append(row)
        samples.append(sample)
        # Drop heavy tables before the next benchmark.
        del responses, valid, responses_arrow, repeat, item_keys, content_lengths

    benchmark_df = pd.DataFrame(benchmarks)
    inventory_df = pd.DataFrame(inventory)
    profiles_df = pd.concat(profiles, ignore_index=True)
    pair_df = pd.concat(pairs, ignore_index=True)
    column_missing = pd.DataFrame(missing).groupby(["table", "column"], as_index=False)[["rows", "missing"]].sum()
    column_missing["missing_pct"] = 100 * column_missing["missing"] / column_missing["rows"]
    lengths = np.concatenate(lengths_all)
    identity_coverage = profiles_df.groupby("identity").agg(
        n_benchmarks=("benchmark", "nunique"), n_subject_rows=("subject_id", "size"),
        example_name=("identity_name", "first")
    ).reset_index()
    table_totals = inventory_df.groupby("table", as_index=False).agg(files=("path", "size"), rows=("rows", "sum"), bytes=("bytes", "sum"))
    modalities = Counter(value.strip() for entry in benchmarks for value in entry["modality"].split(",") if value.strip())
    domains = Counter(value.strip() for entry in benchmarks for value in entry["domain"].split(",") if value.strip())
    observed_rates = benchmark_df["success_rate_binary"].dropna()
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "data_root": str(data_root.resolve()),
        "dataset_repo": manifest["db_repo"], "dataset_revision": manifest["revision"],
        "benchmarks": len(benchmark_df), "files": len(inventory_df),
        "disk_bytes": int(inventory_df["bytes"].sum()),
        "subject_rows": len(profiles_df), "unique_subject_ids_across_benchmarks": profiles_df["subject_id"].nunique(),
        "items": int(benchmark_df["items"].sum()), "responses_raw": int(benchmark_df["responses_raw"].sum()),
        "responses_binary": int(benchmark_df["responses_binary"].sum()),
        "outcomes_null": int(benchmark_df["outcomes_null"].sum()),
        "outcomes_other": int(benchmark_df["outcomes_other"].sum()),
        "outcomes_one": int(benchmark_df["outcomes_one"].sum()),
        "row_weighted_rate_of_one": float(benchmark_df["outcomes_one"].sum() / benchmark_df["responses_binary"].sum()),
        "benchmark_equal_weight_rate_of_one": float(observed_rates.mean()),
        "pair_equal_weight_rate_of_one_eligible": float(pair_df.loc[pair_df.eligible_80_items, "success_rate"].mean()),
        "eligible_pairs_80_items": int(pair_df["eligible_80_items"].sum()),
        "eligible_pairs_80_items_including_null_outcomes": int(benchmark_df["eligible_pairs_including_null_outcomes"].sum()),
        "heuristic_canonical_identities": len(identity_coverage),
        "heuristic_identities_in_one_benchmark": int((identity_coverage["n_benchmarks"] == 1).sum()),
        "content_chars": {f"p{q}": float(np.percentile(lengths, q)) for q in [0, 25, 50, 75, 90, 95, 99, 100]},
        "benchmarks_by_modality": dict(modalities), "benchmarks_by_domain": dict(domains),
        "checks": {field: int(benchmark_df[field].sum()) for field in [
            "duplicate_response_ids", "duplicate_item_ids", "duplicate_subject_ids", "responses_missing_item",
            "responses_missing_subject", "items_without_any_response", "items_without_binary_response",
            "blank_content", "items_with_asset_manifest", "unique_subject_item_links",
            "repeated_subject_item_links", "links_with_both_outcomes", "duplicate_nonempty_content_hashes",
        ]},
    }
    for filename, frame in [
        ("benchmarks.csv", benchmark_df), ("files.csv", inventory_df), ("table_totals.csv", table_totals),
        ("missingness.csv", column_missing), ("subject_benchmark_pairs.csv", pair_df),
        ("identity_coverage.csv", identity_coverage),
    ]:
        frame.to_csv(out / filename, index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "schemas.json").write_text(json.dumps({key: list(variants.values()) for key, variants in schemas.items()}, indent=2))
    (out / "examples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    fig.suptitle("PAIEC public payload — local dataset audit", fontsize=20, fontweight="bold")
    top = benchmark_df.nlargest(10, "responses_raw").sort_values("responses_raw")
    axes[0, 0].barh(top["benchmark"], top["responses_raw"] / 1e6, color="#287c8e")
    axes[0, 0].set(title="Response volume is concentrated", xlabel="Raw response rows (millions)")
    totals = table_totals.sort_values("bytes")
    axes[0, 1].barh(totals["table"], totals["bytes"] / 1e9, color="#d38436")
    axes[0, 1].set(title="Traces dominate storage", xlabel="File size (decimal GB)")
    profile_missing = column_missing[column_missing.table.eq("subjects") & ~column_missing.column.eq("subject_id")].sort_values("missing_pct")
    axes[1, 0].barh(profile_missing["column"], profile_missing["missing_pct"], color="#8c6bb1")
    axes[1, 0].set(title="Model profile fields are often absent", xlabel="Null or blank subject records (%)", xlim=(0, 105))
    bins = np.geomspace(1, max(int(lengths.max()), 2), 45)
    axes[1, 1].hist(np.maximum(lengths, 1), bins=bins, color="#2d927c")
    axes[1, 1].set(xscale="log", title="Item text lengths vary widely", xlabel="Characters in stored content (log scale)", ylabel="Item rows")
    fig.savefig(out / "overview.png", dpi=160)
    plt.close(fig)

    write_report(out, summary, benchmark_df, table_totals, column_missing, identity_coverage)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out.resolve()}", flush=True)


def write_report(out, summary, benchmark_df, table_totals, column_missing, identity_coverage):
    report = make_report(summary, benchmark_df, table_totals, column_missing, identity_coverage)
    (out / "report.md").write_text(report)
    # Standalone browsable report: all figures are embedded, tables are sortable.
    import base64
    encoded = base64.b64encode((out / "overview.png").read_bytes()).decode()
    body = "<h1>PAIEC dataset exploration</h1><p>Full scan of every non-trace table; trace metadata only.</p>"
    body += f'<img alt="Dataset overview" style="width:100%" src="data:image/png;base64,{encoded}">'
    body += '<article>' + MarkdownIt("commonmark", {"html": False}).enable("table").render(report) + '</article>'
    display = benchmark_df[["benchmark", "domain", "modality", "subjects", "items", "responses_raw", "responses_binary", "null_outcome_pct", "success_rate_binary", "eligible_pairs", "content_chars_p50", "content_chars_p99"]]
    body += '<h2>All 61 benchmarks</h2><p>Click column headers to sort.</p>' + display.to_html(index=False, border=0, float_format=lambda x: f"{x:,.3f}", table_id="benchmarks")
    (out / "report.html").write_text('''<!doctype html><html><head><meta charset="utf-8"><title>PAIEC dataset exploration</title>
<style>body{font:16px/1.5 system-ui;margin:40px auto;padding:0 24px;max-width:1400px;color:#203139;background:#fafbf9}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;text-align:left;border-bottom:1px solid #dce2df}th{background:#e7eeea}#benchmarks th{cursor:pointer;position:sticky;top:0}tr:hover{background:#edf4f0}pre{overflow:auto;background:#edf1ef;padding:20px;border-radius:8px}article{background:white;padding:30px;border-radius:12px}h1{font-size:32px}h2{margin-top:36px}code{font-size:.9em}p,li{max-width:1050px}</style></head><body>''' + body + '''<script>document.querySelectorAll('#benchmarks th').forEach((th,i)=>th.onclick=()=>{const tb=th.closest('table').tBodies[0];const rows=[...tb.rows];const dir=th.dataset.dir==='asc'?-1:1;th.dataset.dir=dir===1?'asc':'desc';rows.sort((a,b)=>{const x=a.cells[i].textContent.replaceAll(',',''),y=b.cells[i].textContent.replaceAll(',','');return dir*(x.trim()!==''&&y.trim()!==''&&!isNaN(x)&&!isNaN(y)?Number(x)-Number(y):x.localeCompare(y))});rows.forEach(r=>tb.appendChild(r))});</script></body></html>''')


def make_report(s, benchmarks, tables, missing, identities):
    total_bytes = s["disk_bytes"]
    display_tables = tables.copy()
    display_tables["size_GB"] = (display_tables["bytes"] / 1e9).round(4)
    display_tables["disk_pct"] = (100 * display_tables["bytes"] / total_bytes).round(2)
    display_tables = display_tables.drop(columns="bytes")
    top = benchmarks.nlargest(10, "responses_raw")[["benchmark", "subjects", "items", "responses_raw", "responses_binary", "success_rate_binary", "eligible_pairs"]].copy()
    top["success_rate_binary"] = top["success_rate_binary"].round(3)
    subject_missing = missing[missing.table.eq("subjects")][["column", "rows", "missing", "missing_pct"]].copy()
    subject_missing["missing_pct"] = subject_missing["missing_pct"].round(1)
    null_top = benchmarks[benchmarks.outcomes_null.gt(0)].nlargest(10, "outcomes_null")[["benchmark", "outcomes_null", "null_outcome_pct"]].copy()
    null_top["null_outcome_pct"] = null_top["null_outcome_pct"].round(2)
    schema_counts = []
    for row in benchmarks.itertuples():
        if row.items != row.declared_n_items or row.subjects != row.declared_n_subjects or row.responses_raw != row.declared_n_responses:
            schema_counts.append(row.benchmark)
    modal_counts = ", ".join(f"{k}: {v}" for k, v in s["benchmarks_by_modality"].items())
    check_text = "\n".join(f"- {key}: {value:,}" for key, value in s["checks"].items())
    scales = benchmarks["response_scale"].map(json.loads)
    directions = scales.map(lambda scale: scale.get("direction", "unspecified")).value_counts().to_dict()
    examples = []
    for name in ["bbq", "afrimedqa", "advprompter", "alignment_faking", "braveguard"]:
        row = benchmarks.loc[benchmarks.benchmark.eq(name)].iloc[0]
        scale = json.loads(row.response_scale)
        examples.append({"benchmark": name, "direction": scale.get("direction"), "meaning_of_1": scale.get("meanings", {}).get("1")})
    scale_excerpt = fmt_table(pd.DataFrame(examples))
    return f'''# PAIEC public payload: exploratory data analysis

Dataset revision: `{s['dataset_revision']}`. Generated: {s['generated_utc']}.

## What `payload/data/` is

This is a local download of the competition's public measurement-db release,
not a model's training checkpoint or a single CSV. Each benchmark has its own
`<benchmark>/formatted_tables/` folder. `corpus_manifest.json` records the source
repository, pinned revision, eligible directories, and downloaded files. The
download contains {s['benchmarks']} binary, item-level benchmarks, {s['files']}
Parquet files, and {total_bytes / 1e9:.2f} GB of file bytes ({total_bytes / 2**30:.2f} GiB).
It is the downloaded eligible subset, not the entire upstream repository.

## Tables and their relationships

| Table | One row means | Main information | Join keys |
| --- | --- | --- | --- |
| benchmarks.parquet | A benchmark's metadata record | Description, domain, modality, license, scoring scale and direction, declared counts, coverage | benchmark_id |
| subjects.parquet | A system/configuration evaluated in that benchmark | Display/normalized names, provider, dates, harness/version, reasoning effort, extra settings | subject_id |
| items.parquet | A task/question | Full content, structured item features, assets, grading/reference information, verifier, original ID, content hash | item_id |
| response.parquet | One system's outcome on one item/trial | Numeric outcome, trial, test condition, interactors; this is a score, not the generated answer text | subject_id + item_id; response_id identifies the recorded response |
| traces.parquet | A logged response/interaction | Serialized trace text and identifying fields; may include outputs or execution details, depending on benchmark | response_id; also subject_id, item_id, trial |

Join within each benchmark: subjects -> response <- items; response -> traces
by response_id. Keep benchmark_id as context when combining benchmarks. Repeated
trials mean subject_id + item_id alone is not a unique response key.

`agentic_review_perturbation` has no traces table. All five table types have
consistent column schemas across their existing files; see schemas.json for
every column/type/nullability. Trace joins/content were not scanned or validated.

{fmt_table(display_tables)}

## Overall counts and outcomes

- Subject-benchmark records: **{s['subject_rows']:,}**. These are configurations,
  not necessarily distinct underlying AI models.
- Unique subject_id values across the downloaded records: {s['unique_subject_ids_across_benchmarks']:,}.
- Item rows: **{s['items']:,}**. These are not proven to be globally unique tasks.
- Raw response rows: **{s['responses_raw']:,}**.
- Binary response rows (exactly 0 or 1): **{s['responses_binary']:,}**.
- Missing outcomes: **{s['outcomes_null']:,}** ({100 * s['outcomes_null'] / s['responses_raw']:.2f}% of raw responses).
- Non-null outcomes other than 0/1: {s['outcomes_other']:,}.
- Row-weighted proportion of outcome 1: {100*s['row_weighted_rate_of_one']:.2f}%.
- Equal-benchmark mean proportion of outcome 1: {100*s['benchmark_equal_weight_rate_of_one']:.2f}%.
- Equal-eligible-pair mean proportion of outcome 1: {100*s['pair_equal_weight_rate_of_one_eligible']:.2f}%.
- Subject-benchmark pairs with >=80 distinct items with binary linked outcomes:
  **{s['eligible_pairs_80_items']:,}**.

If rows with missing outcomes are counted as available items, the >=80-item
count would instead be {s['eligible_pairs_80_items_including_null_outcomes']:,}.
That explains the earlier data note's larger pair count; the harness needs
observed binary outcomes.

Important: **1 is the recorded positive outcome, not universally a correct
answer or desirable behavior**. Read each benchmark's response_scale and
description. Safety/attack benchmarks can score attack success, refusal,
or other behavioral outcomes. The predictor's generic “answers correctly”
question may therefore be inappropriate for some benchmarks. The outcome-rate
averages above are numeric summaries across heterogeneous tasks, not universal
model accuracy or competition scores.

Declared outcome directions: {json.dumps(directions)}. Direction alone does
not tell us whether the outcome is accuracy; even higher-is-better can mean
safety or another task-specific objective.

Examples from the actual benchmark metadata:

{scale_excerpt}

## Which benchmarks dominate?

{fmt_table(top)}

The five largest benchmarks contain {100 * benchmarks.nlargest(5, 'responses_raw').responses_raw.sum() / s['responses_raw']:.1f}%
of raw responses. Sampling response rows uniformly will therefore favor those
benchmarks. The harness scores pairs with equal weight, so raw row-level
accuracy and its scoring objective need not agree.

Alignment_faking alone accounts for
{int(benchmarks.loc[benchmarks.benchmark.eq('alignment_faking'), 'eligible_pairs'].iloc[0]):,}
of {s['eligible_pairs_80_items']:,} eligible pairs
({100*benchmarks.loc[benchmarks.benchmark.eq('alignment_faking'), 'eligible_pairs'].iloc[0]/s['eligible_pairs_80_items']:.1f}%).
Consequently, even equal-pair weighting across the full public payload would
heavily weight that one benchmark. Report benchmark-level results as well.

## Missing outcomes

{fmt_table(null_top)}

Null outcomes are unknown, not failures. The evaluation loader drops them;
this explains why its loaded response count may be lower than the Parquet
file row count. Missingness may be systematic, so a binary-only analysis
describes the observed subset.

## Model profiles and coverage

Missing means null or a whitespace-only string; empty JSON strings such as
`{{}}` are not treated as missing. Rates are over subject-benchmark records.

{fmt_table(subject_missing)}

Using the current name fallback and canonical_identity heuristic gives
{s['heuristic_canonical_identities']:,} name keys; {s['heuristic_identities_in_one_benchmark']:,}
({100*s['heuristic_identities_in_one_benchmark']/s['heuristic_canonical_identities']:.1f}%)
appear in only one benchmark. This is approximate name overlap, not resolved
model identity: aliases can remain unmatched, and the heuristic can merge
different configurations. See identity_coverage.csv.

## Item content and modalities

Content length in **characters**, not model tokens, across all item rows:
```json
{json.dumps(s['content_chars'], indent=2)}
```

Benchmark modality tags (a benchmark can have several): {modal_counts}.
Nonblank asset_manifest rows: {s['checks']['items_with_asset_manifest']:,}; this
does not establish that referenced image/audio/video files are downloaded or
that text alone contains enough information to solve the task.

Item content can be plain text, a prompt containing instructions, or a JSON
string. Structured metadata and grading instructions are separate columns.
Reference answers/verifiers are for grading and analysis: do not pass them
as predictor inputs. See examples.json for bounded item/profile excerpts from
each benchmark; these are first-row illustrations, not a representative sample.

Concrete examples inspected: ai2d_test stores an image path and question/options
inside a JSON multimedia_elements object. Babilong can store a context.txt asset
reference rather than embedding the complete context. Egocentric video-derived
tasks can be released as caption-fed text (egoschema is tagged text here).
Thus character counts describe stored content, not necessarily the full original
input presented to the evaluated system. An inventory of non-Parquet files in
the current payload/data contains only corpus_manifest.json: no separate image
or context asset files were downloaded there. Trace strings were not inspected
for embedded media/context; external assets were not fetched.

## Integrity, repetition, and coverage checks

{check_text}

Duplicate IDs above are checked within each benchmark. Repeated subject-item
links count links with more than one observed binary outcome row; trials,
conditions, and interactors may explain repetition. Links with both 0 and 1
are not automatically errors: outcomes can vary across attempts or conditions.
Content-hash duplicates count excess rows sharing a nonempty stored hash within
a benchmark, not proven duplicate tasks. The analysis does not recompute hashes.

Benchmarks whose declared counts differ from actual local table-row counts:
{', '.join(schema_counts) or 'none'}.

## Consequences for Laya experiments

1. Start with a small, semantically coherent text subset whose outcome really
   means correct/incorrect. Inspect response_scale before mixing domains.
2. Include subject identity/configuration and item text/features. Missing model
   metadata limits what a pretrained predictor can infer about that system.
3. Measure tokenizer truncation on the actual input state. Character lengths
   reveal variation but do not establish a suitable token budget.
4. Keep all trials of each item together in acquisition/evaluation splits.
   Check near-duplicate content separately before any fine-tuning split.
5. Distinguish null grades from zero, and equal-pair scoring from row-weighted
   summaries. Report results by benchmark as well as the aggregate.
6. Public-data pretrained results are exploratory: pretraining overlap is
   unknown. Fine-tuning will need explicit held-out benchmarks and no grading
   columns in model inputs.

## Reproduce and inspect

Install the optional plotting dependency in the Laya environment:
`uv pip install --python .venv-laya/bin/python -e '.[eda]'`.
Then run `.venv-laya/bin/python scripts/eda.py` from the project root.

- report.html: standalone report with embedded chart and sortable benchmark table.
- overview.png: four summary charts.
- benchmarks.csv: per-benchmark outcome/length/integrity statistics and metadata.
- table_totals.csv / files.csv: table totals and complete file inventory.
- missingness.csv: missing values for every column of every non-trace table.
- subject_benchmark_pairs.csv: item counts and observed outcome rate per pair.
- identity_coverage.csv: approximate cross-benchmark name coverage.
- schemas.json: complete Parquet schemas.
- examples.json: bounded profile/item examples, excluding trace content.
- summary.json: machine-readable overall results and provenance.

All non-trace rows are scanned. Trace contents are deliberately excluded; their
row counts, sizes, and schemas are taken from Parquet metadata. No model
inference, remote data refresh, exhaustive semantic labeling, cross-benchmark
content deduplication, or asset availability audit is performed.
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "payload/data")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/eda")
    parser.add_argument("--render-only", action="store_true", help="rebuild reports from existing EDA CSV/JSON outputs")
    args = parser.parse_args()
    if args.render_only:
        write_report(args.out, json.loads((args.out / "summary.json").read_text()),
                     pd.read_csv(args.out / "benchmarks.csv"), pd.read_csv(args.out / "table_totals.csv"),
                     pd.read_csv(args.out / "missingness.csv"), pd.read_csv(args.out / "identity_coverage.csv"))
    else:
        run(args.data_root, args.out)
