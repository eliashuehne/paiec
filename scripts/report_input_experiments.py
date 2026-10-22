#!/usr/bin/env python
"""Build a readable local report from completed input-design artifacts."""

import argparse
from html import escape
import json
from pathlib import Path

import pandas as pd


def system_diagnostic(root):
    """Separate metadata sensitivity from the extra task space its removal frees."""
    from paiec.compare import pair_curves
    from paiec.input_audit import audit_state
    from paiec.laya_predictor import build_state, load_questions
    from measure_tokens import load_tokenizer
    from run_eval import build_pairs

    paths = [root / "states" / name / "predictions.csv"
             for name in ("text_v1", "text_v1_no_system")]
    if not all(path.exists() for path in paths):
        return None
    frames = [pd.read_csv(path, dtype={"benchmark": str, "subject_id": str, "item_id": str})
              for path in paths]
    frames = [frame[frame.budget.eq(0)].copy() for frame in frames]
    keys = ["benchmark", "subject_id", "item_id"]
    paired = frames[0].merge(frames[1], on=keys, suffixes=("_full", "_omitted"), validate="one_to_one")
    if len(paired) != len(frames[0]) or len(paired) != len(frames[1]):
        raise ValueError("System ablation samples differ")
    tokenizer = load_tokenizer("english")
    suite = json.loads((root / "states/suite.json").read_text())
    pairs = build_pairs(Path(__file__).resolve().parents[1] / "payload/data", suite)
    lookup = {(pair.benchmark, pair.subject_id): pair for pair in pairs}
    records = []
    config = json.loads((root / "states/text_v1/config.json").read_text())
    question = config["variant"].get("question", load_questions()["correct"])
    for row in paired.itertuples():
        pair = lookup[(row.benchmark, row.subject_id)]
        complete = []
        for omit in (False, True):
            state = build_state(pair.subject, pair.item(row.item_id), [], state_format="text_v1", omit_system=omit)
            fields = audit_state(tokenizer, state, question, state_format="text_v1")["task_fields"]
            complete.append("item_content" in fields and all(value["complete"] for value in fields.values()))
        records.append({"benchmark": row.benchmark, "subject_id": row.subject_id,
                        "item_id": row.item_id, "task_complete_both": all(complete),
                        "absolute_probability_change": abs(row.probability_full - row.probability_omitted)})
    detail = pd.DataFrame(records)
    detail.to_csv(root / "system_ablation_visibility.csv", index=False)
    subsets = {"all_inputs": detail, "complete_task_in_both": detail[detail.task_complete_both]}
    report = {}
    for name, subset in subsets.items():
        if subset.empty:
            continue
        scores = []
        for frame in frames:
            restricted = frame.merge(subset[keys], on=keys, validate="one_to_one")
            curves = pair_curves(restricted)
            scores.append(sum(value[0] for value in curves.values()) / len(curves))
        report[name] = {"inputs": len(subset), "pairs": len(curves),
                        "mean_absolute_probability_change": subset.absolute_probability_change.mean(),
                        "brier_full_system": scores[0], "brier_system_omitted": scores[1]}
    (root / "system_ablation.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    diagnostic = system_diagnostic(root)
    parts = ["<h1>Laya input experiments</h1>",
             "<p>Exploratory public-data comparisons. Lower Brier ALC is better; "
             "always predicting 50% gives 0.2500. Rows within each phase use the same "
             "sample. Different phases may use different sample sizes.</p>"]
    for phase in ("prompts", "states", "evidence", "confirm"):
        rows = []
        for path in sorted((root / phase).glob("*/details.json")):
            d = json.loads(path.read_text())
            scores = pd.read_csv(path.parent / "scores.csv")
            rows.append({"Variant": d["variant"], "Inputs": d["n_eval_inputs"],
                         "Budget 0 Brier": scores[scores.budget.eq(0)].brier.mean(),
                         "Brier ALC": d["brier_alc"], "Forwards": d["forward_items"],
                         "Truncated": d["truncated_items"], "Seconds": d["seconds"],
                         "Run ID": d["run_id"]})
        if rows:
            parts += [f"<h2>{phase.capitalize()}</h2>",
                      pd.DataFrame(rows).to_html(index=False, float_format=lambda x: f"{x:.4f}")]
    if diagnostic:
        parts += ["<h2>System-description diagnostic</h2>",
                  "<p>Removing the system also frees context space. The complete-task subset "
                  "checks sensitivity when all supplied task fields fit in both conditions. "
                  "This subset has a different task mix; its scores are diagnostic.</p>",
                  pd.DataFrame(diagnostic).T.to_html(float_format=lambda x: f"{x:.4f}")]
    audit = root / "audit_final"
    summary = audit / "audit_summary.csv"
    if summary.exists():
        parts += ["<h2>What survived truncation?</h2>",
                  "<p>24 selected cases: shortest, middle and longest task for the first "
                  "system in each of eight benchmarks. These are diagnostic cases, not a "
                  "population estimate. Outcomes visible means their numeric values reached "
                  "the model; it does not establish that the model understood them. "
                  "The token limit includes the question head.</p>",
                  pd.read_csv(summary).to_html(index=False)]
        rows = [json.loads(line) for line in (audit / "retained_inputs.jsonl").open()]
        long_cases = {}
        for row in rows:
            if row["state_format"] == "json" and row["max_len"] == 512 and row["budget"] == 1:
                previous = long_cases.get(row["benchmark"])
                if previous is None or row["state_tokens"] > previous["state_tokens"]:
                    long_cases[row["benchmark"]] = row
        parts += ["<h2>Actual retained inputs</h2>",
                  "<p>Longest selected case per benchmark, at budget 1 and 512 tokens. "
                  "Expand each case to compare the rendered source with what the model retained. "
                  "Reference labels for evaluation items are never part of these inputs.</p>"]
        for benchmark, chosen in long_cases.items():
            parts.append(f"<h3>{escape(benchmark)}</h3>")
            for fmt in ("json", "text_v1"):
                row = next(r for r in rows if r["benchmark"] == benchmark and r["item_id"] == chosen["item_id"]
                           and r["max_len"] == 512 and r["budget"] == 1 and r["state_format"] == fmt)
                title = (f"{fmt}: {row['state_tokens_used']} of {row['state_tokens']} state tokens; "
                         f"{row['revealed_outcomes_used']} of {row['revealed_outcomes']} outcomes visible")
                parts += [f"<details><summary>{escape(title)}</summary>",
                          "<h4>Yes/no question</h4><pre>" + escape(row["question"]["instructions"]) + "</pre>",
                          "<h4>State retained by the tokenizer</h4><pre>" + escape(row["retained_state"]) + "</pre>",
                          "<h4>Full rendered state before truncation</h4><pre>" + escape(row["state"]) + "</pre></details>"]
    html = ('<!doctype html><meta charset="utf-8"><title>Laya input experiments</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#182333}'
            'table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}'
            'th,td{padding:8px;border:1px solid #dce1e8;text-align:left}th{background:#eef3f8}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px}'
            'details{border:1px solid #dce1e8;padding:14px;margin:12px 0}summary{cursor:pointer}</style>'
            + "\n".join(parts))
    path = root / "report.html"
    path.write_text(html)
    print(path.resolve())


if __name__ == "__main__":
    main()
