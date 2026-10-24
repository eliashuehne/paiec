import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from paiec.evidence_workflow import (EvidenceWorkflowConfig, verify_evidence_inputs,
    compare_evidence_controls, select_evidence_candidate, finalize_evidence_workflow)
from paiec.feature_preparation import write_json, write_jsonl
from paiec.head_training import file_hash


class EvidenceWorkflowTests(unittest.TestCase):
    def fixture(self, root):
        source, prepared, out = root / "runs/source", root / "runs/data", root / "runs/workflow"
        source.mkdir(parents=True)
        (prepared / "evidence_plan_v1").mkdir(parents=True)
        out.mkdir(parents=True)
        (root / "configs").mkdir()
        write_json(root / "configs/split.json", {"family_roles": {f: "training" for f in "ABC"}})
        rows, targets, episodes, provenance = [], [], [], []
        for family in "ABC":
            keys = [family + str(i) for i in range(80)]
            for i, key in enumerate(keys):
                line = len(rows)
                rows.append({"row_key": key, "role": "training", "source_line": line,
                             "scope_key": family, "task_group": key})
                targets.append({"row_key": key, "native_binary_value": i % 2})
                provenance.append({"role": "training", "family": family, "jsonl_line": line})
            episodes.append({"scope_key": family, "acquisition_task_groups": keys[:40],
                "query_task_groups": keys[40:], "support_order_row_keys": keys[:31], "query_row_keys": keys[40:]})
        pd.DataFrame(provenance).to_parquet(source / "family.parquet", index=False)
        write_json(source / "manifest.json", {"family_provenance_path": "family.parquet",
            "artifacts": {"family.parquet": file_hash(source / "family.parquet")},
            "data_config": {"split": "configs/split.json"}})
        write_json(prepared / "source_contract.json", {"root": str(root), "prepared": str(source),
            "config": {"source_policy": "provided_dataset_only"},
            "source_manifest_sha256": file_hash(source / "manifest.json"),
            "split_sha256": file_hash(root / "configs/split.json")})
        write_json(prepared / "summary.json", {"source_policy": "provided_dataset_only",
            "external_profile_facts_used": 0, "records_by_role": {"training": len(rows)}})
        write_jsonl(prepared / "examples.jsonl", rows)
        write_jsonl(prepared / "training_targets.jsonl", targets)
        write_jsonl(prepared / "evidence_plan_v1/training_row_index.jsonl", rows)
        write_jsonl(prepared / "evidence_plan_v1/evidence_episode_links.jsonl", episodes)
        write_json(prepared / "manifest.json", {"version": "competition_data_v1", "artifacts": {
            str(p.relative_to(prepared)): file_hash(p) for p in sorted(prepared.rglob("*")) if p.is_file()}})
        return prepared, out, {"prepared": "runs/data"}

    def run_stages(self, root):
        prepared, out, config = self.fixture(root)
        inputs = verify_evidence_inputs(root, config)
        controls = compare_evidence_controls(inputs, out)
        selected = select_evidence_candidate(inputs, controls, out)
        return prepared, out, inputs, controls, selected

    def test_synthetic_training_workflow_never_approves_a_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, out, inputs, controls, selected = self.run_stages(Path(temporary))
            report = finalize_evidence_workflow(inputs, controls, selected, out)
            self.assertEqual(controls["training_scopes"], 3)
            self.assertEqual(selected["training_families"], 3)
            self.assertFalse(report["release_approved"])
            self.assertFalse(report["submission_built"])
            self.assertFalse(report["development_outcomes_used"])
            self.assertFalse(report["sealed_roles_opened"])
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertIn("selection/evidence_policy.json", manifest["artifacts"])
            self.assertIn("fixed_controls/manifest.json", manifest["artifacts"])

    def test_input_and_stage_changes_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared, out, inputs, controls, selected = self.run_stages(Path(temporary))
            policy = out / "selection/evidence_policy.json"
            original = policy.read_text()
            policy.write_text(original + " ")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                finalize_evidence_workflow(inputs, controls, selected, out)
            policy.write_text(original)
            manifest = prepared / "manifest.json"
            manifest.write_text(manifest.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                finalize_evidence_workflow(inputs, controls, selected, out)

    def test_partial_budget_coverage_cannot_select_a_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, out, config = self.fixture(Path(temporary))
            inputs = verify_evidence_inputs(Path(temporary), config)
            controls = compare_evidence_controls(inputs, out)
            controls["excluded_small_retained_scopes"] = 1
            with self.assertRaisesRegex(ValueError, "all evidence budgets"):
                select_evidence_candidate(inputs, controls, out)
            self.assertFalse((out / "selection").exists())
        for config in [{"prepared": "runs/data", "source_policy": "web_enriched"},
                       {"prepared": "../elsewhere"}, {"prepared": "runs/data", "open_development": True}]:
            with self.assertRaises(ValueError):
                EvidenceWorkflowConfig.model_validate(config)

    def test_plan_needs_no_outcomes_or_pipeline_runtime(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            destination = root / "runs" / ("plan-only-" + Path(temporary).name)
            result = subprocess.run([sys.executable, str(root / "scripts/run_competition_pipeline.py"),
                "--workflow", "evidence", "--plan", "--out", str(destination.relative_to(root))],
                cwd=root, capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            self.assertFalse(report["execution_started"])
            self.assertFalse(report["opens_development_outcomes"])
            self.assertEqual(len(report["stages"]), 4)
            self.assertFalse(destination.exists())
