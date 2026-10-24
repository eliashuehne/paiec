import json
from pathlib import Path
import tempfile
import unittest

from paiec.competition_analysis import analyze_training_evidence, numeric_evidence_predictions
from paiec.feature_preparation import read_jsonl, write_json, write_jsonl
from paiec.head_training import file_hash


class CompetitionAnalysisTests(unittest.TestCase):
    def refresh_manifest(self, path):
        write_json(path / "manifest.json", {"version": "competition_data_v1", "artifacts": {
            str(p.relative_to(path)): file_hash(p) for p in path.rglob("*") if p.is_file() and p.name != "manifest.json"}})

    def fixture(self, path):
        (path / "evidence_plan_v1").mkdir(parents=True)
        rows, targets, episodes = [], [], []
        for scope, count in [("eligible", 80), ("small", 40)]:
            keys = [scope + str(i) for i in range(count)]
            for i, key in enumerate(keys):
                rows.append({"row_key": key, "role": "training", "scope_key": scope, "task_group": key})
                targets.append({"row_key": key, "native_binary_value": bool(i % 2)})
            episodes.append({"scope_key": scope, "acquisition_task_groups": keys[:count // 2],
                             "query_task_groups": keys[count // 2:], "support_order_row_keys": keys[:min(31,count // 2)],
                             "query_row_keys": keys[count // 2:]})
        write_jsonl(path / "evidence_plan_v1/training_row_index.jsonl", rows)
        write_jsonl(path / "evidence_plan_v1/evidence_episode_links.jsonl", episodes)
        write_jsonl(path / "training_targets.jsonl", targets)
        write_json(path / "source_contract.json", {"config": {"source_policy": "provided_dataset_only"}})
        write_json(path / "summary.json", {"source_policy": "provided_dataset_only", "external_profile_facts_used": 0,
                                          "records_by_role": {"training": 120}})
        self.refresh_manifest(path)

    def test_query_targets_never_change_probabilities_but_change_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "bundle")
            before = analyze_training_evidence(root / "bundle", root / "first")
            targets = list(read_jsonl(root / "bundle/training_targets.jsonl"))
            for row in targets:
                if row["row_key"].startswith("eligible") and int(row["row_key"][8:]) >= 40:
                    row["native_binary_value"] = False
            write_jsonl(root / "bundle/training_targets.jsonl", targets)
            self.refresh_manifest(root / "bundle")
            after = analyze_training_evidence(root / "bundle", root / "second")
            self.assertEqual((root / "first/pair_budget_predictions.jsonl").read_bytes(),
                             (root / "second/pair_budget_predictions.jsonl").read_bytes())
            self.assertNotEqual(before["brier_alc_equal_visible_scope"]["beta_k5"], after["brier_alc_equal_visible_scope"]["beta_k5"])
            self.assertEqual(before["fully_eligible_retained_scopes"], 1)
            self.assertEqual(before["excluded_small_retained_scopes"], 1)
            self.assertEqual(before["brier_alc_equal_visible_scope"]["constant"], .25)

    def test_research_enrichment_and_unpinned_targets_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "bundle")
            summary = json.loads((root / "bundle/summary.json").read_text())
            summary["external_profile_facts_used"] = 1
            write_json(root / "bundle/summary.json", summary)
            self.refresh_manifest(root / "bundle")
            with self.assertRaisesRegex(ValueError, "research enrichment"):
                analyze_training_evidence(root / "bundle", root / "bad")
            manifest = json.loads((root / "bundle/manifest.json").read_text())
            del manifest["artifacts"]["training_targets.jsonl"]
            write_json(root / "bundle/manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "hash-pinned"):
                analyze_training_evidence(root / "bundle", root / "unpinned")

    def test_only_supplied_binary_evidence_can_change_fixed_controls(self):
        self.assertEqual(numeric_evidence_predictions([]), {"constant": .5, "empirical_mean": .5, "beta_k5": .5})
        self.assertEqual(numeric_evidence_predictions([1])["beta_k5"], 3.5 / 6)
        self.assertEqual(numeric_evidence_predictions([0])["empirical_mean"], 0.)
        for invalid in [[2], ["1"], [.5]]:
            with self.assertRaises(ValueError):
                numeric_evidence_predictions(invalid)
