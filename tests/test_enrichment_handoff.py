import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from paiec.competition_fields import SUBJECT_FIELDS
from paiec.enrichment_handoff import TABLES, export_enrichment, verify_handoff
from paiec.feature_preparation import stable_hash, write_json, write_jsonl, read_jsonl
from paiec.head_training import file_hash


class EnrichmentHandoffTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        evidence = [{"source_id": "s", "locator": "config", "excerpt": "Example documentation"}]
        def fact(value, status="documented"):
            return {"value": value, "status": status, "evidence": [] if value is None else evidence,
                    "note": "Rounded component count; not an exact evaluated-checkpoint total."}
        subject = {k: "" for k in SUBJECT_FIELDS}
        subject.update(normalized_name="Example base model", subject_features_extra="shots=0")
        key = stable_hash(subject)
        tables = {n: [] for n in TABLES}
        resolution = {"model_profile_id": "model.example", "harness_profile_id": "harness.example",
                      "model_identity_basis": "base_family", "exact_evaluated_checkpoint_verified": False}
        tables["systems.jsonl"] = [{"system_key": key, "recorded": subject, "resolution": resolution}]
        tables["typed_run_features.jsonl"] = [{"system_key": key, "recorded": subject, "features": {
            "few_shot_count": fact(0, "recorded"), "enabled_tools": fact(None, "unknown")}}]
        for pid, kind in [("model.example", "model"), ("harness.example", "harness")]:
            filename = "supplemented_" + ("model" if kind == "model" else "harness") + "_profiles.jsonl"
            tables[filename] = [{"profile_id": pid, "entity_type": kind, "identity": {"resolution": "unresolved"},
                "valid_from": None, "valid_until": None, "specifications": {"extra": fact("documented extra"),
                    "unreviewed_inference": fact(123, "inferred")}}]
        tables["reviewed_static_features.jsonl"] = [
            {"profile_id": "model.example", "field": "architecture.total_parameters", "fact": fact(8000),
             "review_status": "reviewed", "review_disposition": "value_confirmed"},
            {"profile_id": "model.example", "field": "release_date", "fact": fact(None, "unknown"),
             "review_status": "reviewed", "review_disposition": "not_publicly_established"},
            {"profile_id": "model.example", "field": "architecture.parameterization", "fact": fact("dense", "inferred"),
             "review_status": "reviewed", "review_disposition": "value_confirmed"}]
        tables["profile_sources.jsonl"] = [{"source_id": "s", "location": "https://example.com/config", "eligibility": "unreviewed"}]
        tables["qualified_harness_observations.jsonl"] = [{"profile_id": "harness.example", "source_id": "s",
            "field": "supported_tools", "value": ["python"], "status": "documented_supported"}]
        for n, rows in tables.items():
            write_jsonl(source / n, rows)
        # This tempting extra file must never be exported, even when present.
        write_jsonl(source / "training_targets.jsonl", [{"label": 1, "private": "do not export"}])
        write_json(source / "manifest.json", {"version": "feature_preparation_v1", "artifacts": {n: file_hash(source / n) for n in TABLES}})
        guide = root / "guide.md"
        guide.write_text("# Collected model/harness information\n")
        return source, guide, file_hash(source / "manifest.json"), key

    def test_fact_status_scope_sources_and_exact_run_mapping_survive_without_model_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, guide, digest, key = self.fixture(root)
            out = root / "enrichment_v1"
            receipt = export_enrichment(source, out, expected_manifest_sha256=digest, guide=guide)
            manifest = verify_handoff(out)
            model = list(read_jsonl(out / "models.jsonl"))[0]
            self.assertEqual(model["identity"]["resolution"], "unresolved")
            self.assertIsNone(model["facts"]["release_date"]["value"])
            self.assertIn("Rounded", model["facts"]["architecture.total_parameters"]["note"])
            self.assertEqual(model["facts"]["architecture.parameterization"]["status"], "inferred")
            self.assertEqual(model["facts"]["extra"]["review_disposition"], "previously_documented")
            self.assertNotIn("unreviewed_inference", model["facts"])
            run = list(read_jsonl(out / "runs.jsonl"))[0]
            self.assertEqual(run["system_key"], key)
            self.assertEqual(run["features"]["few_shot_count"]["value"], 0)
            self.assertIsNone(run["features"]["enabled_tools"]["value"])
            self.assertEqual(receipt["counts"]["inferred_additional_fields_excluded"], 2)
            self.assertFalse(manifest["outcome_labels_included"])
            self.assertFalse(manifest["feature_matrix_included"])
            with zipfile.ZipFile(root / "enrichment_v1.zip") as archive:
                self.assertFalse(any("training_targets" in n or n.endswith(".npy") for n in archive.namelist()))
            with (out / "run_settings.csv").open() as stream:
                rows = {r["field"]: r for r in csv.DictReader(stream)}
            self.assertEqual(rows["few_shot_count"]["value_json"], "0")
            self.assertEqual(rows["enabled_tools"]["value_json"], "null")
            with self.assertRaises(FileExistsError):
                export_enrichment(source, out, expected_manifest_sha256=digest, guide=guide)
            (out / "models.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_handoff(out)

    def test_changed_sources_and_capability_defaults_cannot_become_recorded_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, guide, digest, _ = self.fixture(root)
            rows = list(read_jsonl(source / "typed_run_features.jsonl"))
            rows[0]["features"]["few_shot_count"]["status"] = "documented"
            write_jsonl(source / "typed_run_features.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "source table changed"):
                export_enrichment(source, root / "export", expected_manifest_sha256=digest, guide=guide)
            m = json.loads((source / "manifest.json").read_text())
            m["artifacts"]["typed_run_features.jsonl"] = file_hash(source / "typed_run_features.jsonl")
            write_json(source / "manifest.json", m)
            with self.assertRaisesRegex(ValueError, "Run settings must remain recorded"):
                export_enrichment(source, root / "export", expected_manifest_sha256=file_hash(source / "manifest.json"), guide=guide)
            self.assertFalse((root / "export").exists())
