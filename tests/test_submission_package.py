import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

from paiec.feature_preparation import write_json
from paiec.head_training import file_hash
from paiec.submission_package import build_evidence_submission


class SubmissionPackageTests(unittest.TestCase):
    def fixture(self, path):
        path.mkdir()
        write_json(path / "evidence_policy.json", {"version": "evidence_policy_v1", "source_policy": "provided_dataset_only",
                   "prior": .5, "strength": .5})
        write_json(path / "summary.json", {"version": "evidence_policy_selection_v1", "source_policy": "provided_dataset_only",
            "full_budget_coverage_complete": True, "original_development_outcomes_used": False, "sealed_roles_opened": False,
            "training_families": 3, "policy_sha256": file_hash(path / "evidence_policy.json"), "selected_strength": .5})
        write_json(path / "manifest.json", {"artifacts": {p.name: file_hash(p) for p in path.iterdir() if p.is_file()}})

    def test_allowlisted_package_is_deterministic_and_has_no_payload_or_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "selection")
            first = build_evidence_submission(root / "selection", root / "one")
            second = build_evidence_submission(root / "selection", root / "two")
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            with ZipFile(root / "one/submission.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"model.py", "policy.json", "README.md", "paiec_submission/__init__.py",
                    "paiec_submission/evidence_policy.py", "paiec_submission/competition_fields.py"})
            self.assertEqual(first["external_model_repositories"], 0)
            self.assertFalse(first["uploaded"])

    def test_import_runs_without_installed_packages_and_fills_only_missing_known_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "selection")
            build_evidence_submission(root / "selection", root / "package")
            code = '''import sys
sys.path.insert(0,sys.argv[1])
import model
subject={"normalized_name":"unknown"}
query={"item_content":" Target ","benchmark_id":"opaque"}
assert model.predict([subject,{"item_content":"None"}],[])==.5
assert model.predict([subject,query],[])==.5
from paiec_submission.competition_fields import normalize_fields
s,q=normalize_fields(subject,query)
observed=dict(q,item_content="Observed")
assert model.predict([subject,query],[[[s,observed],1]])==1.25/1.5
assert model.predict([subject,query],[])==.5
try:
 model.predict([dict(subject,display_name="PRIVATE"),query],[])
except ValueError:
 pass
else:
 raise AssertionError("private field accepted")
print("OK")
'''
            run = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(root / "package/submission")],
                                 capture_output=True, text=True, check=True, timeout=10)
            self.assertEqual(run.stdout.strip(), "OK")

    def test_changed_or_incompletely_qualified_selection_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "selection")
            with self.assertRaisesRegex(ValueError, "separate"):
                build_evidence_submission(root / "selection", root / "selection/submission")
            (root / "selection/evidence_policy.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                build_evidence_submission(root / "selection", root / "bad")
