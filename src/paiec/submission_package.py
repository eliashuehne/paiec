"""Allowlisted, reproducible local packaging of a selected evidence policy."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

from .evidence_policy import EvidencePolicy
from .head_training import file_hash
from .feature_preparation import write_json


def build_evidence_submission(selection, out):
    selection, out = Path(selection).resolve(), Path(out)
    if out.resolve().is_relative_to(selection) or selection.is_relative_to(out.resolve()):
        raise ValueError("Keep submission output separate from immutable selection artifacts")
    manifest = json.loads((selection / "manifest.json").read_text())
    if not {"summary.json", "evidence_policy.json"} <= set(manifest["artifacts"]):
        raise ValueError("Selection policy and summary must be hash-pinned")
    for name, sha in manifest["artifacts"].items():
        path = (selection / name).resolve()
        if not path.is_relative_to(selection) or file_hash(path) != sha:
            raise ValueError("Evidence selection artifact changed")
    summary = json.loads((selection / "summary.json").read_text())
    if (summary.get("version") != "evidence_policy_selection_v1" or summary.get("source_policy") != "provided_dataset_only" or
        summary.get("full_budget_coverage_complete") is not True or summary.get("original_development_outcomes_used") is not False or
        summary.get("sealed_roles_opened") is not False or summary.get("training_families", 0) < 3):
        raise ValueError("Selection lacks complete, isolated training-family qualification")
    policy_path = selection / "evidence_policy.json"
    policy_sha = file_hash(policy_path)
    if policy_sha != summary["policy_sha256"]:
        raise ValueError("Policy differs from its selection receipt")
    policy = EvidencePolicy.load(policy_path, expected_sha256=policy_sha)
    if policy.strength != summary["selected_strength"]:
        raise ValueError("Policy hyperparameter differs from selection")
    source = Path(__file__).parent
    model = '''from pathlib import Path
from paiec_submission.evidence_policy import EvidencePolicy
from paiec_submission.competition_fields import normalize_fields

_policy = EvidencePolicy.load(Path(__file__).with_name("policy.json"), expected_sha256=POLICY_SHA)

def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = normalize_fields(*input)
    return _policy.predict([subject, item], labeled)
'''.replace("POLICY_SHA", repr(policy_sha))
    readme = ("# Evidence-policy candidate\n\n"
        "Uses only this call's revealed native binary outcomes from the same complete visible system and anonymous benchmark. "
        "No labels persist between calls. Exact query observations are excluded; legitimate repeated trials are retained. "
        "Missing group identity returns the prior. No network, model repository, external profile, runtime fitting or third-party package is used.\n\n"
        f"Prior: {policy.prior}; strength: {policy.strength}. Selection used only {summary['training_families']} held training families, before original development. "
        "This predicts a shared pair mean and does not model item difficulty. No labeling.py is provided, so acquisition remains the platform default.\n\n"
        "Local packaging and smoke checks do not establish hosted execution or hidden-test performance.\n")
    files = {"model.py": model.encode(), "policy.json": policy_path.read_bytes(), "README.md": readme.encode(),
             "paiec_submission/__init__.py": b'"""Portable supplied-evidence inference."""\n',
             "paiec_submission/evidence_policy.py": (source / "evidence_policy.py").read_bytes(),
             "paiec_submission/competition_fields.py": (source / "competition_fields.py").read_bytes()}
    out.mkdir(parents=True, exist_ok=False)
    extracted = out / "submission"
    for name, data in files.items():
        target = extracted / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    archive_path = out / "submission.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            member = ZipInfo(name)
            member.compress_type = ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, data)
    report = {"version": "evidence_submission_v1", "source_policy": "provided_dataset_only",
        "selection_manifest_sha256": file_hash(selection / "manifest.json"), "policy_sha256": policy_sha,
        "prior": policy.prior, "strength": policy.strength,
        "archive_sha256": file_hash(archive_path), "archive_bytes": archive_path.stat().st_size,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
        "external_model_repositories": 0, "third_party_runtime_dependencies": [],
        "research_metadata_packaged": False, "raw_training_data_packaged": False,
        "uploaded": False, "hosted_execution_verified": False}
    write_json(out / "package_receipt.json", report)
    return report
