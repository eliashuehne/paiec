"""Conservative catalog audit and reversible retirement of marked data previews."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from .feature_preparation import local_path, write_json
from .head_training import file_hash


def file_inventory(path):
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Data retirement does not follow symlinks")
    return {str(p.relative_to(path)): {"sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in sorted(path.rglob("*")) if p.is_file()}


def referenced_paths(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from referenced_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from referenced_paths(child)
    elif isinstance(value, str):
        yield value.rstrip("/")


def live_references(root, relative):
    # Recipes and checkpoint contracts are authoritative; old reports are history.
    files = list((root / "configs").rglob("*.json"))
    files += list((root / "runs").rglob("paiec_training.json"))
    found = []
    for p in files:
        if p.name == "data_versions_v1.json":
            continue
        value = json.loads(p.read_text())
        if any(v == relative or v.startswith(relative + "/") or
               v == str(root / relative) or v.startswith(str(root / relative) + "/") for v in referenced_paths(value)):
            found.append(str(p.relative_to(root)))
    return found


def audit_catalog(root, catalog):
    root = Path(root).resolve()
    entries = []
    for entry in catalog["artifacts"]:
        path = local_path(root, entry["path"])
        files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else []
        status = {**entry, "exists": path.exists(), "logical_bytes": sum(p.stat().st_size for p in files), "files": len(files)}
        if entry.get("archive"):
            archive = local_path(root, entry["archive"]) / "preview.tar.gz"
            status.update(archive_exists=archive.is_file(), archive_bytes=archive.stat().st_size if archive.is_file() else 0)
        entries.append(status)
    return {"version": "data_lifecycle_audit_v1", "artifacts": entries,
            "note": "Logical sizes; filesystem sharing and MLflow copies may differ. Protected data is never retired automatically."}


def retire_preview(root, entry, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if entry.get("lifecycle") != "retirable_preview":
        raise ValueError("Only explicitly cataloged untrained previews can be retired")
    if (root / entry["path"]).is_symlink():
        raise ValueError("Data retirement does not follow symlinks")
    path = local_path(root, entry["path"])
    marker = path / "SUPERSEDED.txt"
    if not marker.is_file() or "Untrained input-development preview" not in marker.read_text():
        raise ValueError("Preview must have an explicit untrained/superseded marker")
    replacement = local_path(root, entry["replacement"])
    if not replacement.is_dir() or replacement == path or replacement.is_relative_to(path):
        raise ValueError("A separate retained release must exist")
    if not out.is_relative_to(root / "runs") or any(out.is_relative_to(p) or p.is_relative_to(out) for p in [path, replacement]):
        raise ValueError("Retirement archive must be separate from preview and retained release")
    references = live_references(root, entry["path"])
    if references:
        raise ValueError("Preview is referenced by live recipes/checkpoints: " + ", ".join(references))
    inventory = file_inventory(path)
    out.mkdir(parents=True, exist_ok=False)
    archive = out / "preview.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(path, arcname=entry["path"], recursive=True)
    archived = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Archive contains a nonregular data member")
            relative = str(Path(member.name).relative_to(entry["path"]))
            digest = hashlib.sha256()
            with source.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            archived[relative] = {"sha256": digest.hexdigest(), "bytes": member.size}
    if archived != inventory or file_inventory(path) != inventory:
        raise ValueError("Preview changed or archive failed byte-for-byte verification")
    if live_references(root, entry["path"]):
        raise ValueError("Preview acquired a live reference during archival")
    report = {"version": "retired_preview_v1", "original_path": entry["path"], "replacement": entry["replacement"],
        "archive_sha256": file_hash(archive), "archive_bytes": archive.stat().st_size,
        "original_bytes": sum(v["bytes"] for v in inventory.values()), "files": inventory,
        "active_recipe_checkpoint_references": [], "archive_verified": True}
    write_json(out / "receipt.json", report)
    shutil.rmtree(path)
    return report


def restore_preview(root, archive_directory):
    root, folder = Path(root).resolve(), Path(archive_directory)
    receipt = json.loads((folder / "receipt.json").read_text())
    archive = folder / "preview.tar.gz"
    if file_hash(archive) != receipt["archive_sha256"]:
        raise ValueError("Retired preview archive changed")
    target = local_path(root, receipt["original_path"])
    if target.exists():
        raise FileExistsError("Restoration never overwrites a data version")
    # Validate the entire archive before creating any restored files.
    with tarfile.open(archive, "r:gz") as source:
        members = {}
        for member in source.getmembers():
            if member.isdir():
                continue
            relative = str(Path(member.name).relative_to(receipt["original_path"]))
            if not member.isfile() or Path(relative).is_absolute() or ".." in Path(relative).parts or relative in members:
                raise ValueError("Unsafe or duplicate archive member")
            data = source.extractfile(member).read()
            if {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} != receipt["files"].get(relative):
                raise ValueError("Archived preview file changed")
            members[relative] = data
    if set(members) != set(receipt["files"]):
        raise ValueError("Archived preview is incomplete")
    target.mkdir(parents=True, exist_ok=False)
    for name, data in members.items():
        p = target / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return {"restored": receipt["original_path"], "files": len(members)}
