"""Report the disk footprint of this project and its HuggingFace caches.

`baseline/tools/prepare_data.py` downloads every table through the global
HuggingFace cache and then copies it into `payload/`. Both copies persist, so a
full build costs roughly twice the payload size. This script shows both, and can
reclaim the cache copy once `payload/` is verified present.

    python scripts/disk_report.py            # read-only
    python scripts/disk_report.py --cleanup  # delete the duplicated dataset cache

The model cache (Laya weights) is never removed: it is the only copy of the
checkpoint and shares no blobs with the dataset cache.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "payload" / "data"
DATASET_REPO = "aims-foundations/measurement-db"


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TiB"


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() or item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help=f"delete the {DATASET_REPO} cache once payload/ exists",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        scan_cache_dir = None

    print(f"project: {ROOT}")
    for label, path in (
        ("payload/data", PAYLOAD),
        ("runs/", ROOT / "runs"),
        (".venv-laya-mlx", ROOT / ".venv-laya-mlx"),
        (".venv-laya", ROOT / ".venv-laya"),
        (".venv", ROOT / ".venv"),
    ):
        size = tree_size(path)
        if size or path.exists():
            print(f"  {label:16s} {human(size):>12s}")

    usage = shutil.disk_usage(ROOT)
    print(f"\nvolume: {human(usage.free)} free of {human(usage.total)}")

    if scan_cache_dir is None:
        print("\nhuggingface_hub not installed here; skipping cache report.")
        print("Use .venv-laya-mlx/bin/python to include it.")
        return 0

    cache = scan_cache_dir()
    print(f"\nHuggingFace cache: {human(cache.size_on_disk)}")
    duplicated = []
    for repo in sorted(cache.repos, key=lambda r: -r.size_on_disk):
        print(f"  {repo.repo_type:8s} {repo.repo_id:45s} {human(repo.size_on_disk):>12s}")
        if repo.repo_id == DATASET_REPO:
            duplicated.append(repo)

    if not duplicated:
        print("\nNo duplicated dataset cache.")
        return 0

    freed = sum(r.size_on_disk for r in duplicated)
    print(f"\n{DATASET_REPO} is cached ({human(freed)}).")

    if not PAYLOAD.is_dir():
        print("payload/ is missing: keep the cache, it is currently the only copy.")
        return 0

    if not args.cleanup:
        print("payload/ exists, so this cache is a duplicate.")
        print("Run with --cleanup to reclaim it (the model cache is left alone).")
        return 0

    revisions = [rev.commit_hash for repo in duplicated for rev in repo.revisions]
    strategy = cache.delete_revisions(*revisions)
    print(f"reclaiming {human(strategy.expected_freed_size)} ...")
    strategy.execute()
    print(f"cache now: {human(scan_cache_dir().size_on_disk)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
