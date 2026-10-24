#!/usr/bin/env python3
"""Audit data roles or reversibly retire explicitly marked untrained previews."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.data_lifecycle import audit_catalog, retire_preview, restore_preview
from paiec.feature_preparation import local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "retire", "restore"])
    parser.add_argument("--name", help="Exact catalog entry name for retirement")
    parser.add_argument("--out", help="Fresh retirement archive under runs/")
    parser.add_argument("--archive", help="Existing retirement archive directory for restoration")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "configs/data/data_versions_v1.json").read_text())
    if args.action == "audit":
        result = audit_catalog(ROOT, catalog)
    elif args.action == "retire":
        if not args.name or not args.out:
            parser.error("Retirement needs --name and a fresh --out archive directory")
        entry = next(e for e in catalog["artifacts"] if e["name"] == args.name)
        result = retire_preview(ROOT, entry, local_path(ROOT, args.out))
    else:
        if not args.archive:
            parser.error("Restoration needs --archive")
        result = restore_preview(ROOT, local_path(ROOT, args.archive))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
