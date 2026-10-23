#!/usr/bin/env python3
"""Fetch the catalog's pinned evidence snapshots without executing downloaded code."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(catalog, cache):
    sources = {s["source_id"]: s for s in map(json.loads, (catalog / "sources.jsonl").read_text().splitlines())}
    cache.mkdir(parents=True, exist_ok=True)
    failures = []
    for snapshot in json.loads((catalog / "snapshots.json").read_text()):
        target = cache / snapshot["cache_file"]
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == snapshot["sha256"]:
            continue
        source = sources[snapshot["source_id"]]
        try:
            request = Request(source["location"], headers={"User-Agent": "PAIEC metadata research"})
            with urlopen(request, timeout=30) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ValueError("Metadata source exceeds 8 MB limit")
            if hashlib.sha256(raw).hexdigest() != snapshot["sha256"]:
                raise ValueError("Source changed: expected saved hash; existing archive preserved")
            target.write_bytes(raw)
        except Exception as error:
            failures.append(f"{snapshot['source_id']}: {error}")
    if failures:
        raise RuntimeError("\n".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("research/system_profiles/collection_v1"))
    parser.add_argument("--cache", type=Path, default=Path("runs/system_profile_sources_20261022"))
    args = parser.parse_args()
    fetch(args.catalog, args.cache)
    print("Evidence snapshots present and verified. No inference or training performed.")
