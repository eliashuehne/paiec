#!/usr/bin/env python3
"""Package existing reviewed system information; no collection, fitting or inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.enrichment_handoff import export_enrichment
from paiec.feature_preparation import local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/enrichment_handoff_v1.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if config["version"] != "system_enrichment_export_v1":
        raise ValueError("Unknown export recipe")
    result = export_enrichment(local_path(ROOT, config["source"]), local_path(ROOT, config["out"]),
        expected_manifest_sha256=config["source_manifest_sha256"], guide=ROOT / config["guide"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
