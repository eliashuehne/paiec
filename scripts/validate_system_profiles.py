#!/usr/bin/env python3
"""Validate research profiles and their evidence links. No inference or networking."""

import argparse
from pathlib import Path
from paiec.system_profiles import load_catalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="research/system_profiles/v1")
    args = parser.parse_args()
    profiles, sources = load_catalog(Path(args.catalog))
    print(
        f"Valid: {len(profiles)} research profiles, {len(sources)} sources. Predictor use remains disabled."
    )
