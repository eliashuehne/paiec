#!/usr/bin/env python3
"""Augment reviewed training inputs to full-budget scope coverage; no model work."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.competition_panel import CompetitionPanelConfig, prepare_competition_panel
from paiec.feature_preparation import local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/competition_panel_v1.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a pipeline configuration under configs/pipelines")
    config = CompetitionPanelConfig.model_validate_json(config_path.read_text())
    out = local_path(ROOT, args.out)
    base = local_path(ROOT, config.base_prepared)
    if out == base or out.is_relative_to(base) or base.is_relative_to(out):
        raise ValueError("Keep augmented sources separate from the immutable base")
    print(json.dumps(prepare_competition_panel(ROOT, config.model_dump(), out), indent=2))


if __name__ == "__main__":
    main()
