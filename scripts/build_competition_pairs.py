#!/usr/bin/env python3
"""Build the subject-benchmark pair matrix over the whole public corpus."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.competition_pairs import PairMatrixConfig, build_pair_matrix, refresh_manifest
from paiec.feature_preparation import local_path, write_json


RECIPE_KEYS = {"version", "data_root", "dataset_revision", "min_items", "require_binary",
               "require_item_granularity", "require_text_modality", "keep_lower_is_better"}


def load_recipe(path):
    config = json.loads(Path(path).read_text())
    if set(config) != RECIPE_KEYS or config["version"] != "competition_pairs_v1":
        raise ValueError("Unknown pair-matrix recipe")
    if config["data_root"] != "payload/data":
        raise ValueError("The pair matrix reads only the public corpus payload")
    return PairMatrixConfig.model_validate(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/competition_pairs_v1.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a repository pipeline configuration")
    recipe = load_recipe(config_path)
    out = local_path(ROOT, args.out)
    if out.exists():
        raise ValueError("Use a fresh output directory for the pair matrix")

    def log(benchmark, cells, admitted):
        print(f"{benchmark:28s} cells={cells:5d} admitted={admitted}", flush=True)

    summary = build_pair_matrix(ROOT, recipe, out, log=log)
    write_json(out / "resolved_config.json", recipe.model_dump())
    refresh_manifest(out)
    print(json.dumps({key: summary[key] for key in
                      ["benchmarks_read", "benchmarks_admitted", "benchmarks_excluded", "cells",
                       "labelled_responses", "cells_by_benchmark", "exclusion_reasons",
                       "distinct_named_models", "named_models_in_multiple_benchmarks",
                       "mean_cell_rate", "mean_bernoulli_variance"]}, indent=2))


if __name__ == "__main__":
    main()
