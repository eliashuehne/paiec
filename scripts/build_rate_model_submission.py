#!/usr/bin/env python3
"""Package a rate-model submission for a run whose gate cleared, or explain why not.

The package is refitted on every admitted pair of the judged direction, never on
a single fold model. Nothing is uploaded and no hosted execution is claimed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.competition_pairs import load_pair_matrix
from paiec.competition_submission import build
from paiec.feature_preparation import local_path, write_json
from paiec.rate_model import (ITEM_FEATURES, ItemFeatureStore, cell_feature_matrix, fit_difficulty,
                              fit_two_way, item_feature_frame)

REQUIRED_GATE = "clears_matched_adaptive_control"


def local_check(package):
    """Run the starting kit's ZIP checker when the baseline copy is present."""
    checker = ROOT / "baseline" / "check_submission_zip.py"
    if not checker.is_file():
        return {"ran": False, "reason": "starting-kit checker is not in the repository"}
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "submission.zip"
        subprocess.run([sys.executable, "-m", "zipfile", "-c", str(archive)]
                       + [str(path) for path in sorted(package.iterdir())],
                       check=True, capture_output=True)
        completed = subprocess.run([sys.executable, str(checker), str(archive), "--static-only"],
                                   capture_output=True, text=True)
    return {"ran": True, "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="rate-model run directory with summary.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = local_path(ROOT, args.run)
    summary = json.loads((run / "summary.json").read_text())
    gate = summary.get("gate", {})
    if gate.get("verdict") != REQUIRED_GATE:
        print(json.dumps({"packaged": False, "reason": "gate did not clear",
                          "verdict": gate.get("verdict"), "failed": gate.get("failed")}, indent=2))
        # A refusal must not look like success to whatever invoked the packager.
        raise SystemExit(1)
    import pyarrow.parquet as pq
    matrix_directory = local_path(ROOT, summary["recipe"]["pair_matrix"])
    matrix = load_pair_matrix(matrix_directory)
    cells = [cell for cell in matrix["cells"]
             if cell["direction"] == summary["recipe"]["ability_direction"]]
    two_way = fit_two_way(cells, name_shrinkage=summary["recipe"]["name_shrinkage"],
                          benchmark_shrinkage=summary["recipe"]["benchmark_shrinkage"],
                          direction_scope=summary["recipe"]["ability_direction"])
    table = pq.read_table(matrix_directory / "items.parquet",
                          columns=["benchmark", "item_id", "content"]).to_pandas()
    store = ItemFeatureStore(item_feature_frame(table))
    features = cell_feature_matrix(cells, matrix["labels"], store)
    difficulty = fit_difficulty(cells, two_way.difficulty, features,
                                alpha=summary["recipe"]["difficulty_alpha"])
    state = {"mu": float(two_way.mu),
             "ability": {name: float(value) for name, value in two_way.ability.items()},
             "intercept": float(difficulty["intercept"]),
             "feature_mean": [float(value) for value in difficulty["mean"]],
             "feature_scale": [float(value) for value in difficulty["scale"]],
             "coefficients": [float(value) for value in difficulty["coefficients"]],
             "prior_strength": float(summary["recipe"]["prior_strength"]),
             "benchmark_weight": float(summary["recipe"]["benchmark_evidence_weight"])}
    out = local_path(ROOT, args.out)
    result = build(state, out, gate=gate)
    result["benchmarks"] = len(two_way.difficulty)
    result["abilities"] = len(two_way.ability)
    result["features"] = list(ITEM_FEATURES)
    result["local_check"] = local_check(out)
    write_json(out / "build_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
