#!/usr/bin/env python3
"""Check fitted control artifacts in an isolated, standard-library-only runtime."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.competition_analysis import verify_competition_bundle
from paiec.conditional_gate import assess_conditional_controls
from paiec.feature_preparation import local_path, read_jsonl, write_json
from paiec.head_training import file_hash


DRIVER = '''import json, math, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from paiec.conditional_predictor import ConditionalPredictor
root = Path(sys.argv[1])
fixtures = json.loads((root / "fixtures.json").read_text())
models = {}
drift = 0.
for fixture in fixtures:
    key = fixture["model"]
    if key not in models:
        models[key] = ConditionalPredictor.load(root / key, expected_sha256=fixture["sha256"])
    value = models[key].predict(fixture["input"], [])
    assert type(value) is float and math.isfinite(value) and 0 <= value <= 1
    drift = max(drift, abs(value - fixture["expected_probability"]))
    assert abs(value - fixture["expected_probability"]) <= 1e-10
print(json.dumps({"fixtures":len(fixtures),"models":len(models),"maximum_probability_drift":drift,
                  "isolated_interpreter":True,"site_packages_disabled":True}))
'''


def qualify(prepared, analysis, out):
    verify_competition_bundle(prepared)
    manifest = json.loads((analysis / "manifest.json").read_text())
    for name, sha in manifest["artifacts"].items():
        path = (analysis / name).resolve()
        if not path.is_relative_to(analysis.resolve()) or file_hash(path) != sha:
            raise ValueError("Control artifact changed")
    report = json.loads((analysis / "summary.json").read_text())
    assessment = assess_conditional_controls(report)
    rows = {r["row_key"]: r for r in read_jsonl(prepared / "examples.jsonl") if r["role"] == "training"}
    systems = {r["system_key"]: r["recorded"] for r in read_jsonl(prepared / "systems.jsonl")}
    tasks = {r["task_key"]: r["recorded"] for r in read_jsonl(prepared / "tasks.jsonl")}
    predictions = list(read_jsonl(analysis / "out_of_family_predictions.jsonl"))
    if {r["row_key"] for r in predictions} != set(rows):
        raise ValueError("Control predictions do not cover the exact training source")
    fixtures = []
    with tempfile.TemporaryDirectory(prefix="paiec-portable-control-") as directory:
        folder = Path(directory)
        (folder / "paiec").mkdir()
        (folder / "paiec/__init__.py").write_text("")
        for name in ["conditional_features.py", "conditional_predictor.py", "competition_fields.py"]:
            shutil.copyfile(ROOT / "src/paiec" / name, folder / "paiec" / name)
        for family in sorted({p["validation_family"] for p in predictions}):
            members = [p for p in predictions if p["validation_family"] == family]
            members.sort(key=lambda p: (len(tasks[rows[p["row_key"]]["task_key"]]["item_content"]), p["row_key"]))
            for variant in ["additive", "conditional"]:
                name = "fold_models/" + family + "/" + variant + ".json"
                model_path = (analysis / name).resolve()
                if not model_path.is_relative_to(analysis.resolve()) or name not in manifest["artifacts"]:
                    raise ValueError("Fold checkpoint is not hash-pinned")
                state = json.loads(model_path.read_text())
                if state["training_provenance"]["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json"):
                    raise ValueError("Control belongs to another prepared dataset")
                destination = folder / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(model_path, destination)
                for member in [members[0], members[len(members) // 2], members[-1]]:
                    row = rows[member["row_key"]]
                    fixtures.append({"model": name, "sha256": manifest["artifacts"][name],
                        "input": [systems[row["system_key"]], dict(tasks[row["task_key"]], benchmark_id=row["benchmark_id"])],
                        "expected_probability": member[variant]})
        write_json(folder / "fixtures.json", fixtures)
        (folder / "driver.py").write_text(DRIVER)
        completed = subprocess.run([sys.executable, "-I", "-S", str(folder / "driver.py"), str(folder)],
                                   capture_output=True, text=True, check=True, timeout=60)
        runtime = json.loads(completed.stdout)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "candidate_assessment.json", assessment)
    result = {"version": "conditional_control_qualification_v1", "runtime": runtime,
        "source_policy": "provided_dataset_only", "analysis_manifest_sha256": file_hash(analysis / "manifest.json"),
        "checks": "shortest, median and longest held inputs per training family and variant",
        "candidate_release_checks_passed": assessment["any_candidate_passed_initial_checks"],
        "hosted_execution_verified": False, "deployment_performed": False,
        "native_laya_inference": False, "gpu_jobs": 0, "paid_calls": 0}
    write_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(local_path(ROOT, args.prepared), local_path(ROOT, args.analysis), local_path(ROOT, args.out)), indent=2))


if __name__ == "__main__":
    main()
