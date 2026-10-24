"""Isolated exact-package qualification, independent of the command-line wrapper."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile
import hashlib

from .competition_analysis import verify_competition_bundle
from .feature_preparation import read_jsonl, write_json
from .head_training import file_hash


DRIVER = '''import json,math,resource,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
import model
data=json.loads(Path(sys.argv[2]).read_text())
def input_for(key):
 row=data["rows"][key]
 return [data["systems"][row["system_key"]],dict(data["tasks"][row["task_key"]],benchmark_id=row["benchmark_id"])]
calls=0; drift=0.; started=time.perf_counter()
for budget in [31,0,7,1,15,3,0]:
 labeled=[]
 for episode in data["episodes"]:
  labeled.extend([[input_for(key),label] for key,label in episode["supports"][:budget]])
 for episode in data["episodes"]:
  observed=[label for _,label in episode["supports"][:budget]]
  expected=(sum(observed)+data["strength"]*data["prior"])/(len(observed)+data["strength"]) if observed else data["prior"]
  probability=model.predict(input_for(episode["query"]),labeled)
  assert type(probability) is float and math.isfinite(probability) and 0<=probability<=1
  drift=max(drift,abs(expected-probability))
  assert abs(expected-probability)<=1e-12
  calls+=1
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"prediction_calls":calls,"scopes":len(data["episodes"]),"budgets_tested_out_of_order":[31,0,7,1,15,3,0],
 "maximum_probability_drift":drift,"prediction_seconds":time.perf_counter()-started,
 "peak_process_rss_bytes":int(rss if sys.platform=="darwin" else rss*1024),
 "global_revealed_evidence_used":True,"isolated_interpreter":bool(sys.flags.isolated),"site_packages_disabled":bool(sys.flags.no_site)}))
'''


def qualify(prepared, package, out):
    verify_competition_bundle(prepared)
    receipt = json.loads((package / "package_receipt.json").read_text())
    if file_hash(package / "submission.zip") != receipt["archive_sha256"]:
        raise ValueError("Submission archive changed")
    with ZipFile(package / "submission.zip") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(receipt["files"]):
            raise ValueError("Submission archive member list differs from the checked runtime")
        for name, digest in receipt["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Archive and extracted runtime differ")
    for name, sha in receipt["files"].items():
        path = (package / "submission" / name).resolve()
        if not path.is_relative_to((package / "submission").resolve()) or file_hash(path) != sha:
            raise ValueError("Extracted submission files changed")
    policy = json.loads((package / "submission/policy.json").read_text())
    if policy["training_provenance"]["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json"):
        raise ValueError("Policy belongs to a different training panel")
    index = {r["row_key"]: r for r in read_jsonl(prepared / "examples.jsonl") if r["role"] == "training"}
    targets = {r["row_key"]: r["native_binary_value"] for r in read_jsonl(prepared / "training_targets.jsonl")}
    episodes, selected = [], set()
    for episode in read_jsonl(prepared / "evidence_plan_v1/evidence_episode_links.jsonl"):
        if len(episode["support_order_row_keys"]) < 31 or not episode["query_row_keys"]:
            raise ValueError("Every scope must support the complete protocol")
        query = episode["query_row_keys"][0]
        supports = episode["support_order_row_keys"]
        if query in supports:
            raise ValueError("Query outcome entered acquisition evidence")
        selected.update([query, *supports])
        episodes.append({"query": query, "supports": [[key, targets[key]] for key in supports]})
    rows = {key: index[key] for key in selected}
    system_keys, task_keys = {r["system_key"] for r in rows.values()}, {r["task_key"] for r in rows.values()}
    fixture = {"rows": rows, "episodes": episodes, "strength": receipt["strength"], "prior": receipt["prior"],
        "systems": {r["system_key"]: r["recorded"] for r in read_jsonl(prepared / "systems.jsonl") if r["system_key"] in system_keys},
        "tasks": {r["task_key"]: r["recorded"] for r in read_jsonl(prepared / "tasks.jsonl") if r["task_key"] in task_keys}}
    with tempfile.TemporaryDirectory(prefix="paiec-submission-qualification-") as temporary:
        folder = Path(temporary)
        write_json(folder / "fixtures.json", fixture)
        (folder / "driver.py").write_text(DRIVER)
        result = subprocess.run([sys.executable, "-I", "-S", str(folder / "driver.py"),
            str(package / "submission"), str(folder / "fixtures.json")], capture_output=True, text=True, check=True, timeout=60)
    report = {"version": "evidence_submission_qualification_v1", "runtime": json.loads(result.stdout),
        "package_sha256": receipt["archive_sha256"], "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
        "query_outcomes_supplied_to_runtime": False, "hosted_execution_verified": False,
        "uploaded": False, "paid_calls": 0, "gpu_jobs": 0}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "qualification.json", report)
    return report
