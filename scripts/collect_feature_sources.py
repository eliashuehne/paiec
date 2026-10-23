#!/usr/bin/env python3
"""Snapshot primary metadata only; never download weights or execute remote code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def retrieve(url):
    request = Request(url, headers={"User-Agent": "PAIEC feature provenance research"})
    with urlopen(request, timeout=25) as response:
        raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Metadata exceeded 2 MB limit")
        return raw, response.geturl()


def collect(entry, out):
    result = dict(entry, files=[], failures=[])
    try:
        raw, _ = retrieve("https://huggingface.co/api/models/" + entry["repository"])
        metadata = json.loads(raw)
        revision = metadata["sha"]
        result["revision"] = revision
        for filename in ["README.md", "config.json"]:
            url = f"https://huggingface.co/{entry['repository']}/resolve/{revision}/{filename}"
            try:
                content, final_url = retrieve(url)
                digest = hashlib.sha256(content).hexdigest()
                relative = entry["profile_id"] + "." + filename
                (out / relative).write_bytes(content)
                result["files"].append({"path": relative, "url": url, "sha256": digest,
                                        "bytes": len(content)})
            except (HTTPError, OSError, ValueError) as error:
                result["failures"].append({"file": filename, "error": str(error)})
    except (HTTPError, OSError, ValueError, KeyError) as error:
        result["failures"].append({"file": "repository_metadata", "error": str(error)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "research/feature_collection/source_plan_v1.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda entry: collect(entry, args.out), plan))
    manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "usage": "research_only",
                "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(), "sources": results,
                "files_collected": sum(len(r["files"]) for r in results),
                "failed_requests": sum(len(r["failures"]) for r in results)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["files_collected", "failed_requests"]}))


if __name__ == "__main__":
    main()
