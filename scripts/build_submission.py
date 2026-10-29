"""Assemble and zip the Codabench submission, then run the official checker.

    python scripts/train_submission.py      # once: fits and saves artifacts/prior_offset.npz
    python scripts/build_submission.py      # writes dist/submission.zip and checks it
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paiec.submission import assemble, build_zip  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "prior_offset.npz")
    parser.add_argument("--zip", type=Path, default=ROOT / "dist" / "submission.zip")
    args = parser.parse_args()
    if not args.artifacts.is_file():
        raise SystemExit(f"{args.artifacts} missing: run scripts/train_submission.py first")

    directory = assemble(ROOT / "dist" / "submission", args.artifacts)
    zip_path = build_zip(directory, args.zip)
    print(f"wrote {zip_path} ({zip_path.stat().st_size:,} bytes)")
    checker = ROOT / "baseline" / "check_submission_zip.py"
    result = subprocess.run([sys.executable, str(checker), str(zip_path)], capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
