"""Download competition-eligible Measurement DB tables into ``payload/data``.

Eligible benchmarks contain binary responses at item granularity. The script
downloads the benchmark metadata first, uses it to find eligible benchmarks,
and then downloads only their subjects, items, and response tables. It skips
traces, assets, and raw files.

The Measurement DB dataset is gated, so authenticate with Hugging Face first:

    hf auth login
    python scripts/download_data.py
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


DATASET_REPOSITORY = "aims-foundations/measurement-db"
DATASET_REVISION = "54074453da7b9911be0f48e4c560be547bc84d21"
TABLE_NAMES = ("subjects", "items", "responses", "response")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DOWNLOAD_WORKERS = 16
DATA_DOWNLOAD_WORKERS = 8
PATH_LOOKUP_BATCH_SIZE = 200


def find_existing_files(api, paths):
    """Return repository files that exist and have a non-zero size."""
    existing_files = []
    for start in range(0, len(paths), PATH_LOOKUP_BATCH_SIZE):
        path_batch = paths[start : start + PATH_LOOKUP_BATCH_SIZE]
        file_info = api.get_paths_info(
            DATASET_REPOSITORY,
            path_batch,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
        existing_files.extend(file for file in file_info if file.size)
    return existing_files


def download_file(path, output_directory):
    """Download one dataset file and preserve its repository-relative path."""
    return hf_hub_download(
        DATASET_REPOSITORY,
        path,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=output_directory,
    )


def benchmark_directory(benchmark_file):
    """Get the benchmark directory that contains a metadata parquet file."""
    return benchmark_file.path.rsplit("/", 1)[0]


def select_eligible_benchmarks(benchmark_files, output_directory):
    """Read metadata and return eligible directories plus manifest details."""
    eligible_directories = []
    benchmark_details = []

    for benchmark_file in benchmark_files:
        directory = benchmark_directory(benchmark_file)
        benchmark = pd.read_parquet(output_directory / benchmark_file.path).iloc[0]
        response_type = benchmark.get("response_type")
        granularity = benchmark.get("granularity")
        is_eligible = response_type == "binary" and granularity == "item"

        benchmark_details.append(
            {
                "dir": directory,
                "response_type": response_type,
                "granularity": granularity,
                "eligible": bool(is_eligible),
            }
        )
        if is_eligible:
            eligible_directories.append(directory)

    return eligible_directories, benchmark_details


def write_manifest(output_directory, eligible_directories, benchmark_details):
    """Record selected benchmarks and the source revision beside the data."""
    manifest = {
        "repo": DATASET_REPOSITORY,
        "revision": DATASET_REVISION,
        "eligible": eligible_directories,
        "all_dirs": benchmark_details,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "payload" / "data",
        help="directory where the downloaded data will be written",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    output_directory = arguments.out.resolve()
    api = HfApi()
    started_at = time.time()

    print("Finding benchmark metadata files...", flush=True)
    repository_entries = api.list_repo_tree(
        DATASET_REPOSITORY,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    benchmark_metadata_paths = [
        f"{entry.path}/formatted_tables/benchmarks.parquet"
        for entry in repository_entries
    ]
    benchmark_files = find_existing_files(api, benchmark_metadata_paths)

    with ThreadPoolExecutor(max_workers=METADATA_DOWNLOAD_WORKERS) as executor:
        list(
            executor.map(
                download_file,
                (file.path for file in benchmark_files),
                (output_directory for _ in benchmark_files),
            )
        )
    print(
        f"Downloaded metadata for {len(benchmark_files)} benchmark directories.",
        flush=True,
    )

    eligible_directories, benchmark_details = select_eligible_benchmarks(
        benchmark_files, output_directory
    )
    print(
        f"Found {len(eligible_directories)} eligible benchmarks "
        "(binary responses at item granularity).",
        flush=True,
    )

    data_paths = [
        f"{directory}/{table_name}.parquet"
        for directory in eligible_directories
        for table_name in TABLE_NAMES
    ]
    data_files = find_existing_files(api, data_paths)
    total_bytes = sum(file.size for file in data_files)
    print(
        f"Downloading {len(data_files)} data files ({total_bytes / 1e9:.2f} GB)...",
        flush=True,
    )

    downloaded_bytes = 0
    with ThreadPoolExecutor(max_workers=DATA_DOWNLOAD_WORKERS) as executor:
        downloads = {
            executor.submit(download_file, file.path, output_directory): file
            for file in data_files
        }
        for completed_count, future in enumerate(as_completed(downloads), start=1):
            future.result()
            downloaded_bytes += downloads[future].size
            if completed_count % 20 == 0 or completed_count == len(data_files):
                elapsed_seconds = time.time() - started_at
                print(
                    f"  {completed_count}/{len(data_files)} files, "
                    f"{downloaded_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB "
                    f"({elapsed_seconds:.0f}s)",
                    flush=True,
                )

    write_manifest(output_directory, eligible_directories, benchmark_details)
    elapsed_seconds = time.time() - started_at
    print(f"Finished in {elapsed_seconds:.0f}s. Data is in {output_directory}", flush=True)


if __name__ == "__main__":
    main()
