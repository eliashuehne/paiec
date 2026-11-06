"""Rate the pilot questions with a hosted model: the Appendix C prompt, through an OpenAI-compatible API.

    python difficulty_api.py --base-url URL --list
    python difficulty_api.py --base-url URL --model MODEL SAMPLE.parquet OUT.parquet [--limit N]

The key comes from the environment variable PILOT_API_KEY. Rating = expected digit
under the next-token probabilities when the API returns them, else the digit it
answers. Finished questions are appended to OUT.parquet.jsonl, so a rerun resumes.
Then: python difficulty_eval.py OUT.parquet
"""
import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROMPT = ("How difficult is the following task for a capable AI system? Reply with a single digit "
          "from 1 (trivial) to 9 (almost impossible).\n\nTask:\n")
MAX_CHARS = 3000  # about the 768 tokens of the pilot
ATTEMPTS = 6


def expected_digit(choice):
    """(rating, source) from one API choice; rating is None when the reply has no digit."""
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    tops = getattr(content[0], "top_logprobs", None) if content else None
    if tops:
        mass = {}
        for top in tops:
            token = top.token.strip()
            if token in tuple("123456789"):
                mass[int(token)] = mass.get(int(token), 0.0) + math.exp(top.logprob)
        if mass:
            return sum(d * p for d, p in mass.items()) / sum(mass.values()), "logprobs"
    found = re.search(r"[1-9]", choice.message.content or "")
    return (float(found.group()), "digit") if found else (None, "none")


def rate(client, model, text, use_logprobs=True):
    messages = [{"role": "user", "content": PROMPT + text[:MAX_CHARS]}]
    extra = {"logprobs": True, "top_logprobs": 20} if use_logprobs else {}
    for attempt in range(ATTEMPTS):
        try:
            reply = client.chat.completions.create(model=model, messages=messages, max_tokens=2,
                                                   temperature=0, **extra)
            return expected_digit(reply.choices[0])
        except Exception as error:
            if extra and "logprob" in str(error).lower():
                extra = {}  # this provider does not offer them
                continue
            if attempt == ATTEMPTS - 1:
                return None, f"error: {' '.join(str(error).split())[:200]}"
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", nargs="?")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=os.environ["PILOT_API_KEY"], timeout=60)
    if args.list:
        print("\n".join(sorted(m.id for m in client.models.list())))
        return

    sample = pd.read_parquet(args.sample)
    if args.limit:  # spread over the benchmarks, for a first look
        sample = sample.sample(args.limit, random_state=0)
    log = Path(args.out + ".jsonl")
    done = {}
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        done = {(r["benchmark"], r["item_id"]): r for r in rows if r["rating"] is not None}
    todo = [row for row in sample.itertuples() if (row.benchmark, row.item_id) not in done]
    print(f"{len(sample)} questions, {len(todo)} to rate with {args.model}", flush=True)
    started, failures = time.time(), 0
    with log.open("a") as handle, ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(rate, client, args.model, row.content): row for row in todo}
        for count, job in enumerate(as_completed(jobs), 1):
            row, (rating, source) = jobs[job], job.result()
            failures += rating is None
            record = {"benchmark": row.benchmark, "item_id": row.item_id, "rating": rating, "source": source}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            done[(row.benchmark, row.item_id)] = record
            if count == 1 or count % 200 == 0:
                print(f"{count}/{len(todo)}, {time.time() - started:.0f}s, {failures} failed, last: {source}", flush=True)

    rated = pd.DataFrame(r for r in done.values() if r["rating"] is not None)
    frame = sample.merge(rated, on=["benchmark", "item_id"])
    frame.drop(columns=["content", "item_features"]).to_parquet(args.out, index=False)
    print(f"{len(frame)} rated -> {args.out}; sources: {frame.source.value_counts().to_dict()}")


if __name__ == "__main__":
    sys.exit(main())
