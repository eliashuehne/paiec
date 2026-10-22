"""Reproducible benchmark-family roles and outcome-blind pilot sampling."""

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .split import stable_seed

ROLES = ("training", "development", "calibration", "confirmation")


class BenchmarkReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    family: str
    status: Literal["eligible", "monitoring", "monitoring_related", "excluded", "deferred"]
    reason: str
    domain: str = "other"
    has_english: bool = True
    required_feature: str | None = None


class HeadDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_revision: str
    seed: int = 20261022
    training_rows: int = Field(default=5000, ge=1)
    calibration_rows: int = Field(default=1000, ge=1)
    subjects_per_benchmark: int = Field(default=4, ge=1)
    min_items: int = Field(default=80, ge=80)
    acquisition_items: int = Field(default=40, ge=31)
    evaluation_items: int = Field(default=120, ge=1)
    reviews: dict[str, BenchmarkReview]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def task_key(content):
    """Conservative task identity across JSON/chat wrappers; never read graders.

    Preserve context/options in dictionaries. For chronological chat lists use
    the last user task. This is an exact normalized-content screen, not a claim
    to recognize translations or every paraphrase.
    """
    text = str(content)
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        value = value["messages"]
    if isinstance(value, list) and all(isinstance(v, dict) for v in value):
        users = [v.get("content", "") for v in value if v.get("role") == "user"]
        if users:
            text = str(users[-1])
    elif isinstance(value, dict):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def role_counts(n):
    """Closest feasible 60/20/10/10 allocation with >=2 held-out families each."""
    if n < 7:
        raise ValueError("Need at least seven independent families for four roles; use a revised grouped-fold design")
    target = np.array([.6, .2, .1, .1])
    choices = []
    for dev in range(2, n):
        for calib in range(2, n - dev):
            for test in range(2, n - dev - calib):
                train = n - dev - calib - test
                if train >= 1:
                    counts = (train, dev, calib, test)
                    choices.append((float(((np.array(counts) / n - target) ** 2).sum()), counts))
    return dict(zip(ROLES, min(choices)[1]))


def assign_roles(families, seed):
    """Metadata-stratified randomization; no outcomes or model scores accepted.

    Each held-out role gets English-containing and short-task families. Training covers every
    broad domain present in the reviewed pool. Within those constraints take the
    first seeded permutation, not the one with the most favorable scores.
    """
    names = sorted(families)
    counts = role_counts(len(names))
    rng = np.random.default_rng(stable_seed(seed, "head-family-roles"))
    domains = {v["domain"] for v in families.values()}
    for attempt in range(10000):
        order = list(rng.permutation(names))
        assigned, start = {}, 0
        for role, count in counts.items():
            assigned.update({name: role for name in order[start:start + count]})
            start += count
        if {families[n]["domain"] for n in names if assigned[n] == "training"} != domains:
            continue
        if any(not any(families[n]["has_english"] for n in names if assigned[n] == role)
               for role in ROLES[1:]):
            continue
        if any(not any(families[n].get("has_short", True) for n in names if assigned[n] == role)
               for role in ROLES[1:]):
            continue
        return assigned, {"counts": counts, "permutation_attempt": attempt + 1}
    raise ValueError("Metadata constraints cannot be met with these families")


def pair_balanced_sample(frame, total, seed):
    """Sample rows without replacement, independently of label values."""
    if frame.empty:
        raise ValueError("No usable response records")
    frame = frame.sort_values(["benchmark", "subject_id", "item_id", "response_id"]).reset_index(drop=True)
    groups = {key: group.index.to_numpy().copy()
              for key, group in frame.groupby(["benchmark", "subject_id"], sort=True)}
    rng = np.random.default_rng(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    chosen, positions = [], defaultdict(int)
    target = min(total, len(frame))
    while len(chosen) < target:
        active = [key for key, indices in groups.items() if positions[key] < len(indices)]
        rng.shuffle(active)
        for key in active:
            chosen.append(groups[key][positions[key]])
            positions[key] += 1
            if len(chosen) == target:
                break
    result = frame.loc[chosen].copy().reset_index(drop=True)
    counts = result.groupby(["benchmark", "subject_id"])["response_id"].transform("size")
    result["sample_weight"] = len(result) / (len(result.groupby(["benchmark", "subject_id"])) * counts)
    return result


def near_duplicates(texts, roles, *, threshold=.9):
    """Approximate cross-role candidate screen: 16-minhash word-3-gram LSH.

    Candidates are verified by exact shingle Jaccard. It is not an exhaustive
    paraphrase detector. Very short tasks use exact task keys elsewhere.
    """
    from zlib import crc32

    rng = np.random.default_rng(219)
    a = rng.integers(1, 2**31, size=16, dtype=np.uint64)
    b = rng.integers(0, 2**32, size=16, dtype=np.uint64)
    buckets, shingles, matches = defaultdict(list), [], []
    for index, text in enumerate(texts):
        words = text.split()
        grams = {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}
        shingles.append(grams)
        if len(grams) < 8:
            continue
        hashes = np.array([crc32(s.encode()) for s in grams], dtype=np.uint64)
        signature = ((hashes[:, None] * a + b) % np.uint64(4294967291)).min(axis=0)
        candidates = set()
        for band in range(4):
            key = (band, tuple(int(v) for v in signature[band * 4:band * 4 + 4]))
            candidates.update(buckets[key])
            buckets[key].append(index)
        for other in sorted(candidates):
            if roles[other] == roles[index]:
                continue
            previous = shingles[other]
            if min(len(grams), len(previous)) / max(len(grams), len(previous)) < threshold:
                continue
            score = len(grams & previous) / len(grams | previous)
            if score >= threshold:
                matches.append((other, index, score))
    return matches
