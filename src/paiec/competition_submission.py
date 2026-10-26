"""Build a starting-kit submission from a fitted rate model, or refuse.

The package is only produced for a state whose gate cleared: shipping a model
that loses to a fixed control would contradict the assessment that produced it.
No upload, no Codabench call and no hosted execution happens here.
"""
import hashlib
import json
import math
from pathlib import Path

from .feature_preparation import write_json
from .rate_model import ITEM_FEATURES

MODEL_SOURCE = r'''"""Predictive evaluator: ability + difficulty prior with budgeted evidence.

Self-contained: standard library plus ``state.json`` beside this file.
"""
import json
import math
import os
import re

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json"),
          encoding="utf-8") as handle:
    STATE = json.load(handle)

_PREFIX = re.compile(r"^[^/]*\s+/\s+")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_NON_KEY = re.compile(r"[^a-z0-9.+_-]+")
_CJK = re.compile("[" + chr(0x3400) + "-" + chr(0x9FFF) + "]")
_CYRILLIC = re.compile("[" + chr(0x0400) + "-" + chr(0x04FF) + "]")
_ARABIC = re.compile("[" + chr(0x0600) + "-" + chr(0x06FF) + "]")
_OPTION = re.compile(r"(?:^|\n)\s*[A-D][\)\.:]\s")


def canonical_name(name):
    text = str(name or "").strip().lower()
    text = _PREFIX.sub("", text)
    text = _PARENTHETICAL.sub(" ", text)
    text = _NON_KEY.sub(" ", text)
    return " ".join(text.split())


def logit(probability):
    p = min(max(float(probability), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-min(max(float(value), -40.0), 40.0)))


def task_features(text):
    text = str(text or "")
    characters = max(len(text), 1)
    words = max(len(re.findall(r"\s+", text)) + 1, 1)
    return [
        math.log1p(characters),
        math.log1p(words),
        characters / words if words else 0.0,
        len(re.findall(r"\d", text)) / characters,
        len(re.findall(r"[A-Z]", text)) / characters,
        math.log1p(text.count("\n")),
        1.0 if "```" in text else 0.0,
        1.0 if "|" in text else 0.0,
        math.log1p(len(_OPTION.findall(text))),
        len(_CJK.findall(text)) / characters,
        len(_CYRILLIC.findall(text)) / characters,
        len(_ARABIC.findall(text)) / characters,
        1.0 if text.startswith("{") else 0.0,
        math.log1p(len(re.findall(r"\d+(?:\.\d+)?", text))),
        math.log1p(text.count("?")),
    ]


def benchmark_difficulty(text):
    features = task_features(text)
    total = STATE["intercept"]
    for value, mean, scale, coefficient in zip(features, STATE["feature_mean"],
                                               STATE["feature_scale"], STATE["coefficients"]):
        total += coefficient * (value - mean) / scale
    return total


def predict(input, labeled=None):
    """Return P(correct) for the supplied subject and item."""
    subject, item = input[0], input[1]
    ability = STATE["ability"].get(canonical_name(subject.get("normalized_name", "")), 0.0)
    prior = sigmoid(STATE["mu"] + ability + benchmark_difficulty(item.get("item_content", "")))
    benchmark = item.get("benchmark_id", "")
    own_successes = own_count = other_successes = other_count = 0
    for records, label in labeled or []:
        observed_subject, observed_item = records[0], records[1]
        if observed_item.get("benchmark_id", "") != benchmark:
            continue
        if observed_subject == subject:
            if observed_item == item:
                continue
            own_successes += int(label)
            own_count += 1
        else:
            other_successes += int(label)
            other_count += 1
    weight = STATE["benchmark_weight"]
    count = own_count + weight * other_count
    if count <= 0:
        return float(prior)
    posterior = (STATE["prior_strength"] * prior + own_successes
                 + weight * other_successes) / (STATE["prior_strength"] + count)
    return float(min(max(posterior, 0.0), 1.0))
'''

LABELING_SOURCE = '''"""Neutral acquisition: keep the platform's own candidate order.

A targeted rule was not measured to beat random acquisition on matched
episodes, so the shipped function reveals every candidate while its budget
lasts rather than importing an unmeasured bias.
"""


def acquisition_function(input, prediction=None, labeled=None, context=None):
    """Return True while the pair still has budget left."""
    if not isinstance(context, dict):
        return True
    remaining = context.get("labels_remaining")
    items = context.get("items_remaining")
    if isinstance(items, int) and items <= 0:
        return False
    return True if not isinstance(remaining, int) else remaining > 0
'''

STATE_KEYS = {"mu", "ability", "intercept", "feature_mean", "feature_scale", "coefficients",
              "prior_strength", "benchmark_weight"}
REQUIRED_GATE = "clears_matched_adaptive_control"


def checked_state(state):
    if set(state) != STATE_KEYS:
        raise ValueError("Fitted state does not match the submission contract")
    for key in ("mu", "intercept", "prior_strength", "benchmark_weight"):
        if not isinstance(state[key], float) or not math.isfinite(state[key]):
            raise ValueError("Scalar state values must be finite floats: " + key)
    if state["prior_strength"] < 0 or state["benchmark_weight"] < 0:
        raise ValueError("Evidence weights cannot be negative")
    for key in ("feature_mean", "feature_scale", "coefficients"):
        if len(state[key]) != len(ITEM_FEATURES) or any(not isinstance(value, float)
                                                       for value in state[key]):
            raise ValueError("Task-feature vectors must match the declared features: " + key)
    if any(value <= 0 for value in state["feature_scale"]):
        raise ValueError("Feature scales must be positive")
    if not isinstance(state["ability"], dict) or any(
            not isinstance(name, str) or not isinstance(value, float)
            for name, value in state["ability"].items()):
        raise ValueError("Ability must be a name-to-offset mapping of floats")
    return state


def build(state, out, *, gate):
    """Write the package; refuse anything the gate did not clear."""
    if gate.get("verdict") != REQUIRED_GATE:
        raise ValueError("Refusing to package a model the gate rejects: "
                         + str(gate.get("verdict")))
    out = Path(out)
    if out.exists():
        raise ValueError("Use a fresh package directory")
    out.mkdir(parents=True)
    checked_state(state)
    write_json(out / "state.json", state)
    (out / "model.py").write_text(MODEL_SOURCE)
    (out / "labeling.py").write_text(LABELING_SOURCE)
    digest = hashlib.sha256((out / "state.json").read_bytes()).hexdigest()
    write_json(out / "package.json", {
        "version": "competition_submission_v1",
        "state_sha256": digest,
        "gate_verdict": gate["verdict"],
        "files": ["model.py", "labeling.py", "state.json"],
        "network_access_needed": False,
        "models_declared": [],
        "hosted_execution_verified": False,
        "uploaded": False,
    })
    return {"package": str(out), "state_sha256": digest, "files": ["model.py", "labeling.py",
                                                                  "state.json"]}


__all__ = ["LABELING_SOURCE", "MODEL_SOURCE", "REQUIRED_GATE", "STATE_KEYS", "build", "checked_state"]
