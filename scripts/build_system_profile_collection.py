#!/usr/bin/env python3
"""Rebuild reviewed research profiles offline; never changes predictor inputs.

The recipe and atomic assertions are checked-in data. Downloaded source snapshots
are an ignored evidence archive. No source code from that archive is executed.
"""

import argparse
import hashlib
import html
import json
import re
import tempfile
from pathlib import Path

from paiec.system_profiles import (
    HarnessSpecifications, ModelSpecifications, Profile, RunSpecifications,
    load_catalog,
)


def normalized_text(text):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())


def set_path(record, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        record = record[part]
    record[parts[-1]] = value


def configured_facts(config):
    """Extract literal dimensions, including multimodal nested text configs.

    Position embeddings are deliberately excluded: they are not a supported
    context-window promise. Expert counts below count routed experts; a shared
    expert, where present, is not included in that configured count.
    """
    prefix = "text_config." if "text_config" in config else ""
    block = config.get("text_config", config)
    fields = {
        "model_type": "family", "num_hidden_layers": "layers",
        "hidden_size": "hidden_size", "num_attention_heads": "attention_heads",
        "num_key_value_heads": "key_value_heads", "num_experts": "experts_total",
        "n_routed_experts": "experts_total",
        "num_experts_per_tok": "experts_active_per_token",
    }
    for key, destination in fields.items():
        if key in block:
            yield destination, block[key], prefix + key, {key: block[key]}
    if "layer_types" in block:
        yield ("attention_types", list(dict.fromkeys(block["layer_types"])),
               prefix + "layer_types", {"layer_types": block["layer_types"]})


def build(catalog, cache):
    recipe = json.loads((catalog / "recipe.json").read_text())
    snapshots = json.loads((catalog / "snapshots.json").read_text())
    inventory = [json.loads(line) for line in (catalog / "subjects.jsonl").read_text().splitlines()]
    supplied = {
        ("measurement-db." + row["benchmark"] + ".subjects", row["subject_id"]): row["subject"]
        for row in inventory
    }
    texts = {}
    for entry in snapshots:
        raw = (cache / entry["cache_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed source snapshot: {entry['source_id']}")
        texts[entry["source_id"]] = raw.decode()

    profiles = {}
    for entry in recipe["profiles"]:
        kind = entry["entity_type"]
        spec = {"model": ModelSpecifications, "harness": HarnessSpecifications,
                "recorded_run": RunSpecifications}[kind]()
        profiles[entry["profile_id"]] = Profile(
            profile_id=entry["profile_id"], entity_type=kind,
            specifications=spec,
        ).model_dump()

    def apply(profile_id, field, value, source_id, locator, excerpt,
              status="documented", note=""):
        if source_id in texts:
            # JSON excerpts use normalized serialization for stable key/value evidence.
            if locator.startswith("json:"):
                obj = json.loads(texts[source_id])
                keys = locator[5:].split(".")
                for key in keys:
                    obj = obj[key]
                if obj != json.loads(excerpt)[keys[-1]]:
                    raise ValueError(f"JSON assertion disagrees: {locator}")
            elif normalized_text(excerpt) not in normalized_text(texts[source_id]):
                raise ValueError(f"Excerpt absent from {source_id}: {excerpt[:100]}")
        else:
            match = re.fullmatch(r"subject_id=([^;]+); column=(\w+)", locator)
            if not match or supplied[(source_id, match[1])][match[2]] != excerpt:
                raise ValueError(f"Recorded excerpt disagrees with subject inventory: {locator}")
        fact = {"value": value, "status": status,
                "evidence": [{"source_id": source_id, "locator": locator,
                              "excerpt": excerpt}], "note": note}
        set_path(profiles[profile_id], field, fact)

    for entry in recipe["profiles"]:
        if source := entry.get("config_source"):
            for field, value, locator, excerpt in configured_facts(json.loads(texts[source])):
                note = ("Published configuration, not measured runtime behavior. "
                        "For multimodal models these are text-component dimensions.")
                if field.startswith("experts_"):
                    note += " Routed-expert count; shared experts are not included."
                apply(entry["profile_id"], "specifications.architecture." + field,
                      value, source, "json:" + locator, json.dumps(excerpt), note=note)
    for line in (catalog / "assertions.jsonl").read_text().splitlines():
        assertion = json.loads(line)
        apply(**assertion)

    for row in inventory:
        record = profiles[row["profile_id"]]
        record["identity"]["resolution"] = "reported_identifier_only"
        record["specifications"]["raw_supplied_subject"] = row["subject"]
        record["specifications"]["model_profile_id"] = row.get("model_profile_id")
        record["specifications"]["harness_profile_id"] = row.get("harness_profile_id")
    # Validate everything before writing any output.
    validated = [Profile.model_validate(p) for p in profiles.values()]
    output = "".join(p.model_dump_json() + "\n" for p in validated)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        (temporary / "sources.jsonl").write_bytes((catalog / "sources.jsonl").read_bytes())
        (temporary / "profiles.jsonl").write_text(output)
        load_catalog(temporary)
    (catalog / "profiles.jsonl").write_text(output)
    return validated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("research/system_profiles/collection_v1"))
    parser.add_argument("--cache", type=Path, default=Path("runs/system_profile_sources_20261022"))
    args = parser.parse_args()
    profiles = build(args.catalog, args.cache)
    print(f"Rebuilt {len(profiles)} research profiles; active prediction inputs are unchanged.")
