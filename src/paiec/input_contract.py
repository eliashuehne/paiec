"""Keep prepared/trained input formats identical at later prediction time."""
import hashlib
import json
from pathlib import Path


def resolve_input_contract(checkpoint_path=None, *, state_format=None, max_len=None,
                           enrichment_policy=None):
    result = {"state_format": state_format or "json", "max_len": max_len,
              "enrichment_policy": enrichment_policy, "profile_root": None}
    if not checkpoint_path:
        return result
    root = Path(checkpoint_path).expanduser().resolve()
    manifest = json.loads((root / "paiec_training.json").read_text())
    contract = manifest.get("input_contract")
    if not contract:
        return result  # Historical checkpoints retain their historical defaults.
    question_path = Path(__file__).resolve().parents[2] / "configs/laya_question.json"
    if hashlib.sha256(question_path.read_bytes()).hexdigest() != contract["question_sha256"]:
        raise ValueError("Active question differs from the checkpoint's training question")
    project = Path(__file__).resolve().parents[2]
    for name, expected in contract.get("renderer_sources", {}).items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint input renderer changed: {name}")
    for field, supplied in [("state_format", state_format), ("max_len", max_len)]:
        if supplied is not None and supplied != contract[field]:
            raise ValueError(f"Checkpoint requires {field}={contract[field]}")
        result[field] = contract[field]
    if result["state_format"] == "enriched_v1":
        policy = root / contract["policy_file"]
        if enrichment_policy is not None and Path(enrichment_policy).resolve() != policy:
            raise ValueError("Use the checkpoint's bundled enrichment policy")
        if hashlib.sha256(policy.read_bytes()).hexdigest() != contract["policy_sha256"]:
            raise ValueError("Bundled enrichment policy changed")
        from .profile_enrichment import EnrichmentPolicy, catalog_fingerprint
        config = EnrichmentPolicy.model_validate_json(policy.read_text())
        if catalog_fingerprint(root / config.catalog) != contract["catalog_sha256"]:
            raise ValueError("Bundled profile catalog changed")
        result.update(enrichment_policy=str(policy), profile_root=str(root))
    return result
