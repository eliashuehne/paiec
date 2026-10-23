import json
from pathlib import Path
import tempfile
import unittest
from paiec.system_profiles import (
    Architecture,
    Evidence,
    Fact,
    Profile,
    ModelSpecifications,
    load_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


class SystemProfileTests(unittest.TestCase):
    def evidence(self):
        return Evidence(
            source_id="source", locator="config.json: num_hidden_layers", excerpt="32"
        )

    def test_known_values_need_evidence_and_unknowns_cannot_hide_values(self):
        with self.assertRaises(ValueError):
            Fact[int](value=32)
        with self.assertRaises(ValueError):
            Fact[int](value=32, status="documented")
        value = Fact[int](value=32, status="documented", evidence=[self.evidence()])
        self.assertEqual(value.value, 32)
        with self.assertRaises(ValueError):
            Fact[int](value=True, status="documented", evidence=[self.evidence()])

    def test_architecture_numeric_consistency(self):
        def number(value):
            return Fact[int](
                value=value, status="documented", evidence=[self.evidence()]
            )

        for kwargs in [
            dict(total_parameters=number(0)),
            dict(total_parameters=number(7), active_parameters_per_token=number(8)),
            dict(experts_total=number(8), experts_active_per_token=number(9)),
        ]:
            with self.assertRaises(ValueError):
                Architecture(**kwargs)

    def test_real_catalog_is_roundtrippable_without_invented_architecture(self):
        profiles, sources = load_catalog(ROOT / "research/system_profiles/v1")
        self.assertEqual(len(profiles), 3)
        self.assertEqual(len(sources), 1)
        model = next(p for p in profiles if p.entity_type == "model")
        self.assertEqual(model.identity.name.value, "MetaMath-Mistral")
        self.assertEqual(model.identity.resolution, "reported_identifier_only")
        self.assertIsNone(model.specifications.architecture.total_parameters.value)
        for profile in profiles:
            self.assertEqual(
                Profile.model_validate_json(profile.model_dump_json()), profile
            )
            self.assertEqual(profile.usage, "research_only")

    def test_wrong_entity_specification_and_extra_fields_rejected(self):
        with self.assertRaises(ValueError):
            Profile(
                profile_id="model.test",
                entity_type="harness",
                specifications=ModelSpecifications(),
            )
        with self.assertRaises(ValueError):
            Architecture(benchmark_score=0.9)

    def test_catalog_checks_sources_links_and_duplicate_ids(self):
        source = ROOT / "research/system_profiles/v1"
        for mutation in ["missing_source", "duplicate", "broken_link"]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = [
                    json.loads(line)
                    for line in (source / "profiles.jsonl").read_text().splitlines()
                ]
                (root / "sources.jsonl").write_text(
                    ""
                    if mutation == "missing_source"
                    else (source / "sources.jsonl").read_text()
                )
                if mutation == "duplicate":
                    rows.append(rows[0])
                elif mutation == "broken_link":
                    rows[-1]["specifications"]["model_profile_id"] = (
                        "model.does-not-exist"
                    )
                (root / "profiles.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                with self.assertRaises(ValueError):
                    load_catalog(root)
