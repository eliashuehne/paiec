import json
from pathlib import Path
import tempfile
import unittest

from paiec.feature_preparation import read_jsonl, write_json, write_jsonl
from paiec.feature_review import audit_source_reviews, audit_recorded_setting_reviews, audit_identity_reviews, fact_value_hash
from paiec.feature_preparation import stable_hash
from paiec.head_training import file_hash


class SourceReviewTests(unittest.TestCase):
    def test_record_review_abstains_for_new_settings_and_never_makes_unknown_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = {"subject_features_extra": "reported_task_generation_size=512"}
            fact = {"value": None, "status": "unknown", "evidence": []}
            rows = [{"system_key": stable_hash(subject), "recorded": subject,
                     "features": {"max_output_tokens": fact, "enabled_tools": fact}}]
            audit = audit_recorded_setting_reviews(rows, temporary)
            self.assertEqual((audit["reviewed"], audit["known_values"]), (2, 0))
            self.assertIsNone(rows[0]["features"]["max_output_tokens"]["value"])
            subject["subject_features_extra"] = 'configuration={"tools":["python"]}'
            rows[0]["system_key"] = stable_hash(subject)
            audit = audit_recorded_setting_reviews(rows, temporary)
            self.assertEqual(audit["pending"], 2)
            self.assertFalse(audit["complete"])

    def fixture(self, root):
        catalog, cache = root / "research/review", root / "runs/archive"
        catalog.mkdir(parents=True)
        cache.mkdir(parents=True)
        (cache / "card.md").write_text("A fixture launch announcement.")
        sha = file_hash(cache / "card.md")
        write_jsonl(catalog / "sources.jsonl", [{"source_id": "card", "sha256": sha}])
        write_json(catalog / "snapshots.json", [{"source_id": "card", "sha256": sha, "cache_file": "card.md"}])
        fact = {"value": None, "status": "unknown", "evidence": [], "note": "Not disclosed here."}
        queue = [{"entity_type": "model", "profile_id": "model.test", "field": "max_output_tokens", "fact": fact}]
        review = {"entity_type": "model", "profile_id": "model.test", "field": "max_output_tokens",
                  "fact_value_sha256": fact_value_hash(fact), "disposition": "not_established_in_reviewed_sources",
                  "rationale": "The fixture announcement establishes no output-token limit.",
                  "checked_documents": [{"source_id": "card", "sha256": sha, "reviewed_scope": "Announcement"}],
                  "reviewer": "fixture"}
        write_jsonl(catalog / "reviews.jsonl", [review])
        registry = "configs/pipelines/sources.json"
        (root / registry).parent.mkdir(parents=True)
        write_json(root / registry, {"review_ledgers": [{"catalog": "research/review", "cache": "runs/archive", "reviews": "reviews.jsonl"}]})
        return queue, registry, catalog, cache

    def test_reviewed_unknown_stays_unknown_and_unreviewed_fields_stay_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, registry, _, _ = self.fixture(root)
            queue.append(dict(queue[0], field="max_input_tokens"))
            before = json.dumps(queue, sort_keys=True)
            report = audit_source_reviews(root, queue, root, source_registry=registry)
            self.assertEqual((report["reviewed"], report["pending"], report["known_values"]), (1, 1, 0))
            self.assertFalse(report["complete"])
            self.assertEqual(before, json.dumps(queue, sort_keys=True))
            self.assertIsNone(next(read_jsonl(root / "source_review_status.jsonl"))["review"]["checked_documents"][0].get("value"))

    def test_changed_values_invalidate_a_review_and_changed_source_bytes_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, registry, _, cache = self.fixture(root)
            queue[0]["fact"] = {"value": 4096, "status": "documented"}
            report = audit_source_reviews(root, queue, root, source_registry=registry)
            self.assertEqual(report["stale"], 1)
            self.assertFalse(report["complete"])
            (cache / "card.md").write_text("Revised announcement")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                audit_source_reviews(root, queue, root, source_registry=registry)

    def test_a_review_cannot_confirm_an_unknown_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, registry, catalog, _ = self.fixture(root)
            review = next(read_jsonl(catalog / "reviews.jsonl"))
            review["disposition"] = "value_confirmed"
            write_jsonl(catalog / "reviews.jsonl", [review])
            with self.assertRaisesRegex(ValueError, "contradicts"):
                audit_source_reviews(root, queue, root, source_registry=registry)

    def test_reviewing_a_known_value_conflict_does_not_confirm_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, registry, catalog, _ = self.fixture(root)
            queue[0]["fact"] = {"value": 4096, "status": "documented"}
            review = next(read_jsonl(catalog / "reviews.jsonl"))
            review.update(fact_value_sha256=fact_value_hash(queue[0]["fact"]),
                          disposition="conflict_unresolved", rationale="Sources disagree on this exact limit.")
            write_jsonl(catalog / "reviews.jsonl", [review])
            report = audit_source_reviews(root, queue, root, source_registry=registry)
            self.assertTrue(report["complete"])
            self.assertEqual(report["reviewed_conflicts"], 1)
            self.assertEqual(report["confirmed_known_values"], 0)
            feature = next(read_jsonl(root / "reviewed_static_features.jsonl"))
            self.assertIsNone(feature["fact"]["value"])
            self.assertEqual(feature["fact"]["status"], "unknown")

    def test_identity_review_retains_an_unbound_private_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "research").mkdir()
            subject = {"normalized_name": "private-finetune", "harness": "fixture"}
            key = stable_hash(subject)
            queue = [{"system_key": key, "entity_type": "model", "field": "identity"}]
            review = {"system_key": key, "entity_type": "model", "checked_subject_sha256": key,
                      "disposition": "reported_private_checkpoint_unverified",
                      "rationale": "A public base checkpoint does not verify this private fine-tune.", "reviewer": "fixture"}
            write_jsonl(root / "research/identity.jsonl", [review])
            report = audit_identity_reviews(root, queue, {key: {"recorded": subject}}, root, ledger="research/identity.jsonl")
            self.assertTrue(report["complete"])
            self.assertEqual(report["unresolved"], 1)
            self.assertFalse(next(read_jsonl(root / "identity_review_status.jsonl"))["identity_resolved"])
            review["disposition"] = "not_supplied"
            write_jsonl(root / "research/identity.jsonl", [review])
            with self.assertRaisesRegex(ValueError, "Identity is reported"):
                audit_identity_reviews(root, queue, {key: {"recorded": subject}}, root, ledger="research/identity.jsonl")

    def test_identity_review_checks_additional_visible_identifiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "research").mkdir()
            subject = {"normalized_name": "", "subject_features_extra": "source_model=visible-model"}
            key = stable_hash(subject)
            review = {"system_key": key, "entity_type": "model", "checked_subject_sha256": key,
                      "disposition": "not_supplied", "rationale": "Fixture", "reviewer": "fixture"}
            write_jsonl(root / "research/identity.jsonl", [review])
            with self.assertRaisesRegex(ValueError, "Extra settings report an identity"):
                audit_identity_reviews(root, [{"system_key": key, "entity_type": "model", "field": "identity"}],
                    {key: {"recorded": subject}}, root, ledger="research/identity.jsonl")
