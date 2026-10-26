"""Stream the full public population without loading response outcomes/traces."""
from functools import lru_cache
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from .contracts import SUBJECT_FIELDS, source_string, stable_key
from .sources import safe_source
from .tables import TableWriter
from .tasks import content_group, effective_split, task_observables

SPLIT_BITS = {"training": 1, "validation": 2, "test": 4}


def stream_population(config, source_root, out, population, progress=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ("systems", "subject_links", "items", "tasks", "task_features", "item_input_links", "response_input_links", "training_eligibility", "content_groups")
    writers = {name: TableWriter(out, name, batch_size=config.batch_size) for name in names}
    db_path = out / "_feature_index.sqlite"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("CREATE TABLE variants (item_key TEXT, interactor_key TEXT, value TEXT, recorded INTEGER, PRIMARY KEY(item_key, interactor_key)) WITHOUT ROWID")
    db.execute("CREATE TABLE tasks (task_key TEXT PRIMARY KEY) WITHOUT ROWID")
    db.execute("CREATE TABLE groups (group_key TEXT PRIMARY KEY, mask INTEGER, n INTEGER) WITHOUT ROWID")
    seen_systems = set()
    accesses = []

    def batches(path, columns, protected=False):
        if set(columns) & {"response", "grading_criterion", "verifier", "display_name", "test_condition"}:
            raise ValueError("Forbidden feature source column")
        if protected and set(columns) - {"subject_id", "item_id", "response_id", "trial"}:
            raise ValueError("Protected input content cannot be opened")
        accesses.append({"path": str(path.relative_to(Path(source_root).resolve())), "columns": columns, "protected": protected})
        reader = pq.ParquetFile(path)
        missing = set(columns) - set(reader.schema_arrow.names)
        if missing:
            raise ValueError("Missing required source columns: " + str(missing))
        yield from reader.iter_batches(batch_size=config.batch_size, columns=columns)

    try:
        for pop in population:
            benchmark, protected = pop["benchmark"], pop["protected"]
            directory = safe_source(source_root, config.data_root + "/" + pop["directory"])
            subjects = {}
            columns = ["subject_id"] if protected else ["subject_id", *SUBJECT_FIELDS]
            for batch in batches(directory / "subjects.parquet", columns, protected):
                for row in batch.to_pylist():
                    sid = source_string(row["subject_id"])
                    if not sid or sid in subjects:
                        raise ValueError("Duplicate or empty source subject ID")
                    subject = {key: source_string(row[key]) for key in SUBJECT_FIELDS} if not protected else None
                    key = stable_key(subject) if subject is not None else None
                    subjects[sid] = key
                    if key and key not in seen_systems:
                        writers["systems"].add({"system_key": key, **subject})
                        seen_systems.add(key)
                    writers["subject_links"].add({"benchmark": benchmark, "subject_id": sid, "system_key": key,
                                                   "status": "withheld_protected" if protected else "recorded"})
            item_keys = {}
            for batch in batches(directory / "items.parquet", ["item_id"], protected):
                for iid in batch.column(0).to_pylist():
                    iid = source_string(iid)
                    if not iid or iid in item_keys:
                        raise ValueError("Duplicate or empty source item ID")
                    item_keys[iid] = stable_key({"benchmark": benchmark, "item_id": iid})
            @lru_cache(maxsize=4096)
            def interactor_key(value):
                return stable_key({"interactors": value})
            columns = ["subject_id", "item_id", "response_id", "trial"] + ([] if protected else ["interactors"])
            position = 0
            for batch in batches(directory / "response.parquet", columns, protected):
                variants = set()
                for row in batch.to_pylist():
                    sid, iid = source_string(row["subject_id"]), source_string(row["item_id"])
                    item_key = item_keys.get(iid)
                    value = source_string(row.get("interactors"))
                    ikey = None if protected else interactor_key(value)
                    status = "withheld_protected" if protected else "orphan_subject_and_item" if sid not in subjects and item_key is None else "orphan_subject" if sid not in subjects else "orphan_item" if item_key is None else "recorded"
                    writers["response_input_links"].add({"benchmark": benchmark, "source_row": position,
                        "response_id": source_string(row["response_id"]), "subject_id": sid, "item_id": iid,
                        "trial": row["trial"], "system_key": subjects.get(sid), "item_key": item_key,
                        "interactor_key": ikey, "status": status})
                    position += 1
                    if item_key is not None and not protected:
                        variants.add((item_key, ikey, value, int(row.get("interactors") is not None)))
                db.executemany("INSERT INTO variants VALUES (?,?,?,?) ON CONFLICT(item_key,interactor_key) DO UPDATE SET recorded=MAX(recorded,excluded.recorded)", variants)
            if position != pop["responses"]:
                raise ValueError("Response census mismatch")
            columns = ["item_id"] if protected else ["item_id", "content", "item_features"]
            position = 0
            for batch in batches(directory / "items.parquet", columns, protected):
                for row in batch.to_pylist():
                    iid = source_string(row["item_id"])
                    item_key = item_keys[iid]
                    content = None if protected else source_string(row["content"])
                    features = None if protected else source_string(row["item_features"])
                    split = pop["split"] if protected else effective_split(content, pop, config.legacy_monitor_seed)
                    group = None if protected else content_group(content)
                    status = "withheld_protected" if protected else "missing_content" if not content.strip() else pop["feature_status"]
                    writers["items"].add({"benchmark": benchmark, "item_id": iid, "item_key": item_key, "source_row": position,
                                          "split": split, "content": content, "item_features": features, "status": status, "group_key": group})
                    position += 1
                    if protected:
                        writers["item_input_links"].add({"benchmark": benchmark, "item_id": iid, "item_key": item_key,
                            "interactor_key": None, "task_key": None, "group_key": None, "split": split,
                            "feature_status": status, "interactors_recorded": False})
                        continue
                    db.execute("INSERT INTO groups VALUES (?,?,1) ON CONFLICT(group_key) DO UPDATE SET mask=mask|excluded.mask,n=n+1", (group, SPLIT_BITS[split]))
                    variants = db.execute("SELECT interactor_key,value,recorded FROM variants WHERE item_key=? ORDER BY interactor_key", (item_key,)).fetchall()
                    if not variants:
                        variants = [(interactor_key(""), "", 0)]
                    for ikey, interactors, recorded in variants:
                        task = {"item_content": content, "item_features": features, "interactors": interactors}
                        task_key = stable_key(task)
                        if db.execute("INSERT OR IGNORE INTO tasks VALUES (?)", (task_key,)).rowcount:
                            writers["tasks"].add({"task_key": task_key, **task})
                            writers["task_features"].add({"task_key": task_key, **task_observables(task)})
                        writers["item_input_links"].add({"benchmark": benchmark, "item_id": iid, "item_key": item_key,
                            "interactor_key": ikey, "task_key": task_key, "group_key": group, "split": split,
                            "feature_status": status, "interactors_recorded": bool(recorded)})
            if position != pop["items"]:
                raise ValueError("Item census mismatch")
            db.execute("DELETE FROM variants")
            db.commit()
            if progress:
                progress({"benchmark": benchmark, "items_done": writers["items"].count, "responses_done": writers["response_input_links"].count})
        links = writers["item_input_links"]
        links.flush()
        # Parquet footer is needed before opening the completed link table.
        link_info = links.close()
        writers.pop("item_input_links")
        for batch in pq.ParquetFile(out / "item_input_links.parquet").iter_batches(
                batch_size=config.batch_size,
                columns=["benchmark", "item_key", "task_key", "group_key", "split", "feature_status"]):
            for row in batch.to_pylist():
                mask = db.execute("SELECT mask FROM groups WHERE group_key=?", (row["group_key"],)).fetchone() if row["group_key"] else None
                conflict = bool(mask and mask[0] & (mask[0] - 1))
                reason = "cross_split_content" if row["split"] == "training" and conflict else row["feature_status"] if row["feature_status"] != "supported_text" else "nontraining_split" if row["split"] != "training" else "eligible"
                writers["training_eligibility"].add({"benchmark": row["benchmark"], "item_key": row["item_key"], "task_key": row["task_key"],
                    "split": row["split"], "eligible": reason == "eligible", "reason": reason})
        for key, mask, n in db.execute("SELECT group_key,mask,n FROM groups ORDER BY group_key"):
            writers["content_groups"].add({"group_key": key, "split_mask": mask, "item_rows": n})
        result = {"tables": {"item_input_links": link_info}, "accesses": accesses}
        for name, writer in writers.items():
            result["tables"][name] = writer.close()
        writers.clear()
        result.update(items=sum(r["items"] for r in population), responses=sum(r["responses"] for r in population))
        return result
    finally:
        for writer in writers.values():
            writer.close()
        db.close()
        db_path.unlink(missing_ok=True)
