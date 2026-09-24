"""Compact corpus progress and failure reports computed inside AWS."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from boto3.dynamodb.conditions import Attr

from .failure_reports import bounded_report, problem_detail

READY_INPUTS = {"fulltext_ready", "model_draft", "draft_failed"}


def _items(table):
    request = {
        "FilterExpression": Attr("id_kind").eq("stem"),
        "ProjectionExpression": "work_id, ingest_status, source_note_status, source_note_problems, "
                                "source_note_at, page_status, page_problems, category",
    }
    while True:
        response = table.scan(**request)
        yield from response.get("Items", [])
        if not response.get("LastEvaluatedKey"):
            return
        request["ExclusiveStartKey"] = response["LastEvaluatedKey"]


def failure_rows(items):
    rows = []
    for item in items:
        if item.get("ingest_status") not in READY_INPUTS:
            continue
        note = item.get("source_note_status")
        if note and note != "source_ready":
            stage, status, problems = "source_note", note, item.get("source_note_problems")
        else:
            continue
        rows.append({"stem": item["work_id"], "stage": stage, "status": status,
                     "category": item.get("category"), "problems": list(problems or [])})
    return sorted(rows, key=lambda row: row["stem"])


def _window(event):
    offset, limit = event.get("offset", 0), event.get("limit", 100)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("offset must be non-negative and limit must be 1 to 100")
    return offset, limit


def report(event, *, table, s3=None, bucket=None, index=None):
    offset, limit = _window(event)
    items = list(_items(table))
    failures = failure_rows(items)
    if "problem_id" in event:
        return problem_detail(event, failures, id_key="stem")
    result = bounded_report(failures, offset=offset, limit=limit,
                            verbose=event.get("action") != "pipeline_failures" or bool(event.get("verbose")),
                            id_key="stem", rows_key="papers")
    if event.get("action") == "pipeline_failures":
        return result

    ready = [item for item in items if item.get("source_note_status") == "source_ready"]
    result.update({"catalogued": len(items), "ready_notes": len(ready),
                   "note_statuses": dict(Counter(item.get("source_note_status") or "unattempted" for item in items)),
                   "notes_by_category": dict(Counter(item.get("category") or "uncategorized" for item in ready))})
    if event.get("check_index"):
        connection, etag = index()
        try:
            indexed = {row[0] for row in connection.execute("SELECT doc_id FROM docs WHERE doc_type = 'note'")}
        finally:
            connection.close()
        modified = s3.head_object(Bucket=bucket, Key="index/wiki-index.sqlite3")["LastModified"]
        if modified.tzinfo is None:
            modified = modified.replace(tzinfo=timezone.utc)
        missing = sorted(item["work_id"] for item in ready if item["work_id"] not in indexed)
        by_id = {item["work_id"]: item for item in items}
        nonready = sorted(indexed - {item["work_id"] for item in ready})
        nonready_status = {stem: (by_id[stem].get("source_note_status") or "unattempted")
                           if stem in by_id else "not_catalogued" for stem in nonready}
        newer, unknown = [], []
        for item in ready:
            try:
                raw_stamp = item.get("source_note_at")
                if not isinstance(raw_stamp, str):
                    raise ValueError("Missing or invalid generation timestamp")
                stamp = datetime.fromisoformat(raw_stamp.replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                if stamp > modified:
                    newer.append(item["work_id"])
            except (KeyError, TypeError, ValueError):
                unknown.append(item["work_id"])
        result["index"] = {"etag": etag, "last_modified": modified.isoformat(), "indexed_notes": len(indexed),
                           "missing_ready_count": len(missing), "missing_ready_sample": missing[:50],
                           "indexed_nonready_count": len(nonready),
                           "indexed_nonready_by_status": dict(Counter(nonready_status.values())),
                           "indexed_nonready_sample": [{"stem": stem, "status": nonready_status[stem],
                                                        "category": by_id.get(stem, {}).get("category")}
                                                       for stem in nonready[:50]],
                           "newer_than_index_count": len(newer), "newer_than_index_sample": sorted(newer)[:50],
                           "unknown_note_timestamp_count": len(unknown),
                           "scope": "Identity coverage and generation timestamps; not content-quality verification."}
    return result
