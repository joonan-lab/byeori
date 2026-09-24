"""Plan metadata review in AWS; keep complete queues in immutable S3 manifests.

The planner never changes catalogue fields or reads scientific content. Stored metadata
differences are review candidates, and PDF/TEI keys are copied only when already observed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

from . import openalex_audit as audit
from .openalex_match import normalize_doi

SCHEMA_VERSION = 1
MAX_RESPONSE_BYTES = 131_072
ISSUE_PRIORITY = {
    "pmid_disagreement": 1, "pmcid_disagreement": 1, "year_difference_over_one": 1,
    "doi_missing": 1, "doi_invalid": 1, "duplicate_stem_doi": 1,
    "missing_pmcid": 2,  # Read legacy immutable plans; new plans allow empty PMCID.
    "missing_references": 2, "authors_at_cap": 2,
    "title_disagreement": 3, "authors_disagreement": 3, "journal_disagreement": 3,
}
METADATA_FIELDS = (
    "doi", "title", "year", "authors", "journal", "pmid", "pmcid",
    "openalex_id", "openalex_status", "openalex_title", "openalex_year", "openalex_authors",
    "openalex_venue", "pmid_openalex", "pmcid_openalex", "openalex_match_run",
    "openalex_authors_returned_count", "openalex_authors_truncated",
    "pdf_key", "pdf_sha256", "tei_key", "grobid_sha256", "extracted_pdf_sha256",
)


def _run_id(event: dict[str, Any]) -> str:
    run_id = event.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("run_id must contain 1..100 letters, digits, underscores or hyphens")
    return run_id


def _key(run_id: str) -> str:
    return f"runs/metadata-review/{run_id}/plan.json"


def _encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def _read(s3: Any, bucket: str, run_id: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=_key(run_id))["Body"]
    except ClientError as exc:
        if missing_ok and exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    try:
        plan = json.loads(body.read())
    finally:
        body.close()
    if plan.get("run_id") != run_id or plan.get("schema_version") != SCHEMA_VERSION or not isinstance(plan.get("items"), list):
        raise ValueError("Stored metadata review plan has an incompatible identity or schema")
    return plan


def _text(value: Any, limit: int = 160) -> str:
    """Bound JSON-escaped bytes, including supplementary Unicode, not only code points."""
    text = str(value).replace("\u00b7", ",")[:limit]
    while len(json.dumps(text, ensure_ascii=True).encode()) > limit:
        text = text[:max(0, len(text) - max(1, len(text) // 8))]
    return text


def _preview(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, list):
        return {"count": len(value), "first": [_text(entry, 100) for entry in value[:2]]}
    return _text(value)


def _bounded(response: dict[str, Any]) -> dict[str, Any]:
    if len(json.dumps(response, ensure_ascii=True).encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("Review response exceeds the bounded response size")
    return response


def _summary(plan: dict[str, Any], *, reused: bool) -> dict[str, Any]:
    samples = {name: [] for name in ISSUE_PRIORITY}
    issue_counts = Counter()
    priority_counts = Counter()
    for item in plan["items"]:
        priority_counts[str(item["priority"])] += 1
        for issue in item["issues"]:
            issue_counts[issue] += 1
            if issue in samples and len(samples[issue]) < 3:
                samples[issue].append({"work_id": _text(item["work_id"], 180), "priority": item["priority"],
                                       "title": _text(item["metadata"].get("title") or "")})
    return _bounded({
        "schema_version": SCHEMA_VERSION, "run_id": plan["run_id"], "plan_key": _key(plan["run_id"]),
        "status": "planned", "reused": reused, "scope": "all_stem", "canonical_writes": False,
        "created_at": _text(plan["created_at"], 50), "catalogued_stems": int(plan["catalogued_stems"]),
        "scan_pages": int(plan["scan_pages"]), "planned_papers": len(plan["items"]),
        "issue_counts": {name: issue_counts[name] for name in ISSUE_PRIORITY},
        "priority_counts": {str(priority): priority_counts[str(priority)] for priority in (1, 2, 3)},
        "duplicate_doi_groups": int(plan["duplicate_doi_groups"]),
        "missing_pdf_key": sum(not item["metadata"].get("pdf_key") for item in plan["items"]),
        "missing_tei_key": sum(not item["metadata"].get("tei_key") for item in plan["items"]),
        "samples": samples,
        "limitations": [
            "Plan membership is a metadata review queue, not confirmed source errors or execution completion.",
            "One paper can have several issues; priority is the smallest issue priority number.",
            "Author comparison uses first surnames only, and unmarked 30-name lists may be truncated.",
            "PDF/TEI availability is not checked; only observed catalogue keys and hashes are recorded.",
            "The manifest is immutable; the same run_id reuses its snapshot without resetting worker progress.",
            "A DynamoDB consistent scan is not a transactional snapshot.",
        ],
    })


def _different(row: dict[str, Any], field: str) -> bool:
    own_key, other_key = audit.FIELDS[field]
    own, other = row.get(own_key), row.get(other_key)
    if not (audit._present(own) and audit._present(other)):
        return False
    left, right = audit._comparable(field, own), audit._comparable(field, other)
    return left is not None and right is not None and left != right


def _issues(row: dict[str, Any]) -> list[str]:
    issues = [f"{field}_disagreement" for field in ("pmid", "pmcid", "title", "authors", "journal") if _different(row, field)]
    doi = normalize_doi(row.get("doi"))
    valid_doi = bool(doi and re.fullmatch(r"10\.[0-9]{4,9}/\S+", doi))
    if not doi:
        issues.append("doi_missing")
    elif not valid_doi:
        issues.append("doi_invalid")
    if _different(row, "year"):
        try:
            if abs(int(row["year"]) - int(row["openalex_year"])) > 1:
                issues.append("year_difference_over_one")
        except (TypeError, ValueError, OverflowError):
            pass  # A nonnumeric year is not evidence for a measured >1-year difference.
    if row.get("openalex_status") == "matched" and not row.get("openalex_referenced_works"):
        issues.append("missing_references")
    authors = row.get("openalex_authors")
    if isinstance(authors, list) and len(authors) == 30 and row.get("openalex_authors_truncated") is not False:
        issues.append("authors_at_cap")
    return issues


def plan_metadata_review(event: dict[str, Any], *, table: Any, s3: Any, bucket: str) -> dict[str, Any]:
    """Create a one-item-per-stem immutable queue in S3, or reuse the existing plan."""
    run_id = _run_id(event)
    if not bucket:
        raise ValueError("bucket must be explicitly configured")
    existing = _read(s3, bucket, run_id, missing_ok=True)
    if existing is not None:
        return _summary(existing, reused=True)
    fields = sorted({"work_id", "id_kind", "openalex_referenced_works", *METADATA_FIELDS})
    names = {f"#f{i}": field for i, field in enumerate(fields)}
    request: dict[str, Any] = {
        "FilterExpression": Attr("id_kind").eq("stem"), "ConsistentRead": True, "Limit": 200,
        "ProjectionExpression": ", ".join(names), "ExpressionAttributeNames": names,
    }
    rows: dict[str, dict[str, Any]] = {}
    doi_stems: dict[str, list[str]] = {}
    pages = 0
    while True:
        page = table.scan(**request)
        pages += 1
        for row in page.get("Items", []):
            if row.get("id_kind") != "stem":
                continue
            work_id = row.get("work_id")
            if not isinstance(work_id, str) or not work_id:
                raise ValueError("Stem catalogue row lacks a work_id")
            # DynamoDB should return each key once; reject conflicting observations rather than
            # silently planning two versions of one paper during concurrent catalogue changes.
            if work_id in rows:
                if rows[work_id] != row:
                    raise ValueError("Stem metadata changed during scan")
                continue
            rows[work_id] = row
            doi = normalize_doi(row.get("doi"))
            if doi and re.fullmatch(r"10\.[0-9]{4,9}/\S+", doi):
                doi_stems.setdefault(doi, []).append(work_id)
        cursor = page.get("LastEvaluatedKey")
        if not cursor:
            break
        request["ExclusiveStartKey"] = cursor

    items = []
    for work_id, row in rows.items():
        doi = normalize_doi(row.get("doi"))
        duplicates = sorted(doi_stems.get(doi, [])) if doi else []
        issues = _issues(row)
        if len(duplicates) > 1:
            issues.append("duplicate_stem_doi")
        if not issues:
            continue
        issues = sorted(set(issues), key=lambda issue: (ISSUE_PRIORITY[issue], issue))
        metadata = {field: row[field] for field in METADATA_FIELDS if field in row}
        refs = row.get("openalex_referenced_works")
        metadata["openalex_references_count"] = len(refs) if isinstance(refs, list) else None
        metadata["openalex_references_sha256"] = hashlib.sha256(_encode(refs)).hexdigest()
        items.append({"work_id": work_id, "issues": issues, "priority": min(ISSUE_PRIORITY[issue] for issue in issues),
                      "metadata": metadata, "normalized_doi": doi,
                      "input_sha256": hashlib.sha256(_encode({field: row[field] for field in fields if field in row})).hexdigest(),
                      "duplicate_doi_stems": duplicates if len(duplicates) > 1 else []})
    items.sort(key=lambda item: (item["priority"], item["work_id"]))
    plan = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "scope": "all_stem",
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(), "catalogued_stems": len(rows),
            "scan_pages": pages, "duplicate_doi_groups": sum(len(stems) > 1 for stems in doi_stems.values()),
            "items": items}
    try:
        s3.put_object(Bucket=bucket, Key=_key(run_id), Body=_encode(plan), ContentType="application/json", IfNoneMatch="*")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
            raise
        # Another invocation may have committed first. Its snapshot is authoritative for this ID.
        existing = _read(s3, bucket, run_id)
        return _summary(existing, reused=True)
    return _summary(plan, reused=False)


def metadata_review_summary(event: dict[str, Any], *, s3: Any, bucket: str) -> dict[str, Any]:
    """Return the immutable plan summary; execution progress belongs to the review worker."""
    return _summary(_read(s3, bucket, _run_id(event)), reused=True)


def inspect_metadata_review(event: dict[str, Any], *, s3: Any, bucket: str) -> dict[str, Any]:
    """Return at most ten bounded item previews; filtering and slicing happen in AWS."""
    run_id = _run_id(event)
    offset, limit = event.get("offset", 0), event.get("limit", 5)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("limit must be an integer from 1 to 10")
    issue = event.get("issue")
    if issue is not None and (not isinstance(issue, str) or issue not in ISSUE_PRIORITY):
        raise ValueError("issue must name a supported review issue")
    plan = _read(s3, bucket, run_id)
    selected = [(index, item) for index, item in enumerate(plan["items"]) if issue is None or issue in item["issues"]]
    previews = []
    for index, item in selected[offset:offset + limit]:
        previews.append({"item_index": index, "work_id": _text(item["work_id"], 180), "priority": item["priority"],
                         "issues": item["issues"], "input_sha256": item["input_sha256"],
                         "normalized_doi": _text(item["normalized_doi"] or ""),
                         "metadata": {field: _preview(value) for field, value in item["metadata"].items()
                                      if field in METADATA_FIELDS or field in ("openalex_references_count", "openalex_references_sha256")},
                         "duplicate_doi_stems": _preview(item["duplicate_doi_stems"])})
    end = offset + len(previews)
    return _bounded({"schema_version": SCHEMA_VERSION, "run_id": run_id, "plan_key": _key(run_id),
                     "preview": True, "string_json_byte_limit": 180, "metadata_string_json_byte_limit": 160,
                     "issue": issue, "total": len(selected),
                     "offset": offset, "limit": limit, "next_offset": end if end < len(selected) else None,
                     "items": previews})
