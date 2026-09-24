"""Read-only, AWS-side audit of stored OpenAlex metadata and one match checkpoint.

This module never fetches OpenAlex or reads originals, extractions, notes or indexes. A
caller must execute it in AWS; only its bounded summary should leave AWS. Comparisons
identify review candidates, not errors in the original source or confirmed bad matches.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import openalex_match as matcher

MAX_SAMPLES = 3
MAX_RESPONSE_BYTES = 131_072
SCAN_SIZE = 200
FIELDS = {
    "title": ("title", "openalex_title"),
    "year": ("year", "openalex_year"),
    "authors": ("authors", "openalex_authors"),
    "journal": ("journal", "openalex_venue"),
    "type": ("type", "openalex_type"),
    "pmid": ("pmid", "pmid_openalex"),
    "pmcid": ("pmcid", "pmcid_openalex"),
    "references": ("referenced_works", "openalex_referenced_works"),
    "oa": ("is_open_access", "openalex_is_oa"),
}
STATUSES = ("matched", "unmatched", "unattempted", "error", "pending", "other")


def _present(value: Any) -> bool:
    # False is a reported OA value, not a missing field.
    return value is not None and value != "" and value != [] and value != {}


def _clip(value: Any, size: int = 160) -> str:
    return str(value)[:size]


def _status(row: dict[str, Any]) -> str:
    value = row.get("openalex_status")
    if not _present(value):
        return "unattempted"
    return value if value in STATUSES else "other"


def _openalex_id(value: Any) -> str | None:
    text = re.sub(r"^https?://openalex\.org/", "", str(value or "").strip())
    return text if re.fullmatch(r"W[0-9]+", text) else None


def _identifier(value: Any, field: str) -> str:
    text = str(value or "").strip().rstrip("/").rsplit("/", 1)[-1]
    return re.sub(rf"^{field}:", "", text, flags=re.I).upper()


def _surname(value: Any) -> str | None:
    """First surname only: roster lengths and truncated 'et al.' lists are incomparable."""
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        return None
    first = re.split(r"\s*(?:;|,|\band\b|&|\bet\s+al\.?|\u00b7)\s*", value)[0].strip()
    # Skip organizational authors and uninterpretable frontmatter containers.
    if re.search(r"consortium|collaboration|group|committee|\[|\{|\]", first, re.I):
        return None
    tokens = first.split()
    while tokens and re.fullmatch(r"(?:[A-Z]\.?[-]?){1,4}", tokens[-1]):
        tokens.pop()
    if not tokens:
        return None
    surname = matcher._squash(tokens[-1])
    return surname if len(surname) > 1 else None


def _comparable(field: str, value: Any) -> Any:
    if field == "authors":
        return _surname(value)
    if field in ("pmid", "pmcid"):
        return _identifier(value, field)
    if field == "references":
        if not isinstance(value, list) or any(_openalex_id(v) is None for v in value):
            return None
        return frozenset(_openalex_id(v) for v in value)
    if field == "oa":
        return value if isinstance(value, bool) else None
    return matcher._squash(value) or None


def _preview(value: Any) -> Any:
    if isinstance(value, list):
        return {"length": len(value), "first": [_clip(v, 80) for v in value[:2]]}
    return value if isinstance(value, bool) else _clip(value)


def _scope(row: dict[str, Any]) -> str:
    if row.get("id_kind") == "stem":
        return "stem"
    return "legacy_openalex" if _openalex_id(row.get("work_id")) else "other"


def _empty_scope() -> dict[str, Any]:
    return {"rows": 0, "status": dict.fromkeys(STATUSES, 0), "without_doi": 0,
            "with_openalex_id": 0, "source_note_ready": 0, "source_note_failed": 0}


def audit_catalogue(*, table: Any, s3: Any = None, bucket: str | None = None,
                    run_id: str | None = None, sample_limit: int = MAX_SAMPLES) -> dict[str, Any]:
    """Scan projected metadata in AWS, returning fixed counts and bounded deterministic samples.

    A consistent DynamoDB scan is not a transactional snapshot. Run after matching settles
    to reconcile a checkpoint. Only the selected run is grouped; arbitrary per-run dictionaries
    or arbitrary status/venue/type values are never copied into the response.
    """
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or not 0 <= sample_limit <= MAX_SAMPLES:
        raise ValueError("sample_limit must be an integer from 0 to 3")
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("run_id must contain 1..100 letters, digits, underscores or hyphens")
    if run_id is not None and (s3 is None or not bucket):
        raise ValueError("s3 and bucket are required to inspect a run checkpoint")

    report: dict[str, Any] = {
        "schema_version": 1, "read_only": True, "scan_pages": 0,
        "scopes": {name: _empty_scope() for name in ("stem", "legacy_openalex", "other")},
        "coverage": {field: {"catalogue": 0, "openalex": 0, "catalogue_missing_openalex_present": 0,
                              "both_present": 0, "compared": 0, "incomparable": 0,
                              "review_candidates": 0} for field in FIELDS},
        "run_membership": {name: dict.fromkeys(STATUSES, 0) for name in ("requested_run", "other_run", "no_run")},
        "review_candidates": {}, "samples": {}, "review_candidate_stem_rows": 0,
        "coverage_scope": "All id_kind=stem rows, including unmatched and unattempted rows",
        "matched_without_references": 0, "openalex_author_lists_at_cap": 0,
        "openalex_reference_entries": 0, "invalid_reference_entries": 0,
        "year_difference": {"one_year": 0, "more_than_one_year": 0, "non_numeric": 0},
        "limitations": [
            "Coverage and paired comparisons apply only to id_kind=stem rows.",
            "Catalogue PMID/PMCID may already be gap-filled; original missingness cannot be reconstructed.",
            "Stored DOI/OpenAlex associations are checked for format and duplicates, not re-fetched identity.",
            "All disagreements are review candidates; original PDFs have not been inspected by this audit.",
            "Author comparison uses first surnames only; OpenAlex author lists may be capped at 30.",
            "An absent reference list does not distinguish zero references from missing upstream metadata.",
            "Consistent scans are not transactional snapshots across concurrent changes.",
        ],
    }
    samples: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    review_rows: set[str] = set()

    def sample(issue: str, row: dict[str, Any], **details: Any) -> None:
        if not sample_limit:
            return
        work_id = str(row.get("work_id") or "")
        item = {"work_id": _clip(work_id, 180), **details}
        bucket_samples = samples.setdefault(issue, [])
        bucket_samples.append((work_id, item))
        bucket_samples.sort(key=lambda entry: (entry[0], json.dumps(entry[1], sort_keys=True)))
        del bucket_samples[sample_limit:]

    def issue(name: str, row: dict[str, Any], **details: Any) -> None:
        review_rows.add(str(row.get("work_id") or ""))
        report["review_candidates"][name] = report["review_candidates"].get(name, 0) + 1
        sample(name, row, **details)

    # Retain join keys and at most three row identifiers per group inside AWS, never in response.
    identities: dict[str, dict[str, dict[str, Any]]] = {"doi": {}, "openalex_id": {}}

    def remember(field: str, key: str | None, row: dict[str, Any], scope: str) -> None:
        if not key:
            return
        group = identities[field].setdefault(key, {"rows": 0, "scopes": {}, "work_ids": []})
        group["rows"] += 1
        group["scopes"][scope] = group["scopes"].get(scope, 0) + 1
        group["work_ids"] = sorted(group["work_ids"] + [_clip(row.get("work_id"), 180)])[:MAX_SAMPLES]

    projected = {"work_id", "id_kind", "doi", "openalex_status", "openalex_id", "openalex_match_run", "source_note_status"}
    projected.update(name for pair in FIELDS.values() for name in pair)
    aliases = {f"#f{i}": name for i, name in enumerate(sorted(projected))}
    request: dict[str, Any] = {"Limit": SCAN_SIZE, "ConsistentRead": True,
                              "ProjectionExpression": ", ".join(aliases), "ExpressionAttributeNames": aliases}
    while True:
        page = table.scan(**request)
        report["scan_pages"] += 1
        for row in page.get("Items", []):
            scope_name = _scope(row)
            scope = report["scopes"][scope_name]
            scope["rows"] += 1
            status = _status(row)
            scope["status"][status] += 1
            doi = matcher.normalize_doi(row.get("doi"))
            scope["without_doi"] += int(not doi)
            stored_id = row.get("openalex_id")
            identity = _openalex_id(stored_id)
            scope["with_openalex_id"] += int(identity is not None)
            scope["source_note_ready"] += int(row.get("source_note_status") == "source_ready")
            scope["source_note_failed"] += int(row.get("source_note_status") == "source_failed")
            remember("doi", doi if doi and re.fullmatch(r"10\.[0-9]{4,9}/\S+", doi) else None, row, scope_name)
            remember("openalex_id", identity or (_openalex_id(row.get("work_id")) if scope_name == "legacy_openalex" else None), row, scope_name)
            if scope_name != "stem":
                continue
            membership = "no_run" if not row.get("openalex_match_run") else (
                "requested_run" if run_id is not None and row["openalex_match_run"] == run_id else "other_run")
            report["run_membership"][membership][status] += 1
            if doi and not re.fullmatch(r"10\.[0-9]{4,9}/\S+", doi):
                issue("invalid_doi", row, doi=_clip(doi))
            if _present(stored_id) and identity is None:
                issue("invalid_openalex_id", row, openalex_id=_clip(stored_id))
            if status == "matched" and identity is None:
                issue("matched_without_valid_openalex_id", row)
            if status != "matched":
                sample(status, row, doi=_clip(doi or ""), openalex_id=_clip(stored_id or ""))
            if not doi:
                sample("without_doi", row)
            authors = row.get("openalex_authors")
            report["openalex_author_lists_at_cap"] += int(isinstance(authors, list) and len(authors) == 30)
            refs = row.get("openalex_referenced_works")
            if status == "matched" and not refs:
                report["matched_without_references"] += 1
                sample("matched_without_references", row)
            if isinstance(refs, list):
                report["openalex_reference_entries"] += len(refs)
                invalid_refs = [v for v in refs if _openalex_id(v) is None]
                report["invalid_reference_entries"] += len(invalid_refs)
                if invalid_refs:
                    issue("invalid_reference_ids", row, invalid_count=len(invalid_refs), first=_clip(invalid_refs[0]))
            elif _present(refs):
                issue("invalid_reference_container", row)
            for field, (own_name, oa_name) in FIELDS.items():
                own, oa = row.get(own_name), row.get(oa_name)
                own_present, oa_present = _present(own), _present(oa)
                coverage = report["coverage"][field]
                coverage["catalogue"] += int(own_present)
                coverage["openalex"] += int(oa_present)
                coverage["catalogue_missing_openalex_present"] += int(oa_present and not own_present)
                if not (own_present and oa_present):
                    continue
                coverage["both_present"] += 1
                left, right = _comparable(field, own), _comparable(field, oa)
                if left is None or right is None:
                    coverage["incomparable"] += 1
                    continue
                coverage["compared"] += 1
                if left != right:
                    coverage["review_candidates"] += 1
                    issue(f"{field}_disagreement", row, catalogue=_preview(own), openalex=_preview(oa))
                    if field == "year":
                        try:
                            difference = abs(int(own) - int(oa))
                            key = "one_year" if difference == 1 else "more_than_one_year"
                        except (TypeError, ValueError, OverflowError):
                            key = "non_numeric"
                        report["year_difference"][key] += 1
        cursor = page.get("LastEvaluatedKey")
        if not cursor:
            break
        request["ExclusiveStartKey"] = cursor

    report["duplicate_identities"] = {}
    for field, groups in identities.items():
        counts = {"groups": 0, "rows_in_groups": 0, "stem_groups": 0,
                  "legacy_groups": 0, "cross_scope_groups": 0}
        for key, group in groups.items():
            if group["rows"] < 2:
                continue
            counts["groups"] += 1
            counts["rows_in_groups"] += group["rows"]
            counts["stem_groups"] += int(group["scopes"].get("stem", 0) > 1)
            counts["legacy_groups"] += int(group["scopes"].get("legacy_openalex", 0) > 1)
            counts["cross_scope_groups"] += int(len(group["scopes"]) > 1)
            sample(f"duplicate_{field}", {"work_id": key}, identity=_clip(key), **group)
        report["duplicate_identities"][field] = counts

    if run_id is not None:
        key = f"runs/openalex-match/{run_id}.json"
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
        try:
            checkpoint = json.loads(body.read())
        finally:
            body.close()
        if checkpoint.get("run_id") != run_id:
            raise ValueError("Checkpoint run_id does not match the requested run")
        counters = ("catalogued", "selected", "without_doi", "skipped_matched", "matched",
                    "unmatched", "errors", "conflicts", "requests", "disagreements")
        summary = {name: int(checkpoint.get(name, 0)) for name in counters}
        summary.update(run_id=run_id, done=bool(checkpoint.get("done")), pending=len(checkpoint.get("pending") or []),
                       started_at=_clip(checkpoint.get("started_at") or "", 40),
                       finished_at=_clip(checkpoint.get("finished_at") or "", 40), progress_key=key)
        summary["unaccounted_selected"] = summary["selected"] - sum(summary[name] for name in ("matched", "unmatched", "errors", "conflicts"))
        summary["table_requested_run_matched"] = report["run_membership"]["requested_run"]["matched"]
        summary["table_requested_run_unmatched"] = report["run_membership"]["requested_run"]["unmatched"]
        report["checkpoint"] = summary
        for entry in checkpoint.get("error_details") or []:
            sample("checkpoint_error", {"work_id": entry.get("stem")}, doi=_clip(entry.get("doi") or ""), error=_clip(entry.get("error") or ""))
        for entry in checkpoint.get("conflict_details") or []:
            sample("checkpoint_conflict", {"work_id": entry.get("stem")},
                   doi=_clip(entry.get("doi") or ""), reason=_clip(entry.get("reason") or ""))
        report["limitations"].append("Checkpoint errors/conflicts are run events; failed rows need not carry an error status in DynamoDB.")

    report["samples"] = {key: [value for _, value in entries] for key, entries in sorted(samples.items())}
    report["review_candidate_stem_rows"] = len(review_rows)
    report["coverage_denominator"] = report["scopes"]["stem"]["rows"]
    report["total_rows"] = sum(value["rows"] for value in report["scopes"].values())
    if len(json.dumps(report, ensure_ascii=True).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Audit response exceeds the bounded summary size")
    return report
