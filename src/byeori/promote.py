"""Review a draft inside AWS and publish its evidence note directly to S3.

The PDF and draft remain in AWS. An explicitly supplied reviewed_text may replace the
body during review; hashes and the DynamoDB condition still protect the source identity.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Protocol

from .config import Settings
from .validation import SOURCE_SECTIONS

REVIEW_METHODS = ("agent-session", "validator", "person", "bedrock-review")
REQUIRED_SECTIONS = SOURCE_SECTIONS + ("## Related pages",)
# Matches ingest_lambda.DRAFT_FORBIDDEN, which says why "as an ai" alone is not the pattern.
PLACEHOLDER = re.compile(r"\[(?:TODO|TBD|placeholder|insert|citation needed)[^\]]*\]|lorem ipsum|"
                         r"\bas an ai(?: language model| assistant)?,? i\b", re.I)
DRAFT_KEY_PREFIX = "wiki/drafts/"
REVIEWED_KEY_PREFIX = "wiki/sources/"
REVIEW_FIELDS = ("review_status", "reviewed_by", "reviewed_at", "review_method", "review_note",
                 "reviewed_key", "reviewed_sha256", "draft_edited_in_review")


class ReviewStore(Protocol):
    """The AWS-side storage interface for promotion; tests substitute a fake."""

    def get_item(self, work_id: str) -> dict[str, Any]: ...
    def get_text(self, key: str) -> str: ...
    def put_text(self, key: str, text: str, *, create_only: bool = False) -> dict[str, str]: ...
    def mark_reviewed(self, work_id: str, fields: dict[str, Any], *,
                      expected_draft_sha256: str) -> dict[str, Any]: ...


def split_page(text: str) -> tuple[list[str], str]:
    """Return the frontmatter lines (without the --- fences) and the body."""
    if not text.startswith("---\n"):
        raise ValueError("draft has no code-written frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("draft frontmatter is not closed")
    return text[4:end].splitlines(), text[end + len("\n---\n"):]


def parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    """Parse the flat ``key: value`` frontmatter the Lambda writes (JSON scalars or bare words)."""
    fields: dict[str, Any] = {}
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        key, separator, raw = line.partition(":")
        if not separator:
            raise ValueError(f"unreadable frontmatter line: {line!r}")
        raw = raw.strip()
        try:
            fields[key.strip()] = json.loads(raw) if raw else ""
        except json.JSONDecodeError:
            fields[key.strip()] = raw
    return fields


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def check_draft_body(body: str) -> list[str]:
    """AWS re-check of the draft validator so a hand-edited draft cannot skip it."""
    problems = [f"missing section {section}" for section in REQUIRED_SECTIONS if section not in body]
    if not body.lstrip().startswith("## Citation"):
        problems.append("note must start with ## Citation")
    if PLACEHOLDER.search(body):
        problems.append("placeholder or filler text present")
    if len(body) < 1500:
        problems.append("note is too short to be a full evidence note")
    return problems


def promote_draft_in_worker(
    settings: Settings,
    work_id: str,
    *,
    reviewer: str,
    method: str = "agent-session",
    note: str | None = None,
    edited: bool = False,
    reviewed_text: str | None = None,
    dry_run: bool = False,
    store: ReviewStore | None = None,
) -> dict[str, Any]:
    """Accept a model draft as a reviewed evidence note. Raises ValueError when a check fails."""
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("--reviewer must name who reviewed the draft")
    if method not in REVIEW_METHODS:
        raise ValueError(f"review method must be one of {', '.join(REVIEW_METHODS)}")
    checks: list[str] = []

    if store is None or not settings.aws_bucket or not settings.aws_table:
        raise RuntimeError("Promotion requires the configured AWS bucket and table")
    if edited and reviewed_text is None:
        raise ValueError("edited=True requires explicitly supplied reviewed_text")
    draft_key = f"{DRAFT_KEY_PREFIX}{work_id}.md"
    item = store.get_item(work_id)
    draft_key = item.get("draft_key") or draft_key
    draft_text = reviewed_text if reviewed_text is not None else store.get_text(draft_key)
    checks.append(f"read draft from s3://{settings.aws_bucket}/{draft_key}" if reviewed_text is None
                  else "read explicitly supplied reviewed text")
    reviewed_input_sha256 = sha256_text(draft_text)
    frontmatter_lines, body = split_page(draft_text)
    fields = parse_frontmatter(frontmatter_lines)

    if fields.get("work_id") != work_id:
        raise ValueError(f"draft frontmatter is for {fields.get('work_id')!r}, not {work_id}")
    if fields.get("draft_status") != "model_draft":
        raise ValueError(f"draft_status is {fields.get('draft_status')!r}; only model_draft can be promoted")
    if fields.get("draft_problems"):
        raise ValueError(f"draft recorded validator problems: {fields['draft_problems']}")
    if fields.get("review_status") != "unreviewed":
        raise ValueError(f"draft review_status is {fields.get('review_status')!r}, expected unreviewed")
    for name in ("pdf_sha256", "grobid_sha256", "ingest_model", "ingest_harness"):
        if not fields.get(name):
            raise ValueError(f"draft frontmatter lacks provenance field {name}")
    checks.append("frontmatter: model_draft, unreviewed, provenance present")
    problems = check_draft_body(body)
    if problems:
        raise ValueError("draft body fails the evidence-note check: " + "; ".join(problems))
    checks.append(f"body: {len(REQUIRED_SECTIONS)} required sections, no placeholders")

    expected_sha256: str | None = None
    if store is not None and settings.aws_table:
        item = store.get_item(work_id)
        if item.get("ingest_status") != "model_draft":
            raise ValueError(f"DynamoDB ingest_status is {item.get('ingest_status')!r}, expected model_draft")
        if item.get("review_status") not in (None, "unreviewed"):
            raise ValueError(f"DynamoDB review_status is already {item.get('review_status')!r}")
        expected_sha256 = item.get("draft_sha256")
        if not expected_sha256:
            raise ValueError("DynamoDB item has no draft_sha256; run aws-draft-page again")
        for name in ("pdf_sha256", "grobid_sha256"):
            if item.get(name) != fields.get(name):
                raise ValueError(f"{name} in the draft does not match DynamoDB; the ingest changed")
        if item.get("draft_key"):
            draft_key = item["draft_key"]
        checks.append("dynamodb: model_draft, pdf and GROBID hashes match the draft")
    draft_edited = expected_sha256 is not None and reviewed_input_sha256 != expected_sha256
    if draft_edited and not edited:
        raise ValueError(
            f"reviewed draft sha256 {reviewed_input_sha256[:12]} differs from the recorded draft {expected_sha256[:12]}; "
            "pass --edited if you revised the draft during review, otherwise re-read it from S3"
        )
    checks.append("draft hash: edited by the reviewer" if draft_edited
                  else "draft hash: matches the recorded draft" if expected_sha256 else "draft hash: no DynamoDB record to compare")

    reviewed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    reviewed_key = f"{REVIEWED_KEY_PREFIX}{work_id}.md"
    record: dict[str, Any] = {
        "review_status": "reviewed",
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
        "review_method": method,
        "draft_key": draft_key,
        "draft_sha256": expected_sha256 or reviewed_input_sha256,
        "draft_edited_in_review": draft_edited,
    }
    if note:
        record["review_note"] = note.strip()
    kept = [line for line in frontmatter_lines
            if line.split(":", 1)[0].strip() not in record]
    page_lines = ["---", *kept, *(f"{key}: {_yaml_scalar(value)}" for key, value in record.items()), "---", ""]
    page = "\n".join(page_lines) + body
    reviewed_sha256 = sha256_text(page)

    result: dict[str, Any] = {
        "work_id": work_id,
        "dry_run": dry_run,
        "reviewed_path": f"s3://{settings.aws_bucket}/{reviewed_key}",
        "reviewed_key": reviewed_key,
        "reviewed_sha256": reviewed_sha256,
        "draft_key": draft_key,
        "draft_sha256": record["draft_sha256"],
        "draft_edited_in_review": draft_edited,
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
        "review_method": method,
        "checks": checks,
        "s3": None,
        "dynamodb": None,
    }
    if dry_run:
        return result

    result["s3"] = store.put_text(reviewed_key, page, create_only=True)
    if store is not None and settings.aws_table and expected_sha256:
        fields_for_table = {key: record[key] for key in REVIEW_FIELDS if key in record}
        fields_for_table["reviewed_key"] = reviewed_key
        fields_for_table["reviewed_sha256"] = reviewed_sha256
        result["dynamodb"] = store.mark_reviewed(work_id, fields_for_table,
                                                 expected_draft_sha256=expected_sha256)
    return result


def promote_draft(settings: Settings, work_id: str, *, reviewer: str,
                  method: str = "agent-session", note: str | None = None, edited: bool = False,
                  reviewed_text: str | None = None, dry_run: bool = False, store=None) -> dict[str, Any]:
    """Ask AWS to validate and promote; never fetch the draft into the client."""
    from .aws_store import AwsStore
    return (store or AwsStore(settings)).promote_draft(
        work_id, reviewer=reviewer, method=method, note=note, edited=edited,
        reviewed_text=reviewed_text, dry_run=dry_run)
