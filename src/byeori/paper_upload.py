"""Store one PDF in S3 `papers/{stem}/`: the original, its meta.json and a DynamoDB catalog item.

Moved out of `scripts/upload_to_s3_folder.py` (task 9b, 2026-09-24) so the same code backs both the
lab's shared-folder bulk upload and the installer's `upload-pdf` subcommand for a lab with no
shared folder at all. Nothing local is moved, renamed or deleted, and a paper whose stored PDF
already carries the same digest is reported as `already_present` without re-uploading; its
meta.json and catalogue fields keep the values already there and only missing keys are filled.
Identity beyond the file name is settled later by `resolve-ids` and the extraction; the file name
alone is never treated as proof of which paper this is.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from .config import Settings
from .papers import _s3, _table, sha256_file


def _stored_meta(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return {}
        raise


STEM_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{2,200}")   # what the ingest Lambda's source_note accepts


def upload_one(settings: Settings, pdf: Path, *, stem: str | None = None, source: str = "upload-pdf") -> dict[str, Any]:
    """Store one PDF. ``stem`` names a file whose own name is not an llm-wiki stem (a publisher's
    ``PIIS0092867421013398.pdf`` in the user's Downloads), and ``source`` records where it came from;
    the file itself is only read, never renamed or moved."""
    if stem is None:
        stem = pdf.stem
        if not STEM_PATTERN.fullmatch(stem):
            raise ValueError(f"the file name {pdf.name!r} is not a lowercase stem (a-z, 0-9, hyphens); "
                             f"pass --stem author-year-words")
    elif not STEM_PATTERN.fullmatch(stem):
        raise ValueError(f"{stem!r} is not a lowercase stem (a-z, 0-9, hyphens, at least three characters)")
    digest = sha256_file(pdf)
    pdf_key, meta_key = f"papers/{stem}/original.pdf", f"papers/{stem}/meta.json"
    s3 = _s3()
    state = "uploaded"
    try:
        head = s3.head_object(Bucket=settings.aws_bucket, Key=pdf_key)
        state = "already_present" if head.get("Metadata", {}).get("sha256") == digest else "replaced_digest_differs"
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
            raise
    if state == "replaced_digest_differs":
        # A different file under a stem that already exists is never written over an original.
        return {"stem": stem, "state": "conflict_existing_original", "bytes": pdf.stat().st_size, "pdf_sha256": digest}
    if state == "uploaded":
        s3.upload_file(str(pdf), settings.aws_bucket, pdf_key,
                       ExtraArgs={"ContentType": "application/pdf", "Metadata": {"sha256": digest, "stem": stem}})
    meta = {"stem": stem, "pdf_key": pdf_key, "pdf_sha256": digest, "pdf_bytes": pdf.stat().st_size,
            "source": source, "source_collection": source, "id_kind": "stem",
            "uploaded_at": datetime.now(UTC).replace(microsecond=0).isoformat()}
    # An original already stored with these bytes may carry what later steps merged in (identity,
    # journal verdict from backfill_to_s3_identity.py), so its existing keys win and only gaps are filled.
    keep = state == "already_present"
    existing = _stored_meta(s3, settings.aws_bucket, meta_key) if keep else {}
    merged = meta | existing
    if merged != existing:
        s3.put_object(Bucket=settings.aws_bucket, Key=meta_key, ContentType="application/json",
                      Body=json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8"))
    if settings.aws_table:
        item = dict(meta)
        names = {f"#f{index}": key for index, key in enumerate(item)}
        values = {f":v{index}": value for index, value in enumerate(item.values())}
        names["#st"], values[":status"] = "ingest_status", "pdf_uploaded"
        assign = "{0} = if_not_exists({0}, {1})" if keep else "{0} = {1}"
        _table(settings).update_item(
            Key={"work_id": stem},
            UpdateExpression="SET " + ", ".join(assign.format(name, f":v{name[2:]}") for name in names if name != "#st")
                             + ", #st = if_not_exists(#st, :status)",
            ExpressionAttributeNames=names, ExpressionAttributeValues=values)
    return {"stem": stem, "state": state, "bytes": meta["pdf_bytes"], "pdf_sha256": digest}
