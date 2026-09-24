from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .openalex import normalize_doi, normalize_work_id


def _plain_numbers(value):
    """DynamoDB reads must remain JSON-serializable when republished to S3."""
    return json.loads(json.dumps(value, default=lambda number: int(number) if number % 1 == 0 else float(number)))


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def slugify(value: str, limit: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:limit].rstrip("-") or "untitled"


def candidate_stem(work: dict[str, Any]) -> str:
    author = (work.get("authors") or ["unknown"])[0].split()[-1]
    year = work.get("publication_year") or "undated"
    title = work.get("title") or "untitled"
    return slugify(f"{author}-{year}-{title}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AwsCatalog:
    """Candidate metadata in DynamoDB and S3, with no client-side database or PDF copy."""

    def __init__(self, settings: Settings, store=None) -> None:
        from .aws_store import AwsStore
        if not settings.aws_table or not settings.aws_bucket:
            raise RuntimeError("Configure AWS_KIRO_WIKI_TABLE and AWS_KIRO_WIKI_BUCKET; local catalogs are retired")
        self.settings = settings
        self.store = store or AwsStore(settings)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def get_candidate(self, identifier: str) -> dict[str, Any]:
        from boto3.dynamodb.conditions import Key
        work_id = normalize_work_id(identifier)
        doi = normalize_doi(identifier)
        if work_id:
            item = self.store.get_item(work_id)
        elif doi:
            table = self.store.session.resource("dynamodb").Table(self.settings.aws_table)
            rows = table.query(IndexName="doi-index", KeyConditionExpression=Key("doi").eq(doi)).get("Items", [])
            item = next((r for r in rows if r.get("record")), {})
        else:
            raise ValueError("identifier must be an OpenAlex work ID or DOI")
        if not item or not item.get("record"):
            raise KeyError(f"candidate not found in AWS: {identifier}")
        return _plain_numbers(item)

    def save_candidate(self, work: dict[str, Any]) -> dict[str, Any]:
        work_id = normalize_work_id(work.get("work_id"))
        if not work_id or not work.get("title"):
            raise ValueError("candidate requires work_id and title")
        doi = normalize_doi(work.get("doi"))
        if doi:
            try:
                same_doi = self.get_candidate(doi)
            except KeyError:
                same_doi = None
            if same_doi and same_doi["work_id"] != work_id:
                raise ValueError(f"DOI {doi} is already stored as {same_doi['work_id']}")
            # S3 conditional writes serialize ownership even while the DOI GSI lags.
            # Keep a failed save's reservation so the same work can safely retry.
            owner_key = f"candidates/doi-owners/{hashlib.sha256(doi.encode()).hexdigest()}.json"
            try:
                self.store.put_text(owner_key, json.dumps({"doi": doi, "work_id": work_id}),
                                    create_only=True, content_type="application/json")
            except FileExistsError:
                owner = json.loads(self.store.get_text(owner_key))
                if owner.get("work_id") != work_id:
                    raise ValueError(f"DOI {doi} is already stored as {owner.get('work_id')}") from None
        existing = self.store.get_item(work_id)
        timestamp = utc_now()
        candidate = {"work_id": work_id, "doi": doi, "title": work["title"],
                     "publication_year": work.get("publication_year"),
                     "status": existing.get("status") or "candidate",
                     "stem": existing.get("stem") or f"{candidate_stem(work)}-{work_id.lower()}",
                     "record": work, "created_at": existing.get("created_at") or timestamp,
                     "updated_at": timestamp}
        self.store.push_candidate(candidate)
        self.store.upload_json(work, f"candidates/{work_id}.json")
        return candidate

    def list_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        from boto3.dynamodb.conditions import Attr
        table = self.store.session.resource("dynamodb").Table(self.settings.aws_table)
        condition = Attr("record").exists()
        if status:
            condition &= Attr("status").eq(status)
        request = {"FilterExpression": condition}
        items = []
        while True:
            page = table.scan(**request)
            items.extend(page.get("Items", []))
            if not page.get("LastEvaluatedKey"):
                return _plain_numbers(items)
            request["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def search_corpus(self, **filters):
        return self.store.search_corpus(**filters)["items"]

    def attach_pdf(self, identifier: str, source_path: Path) -> dict[str, Any]:
        candidate = self.get_candidate(identifier)
        with source_path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError(f"not a PDF file: {source_path}")
        digest = sha256_file(source_path)
        key = f"papers/{candidate['stem']}/original.pdf"
        from botocore.exceptions import ClientError
        try:
            with source_path.open("rb") as handle:
                self.store.session.client("s3").put_object(
                    Bucket=self.settings.aws_bucket, Key=key, Body=handle,
                    ContentType="application/pdf", IfNoneMatch="*",
                )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"PreconditionFailed", "412"}:
                raise FileExistsError(f"s3://{self.settings.aws_bucket}/{key} already exists") from exc
            raise
        self.store.session.resource("dynamodb").Table(self.settings.aws_table).update_item(
            Key={"work_id": candidate["work_id"]},
            UpdateExpression="SET #status = :status, attached_pdf_key = :key, attached_pdf_sha256 = :sha",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":status": "pdf_attached", ":key": key, ":sha": digest},
        )
        return {"work_id": candidate["work_id"], "path": f"s3://{self.settings.aws_bucket}/{key}",
                "sha256": digest, "status": "pdf_attached"}


# Kept as the public name for callers; there is no local catalog backend.
Catalog = AwsCatalog
