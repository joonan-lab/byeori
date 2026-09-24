"""One CSV row per PDF the catalogue records in S3.

Paper and candidate metadata is the client-readable surface of this project: wiki content
and its computation stay in AWS, while catalogue metadata is read from DynamoDB and S3
(user, 2026-09-19). This export projects the identification fields of every catalogue row
that carries a stored PDF key and writes them as one CSV file. It reads no PDF, no
extraction, no wiki page and no search index.

The file answers one question: which papers does Byeori already hold? It is a worksheet
for deciding what a newly collected PDF adds, not a mirror of the catalogue. No tool here
reads it back, and the row states what the catalogue records rather than the result of
opening each object in S3.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr

from .config import Settings

# Identification first, then how far the paper got, then the stored object. `doi` and
# `pdf_sha256` are the two columns that decide whether a new PDF is already here.
COLUMNS = (
    "work_id", "id_kind", "title", "authors", "year", "journal", "doi", "pmid", "pmcid",
    "openalex_id", "openalex_status", "openalex_venue", "openalex_year",
    "category", "document_type", "source_collection",
    "ingest_status", "source_note_status", "source_note_key",
    "pdf_key", "pdf_filename", "pdf_bytes", "pdf_sha256", "uploaded_at",
)
# A paper ingested from OpenAlex keeps its metadata inside the stored candidate record
# instead of in the row's own attributes. Those rows are the ones with an empty `id_kind`.
RECORD_FALLBACK = {"authors": "authors", "year": "publication_year",
                   "journal": "source", "document_type": "type"}


def _cell(value: Any) -> str:
    """A catalogue attribute as one spreadsheet cell; lists keep their order."""
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return str(int(value) if value % 1 == 0 else float(value))
    if isinstance(value, (list, tuple)):
        return "; ".join(_cell(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _row(item: dict[str, Any]) -> dict[str, str]:
    record = item.get("record") or {}
    row = {}
    for column in COLUMNS:
        value = item.get(column)
        if value in (None, "") and column in RECORD_FALLBACK:
            value = record.get(RECORD_FALLBACK[column])
        row[column] = _cell(value)
    return row


def _projection() -> tuple[str, dict[str, str]]:
    """The export columns, plus the record fields a candidate row keeps nested."""
    names = {f"#c{index}": column for index, column in enumerate(COLUMNS)}
    paths = list(names)
    for key in dict.fromkeys(RECORD_FALLBACK.values()):
        token = next((name for name, attribute in names.items() if attribute == key), f"#r{len(names)}")
        names[token] = key
        paths.append(f"#rec.{token}")
    names["#rec"] = "record"
    return ", ".join(paths), names


def _log(message: str) -> None:
    print(f"{datetime.now(UTC).replace(microsecond=0).isoformat()} {message}", file=sys.stderr, flush=True)


def scan_stored_papers(settings: Settings, *, table: Any = None) -> dict[str, Any]:
    """Every catalogue row with a PDF key, projected to the export columns."""
    if not settings.aws_table:
        raise RuntimeError("AWS_KIRO_WIKI_TABLE is not configured")
    table = table or boto3.Session(region_name=settings.aws_region).resource("dynamodb").Table(settings.aws_table)
    projection, names = _projection()
    request: dict[str, Any] = {
        "FilterExpression": Attr("pdf_key").exists(),
        "ProjectionExpression": projection,
        "ExpressionAttributeNames": names,
        "ConsistentRead": True,
        "ReturnConsumedCapacity": "TOTAL",
    }
    rows: list[dict[str, str]] = []
    scanned = 0
    capacity = 0.0
    while True:
        response = table.scan(**request)
        rows.extend(_row(item) for item in response.get("Items", []))
        scanned += response.get("ScannedCount", 0)
        capacity += float(response.get("ConsumedCapacity", {}).get("CapacityUnits", 0))
        if not response.get("LastEvaluatedKey"):
            break
        request["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        if len(rows) % 2000 < 200:
            _log(f"{len(rows)} papers read ({scanned} rows scanned)")
    rows.sort(key=lambda row: row["work_id"])
    return {"rows": rows, "scanned_count": scanned, "read_capacity_units": capacity}


def export_papers_csv(settings: Settings, destination: Path, *, force: bool = False,
                      table: Any = None) -> dict[str, Any]:
    """Write the CSV and return a receipt of what was written."""
    destination = Path(destination).expanduser()
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} exists; pass --force to replace it")
    scan = scan_stored_papers(settings, table=table)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(scan["rows"])
    written = destination.read_bytes()
    return {
        "output": str(destination),
        "rows": len(scan["rows"]),
        "bytes": len(written),
        "sha256": hashlib.sha256(written).hexdigest(),
        "columns": list(COLUMNS),
        "table": settings.aws_table,
        "scanned_count": scan["scanned_count"],
        "read_capacity_units": scan["read_capacity_units"],
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "basis": "catalogue rows with a pdf_key; no S3 object was opened",
    }
