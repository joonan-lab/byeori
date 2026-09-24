"""AWS-only reference intake and bounded synthesis failure reports."""
from __future__ import annotations

import hashlib
import urllib.request

from boto3.dynamodb.conditions import Attr

from .failure_reports import bounded_report, problem_detail

HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
HGNC_KEY = "reference/hgnc.tsv"


def reference(event, *, s3, bucket):
    url = event.get("url") or HGNC_URL
    if url != HGNC_URL:
        raise ValueError("Reference intake only accepts the configured HGNC complete-set URL")
    with urllib.request.urlopen(url, timeout=180) as response:
        raw = response.read(50_000_001)
    if len(raw) > 50_000_000:
        raise ValueError("HGNC reference exceeds the 50 MB input limit")
    lines = raw.decode("utf-8").splitlines()
    columns = {name: i for i, name in enumerate(lines[0].split("\t"))} if lines else {}
    wanted = ("symbol", "name", "alias_symbol", "prev_symbol")
    if any(name not in columns for name in wanted):
        raise ValueError("HGNC reference is missing required columns")
    kept = ["\t".join(wanted)]
    for line in lines[1:]:
        row = line.split("\t")
        if len(row) <= columns["symbol"] or not row[columns["symbol"]].strip():
            continue
        kept.append("\t".join(row[columns[name]] if len(row) > columns[name] else "" for name in wanted))
    if len(kept) < 1000:
        raise ValueError("HGNC complete set unexpectedly contains fewer than 999 genes")
    body = ("\n".join(kept) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    result = s3.put_object(Bucket=bucket, Key=HGNC_KEY, Body=body, ContentType="text/tab-separated-values",
                          Metadata={"sha256": digest})
    return {"key": HGNC_KEY, "genes": len(kept) - 1, "bytes": len(body), "sha256": digest,
            "source": url, "version_id": result.get("VersionId"), "execution": "aws"}


def failures(event, *, table):
    offset, limit = event.get("offset", 0), event.get("limit", 100)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("offset must be non-negative and limit must be 1 to 100")
    request = {"FilterExpression": Attr("id_kind").is_in(["concept", "subtopic", "category"]) &
                                   Attr("synthesis_status").eq("failed"),
               "ProjectionExpression": "work_id, id_kind, problems, calls, updated_at"}
    rows = []
    while True:
        response = table.scan(**request)
        rows.extend({"id": item["work_id"], "kind": item.get("id_kind"), "problems": list(item.get("problems") or []),
                     "calls": int(item.get("calls") or 0), "updated_at": item.get("updated_at")}
                    for item in response.get("Items", []))
        if not response.get("LastEvaluatedKey"):
            break
        request["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    rows.sort(key=lambda row: row["id"])
    if "problem_id" in event:
        return {**problem_detail(event, rows, id_key="id"), "execution": "aws"}
    return {**bounded_report(rows, offset=offset, limit=limit, verbose=bool(event.get("verbose")),
                             id_key="id", rows_key="pages"), "execution": "aws"}
