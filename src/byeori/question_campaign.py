"""Durable dispatch bookkeeping for an AWS-resident list of research questions."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone

from botocore.exceptions import BotoCoreError, ClientError


MISSING = {"404", "NoSuchKey", "NotFound"}
CONFLICT = {"412", "PreconditionFailed", "ConditionalRequestConflict"}
AUTH_ERRORS = {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation", "ExpiredToken",
               "ExpiredTokenException", "InvalidAccessKeyId", "InvalidClientTokenId",
               "SignatureDoesNotMatch", "UnrecognizedClientException"}
MODEL_OPERATIONS = {"Converse", "ConverseStream", "InvokeModel", "InvokeModelWithResponseStream"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _location(event):
    run_id = event.get("run_id", "")
    if not isinstance(run_id, str) or len(run_id) > 100 or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", run_id):
        raise ValueError("run_id must be a lowercase slug of at most 100 characters")
    prefix = f"runs/questions/{run_id}"
    manifest_key = event.get("manifest_key", f"{prefix}/manifest.json")
    if manifest_key != f"{prefix}/manifest.json":
        raise ValueError("manifest_key must be this run's runs/questions/<run_id>/manifest.json")
    return run_id, manifest_key, prefix


def _item(value):
    if not isinstance(value, dict):
        raise ValueError("Each manifest item must be an object")
    ident = str(value.get("id", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", ident):
        raise ValueError("Question id must contain only letters, digits, underscores or hyphens")
    question, origin = value.get("question"), value.get("origin", "")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Every question must have nonempty text")
    if not isinstance(origin, str):
        raise ValueError("Question origin must be a string")
    return {"id": ident, "question": question, "origin": origin}


def _read(s3, bucket, key, *, optional=False):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if optional and exc.response.get("Error", {}).get("Code") in MISSING:
            return None
        raise
    body = response["Body"]
    try:
        return json.loads(body.read())
    finally:
        body.close()


def _create(s3, bucket, key, value):
    return s3.put_object(Bucket=bucket, Key=key,
                         Body=(json.dumps(value, ensure_ascii=False, default=str) + "\n").encode(),
                         ContentType="application/json", IfNoneMatch="*")


def _same_item(stored, item, run_id, manifest_key):
    if not isinstance(stored, dict) or any(stored.get(key) != value for key, value in
            {**item, "run_id": run_id, "manifest_key": manifest_key}.items()):
        raise ValueError("Question identity changed inside an existing campaign; keep the manifest immutable")


def _metadata(receipt, receipt_key, *, reused):
    result = receipt["result"]
    return {"id": receipt["id"], "status": result.get("status", "answer_failed"),
            "question_key": result.get("question_key"), "pages_count": len(result.get("pages_written") or []),
            "estimated_usd": result.get("estimated_usd"), "trace_key": result.get("trace_key"),
            "reused": reused}


def _receipt(s3, bucket, key, item, run_id, manifest_key):
    receipt = _read(s3, bucket, key, optional=True)
    if receipt is not None:
        _same_item(receipt, item, run_id, manifest_key)
        if receipt.get("terminal") is not True or not isinstance(receipt.get("result"), dict):
            raise ValueError("An existing question receipt is not terminal; inspect it before continuing")
    return receipt


def _manifest(s3, bucket, key):
    manifest = _read(s3, bucket, key)
    if not isinstance(manifest, list):
        raise ValueError("The manifest must be a JSON list of questions")
    manifest = [_item(row) for row in manifest]
    if len({row["id"] for row in manifest}) != len(manifest):
        raise ValueError("Question ids must be unique within a manifest")
    return manifest


def plan_batch(event, *, s3, bucket):
    """Read a manifest window and skip receipts or unresolved claims, without model calls."""
    run_id, manifest_key, prefix = _location(event)
    offset, batch_size, concurrency = event.get("offset", 0), event.get("batch_size", 24), event.get("concurrency", 6)
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if type(batch_size) is not int or not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and 100")
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError("concurrency must be between 1 and 32")
    manifest = _manifest(s3, bucket, manifest_key)
    total = len(manifest)
    if offset > total:
        raise ValueError("offset exceeds the manifest length")
    next_offset = min(offset + batch_size, total)
    items, skipped, claimed = [], [], []
    for item in manifest[offset:next_offset]:
        receipt_key = f"{prefix}/results/{item['id']}.json"
        receipt = _receipt(s3, bucket, receipt_key, item, run_id, manifest_key)
        if receipt is not None:
            skipped.append(_metadata(receipt, receipt_key, reused=True))
            continue
        claim_key = f"{prefix}/claims/{item['id']}.json"
        claim = _read(s3, bucket, claim_key, optional=True)
        if claim is not None:
            _same_item(claim, item, run_id, manifest_key)
            claimed.append({"id": item["id"], "status": "claimed", "claim_key": claim_key})
            continue
        items.append({"run_id": run_id, "manifest_key": manifest_key, **item})
    return {"run_id": run_id, "manifest_key": manifest_key, "offset": offset, "next_offset": next_offset,
            "total": total, "items": items, "done": offset >= total,
            "batch_size": batch_size, "concurrency": concurrency, "skipped": skipped, "claimed": claimed}


def run_one(event, *, s3, bucket, answer_callback):
    """Claim one question, invoke the existing answer path once, and retain its full result in S3."""
    run_id, manifest_key, prefix = _location(event)
    item = _item(event)
    receipt_key = f"{prefix}/results/{item['id']}.json"
    receipt = _receipt(s3, bucket, receipt_key, item, run_id, manifest_key)
    if receipt is not None:
        return _metadata(receipt, receipt_key, reused=True)
    claim_key = f"{prefix}/claims/{item['id']}.json"
    identity = {"run_id": run_id, "manifest_key": manifest_key, **item}
    try:
        _create(s3, bucket, claim_key, {**identity, "claimed_at": _now()})
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in CONFLICT:
            raise
        receipt = _receipt(s3, bucket, receipt_key, item, run_id, manifest_key)
        if receipt is not None:
            return _metadata(receipt, receipt_key, reused=True)
        claim = _read(s3, bucket, claim_key)
        _same_item(claim, item, run_id, manifest_key)
        return {"id": item["id"], "status": "claimed",
                "question_key": None, "pages_count": 0, "estimated_usd": None, "trace_key": None,
                "reused": True}

    try:
        result = answer_callback({"action": "answer_question", "title": item["question"]})
        if not isinstance(result, dict):
            raise ValueError("The answer callback returned a non-object result")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in AUTH_ERRORS or exc.operation_name not in MODEL_OPERATIONS:
            raise
        result = {"status": "answer_failed", "error": str(exc), "error_type": type(exc).__name__}
    except BotoCoreError:
        # Transport errors may belong to storage, so do not disguise them as a model result.
        raise
    except Exception as exc:
        result = {"status": "answer_failed", "error": str(exc), "error_type": type(exc).__name__}
    receipt = {**identity, "terminal": True, "finished_at": _now(), "result": result}
    _create(s3, bucket, receipt_key, receipt)
    return _metadata(receipt, receipt_key, reused=False)


def progress(event, *, s3, bucket):
    """Aggregate durable receipts inside AWS; never return answer or scientific page bodies."""
    run_id, manifest_key, prefix = _location(event)
    manifest = _manifest(s3, bucket, manifest_key)
    expected = {item["id"]: item for item in manifest}
    receipts, claimed_ids = [], set()
    paginator = s3.get_paginator("list_objects_v2")
    for folder in ("results", "claims"):
        directory = f"{prefix}/{folder}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=directory):
            for obj in page.get("Contents", []):
                ident = obj["Key"][len(directory):].removesuffix(".json")
                if ident not in expected or obj["Key"] != f"{directory}{ident}.json":
                    continue
                if folder == "results":
                    receipt = _receipt(s3, bucket, obj["Key"], expected[ident], run_id, manifest_key)
                    if receipt is not None:
                        receipts.append(receipt)
                else:
                    claimed_ids.add(ident)
    completed_ids = {receipt["id"] for receipt in receipts}
    claimed_ids -= completed_ids
    statuses, page_keys = Counter(), set()
    answer_saved = pages_created = pages_updated = cost_unavailable = 0
    total_cost = total_seconds = 0.0
    for receipt in receipts:
        result = receipt["result"]
        statuses[str(result.get("status", "answer_failed"))] += 1
        answer_saved += bool(result.get("question_key") and str(result.get("answer") or "").strip())
        cost = result.get("estimated_usd")
        cost_unavailable += cost is None
        total_cost += float(cost or 0)
        total_seconds += float(result.get("seconds") or 0)
        for page in result.get("pages_written") or []:
            page_keys.add(page["key"])
            pages_created += page.get("replaced") is False
            pages_updated += page.get("replaced") is True
    recent = sorted(receipts, key=lambda receipt: (receipt.get("finished_at", ""), receipt["id"]), reverse=True)[:5]
    return {"run_id": run_id, "total": len(manifest), "completed": len(receipts),
            "claimed": len(claimed_ids), "not_started": len(manifest) - len(receipts) - len(claimed_ids),
            "statuses": dict(statuses), "answer_saved": answer_saved,
            "total_estimated_usd": round(total_cost, 5), "cost_unavailable": cost_unavailable,
            "total_seconds": round(total_seconds, 1), "pages_created": pages_created,
            "pages_updated": pages_updated, "distinct_pages": len(page_keys),
            "recent": [_metadata(receipt, "", reused=True) for receipt in recent]}
