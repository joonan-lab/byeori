"""Bounded OpenAlex matching steps with the continuation stored only in S3."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import uuid
from datetime import UTC, datetime
from typing import Any

from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

from . import openalex_match as matcher

MAX_SINGLETONS = 4
MAX_ATTEMPTS = 3
COUNTERS = ("catalogued", "selected", "without_doi", "skipped_matched", "matched",
            "unmatched", "errors", "conflicts", "requests", "disagreements")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _options(event: dict[str, Any]) -> dict[str, Any]:
    batch, limit = int(event.get("batch", 50)), int(event.get("limit", 0))
    if not 1 <= batch <= 50 or limit < 0:
        raise ValueError("batch must be 1..50 and limit must be nonnegative")
    return {"batch": batch, "limit": limit, "refresh": bool(event.get("refresh", False))}


def _scan(table: Any, state: dict[str, Any], *, size: int) -> dict[str, Any]:
    request: dict[str, Any] = {
        "FilterExpression": Attr("id_kind").eq("stem"), "Limit": size,
        "ProjectionExpression": "work_id, doi, pmid, pmcid, openalex_id, openalex_status, #t, #y",
        "ExpressionAttributeNames": {"#t": "title", "#y": "year"},
    }
    if state.get("cursor"):
        request["ExclusiveStartKey"] = state["cursor"]
    return table.scan(**request)


def _select(rows: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    pending = []
    for row in rows:
        state["catalogued"] += 1
        if not matcher.normalize_doi(row.get("doi")):
            state["without_doi"] += 1
        elif not state["refresh"] and row.get("openalex_status") == "matched":
            state["skipped_matched"] += 1
        elif not state["limit"] or state["selected"] < state["limit"]:
            pending.append(row)
            state["selected"] += 1
    return pending


def plan_match(event: dict[str, Any], *, table: Any) -> dict[str, Any]:
    """Read-only AWS selection summary. Never fetch OpenAlex or persist a run."""
    state = {**dict.fromkeys(COUNTERS, 0), **_options(event)}
    while True:
        page = _scan(table, state, size=500)
        _select(page.get("Items", []), state)
        state["cursor"] = page.get("LastEvaluatedKey")
        if not state["cursor"]:
            break
    return {**{k: state[k] for k in COUNTERS[:4]}, **_options(event), "dry_run": True}


def _key(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("run_id must contain 1..100 letters, digits, underscores or hyphens")
    return f"runs/openalex-match/{run_id}.json"


def _summary(state: dict[str, Any], key: str) -> dict[str, Any]:
    wait = max(1, math.ceil(state.get("blocked_until", 0) - time.time()))
    return {"run_id": state["run_id"], "done": state.get("done", False),
            "wait_seconds": 0 if state.get("done") else wait, "progress_key": key,
            "started_at": state["started_at"], "finished_at": state.get("finished_at"),
            **{name: state[name] for name in COUNTERS}, "pending": len(state["pending"])}


def _save(state: dict[str, Any], *, s3: Any, bucket: str, key: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, ContentType="application/json",
                  Body=json.dumps(state, default=str).encode("utf-8"))


def _credit_wait(state: dict[str, Any]) -> None:
    remaining = matcher.RATE.get("remaining")
    if remaining is not None and remaining <= matcher.CREDIT_FLOOR:
        reset = matcher.RATE.get("reset", 60)
        # Responses have used seconds; accept an absolute epoch as well.
        seconds = reset - time.time() if reset > 1_000_000_000 else reset
        state["blocked_until"] = time.time() + max(1, seconds) + 2
    retry_after = matcher.RATE.get("retry_after")
    if retry_after is not None:
        state["blocked_until"] = max(state.get("blocked_until", 0), time.time() + max(1, retry_after))


def _error(state: dict[str, Any], paper: dict[str, Any], message: str) -> None:
    state["errors"] += 1
    # Retain bounded diagnostics in S3, never a local candidate catalogue.
    state["error_details"] = (state.get("error_details", []) + [
        {"stem": paper["work_id"], "doi": paper["doi"], "error": message[:300]}
    ])[-100:]


def _conflict(state: dict[str, Any], paper: dict[str, Any], reason: str, **details: Any) -> None:
    state["conflicts"] += 1
    state["conflict_details"] = (state.get("conflict_details", []) + [
        {"stem": str(paper["work_id"])[:200], "doi": str(paper["doi"])[:300], "reason": reason, **details}
    ])[-100:]


def _retry(state: dict[str, Any], exc: Exception) -> bool:
    state["attempt"] = state.get("attempt", 0) + 1
    retryable = (isinstance(exc, (urllib.error.URLError, TimeoutError)) or
                 isinstance(exc, matcher.OpenAlexRequestError) and exc.retryable)
    _credit_wait(state)
    if retryable and state["attempt"] < MAX_ATTEMPTS:
        state["blocked_until"] = max(state.get("blocked_until", 0), time.time() + 2 ** state["attempt"])
        return True
    return False


def _record(table: Any, state: dict[str, Any], paper: dict[str, Any], work: dict[str, Any] | None) -> None:
    doi = matcher.normalize_doi(paper["doi"])
    if work and work.get("_ambiguous_openalex_ids"):
        identities = work["_ambiguous_openalex_ids"]
        _conflict(state, paper, "ambiguous_doi", openalex_ids=[str(value)[:120] for value in identities[:20]],
                  identity_count=len(identities))
        return
    if work is not None and (matcher.normalize_doi(work.get("doi")) != doi or
                             not matcher.normalize_openalex_id(work.get("id"))):
        _error(state, paper, "OpenAlex response identity does not match the requested DOI/work ID")
        return
    if work and paper.get("openalex_id") and (
            matcher.normalize_openalex_id(paper["openalex_id"]) != matcher.normalize_openalex_id(work["id"])):
        _conflict(state, paper, "openalex_id_mismatch", existing_openalex_id=str(paper["openalex_id"])[:120],
                  returned_openalex_id=matcher.normalize_openalex_id(work["id"])[:120])
        return
    fields = matcher.work_fields(work) if work else {"openalex_status": "unmatched", "openalex_matched_at": _now()}
    fields["openalex_match_run"] = state["run_id"]
    for name in ("pmid", "pmcid"):
        if not paper.get(name) and fields.get(f"{name}_openalex"):
            fields[name] = fields[f"{name}_openalex"]
    names = {f"#f{i}": field for i, field in enumerate(fields)}
    values = {f":v{i}": value for i, value in enumerate(fields.values())}
    updates = [f"#f{i} = :v{i}" for i, field in enumerate(fields)]
    names.update({"#doi": "doi", "#run": "openalex_match_run"})
    values.update({":doi": paper["doi"], ":run": state["run_id"]})
    condition = "attribute_exists(work_id) AND #doi = :doi AND (attribute_not_exists(#run) OR #run <> :run)"
    guarded_fields = (["openalex_id"] if work else []) + [field for field in ("pmid", "pmcid") if field in fields]
    for field in guarded_fields:
        names[f"#old_{field}"] = field
        if field in paper:
            values[f":old_{field}"] = paper[field]
            condition += f" AND #old_{field} = :old_{field}"
        else:
            condition += f" AND attribute_not_exists(#old_{field})"
    if not state["refresh"]:
        names["#status"], values[":matched"] = "openalex_status", "matched"
        condition += " AND (attribute_not_exists(#status) OR #status <> :matched)"
    try:
        table.update_item(Key={"work_id": paper["work_id"]}, UpdateExpression="SET " + ", ".join(updates),
                          ExpressionAttributeNames=names, ExpressionAttributeValues=values,
                          ConditionExpression=condition)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        current = table.get_item(Key={"work_id": paper["work_id"]}, ConsistentRead=True).get("Item", {})
        if current.get("openalex_match_run") != state["run_id"] or current.get("doi") != paper["doi"]:
            changes = {field: {"selected": str(paper.get(field, ""))[:120],
                               "current": str(current.get(field, ""))[:120]}
                       for field in ("doi", "openalex_id", "pmid", "pmcid")
                       if paper.get(field) != current.get(field)}
            _conflict(state, paper, "catalogue_changed", changes=changes)
            return
        # A previous attempt saved DynamoDB but failed to checkpoint S3. Count it once here.
        fields = current
    state["matched" if fields["openalex_status"] == "matched" else "unmatched"] += 1
    for field in ("title", "year", "pmid", "pmcid"):
        ours = paper.get(field)
        theirs = fields.get(f"{field}_openalex" if field in ("pmid", "pmcid") else f"openalex_{field}")
        if field in ("pmid", "pmcid"):
            # IDs are exact comparisons after removing their URL wrapper and case convention.
            normal_ours = str(ours or "").strip().rstrip("/").rsplit("/", 1)[-1].upper()
            normal_theirs = str(theirs or "").strip().rstrip("/").rsplit("/", 1)[-1].upper()
        else:
            normal_ours, normal_theirs = matcher._squash(ours), matcher._squash(theirs)
        if ours and theirs and normal_ours != normal_theirs:
            state["disagreements"] += 1
            state["disagreement_details"] = (state.get("disagreement_details", []) + [
                {"stem": paper["work_id"], "field": field, "ours": str(ours)[:120], "openalex": str(theirs)[:120]}
            ])[-100:]


def handle_match_batch(event: dict[str, Any], *, table: Any, s3: Any, bucket: str,
                       context: Any = None) -> dict[str, Any]:
    """Advance one scan/batch and at most four singleton lookups; Step Functions owns waiting."""
    run_id = str(event.get("run_id") or uuid.uuid4())
    key = _key(run_id)
    try:
        state = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise
        state = {"run_id": run_id, "started_at": _now(), **_options(event),
                 **dict.fromkeys(COUNTERS, 0), "pending": [], "scan_done": False}
    if state.get("done") or state.get("blocked_until", 0) > time.time():
        return _summary(state, key)
    matcher.RATE.clear()  # Lambda warm-container state never crosses runs or windows.
    state.pop("blocked_until", None)
    if not state["pending"] and not state["scan_done"]:
        page = _scan(table, state, size=state["batch"])
        state["pending"] = _select(page.get("Items", []), state)
        state["cursor"] = page.get("LastEvaluatedKey")
        state["scan_done"] = not state["cursor"] or bool(state["limit"] and state["selected"] >= state["limit"])
        state["needs_batch"] = bool(state["pending"])
        state["attempt"] = 0
        # Persist the selected page before writes so retries can recover partial progress.
        _save(state, s3=s3, bucket=bucket, key=key)
    if state["pending"] and state.get("needs_batch"):
        state["requests"] += 1
        try:
            found = matcher.fetch_batch([matcher.normalize_doi(p["doi"]) for p in state["pending"]], attempts=1)
        except (matcher.OpenAlexRequestError, urllib.error.URLError, TimeoutError) as exc:
            if not _retry(state, exc):
                for paper in state["pending"]:
                    _error(state, paper, str(exc))
                state["pending"] = []
        else:
            missed = []
            for paper in state["pending"]:
                work = found.get(matcher.normalize_doi(paper["doi"]))
                if work is None:
                    missed.append(paper)
                else:
                    _record(table, state, paper, work)
            state.update(pending=missed, needs_batch=False, attempt=0)
            _credit_wait(state)
    for _ in range(MAX_SINGLETONS):
        if not state["pending"] or state.get("needs_batch") or state.get("blocked_until", 0) > time.time():
            break
        if context and context.get_remaining_time_in_millis() < 65_000:
            break
        paper = state["pending"][0]
        state["requests"] += 1
        matcher.RATE.clear()
        try:
            work = matcher.fetch_one(matcher.normalize_doi(paper["doi"]), attempts=1)
        except (matcher.OpenAlexRequestError, urllib.error.URLError, TimeoutError) as exc:
            if _retry(state, exc):
                break
            _error(state, paper, str(exc))
        else:
            _record(table, state, paper, work)
            _credit_wait(state)
        state["pending"].pop(0)
        state["attempt"] = 0
    state["done"] = state["scan_done"] and not state["pending"]
    if state["done"]:
        state["finished_at"] = _now()
    _save(state, s3=s3, bucket=bucket, key=key)
    return _summary(state, key)
