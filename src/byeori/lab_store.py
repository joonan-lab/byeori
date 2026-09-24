"""Control records and receipts for the student question workflow (docs/LAB-QUESTION-WORKFLOW.md).

This is the only module that knows DynamoDB expressions. Every other lab module talks to a
``TablePort``: get, create-only put, revision-checked update, revision-checked delete, prefix
query and an all-or-nothing transaction over those operations. ``DynamoTable`` implements the
port with the boto3 client; ``tests/lab_fakes.MemoryTable`` implements it in memory with the same
conditional semantics.

Receipts go to S3 under ``runs/lab-questions/{job_id}/`` only, written once. This module never
touches ``wiki/``, ``papers/`` or ``index/`` and never imports the campaign's Lambda code.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from botocore.exceptions import ClientError

RECEIPT_PREFIX = "runs/lab-questions/"
FORBIDDEN_WRITE_PREFIXES = ("wiki/", "papers/", "index/", "sources/", "runs/questions/", "runs/agents/")
# The one wiki prefix the student path may write (user, 2026-09-22). Answers are kept as Markdown
# so a reader can follow them, and the index builder skips this prefix, so a question never
# competes with a source note or a synthesis page for a search result. Scientific pages stay
# closed to this path: ``PageWriter`` refuses every key outside it, and nothing here edits
# ``wiki/sources/``, ``wiki/overviews/``, ``wiki/concepts/`` or ``wiki/questions/``.
ANSWER_PAGE_PREFIX = "wiki/lab-questions/"
MISSING_OBJECT = frozenset({"NoSuchKey", "404", "NotFound"})
PAGE_CONFLICT = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})


class ConditionFailed(Exception):
    """A create-only put found an existing item, or a revision check did not match."""


class TransactionConflict(ConditionFailed):
    """DynamoDB cancelled the write because another transaction held the same item.

    botocore does not retry this code. It is a ``ConditionFailed`` so the bounded re-read and
    re-plan loops the lab modules already run around hot items cover it without new handlers.
    """


class StoreError(Exception):
    """Base for lab store failures that carry a stable ``code``."""

    code = "store_error"

    def __init__(self, message: str = "", *, code: str | None = None):
        super().__init__(message or self.__class__.__name__)
        if code:
            self.code = code


@dataclass(frozen=True)
class Put:
    """Create ``item`` only if no item with the same ``pk``/``sk`` exists."""

    item: dict[str, Any]


@dataclass(frozen=True)
class Update:
    """Set ``changes`` on an existing item whose ``revision`` equals ``expected_revision``.

    The store bumps ``revision`` by one and sets ``updated_at``; callers never set those.
    """

    pk: str
    sk: str
    expected_revision: int
    changes: dict[str, Any]


@dataclass(frozen=True)
class Check:
    """Require that an item exists with ``expected_revision`` (no write)."""

    pk: str
    sk: str
    expected_revision: int


@dataclass(frozen=True)
class Delete:
    """Remove an existing item whose ``revision`` equals ``expected_revision``.

    Only finished pointer rows (a sent outbox pointer, for example) are deleted; records that
    carry state or consent are closed with an ``Update`` so their history stays readable.
    """

    pk: str
    sk: str
    expected_revision: int


Operation = Put | Update | Check | Delete


class TablePort(Protocol):
    def get(self, pk: str, sk: str) -> dict[str, Any] | None: ...

    def put(self, item: dict[str, Any]) -> dict[str, Any]: ...

    def update(self, pk: str, sk: str, expected_revision: int, changes: dict[str, Any]) -> dict[str, Any]: ...

    def delete(self, pk: str, sk: str, expected_revision: int) -> None: ...

    def query(self, pk: str, *, sk_prefix: str = "", limit: int = 100, start_after: str | None = None,
              ascending: bool = True) -> tuple[list[dict[str, Any]], str | None]: ...

    def transact(self, operations: list[Operation]) -> None: ...


# ---------------------------------------------------------------------------------------------
# Canonical encoding, identifiers and time
# ---------------------------------------------------------------------------------------------

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def now_iso(now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def new_id() -> str:
    return uuid.uuid4().hex


def period_for(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m")


def day_for(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%d")


def question_hash(question: str, context_hash: str) -> str:
    """Hash the exact question with its conversation context; never normalise away conditions."""
    return digest({"question": question, "context_hash": context_hash})


def context_hash(context: list[dict[str, str]] | None) -> str:
    return digest(context or [])


# ---------------------------------------------------------------------------------------------
# Record keys (see the plan's record table)
# ---------------------------------------------------------------------------------------------

class keys:
    @staticmethod
    def member(member_id: str) -> tuple[str, str]:
        return f"MEMBER#{member_id}", "PROFILE"

    @staticmethod
    def principal(principal_id: str) -> tuple[str, str]:
        return f"PRINCIPAL#{principal_id}", "MEMBER"

    @staticmethod
    def idempotency(member_id: str, kind: str, request_id: str) -> tuple[str, str]:
        return f"IDEMP#{member_id}#{kind}#{request_id}", "KEY"

    @staticmethod
    def job(job_id: str) -> tuple[str, str]:
        return f"JOB#{job_id}", "META"

    @staticmethod
    def member_job(member_id: str, created_at: str, job_id: str) -> tuple[str, str]:
        return f"MEMBER#{member_id}", f"JOB#{created_at}#{job_id}"

    @staticmethod
    def day_record(day: str, created_at: str, job_id: str) -> tuple[str, str]:
        return f"RECORDS#{day}", f"{created_at}#{job_id}"

    @staticmethod
    def session(session_id: str) -> tuple[str, str]:
        return f"SESSION#{session_id}", "META"

    @staticmethod
    def verdict(job_id: str) -> tuple[str, str]:
        return f"JOB#{job_id}", "TRIAGE"

    @staticmethod
    def verdict_by_input(input_hash: str) -> tuple[str, str]:
        return f"VERDICT#{input_hash}", "META"

    @staticmethod
    def offer(offer_id: str) -> tuple[str, str]:
        return f"OFFER#{offer_id}", "META"

    @staticmethod
    def job_offer(job_id: str, offer_id: str) -> tuple[str, str]:
        return f"JOB#{job_id}", f"OFFER#{offer_id}"

    @staticmethod
    def approval(approval_id: str) -> tuple[str, str]:
        return f"APPROVAL#{approval_id}", "META"

    @staticmethod
    def outbox(outbox_id: str) -> tuple[str, str]:
        return f"OUTBOX#{outbox_id}", "META"

    @staticmethod
    def outbox_pending(created_at: str, outbox_id: str) -> tuple[str, str]:
        return "OUTBOX", f"PENDING#{created_at}#{outbox_id}"

    @staticmethod
    def budget(scope: str) -> tuple[str, str]:
        return f"BUDGET#{scope}", "META"

    @staticmethod
    def reservation(reservation_id: str) -> tuple[str, str]:
        return f"RESERVATION#{reservation_id}", "META"

    @staticmethod
    def candidate(candidate_id: str) -> tuple[str, str]:
        return f"CANDIDATE#{candidate_id}", "META"

    @staticmethod
    def candidate_pointer(created_at: str, candidate_id: str) -> tuple[str, str]:
        return "CANDIDATES", f"{created_at}#{candidate_id}"

    @staticmethod
    def paper_request(request_id: str) -> tuple[str, str]:
        return f"PAPERREQ#{request_id}", "META"

    @staticmethod
    def paper_request_pointer(created_at: str, request_id: str) -> tuple[str, str]:
        return "PAPERREQS#ALL", f"REQ#{created_at}#{request_id}"

    @staticmethod
    def paper_request_doi(doi: str) -> tuple[str, str]:
        return f"PAPERREQDOI#{doi}", "META"

    @staticmethod
    def round(round_id: str) -> tuple[str, str]:
        return f"ROUND#{round_id}", "META"

    # One row per question the wiki could not answer, whether or not the member asked for
    # collection (user, 2026-09-22). The pointer is what the professor's queue reads.
    @staticmethod
    def collection_gap(job_id: str) -> tuple[str, str]:
        return f"COLLECT#{job_id}", "META"

    @staticmethod
    def collection_gap_pointer(created_at: str, job_id: str) -> tuple[str, str]:
        return "COLLECT#ALL", f"GAP#{created_at}#{job_id}"


def new_item(pk: str, sk: str, created_at: str, **attributes: Any) -> dict[str, Any]:
    """A fresh record: revision 1, created/updated stamps, plus the caller's attributes."""
    return {"pk": pk, "sk": sk, "revision": 1, "created_at": created_at, "updated_at": created_at, **attributes}


# ---------------------------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------------------------

def receipt_key(job_id: str, name: str) -> str:
    if not job_id or "/" in job_id or "/" in name or not name.endswith(".json"):
        raise ValueError("receipt names are simple JSON file names under one job")
    return f"{RECEIPT_PREFIX}{job_id}/{name}"


class ReceiptWriter:
    """Write immutable JSON receipts under runs/lab-questions/ and nothing else."""

    def __init__(self, s3, bucket: str):
        self.s3, self.bucket = s3, bucket
        self.written: list[str] = []

    def put_json(self, key: str, value: Any) -> dict[str, Any]:
        if not key.startswith(RECEIPT_PREFIX) or key.startswith(FORBIDDEN_WRITE_PREFIXES) or ".." in key:
            raise ValueError(f"Receipts may only be written under {RECEIPT_PREFIX}: {key}")
        body = canonical(value)
        response = self.s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                                      ContentType="application/json", IfNoneMatch="*")
        self.written.append(key)
        return {"key": key, "sha256": digest_bytes(body), "bytes": len(body),
                "etag": response.get("ETag", ""), "version_id": response.get("VersionId")}

    def get_json(self, key: str) -> Any:
        if not key.startswith(RECEIPT_PREFIX):
            raise ValueError(f"Receipts live under {RECEIPT_PREFIX}: {key}")
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"]
        try:
            return json.loads(body.read().decode("utf-8"))
        finally:
            body.close()


class PageWriter:
    """Write Markdown under wiki/lab-questions/ and nothing else.

    Separate from ``ReceiptWriter`` because the two have opposite rules: a receipt is immutable
    JSON created once, while a page here is Markdown that may be replaced under the ETag the
    caller read. Both refuse everything outside their one prefix, so the answer worker still
    cannot reach a scientific page, an original or the index.
    """

    def __init__(self, s3, bucket: str):
        self.s3, self.bucket = s3, bucket
        self.written: list[str] = []

    def _check(self, key: str) -> None:
        if not isinstance(key, str) or not key.startswith(ANSWER_PAGE_PREFIX) or ".." in key:
            raise ValueError(f"Lab question pages live under {ANSWER_PAGE_PREFIX}: {key!r}")
        if not key.endswith(".md"):
            raise ValueError(f"Lab question pages are Markdown: {key!r}")

    def get_markdown(self, key: str) -> tuple[str, str] | None:
        """The page's text and ETag, or ``None`` when it does not exist yet."""
        self._check(key)
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in MISSING_OBJECT:
                return None
            raise
        body = response["Body"]
        try:
            return body.read().decode("utf-8"), response.get("ETag", "")
        finally:
            body.close()

    def put_markdown(self, key: str, text: str, *, expected_etag: str | None = None,
                     create_only: bool = False) -> dict[str, Any]:
        """Write the page. ``create_only`` fails if it exists; ``expected_etag`` fails if it moved."""
        self._check(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"A lab question page needs Markdown: {key!r}")
        if create_only and expected_etag is not None:
            raise ValueError("Use either create_only or expected_etag")
        conditions: dict[str, str] = {}
        if create_only:
            conditions["IfNoneMatch"] = "*"
        elif expected_etag is not None:
            conditions["IfMatch"] = expected_etag
        body = text.encode("utf-8")
        try:
            response = self.s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                                          ContentType="text/markdown; charset=utf-8", **conditions)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in PAGE_CONFLICT:
                raise ConditionFailed(f"{key} changed while it was being written") from exc
            raise
        self.written.append(key)
        return {"key": key, "sha256": digest_bytes(body), "bytes": len(body),
                "etag": response.get("ETag", ""), "version_id": response.get("VersionId")}


# ---------------------------------------------------------------------------------------------
# DynamoDB implementation of the port (boto3 client API). Added by Task 4 with its tests.
# ---------------------------------------------------------------------------------------------

@dataclass
class DynamoTable:
    """Implements ``TablePort`` on a boto3 DynamoDB *client* and a table name."""

    client: Any
    table_name: str
    serializer: Any = field(default=None, repr=False)
    deserializer: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.serializer is None or self.deserializer is None:
            from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
            self.serializer, self.deserializer = TypeSerializer(), TypeDeserializer()

    def _to(self, item: dict[str, Any]) -> dict[str, Any]:
        return {k: self.serializer.serialize(_dynamo_safe(v)) for k, v in item.items()}

    def _from(self, item: dict[str, Any]) -> dict[str, Any]:
        return {k: _python_safe(self.deserializer.deserialize(v)) for k, v in item.items()}

    def get(self, pk, sk):
        response = self.client.get_item(TableName=self.table_name, Key=self._to({"pk": pk, "sk": sk}),
                                        ConsistentRead=True)
        item = response.get("Item")
        return self._from(item) if item else None

    def put(self, item):
        self._require_keys(item)
        try:
            self.client.put_item(TableName=self.table_name, Item=self._to(item),
                                 ConditionExpression="attribute_not_exists(pk)")
        except Exception as exc:  # noqa: BLE001
            _raise_condition(exc)
        return item

    def update(self, pk, sk, expected_revision, changes):
        self._require_changes(changes)
        names = {f"#a{i}": name for i, name in enumerate(changes)}
        values = {f":v{i}": _dynamo_safe(value) for i, value in enumerate(changes.values())}
        sets = ", ".join(f"{placeholder} = :v{i}" for i, placeholder in enumerate(names))
        expression = f"SET {sets}, #rev = #rev + :one, #upd = :updated"
        try:
            response = self.client.update_item(
                TableName=self.table_name, Key=self._to({"pk": pk, "sk": sk}),
                UpdateExpression=expression,
                ConditionExpression="attribute_exists(pk) AND #rev = :expected",
                ExpressionAttributeNames={**names, "#rev": "revision", "#upd": "updated_at"},
                ExpressionAttributeValues=self._to({**values, ":one": 1, ":expected": int(expected_revision),
                                                    ":updated": now_iso()}),
                ReturnValues="ALL_NEW")
        except Exception as exc:  # noqa: BLE001
            _raise_condition(exc)
        return self._from(response["Attributes"])

    def delete(self, pk, sk, expected_revision):
        try:
            self.client.delete_item(TableName=self.table_name, Key=self._to({"pk": pk, "sk": sk}),
                                    **self._revision_condition(expected_revision))
        except Exception as exc:  # noqa: BLE001
            _raise_condition(exc)

    def query(self, pk, *, sk_prefix="", limit=100, start_after=None, ascending=True):
        if not 1 <= int(limit) <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        request = {"TableName": self.table_name, "ConsistentRead": True, "Limit": int(limit),
                   "ScanIndexForward": bool(ascending)}
        values = {":pk": pk}
        if sk_prefix:
            request["KeyConditionExpression"] = "pk = :pk AND begins_with(sk, :prefix)"
            values[":prefix"] = sk_prefix
        else:
            request["KeyConditionExpression"] = "pk = :pk"
        request["ExpressionAttributeValues"] = self._to(values)
        if start_after is not None:
            request["ExclusiveStartKey"] = self._to({"pk": pk, "sk": start_after})
        response = self.client.query(**request)
        items = [self._from(item) for item in response.get("Items", [])]
        last = response.get("LastEvaluatedKey")
        return items, (self._from(last)["sk"] if last else None)

    def transact(self, operations):
        if not operations:
            raise ValueError("a transaction needs at least one operation")
        if len(operations) > 100:
            raise ValueError("DynamoDB transactions accept at most 100 items")
        entries = []
        for op in operations:
            if isinstance(op, Put):
                self._require_keys(op.item)
                entries.append({"Put": {"TableName": self.table_name, "Item": self._to(op.item),
                                        "ConditionExpression": "attribute_not_exists(pk)"}})
            elif isinstance(op, Update):
                self._require_changes(op.changes)
                names = {f"#a{i}": name for i, name in enumerate(op.changes)}
                values = {f":v{i}": _dynamo_safe(value) for i, value in enumerate(op.changes.values())}
                sets = ", ".join(f"{placeholder} = :v{i}" for i, placeholder in enumerate(names))
                entries.append({"Update": {
                    "TableName": self.table_name, "Key": self._to({"pk": op.pk, "sk": op.sk}),
                    "UpdateExpression": f"SET {sets}, #rev = #rev + :one, #upd = :updated",
                    "ConditionExpression": "attribute_exists(pk) AND #rev = :expected",
                    "ExpressionAttributeNames": {**names, "#rev": "revision", "#upd": "updated_at"},
                    "ExpressionAttributeValues": self._to({**values, ":one": 1, ":expected": int(op.expected_revision),
                                                           ":updated": now_iso()})}})
            elif isinstance(op, Check):
                entries.append({"ConditionCheck": {
                    "TableName": self.table_name, "Key": self._to({"pk": op.pk, "sk": op.sk}),
                    **self._revision_condition(op.expected_revision)}})
            elif isinstance(op, Delete):
                entries.append({"Delete": {
                    "TableName": self.table_name, "Key": self._to({"pk": op.pk, "sk": op.sk}),
                    **self._revision_condition(op.expected_revision)}})
            else:
                raise TypeError(f"Unsupported transaction operation: {op!r}")
        try:
            self.client.transact_write_items(TransactItems=entries)
        except Exception as exc:  # noqa: BLE001
            _raise_condition(exc)

    def _revision_condition(self, expected_revision):
        """The condition shared by Check, Delete and delete_item: the item exists at this revision."""
        return {"ConditionExpression": "attribute_exists(pk) AND #rev = :expected",
                "ExpressionAttributeNames": {"#rev": "revision"},
                "ExpressionAttributeValues": self._to({":expected": int(expected_revision)})}

    @staticmethod
    def _require_keys(item):
        if not isinstance(item.get("pk"), str) or not isinstance(item.get("sk"), str):
            raise ValueError("items need string pk and sk")
        if item.get("revision") != 1:
            raise ValueError("new items start at revision 1")

    @staticmethod
    def _require_changes(changes):
        if not changes:
            raise ValueError("an update needs at least one change")
        if {"pk", "sk", "revision", "updated_at"} & set(changes):
            raise ValueError("pk, sk, revision and updated_at are managed by the store")


def _dynamo_safe(value):
    """DynamoDB rejects floats; money is already integer micro-USD, probabilities become strings."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str, bytes)):
        return value
    if isinstance(value, float):
        return {"__float__": repr(value)}
    if isinstance(value, dict):
        return {str(k): _dynamo_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dynamo_safe(v) for v in value]
    raise TypeError(f"Unsupported attribute type: {type(value).__name__}")


def _python_safe(value):
    from decimal import Decimal
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        return {k: _python_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_python_safe(v) for v in value]
    return value


def _raise_condition(exc):
    """Map DynamoDB's conditional outcomes to the port's exceptions; anything else propagates.

    ``ConditionalCheckFailedException`` on a single item, or a cancelled transaction whose reasons
    include ``ConditionalCheckFailed``, becomes ``ConditionFailed``: a logical outcome the caller
    re-reads and re-plans around. ``TransactionConflictException`` on a single item, or a
    cancellation whose reasons include ``TransactionConflict``, becomes ``TransactionConflict``:
    another transaction held the item, botocore does not retry the code, and the same bounded
    re-plan loop is the right response. When one item failed its condition and another saw a
    conflict, the definite condition failure wins. ThrottlingError, ProvisionedThroughputExceeded
    and ValidationError reasons stay the raw ``ClientError`` for the caller (or SQS) to retry or
    surface.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        raise exc
    code = response.get("Error", {}).get("Code", "")
    reasons = {reason.get("Code") for reason in response.get("CancellationReasons") or [] if isinstance(reason, dict)}
    if code == "ConditionalCheckFailedException":
        raise ConditionFailed(code) from exc
    if code == "TransactionConflictException":
        raise TransactionConflict(code) from exc
    if code == "TransactionCanceledException":
        if "ConditionalCheckFailed" in reasons:
            raise ConditionFailed(code) from exc
        if "TransactionConflict" in reasons:
            raise TransactionConflict(code) from exc
    raise exc
