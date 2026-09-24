"""Port semantics, DynamoDB expression building and receipt boundaries of byeori.lab_store."""
from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta, timezone

import pytest
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from byeori import lab_store
from byeori.lab_store import (
    FORBIDDEN_WRITE_PREFIXES,
    RECEIPT_PREFIX,
    Check,
    ConditionFailed,
    Delete,
    DynamoTable,
    Put,
    ReceiptWriter,
    StoreError,
    TransactionConflict,
    Update,
    canonical,
    context_hash,
    day_for,
    digest,
    digest_bytes,
    keys,
    new_id,
    new_item,
    now_iso,
    period_for,
    question_hash,
    receipt_key,
)
from lab_fakes import MemoryS3, MemoryTable

STAMP = "2026-09-21T09:00:00+00:00"


def item(pk: str, sk: str, **attributes):
    return new_item(pk, sk, STAMP, **attributes)


# ---------------------------------------------------------------------------------------------
# Port semantics (checked against the in-memory implementation every other lab test uses)
# ---------------------------------------------------------------------------------------------

def test_put_of_existing_key_raises_condition_failed_and_keeps_first_item():
    table = MemoryTable()
    table.put(item("JOB#1", "META", status="received"))
    with pytest.raises(ConditionFailed):
        table.put(item("JOB#1", "META", status="queued"))
    assert table.get("JOB#1", "META")["status"] == "received"


def test_update_with_stale_revision_raises_and_leaves_item_unchanged():
    table = MemoryTable()
    table.put(item("JOB#1", "META", status="received"))
    updated = table.update("JOB#1", "META", 1, {"status": "queued"})
    assert updated["revision"] == 2 and updated["status"] == "queued"
    with pytest.raises(ConditionFailed):
        table.update("JOB#1", "META", 1, {"status": "running"})
    current = table.get("JOB#1", "META")
    assert current["status"] == "queued" and current["revision"] == 2
    assert current["created_at"] == STAMP


def test_update_bumps_revision_and_rejects_managed_fields():
    table = MemoryTable()
    table.put(item("JOB#1", "META", status="received"))
    for managed in ("pk", "sk", "revision", "updated_at"):
        with pytest.raises(ValueError):
            table.update("JOB#1", "META", 1, {managed: "x"})
    with pytest.raises(ValueError):
        table.update("JOB#1", "META", 1, {})
    assert table.get("JOB#1", "META")["revision"] == 1


def test_transact_is_all_or_nothing_when_a_check_fails():
    table = MemoryTable()
    table.put(item("JOB#1", "META", status="received"))
    with pytest.raises(ConditionFailed):
        table.transact([
            Put(item("OUTBOX#1", "META", kind="answer")),
            Update("JOB#1", "META", 1, {"status": "queued"}),
            Check("BUDGET#lab:2026-09", "META", 1),  # does not exist -> whole transaction fails
        ])
    assert table.get("OUTBOX#1", "META") is None
    assert table.get("JOB#1", "META")["status"] == "received"
    assert table.transactions == [[Put(item("JOB#1", "META", status="received"))]]


def test_transact_applies_every_operation_together():
    table = MemoryTable()
    table.put(item("JOB#1", "META", status="received"))
    table.put(item("MEMBER#m1", "PROFILE", active=True))
    table.transact([
        Check("MEMBER#m1", "PROFILE", 1),
        Update("JOB#1", "META", 1, {"status": "queued"}),
        Put(item("OUTBOX#1", "META", kind="answer")),
    ])
    assert table.get("JOB#1", "META")["status"] == "queued"
    assert table.get("OUTBOX#1", "META")["kind"] == "answer"
    with pytest.raises(ValueError):
        table.transact([])
    with pytest.raises(ValueError):
        table.transact([Put(item(f"X#{i}", "META")) for i in range(101)])


def test_query_by_pk_and_sk_prefix_paginates_in_sk_order():
    table = MemoryTable()
    for stamp in ("2026-09-21T09:00:02", "2026-09-21T09:00:00", "2026-09-21T09:00:01"):
        table.put(item("MEMBER#m1", f"JOB#{stamp}#{stamp[-1]}", job_id=stamp[-1]))
    table.put(item("MEMBER#m1", "PROFILE", member_id="m1"))
    table.put(item("MEMBER#m2", "JOB#2026-09-21T09:00:00#9", job_id="9"))

    first, cursor = table.query("MEMBER#m1", sk_prefix="JOB#", limit=2)
    assert [row["job_id"] for row in first] == ["0", "1"]
    assert cursor == "JOB#2026-09-21T09:00:01#1"
    second, end = table.query("MEMBER#m1", sk_prefix="JOB#", limit=2, start_after=cursor)
    assert [row["job_id"] for row in second] == ["2"] and end is None

    everything, _ = table.query("MEMBER#m1")
    assert [row["sk"] for row in everything][-1] == "PROFILE"
    newest, _ = table.query("MEMBER#m1", sk_prefix="JOB#", limit=1, ascending=False)
    assert newest[0]["job_id"] == "2"
    with pytest.raises(ValueError):
        table.query("MEMBER#m1", limit=0)


def test_query_returns_a_cursor_whenever_the_page_is_exactly_full():
    """DynamoDB sets LastEvaluatedKey when a page holds Limit items, even if nothing follows."""
    table = MemoryTable()
    for i in range(2):
        table.put(item("OUTBOX", f"PENDING#{i}", outbox_id=str(i)))
    page, cursor = table.query("OUTBOX", sk_prefix="PENDING#", limit=2)
    assert [row["outbox_id"] for row in page] == ["0", "1"] and cursor == "PENDING#1"
    rest, end = table.query("OUTBOX", sk_prefix="PENDING#", limit=2, start_after=cursor)
    assert rest == [] and end is None
    short, end = table.query("OUTBOX", sk_prefix="PENDING#", limit=3)
    assert len(short) == 2 and end is None
    newest, cursor = table.query("OUTBOX", sk_prefix="PENDING#", limit=1, ascending=False)
    assert newest[0]["outbox_id"] == "1" and cursor == "PENDING#1"
    older, end = table.query("OUTBOX", sk_prefix="PENDING#", limit=1, ascending=False, start_after=cursor)
    assert older[0]["outbox_id"] == "0" and end == "PENDING#0"
    empty, end = table.query("OUTBOX", sk_prefix="PENDING#", limit=1, ascending=False, start_after=end)
    assert empty == [] and end is None


def test_delete_removes_the_item_only_at_the_expected_revision():
    table = MemoryTable()
    table.put(item("OUTBOX", "PENDING#1#x1", outbox_id="x1"))
    table.update("OUTBOX", "PENDING#1#x1", 1, {"status": "sent"})
    with pytest.raises(ConditionFailed):
        table.delete("OUTBOX", "PENDING#1#x1", 1)
    assert table.get("OUTBOX", "PENDING#1#x1")["status"] == "sent"
    assert table.delete("OUTBOX", "PENDING#1#x1", 2) is None
    assert table.get("OUTBOX", "PENDING#1#x1") is None
    with pytest.raises(ConditionFailed):
        table.delete("OUTBOX", "PENDING#1#x1", 2)
    assert table.transactions[-1] == [Delete("OUTBOX", "PENDING#1#x1", 2)]


def test_transact_delete_is_all_or_nothing_and_applied_at_commit():
    table = MemoryTable()
    table.put(item("OUTBOX", "PENDING#1#x1", outbox_id="x1"))
    table.put(item("OUTBOX#x1", "META", status="pending"))
    with pytest.raises(ConditionFailed):
        table.transact([
            Delete("OUTBOX", "PENDING#1#x1", 1),
            Update("OUTBOX#x1", "META", 2, {"status": "done"}),  # stale revision -> nothing is deleted
        ])
    assert table.get("OUTBOX", "PENDING#1#x1")["outbox_id"] == "x1"
    with pytest.raises(ConditionFailed):
        table.transact([Update("OUTBOX#x1", "META", 1, {"status": "done"}), Delete("OUTBOX", "PENDING#0#x0", 1)])
    assert table.get("OUTBOX#x1", "META")["status"] == "pending"
    table.transact([
        Delete("OUTBOX", "PENDING#1#x1", 1),
        Update("OUTBOX#x1", "META", 1, {"status": "done"}),
        Put(item("OUTBOX", "PENDING#2#x2", outbox_id="x2")),
    ])
    assert table.get("OUTBOX", "PENDING#1#x1") is None
    assert table.get("OUTBOX#x1", "META")["status"] == "done"
    assert table.get("OUTBOX", "PENDING#2#x2")["outbox_id"] == "x2"
    with pytest.raises(ValueError):
        table.transact([Delete("OUTBOX#x1", "META", 2), Update("OUTBOX#x1", "META", 2, {"status": "x"})])
    with pytest.raises(ValueError):
        table.transact([Delete("OUTBOX", "PENDING#2#x2", 1), Put(item("OUTBOX", "PENDING#2#x2", outbox_id="x2"))])
    assert table.get("OUTBOX#x1", "META")["status"] == "done"
    assert table.get("OUTBOX", "PENDING#2#x2")["revision"] == 1


def test_get_returns_copies_so_callers_cannot_mutate_the_store():
    table = MemoryTable()
    table.put(item("JOB#1", "META", usage={"inputTokens": 1}))
    fetched = table.get("JOB#1", "META")
    fetched["usage"]["inputTokens"] = 999
    assert table.get("JOB#1", "META")["usage"]["inputTokens"] == 1
    assert table.get("JOB#missing", "META") is None


# ---------------------------------------------------------------------------------------------
# Fakes self-test: the S3 409 path callers use to prove their conditional-write handling
# ---------------------------------------------------------------------------------------------

def test_memory_s3_conflict_keys_raise_409_once_then_accept_the_write():
    key = "runs/lab-questions/job1/answer.json"
    s3 = MemoryS3(conflict_keys={key})
    with pytest.raises(ClientError) as failure:
        s3.put_object(Bucket="b", Key=key, Body=b"{}", IfNoneMatch="*")
    assert failure.value.response["Error"]["Code"] == "ConditionalRequestConflict"
    assert failure.value.response["ResponseMetadata"]["HTTPStatusCode"] == 409
    assert key not in s3.objects and s3.writes == [] and s3.conflict_keys == set()
    s3.put_object(Bucket="b", Key=key, Body=b"{}", IfNoneMatch="*")
    assert s3.objects[key] == b"{}" and s3.writes == [(key, {"IfNoneMatch": "*"})]
    with pytest.raises(ClientError) as repeat:
        s3.put_object(Bucket="b", Key=key, Body=b"{}", IfNoneMatch="*")
    assert repeat.value.response["Error"]["Code"] == "PreconditionFailed"
    other = "runs/lab-questions/job1/evidence.json"
    s3.put_object(Bucket="b", Key=other, Body=b"1", IfNoneMatch="*")
    assert s3.objects[other] == b"1"
    plain = MemoryS3()
    plain.put_object(Bucket="b", Key=key, Body=b"x", IfNoneMatch="*")
    assert plain.writes == [(key, {"IfNoneMatch": "*"})] and plain.conflict_keys == set()


def test_receipt_writer_surfaces_the_409_conflict_without_recording_a_write():
    key = receipt_key("job1", "answer.json")
    s3 = MemoryS3(conflict_keys={key})
    writer = ReceiptWriter(s3, "bucket")
    with pytest.raises(ClientError) as failure:
        writer.put_json(key, {"answer": "결과"})
    assert failure.value.response["ResponseMetadata"]["HTTPStatusCode"] == 409
    assert writer.written == [] and s3.writes == []
    result = writer.put_json(key, {"answer": "결과"})
    assert result["version_id"] == "v1" and writer.written == [key]
    assert writer.get_json(key) == {"answer": "결과"}


# ---------------------------------------------------------------------------------------------
# DynamoTable: exact boto3 client calls against a recording stub
# ---------------------------------------------------------------------------------------------

class RecordingClient:
    """Records every boto3 client call; ``failures`` are raised in order, one per call."""

    def __init__(self, *, failures=(), items=None, attributes=None, query_result=None):
        self.calls: list[tuple[str, dict]] = []
        self.failures = list(failures)
        self.items = items or {}
        self.attributes = attributes
        self.query_result = query_result or {}

    def _record(self, name, request):
        self.calls.append((name, request))
        if self.failures:
            raise self.failures.pop(0)

    def put_item(self, **request):
        self._record("put_item", request)
        return {}

    def update_item(self, **request):
        self._record("update_item", request)
        return {"Attributes": self.attributes}

    def transact_write_items(self, **request):
        self._record("transact_write_items", request)
        return {}

    def delete_item(self, **request):
        self._record("delete_item", request)
        return {}

    def get_item(self, **request):
        self._record("get_item", request)
        key = (request["Key"]["pk"]["S"], request["Key"]["sk"]["S"])
        return {"Item": self.items[key]} if key in self.items else {}

    def query(self, **request):
        self._record("query", request)
        return self.query_result


def serialize(value: dict) -> dict:
    return {k: TypeSerializer().serialize(v) for k, v in value.items()}


def client_error(code: str, operation: str, **extra) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}, **extra}, operation)


def test_dynamo_put_uses_create_only_condition_and_serializes_the_item():
    client = RecordingClient()
    table = DynamoTable(client, "control")
    stored = table.put(item("JOB#1", "META", status="received", usage=None, active=True, tokens=[1, 2]))
    assert stored["status"] == "received"
    name, request = client.calls[0]
    assert name == "put_item"
    assert request["TableName"] == "control"
    assert request["ConditionExpression"] == "attribute_not_exists(pk)"
    assert request["Item"]["pk"] == {"S": "JOB#1"} and request["Item"]["sk"] == {"S": "META"}
    assert request["Item"]["revision"] == {"N": "1"}
    assert request["Item"]["usage"] == {"NULL": True}
    assert request["Item"]["active"] == {"BOOL": True}
    assert request["Item"]["tokens"] == {"L": [{"N": "1"}, {"N": "2"}]}


def test_dynamo_put_requires_string_keys_and_revision_one():
    table = DynamoTable(RecordingClient(), "control")
    with pytest.raises(ValueError):
        table.put({"pk": "JOB#1", "sk": "META", "revision": 2})
    with pytest.raises(ValueError):
        table.put({"pk": 1, "sk": "META", "revision": 1})
    assert table.client.calls == []


def test_dynamo_update_checks_revision_and_bumps_it_in_the_expression():
    attributes = serialize({"pk": "JOB#1", "sk": "META", "revision": 2, "status": "queued", "updated_at": STAMP})
    client = RecordingClient(attributes=attributes)
    table = DynamoTable(client, "control")
    result = table.update("JOB#1", "META", 1, {"status": "queued", "lease_until": None})
    assert result == {"pk": "JOB#1", "sk": "META", "revision": 2, "status": "queued", "updated_at": STAMP}
    name, request = client.calls[0]
    assert name == "update_item"
    assert request["Key"] == {"pk": {"S": "JOB#1"}, "sk": {"S": "META"}}
    assert request["ConditionExpression"] == "attribute_exists(pk) AND #rev = :expected"
    assert request["ExpressionAttributeNames"]["#rev"] == "revision"
    assert request["ExpressionAttributeNames"]["#upd"] == "updated_at"
    assert request["ExpressionAttributeValues"][":expected"] == {"N": "1"}
    assert request["ExpressionAttributeValues"][":one"] == {"N": "1"}
    assert request["ReturnValues"] == "ALL_NEW"
    expression = request["UpdateExpression"]
    assert expression.startswith("SET ") and "#rev = #rev + :one" in expression and "#upd = :updated" in expression
    placeholders = dict(re.findall(r"(#a\d+) = (:v\d+)", expression))
    assert {request["ExpressionAttributeNames"][k] for k in placeholders} == {"status", "lease_until"}
    assert request["ExpressionAttributeValues"][":v1"] == {"NULL": True}
    with pytest.raises(ValueError):
        table.update("JOB#1", "META", 1, {"revision": 5})
    with pytest.raises(ValueError):
        table.update("JOB#1", "META", 1, {})


def test_dynamo_transact_builds_put_update_and_condition_check_entries():
    client = RecordingClient()
    table = DynamoTable(client, "control")
    table.transact([
        Put(item("OUTBOX#1", "META", kind="answer")),
        Update("JOB#1", "META", 3, {"status": "queued"}),
        Check("MEMBER#m1", "PROFILE", 1),
    ])
    name, request = client.calls[0]
    assert name == "transact_write_items"
    entries = request["TransactItems"]
    assert [list(entry)[0] for entry in entries] == ["Put", "Update", "ConditionCheck"]
    put = entries[0]["Put"]
    assert put["TableName"] == "control" and put["ConditionExpression"] == "attribute_not_exists(pk)"
    assert put["Item"]["kind"] == {"S": "answer"}
    update = entries[1]["Update"]
    assert update["Key"] == {"pk": {"S": "JOB#1"}, "sk": {"S": "META"}}
    assert update["ConditionExpression"] == "attribute_exists(pk) AND #rev = :expected"
    assert update["ExpressionAttributeValues"][":expected"] == {"N": "3"}
    assert "#rev = #rev + :one" in update["UpdateExpression"]
    check = entries[2]["ConditionCheck"]
    assert check["Key"] == {"pk": {"S": "MEMBER#m1"}, "sk": {"S": "PROFILE"}}
    assert check["ConditionExpression"] == "attribute_exists(pk) AND #rev = :expected"
    assert check["ExpressionAttributeNames"] == {"#rev": "revision"}
    assert check["ExpressionAttributeValues"] == {":expected": {"N": "1"}}


def test_dynamo_transact_builds_delete_entry_with_the_revision_condition():
    client = RecordingClient()
    table = DynamoTable(client, "control")
    table.transact([Delete("OUTBOX", "PENDING#1#x1", 4), Update("OUTBOX#x1", "META", 2, {"status": "done"})])
    name, request = client.calls[0]
    assert name == "transact_write_items"
    entries = request["TransactItems"]
    assert [list(entry)[0] for entry in entries] == ["Delete", "Update"]
    assert entries[0]["Delete"] == {
        "TableName": "control",
        "Key": {"pk": {"S": "OUTBOX"}, "sk": {"S": "PENDING#1#x1"}},
        "ConditionExpression": "attribute_exists(pk) AND #rev = :expected",
        "ExpressionAttributeNames": {"#rev": "revision"},
        "ExpressionAttributeValues": {":expected": {"N": "4"}},
    }


def test_dynamo_delete_uses_delete_item_with_the_revision_condition():
    client = RecordingClient()
    table = DynamoTable(client, "control")
    assert table.delete("OUTBOX", "PENDING#1#x1", 4) is None
    name, request = client.calls[0]
    assert name == "delete_item"
    assert request == {
        "TableName": "control",
        "Key": {"pk": {"S": "OUTBOX"}, "sk": {"S": "PENDING#1#x1"}},
        "ConditionExpression": "attribute_exists(pk) AND #rev = :expected",
        "ExpressionAttributeNames": {"#rev": "revision"},
        "ExpressionAttributeValues": {":expected": {"N": "4"}},
    }
    stale = DynamoTable(RecordingClient(failures=[client_error("ConditionalCheckFailedException", "DeleteItem")]), "control")
    with pytest.raises(ConditionFailed) as failure:
        stale.delete("OUTBOX", "PENDING#1#x1", 3)
    assert type(failure.value) is ConditionFailed and isinstance(failure.value.__cause__, ClientError)
    throttled = DynamoTable(RecordingClient(failures=[client_error("ProvisionedThroughputExceededException", "DeleteItem")]), "control")
    with pytest.raises(ClientError):
        throttled.delete("OUTBOX", "PENDING#1#x1", 3)


def test_dynamo_transact_rejects_bad_input_before_calling_aws():
    client = RecordingClient()
    table = DynamoTable(client, "control")
    with pytest.raises(ValueError):
        table.transact([])
    with pytest.raises(ValueError):
        table.transact([Put(item(f"X#{i}", "META")) for i in range(101)])
    with pytest.raises(TypeError):
        table.transact(["not an operation"])
    with pytest.raises(ValueError):
        table.transact([Update("JOB#1", "META", 1, {"pk": "other"})])
    assert client.calls == []


def test_dynamo_maps_conditional_check_failed_to_condition_failed():
    client = RecordingClient(failures=[client_error("ConditionalCheckFailedException", "PutItem"),
                                       client_error("ConditionalCheckFailedException", "UpdateItem")])
    table = DynamoTable(client, "control")
    with pytest.raises(ConditionFailed) as put_failure:
        table.put(item("JOB#1", "META"))
    assert isinstance(put_failure.value.__cause__, ClientError)
    with pytest.raises(ConditionFailed):
        table.update("JOB#1", "META", 1, {"status": "queued"})


def test_dynamo_maps_transaction_cancelled_with_cancellation_reasons():
    reasons = [{"Code": "None"}, {"Code": "ConditionalCheckFailed", "Message": "The conditional request failed"}]
    client = RecordingClient(failures=[client_error("TransactionCanceledException", "TransactWriteItems",
                                                    CancellationReasons=reasons)])
    table = DynamoTable(client, "control")
    with pytest.raises(ConditionFailed) as failure:
        table.transact([Put(item("A#1", "META")), Check("B#1", "META", 1)])
    assert "TransactionCanceledException" in str(failure.value)
    assert failure.value.__cause__.response["CancellationReasons"] == reasons


def test_dynamo_reraises_errors_that_are_not_condition_failures():
    client = RecordingClient(failures=[client_error("ProvisionedThroughputExceededException", "PutItem"),
                                       client_error("ValidationException", "TransactWriteItems")])
    table = DynamoTable(client, "control")
    with pytest.raises(ClientError):
        table.put(item("JOB#1", "META"))
    with pytest.raises(ClientError):
        table.transact([Put(item("JOB#1", "META"))])


def test_dynamo_get_reads_consistently_and_round_trips_python_values():
    original = item("JOB#1", "META", usd_micros=7500, probability=0.995, usage={"inputTokens": 12, "cache": None},
                    scopes=["lab:2026-09", "member:m1:2026-09"], active=False, note="한국어")
    table = DynamoTable(RecordingClient(), "control")
    stored = table._to(original)
    assert stored["usd_micros"] == {"N": "7500"}
    assert stored["probability"] == {"M": {"__float__": {"S": "0.995"}}}
    client = RecordingClient(items={("JOB#1", "META"): stored})
    table = DynamoTable(client, "control")
    fetched = table.get("JOB#1", "META")
    assert fetched == original
    assert isinstance(fetched["usd_micros"], int) and isinstance(fetched["probability"], float)
    name, request = client.calls[0]
    assert name == "get_item" and request["ConsistentRead"] is True
    assert request["Key"] == {"pk": {"S": "JOB#1"}, "sk": {"S": "META"}}
    assert table.get("JOB#none", "META") is None
    with pytest.raises(TypeError):
        table._to({"pk": "X", "sk": "Y", "revision": 1, "when": datetime.now(UTC)})


def test_dynamo_query_builds_key_condition_and_pagination():
    rows = [serialize(item("MEMBER#m1", f"JOB#{i}", job_id=str(i))) for i in range(2)]
    client = RecordingClient(query_result={"Items": rows, "LastEvaluatedKey": serialize({"pk": "MEMBER#m1", "sk": "JOB#1"})})
    table = DynamoTable(client, "control")
    items, cursor = table.query("MEMBER#m1", sk_prefix="JOB#", limit=2, start_after="JOB#-1", ascending=False)
    assert [row["job_id"] for row in items] == ["0", "1"] and cursor == "JOB#1"
    name, request = client.calls[0]
    assert name == "query"
    assert request["KeyConditionExpression"] == "pk = :pk AND begins_with(sk, :prefix)"
    assert request["ExpressionAttributeValues"] == {":pk": {"S": "MEMBER#m1"}, ":prefix": {"S": "JOB#"}}
    assert request["ExclusiveStartKey"] == {"pk": {"S": "MEMBER#m1"}, "sk": {"S": "JOB#-1"}}
    assert request["Limit"] == 2 and request["ScanIndexForward"] is False and request["ConsistentRead"] is True

    client = RecordingClient(query_result={"Items": []})
    table = DynamoTable(client, "control")
    items, cursor = table.query("OUTBOX")
    assert items == [] and cursor is None
    request = client.calls[0][1]
    assert request["KeyConditionExpression"] == "pk = :pk" and "ExclusiveStartKey" not in request
    with pytest.raises(ValueError):
        table.query("OUTBOX", limit=1001)


def test_dynamo_deserializer_turns_integral_decimals_into_ints():
    deserialized = TypeDeserializer().deserialize({"N": "42"})
    assert lab_store._python_safe(deserialized) == 42 and isinstance(lab_store._python_safe(deserialized), int)
    assert lab_store._python_safe({"__float__": "0.5"}) == 0.5
    assert lab_store._python_safe([{"__float__": "1.5"}, {"a": deserialized}]) == [1.5, {"a": 42}]


# ---------------------------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------------------------

def test_receipt_writer_writes_once_under_the_lab_prefix_with_if_none_match():
    s3 = MemoryS3()
    writer = ReceiptWriter(s3, "bucket")
    key = receipt_key("job1", "answer.json")
    assert key == "runs/lab-questions/job1/answer.json"
    result = writer.put_json(key, {"answer": "결과", "usd_micros": 7500})
    assert s3.writes == [(key, {"IfNoneMatch": "*"})]
    assert result["key"] == key and result["bytes"] == len(canonical({"answer": "결과", "usd_micros": 7500}))
    assert result["sha256"] == digest({"usd_micros": 7500, "answer": "결과"})
    assert result["etag"] == s3.etag(key) and result["version_id"] == "v1"
    assert writer.written == [key]
    assert writer.get_json(key) == {"answer": "결과", "usd_micros": 7500}
    with pytest.raises(ClientError) as failure:
        writer.put_json(key, {"answer": "overwrite"})
    assert failure.value.response["Error"]["Code"] == "PreconditionFailed"
    assert s3.json(key)["answer"] == "결과"


@pytest.mark.parametrize("key", [
    "wiki/sources/paper.md",
    "wiki/questions/q.json",
    "papers/stem/original.pdf",
    "index/wiki-index-v2.sqlite3",
    "runs/questions/abc/answer.json",
    "runs/agents/x.json",
    "sources/paper.md",
    "runs/lab-questions/../wiki/sources/paper.json",
    "runs/lab-question/job/answer.json",
    "",
])
def test_receipt_writer_refuses_keys_outside_the_lab_prefix(key):
    s3 = MemoryS3()
    writer = ReceiptWriter(s3, "bucket")
    with pytest.raises(ValueError):
        writer.put_json(key, {"x": 1})
    assert s3.writes == [] and writer.written == []
    with pytest.raises(ValueError):
        writer.get_json("wiki/sources/paper.md")
    assert s3.reads == []


def test_forbidden_prefixes_cover_the_wiki_originals_and_index():
    assert {"wiki/", "papers/", "index/"} <= set(FORBIDDEN_WRITE_PREFIXES)
    assert RECEIPT_PREFIX == "runs/lab-questions/"
    assert not RECEIPT_PREFIX.startswith(FORBIDDEN_WRITE_PREFIXES)


@pytest.mark.parametrize("job_id,name", [("", "answer.json"), ("a/b", "answer.json"), ("job", "sub/answer.json"),
                                         ("job", "answer.txt"), ("job", "answer")])
def test_receipt_key_rejects_paths_and_non_json_names(job_id, name):
    with pytest.raises(ValueError):
        receipt_key(job_id, name)


# ---------------------------------------------------------------------------------------------
# Canonical encoding, hashing, identifiers, time
# ---------------------------------------------------------------------------------------------

def test_canonical_and_digest_are_independent_of_key_order():
    left = {"b": [1, {"y": 2, "x": 1}], "a": "가"}
    right = {"a": "가", "b": [1, {"x": 1, "y": 2}]}
    assert canonical(left) == canonical(right) == b'{"a":"\xea\xb0\x80","b":[1,{"x":1,"y":2}]}'
    assert digest(left) == digest(right) == digest_bytes(canonical(right))
    assert len(digest(left)) == 64 and set(digest(left)) <= set("0123456789abcdef")
    assert digest([1, 2]) != digest([2, 1])
    with pytest.raises(ValueError):
        canonical({"p": math.nan})


def test_question_hash_keeps_negations_numbers_and_context_apart():
    context = [{"role": "user", "text": "SFARI 코호트에서"}]
    assert context_hash(None) == context_hash([]) == digest([])
    base = question_hash("CHD8 변이는 대두증과 관련이 있나요?", context_hash(context))
    assert base == digest({"question": "CHD8 변이는 대두증과 관련이 있나요?", "context_hash": context_hash(context)})
    assert base != question_hash("CHD8 변이는 대두증과 관련이 없나요?", context_hash(context))
    assert base != question_hash("CHD8 변이는 대두증과 관련이 있나요?", context_hash(None))
    assert question_hash("n = 120", "c") != question_hash("n = 12", "c")


def test_new_item_starts_at_revision_one_with_matching_stamps():
    record = new_item("JOB#1", "META", STAMP, job_id="1", status="received")
    assert record == {"pk": "JOB#1", "sk": "META", "revision": 1, "created_at": STAMP, "updated_at": STAMP,
                      "job_id": "1", "status": "received"}


def test_new_id_is_uuid4_hex():
    ids = {new_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(re.fullmatch(r"[0-9a-f]{32}", value) and value[12] == "4" for value in ids)


def test_now_iso_normalises_to_utc_and_requires_a_timezone():
    seoul = datetime(2026, 9, 21, 18, 30, 0, 123456, tzinfo=timezone(timedelta(hours=9)))
    assert now_iso(seoul) == "2026-09-21T09:30:00.123456+00:00"
    assert now_iso(datetime(2026, 9, 21, 9, 0, tzinfo=UTC)) == "2026-09-21T09:00:00.000000+00:00"
    with pytest.raises(ValueError):
        now_iso(datetime(2026, 9, 21, 9, 0))
    assert now_iso().endswith("+00:00")


def test_period_and_day_follow_utc():
    late = datetime(2026, 10, 1, 3, 0, tzinfo=timezone(timedelta(hours=9)))  # 2026-09-30T18:00 UTC
    assert period_for(late) == "2026-09" and day_for(late) == "2026-09-30"
    assert period_for(datetime(2026, 10, 1, 0, 0, tzinfo=UTC)) == "2026-10"


def test_store_error_carries_a_stable_code():
    assert StoreError().code == "store_error"
    assert StoreError("x", code="custom").code == "custom"
    assert str(StoreError()) == "StoreError"


# ---------------------------------------------------------------------------------------------
# Key builders (the plan's record table)
# ---------------------------------------------------------------------------------------------

def test_key_builders_produce_the_record_table():
    assert keys.member("m1") == ("MEMBER#m1", "PROFILE")
    assert keys.principal("AIDA1") == ("PRINCIPAL#AIDA1", "MEMBER")
    assert keys.idempotency("m1", "answer", "r1") == ("IDEMP#m1#answer#r1", "KEY")
    assert keys.job("j1") == ("JOB#j1", "META")
    assert keys.member_job("m1", STAMP, "j1") == ("MEMBER#m1", f"JOB#{STAMP}#j1")
    assert keys.day_record("2026-09-21", STAMP, "j1") == ("RECORDS#2026-09-21", f"{STAMP}#j1")
    assert keys.session("s1") == ("SESSION#s1", "META")
    assert keys.verdict("j1") == ("JOB#j1", "TRIAGE")
    assert keys.verdict_by_input("h" * 64) == (f"VERDICT#{'h' * 64}", "META")
    assert keys.offer("o1") == ("OFFER#o1", "META")
    assert keys.job_offer("j1", "o1") == ("JOB#j1", "OFFER#o1")
    assert keys.approval("a1") == ("APPROVAL#a1", "META")
    assert keys.outbox("x1") == ("OUTBOX#x1", "META")
    assert keys.outbox_pending(STAMP, "x1") == ("OUTBOX", f"PENDING#{STAMP}#x1")
    assert keys.budget("job:j1") == ("BUDGET#job:j1", "META")
    assert keys.budget("lab:2026-09") == ("BUDGET#lab:2026-09", "META")
    assert keys.budget("member:m1:2026-09") == ("BUDGET#member:m1:2026-09", "META")
    assert keys.reservation("r1") == ("RESERVATION#r1", "META")
    assert keys.candidate("c1") == ("CANDIDATE#c1", "META")
    assert keys.candidate_pointer(STAMP, "c1") == ("CANDIDATES", f"{STAMP}#c1")
    assert keys.round("rd1") == ("ROUND#rd1", "META")


def test_pointer_keys_sort_by_creation_time_under_one_partition():
    table = MemoryTable()
    stamps = ["2026-09-21T09:00:00.000000+00:00", "2026-09-21T09:00:01.000000+00:00", "2026-09-21T10:00:00.000000+00:00"]
    for i, stamp in enumerate(reversed(stamps)):
        table.put(item(*keys.day_record("2026-09-21", stamp, f"j{i}"), job_id=f"j{i}"))
        table.put(item(*keys.outbox_pending(stamp, f"x{i}"), outbox_id=f"x{i}", kind="answer"))
    rows, _ = table.query("RECORDS#2026-09-21")
    assert [row["sk"].split("#")[0] for row in rows] == stamps
    pending, _ = table.query("OUTBOX", sk_prefix="PENDING#")
    assert [row["outbox_id"] for row in pending] == ["x2", "x1", "x0"]


def test_dynamo_reraises_transaction_cancelled_for_throttling_and_validation():
    """ThrottlingError, capacity and validation reasons are never a logical condition failure."""
    for reason in ("ThrottlingError", "ProvisionedThroughputExceeded", "ValidationError"):
        reasons = [{"Code": "None"}, {"Code": reason, "Message": reason}]
        client = RecordingClient(failures=[client_error("TransactionCanceledException", "TransactWriteItems",
                                                        CancellationReasons=reasons)])
        table = DynamoTable(client, "control")
        with pytest.raises(ClientError) as failure:
            table.transact([Put(item("A#1", "META")), Check("B#1", "META", 1)])
        assert not isinstance(failure.value, ConditionFailed)
        assert failure.value.response["CancellationReasons"] == reasons
    single = DynamoTable(RecordingClient(failures=[client_error("ThrottlingException", "PutItem"),
                                                   client_error("ValidationException", "UpdateItem")]), "control")
    with pytest.raises(ClientError) as put_failure:
        single.put(item("A#1", "META"))
    assert not isinstance(put_failure.value, ConditionFailed)
    with pytest.raises(ClientError) as update_failure:
        single.update("A#1", "META", 1, {"status": "x"})
    assert not isinstance(update_failure.value, ConditionFailed)


# ---------------------------------------------------------------------------------------------
# TransactionConflict: a concurrent transaction on the same item is re-planned, not surfaced raw
# ---------------------------------------------------------------------------------------------

def test_transaction_conflict_is_a_condition_failed_subclass():
    assert issubclass(TransactionConflict, ConditionFailed)
    assert isinstance(TransactionConflict("x"), ConditionFailed)
    assert not isinstance(ConditionFailed("x"), TransactionConflict)


def test_dynamo_maps_single_item_transaction_conflict_exception():
    client = RecordingClient(failures=[client_error("TransactionConflictException", "PutItem"),
                                       client_error("TransactionConflictException", "UpdateItem"),
                                       client_error("TransactionConflictException", "DeleteItem")])
    table = DynamoTable(client, "control")
    with pytest.raises(TransactionConflict) as put_failure:
        table.put(item("JOB#1", "META"))
    assert isinstance(put_failure.value.__cause__, ClientError)
    assert put_failure.value.__cause__.response["Error"]["Code"] == "TransactionConflictException"
    with pytest.raises(TransactionConflict):
        table.update("JOB#1", "META", 1, {"status": "queued"})
    with pytest.raises(TransactionConflict):
        table.delete("JOB#1", "META", 1)
    assert [name for name, _ in client.calls] == ["put_item", "update_item", "delete_item"]


def test_dynamo_maps_cancelled_transaction_with_a_conflict_reason_to_transaction_conflict():
    reasons = [{"Code": "None"}, {"Code": "TransactionConflict", "Message": "Transaction is ongoing for the item"}]
    client = RecordingClient(failures=[client_error("TransactionCanceledException", "TransactWriteItems",
                                                    CancellationReasons=reasons)])
    table = DynamoTable(client, "control")
    with pytest.raises(TransactionConflict) as failure:
        table.transact([Put(item("A#1", "META")), Update("BUDGET#lab:2026-09", "META", 7, {"reserved_micros": 1})])
    assert isinstance(failure.value, ConditionFailed)
    assert "TransactionCanceledException" in str(failure.value)
    assert failure.value.__cause__.response["CancellationReasons"] == reasons
    mixed = [{"Code": "TransactionConflict"}, {"Code": "ThrottlingError"}]
    client = RecordingClient(failures=[client_error("TransactionCanceledException", "TransactWriteItems",
                                                    CancellationReasons=mixed)])
    with pytest.raises(TransactionConflict):
        DynamoTable(client, "control").transact([Put(item("A#1", "META")), Check("B#1", "META", 1)])


def test_conditional_check_failed_outranks_a_conflict_in_the_same_cancellation():
    """A definite condition failure is reported as such even when another item saw a conflict."""
    reasons = [{"Code": "ConditionalCheckFailed", "Message": "The conditional request failed"},
               {"Code": "TransactionConflict", "Message": "Transaction is ongoing for the item"}]
    client = RecordingClient(failures=[client_error("TransactionCanceledException", "TransactWriteItems",
                                                    CancellationReasons=reasons)])
    table = DynamoTable(client, "control")
    with pytest.raises(ConditionFailed) as failure:
        table.transact([Put(item("A#1", "META")), Check("B#1", "META", 1)])
    assert type(failure.value) is ConditionFailed
    assert not isinstance(failure.value, TransactionConflict)


def test_bounded_replan_loops_catch_a_transaction_conflict_as_condition_failed():
    """The retry pattern every lab module uses (``except ConditionFailed``) now covers conflicts."""
    client = RecordingClient(failures=[client_error("TransactionConflictException", "TransactWriteItems")])
    table = DynamoTable(client, "control")
    attempts = 0
    for _attempt in range(2):
        try:
            table.transact([Update("BUDGET#lab:2026-09", "META", 1, {"reserved_micros": 1})])
            break
        except ConditionFailed:
            attempts += 1
    assert attempts == 1 and len(client.calls) == 2
