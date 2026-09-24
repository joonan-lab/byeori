from __future__ import annotations

import io
import json
import urllib.error
from copy import deepcopy

import pytest
from botocore.exceptions import ClientError

from byeori import openalex_match as matcher
from byeori import openalex_match_lambda as worker


class MemoryS3:
    def __init__(self):
        self.objects = {}

    def get_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[kwargs["Key"]])}

    def put_object(self, **kwargs):
        self.objects[kwargs["Key"]] = kwargs["Body"]


class MemoryTable:
    def __init__(self, rows):
        self.rows = {row["work_id"]: deepcopy(row) for row in rows}
        self.scans, self.updates = [], []

    def scan(self, **kwargs):
        self.scans.append(kwargs)
        rows = list(self.rows.values())
        start = int(kwargs.get("ExclusiveStartKey", {}).get("work_id", 0))
        end = min(start + kwargs["Limit"], len(rows))
        result = {"Items": deepcopy(rows[start:end])}
        if end < len(rows):
            result["LastEvaluatedKey"] = {"work_id": str(end)}
        return result

    def update_item(self, **kwargs):
        self.updates.append(kwargs)
        row = self.rows[kwargs["Key"]["work_id"]]
        names, values = kwargs["ExpressionAttributeNames"], kwargs["ExpressionAttributeValues"]
        if (row["doi"] != values[":doi"] or row.get("openalex_match_run") == values[":run"] or
                ":matched" in values and row.get("openalex_status") == "matched"):
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        for field in ("openalex_id", "pmid", "pmcid"):
            if f"#old_{field}" not in names:
                continue
            if ((f":old_{field}" in values and (field not in row or row[field] != values[f":old_{field}"])) or
                    (f":old_{field}" not in values and field in row)):
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        for name, field in names.items():
            if name.startswith("#f"):
                value = values[":v" + name[2:]]
                row[field] = value

    def get_item(self, **kwargs):
        return {"Item": deepcopy(self.rows.get(kwargs["Key"]["work_id"], {}))}


def paper(i, **extra):
    return {"work_id": f"paper-{i}", "doi": f"10.123/{i}", "title": f"Title {i}", "year": 2026, **extra}


def work(doi):
    return {"id": "https://openalex.org/W123", "doi": doi, "display_name": "External title",
            "publication_year": 2025, "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"}}


def advance(s3, table, **event):
    return worker.handle_match_batch({"run_id": "test-run", **event}, table=table, s3=s3, bucket="bucket")


def test_match_batch_preserves_authoritative_fields_and_skips_already_matched(monkeypatch):
    table = MemoryTable([paper(1, pmid="authoritative", authors="Original Author", journal="Original Journal"),
                         paper(2, openalex_status="matched"), paper(3, doi="")])
    s3 = MemoryS3()
    calls = []
    monkeypatch.setattr(matcher, "fetch_batch", lambda dois, **kwargs: calls.append(dois) or {doi: work(doi) for doi in dois})
    result = advance(s3, table)
    assert result["done"] and result["matched"] == 1 and result["selected"] == 1
    assert result["skipped_matched"] == 1 and result["without_doi"] == 1
    assert calls == [["10.123/1"]]
    row = table.rows["paper-1"]
    assert row["title"] == "Title 1" and row["year"] == 2026 and row["doi"] == "10.123/1"
    assert row["authors"] == "Original Author" and row["journal"] == "Original Journal"
    assert row["pmid"] == "authoritative" and row["pmid_openalex"] == "123"
    assert row["openalex_title"] == "External title"
    assert advance(s3, table) == result  # Finished executions do not rescan or refetch.
    assert len(calls) == 1 and len(table.scans) == 1


def test_batch_misses_are_singleton_checked_in_bounded_resumable_steps(monkeypatch):
    table, s3, calls = MemoryTable([paper(i) for i in range(8)]), MemoryS3(), []
    monkeypatch.setattr(matcher, "fetch_batch", lambda *a, **k: {})
    monkeypatch.setattr(matcher, "fetch_one", lambda doi, **kwargs: calls.append(doi))
    first = advance(s3, table)
    assert not first["done"] and first["pending"] == 4 and first["unmatched"] == 4
    assert first["requests"] == 5 and len(calls) == worker.MAX_SINGLETONS
    second = advance(s3, table)
    assert second["done"] and second["unmatched"] == 8 and second["requests"] == 9
    assert len(table.scans) == 1
    assert all(row["openalex_status"] == "unmatched" for row in table.rows.values())


def test_scan_continuation_and_limit_stay_in_aws(monkeypatch):
    table, s3 = MemoryTable([paper(i) for i in range(6)]), MemoryS3()
    monkeypatch.setattr(matcher, "fetch_batch", lambda dois, **kwargs: {doi: work(doi) for doi in dois})
    assert not advance(s3, table, batch=2, limit=3)["done"]
    last = advance(s3, table)
    assert last["done"] and last["matched"] == 3
    assert [scan["Limit"] for scan in table.scans] == [2, 2]
    assert table.scans[1]["ExclusiveStartKey"] == {"work_id": "2"}
    assert json.loads(s3.objects[last["progress_key"]])["pending"] == []
    assert not any(isinstance(value, list) for value in last.values())


def test_rate_limit_wait_survives_lambda_container_and_respects_retry_after(monkeypatch):
    table, s3, calls = MemoryTable([paper(1)]), MemoryS3(), []
    clock = [1000.0]
    monkeypatch.setattr(worker.time, "time", lambda: clock[0])
    def limited(*args, **kwargs):
        calls.append(1)
        matcher.RATE.update(remaining=2, reset=60, retry_after=90)
        raise matcher.OpenAlexRequestError(429)
    monkeypatch.setattr(matcher, "fetch_batch", limited)
    first = advance(s3, table)
    assert first["wait_seconds"] == 90 and first["requests"] == 1 and first["errors"] == 0
    matcher.RATE.clear()  # The next invocation may be in another Lambda container.
    assert advance(s3, table)["wait_seconds"] == 90 and len(calls) == 1
    clock[0] += 90
    advance(s3, table)
    clock[0] += 90
    final = advance(s3, table)
    assert final["done"] and final["errors"] == 1 and final["requests"] == 3
    assert len(calls) == worker.MAX_ATTEMPTS and not table.updates


def test_credit_floor_delays_singleton_checks_without_marking_absent(monkeypatch):
    table, s3 = MemoryTable([paper(1)]), MemoryS3()
    def batch(*a, **k):
        matcher.RATE.update(remaining=worker.matcher.CREDIT_FLOOR, reset=120)
        return {}
    monkeypatch.setattr(matcher, "fetch_batch", batch)
    monkeypatch.setattr(matcher, "fetch_one", lambda *a, **k: pytest.fail("Credits are reserved"))
    first = advance(s3, table)
    assert first["pending"] == 1 and first["unmatched"] == 0 and first["wait_seconds"] >= 121


def test_identity_mismatch_from_singleton_never_updates_catalogue(monkeypatch):
    table, s3 = MemoryTable([paper(1)]), MemoryS3()
    monkeypatch.setattr(matcher, "fetch_batch", lambda *a, **k: {})
    monkeypatch.setattr(matcher, "fetch_one", lambda *a, **k: work("10.123/wrong"))
    final = advance(s3, table)
    assert final["done"] and final["errors"] == 1 and not table.updates


def test_retry_after_dynamodb_write_does_not_overwrite_or_lose_count(monkeypatch):
    table, s3 = MemoryTable([paper(1)]), MemoryS3()
    monkeypatch.setattr(matcher, "fetch_batch", lambda dois, **kwargs: {doi: work(doi) for doi in dois})
    original = s3.put_object
    writes = []
    def failed_checkpoint(**kwargs):
        writes.append(1)
        if len(writes) == 2:
            raise RuntimeError("S3 unavailable")
        return original(**kwargs)
    monkeypatch.setattr(s3, "put_object", failed_checkpoint)
    with pytest.raises(RuntimeError, match="S3 unavailable"):
        advance(s3, table)
    assert table.rows["paper-1"]["openalex_match_run"] == "test-run"
    monkeypatch.setattr(s3, "put_object", original)
    final = advance(s3, table)
    assert final["done"] and final["matched"] == 1 and final["conflicts"] == 0


def test_conditional_update_preserves_concurrent_identity_edits(monkeypatch):
    table, s3 = MemoryTable([paper(1)]), MemoryS3()
    def fetch(dois, **kwargs):
        table.rows["paper-1"]["doi"] = "10.123/corrected"
        return {doi: work(doi) for doi in dois}
    monkeypatch.setattr(matcher, "fetch_batch", fetch)
    final = advance(s3, table)
    assert final["conflicts"] == 1 and final["matched"] == 0
    assert "openalex_id" not in table.rows["paper-1"]


def test_ambiguous_doi_is_held_for_review_without_catalogue_changes(monkeypatch):
    table, s3 = MemoryTable([paper(1)]), MemoryS3()
    monkeypatch.setattr(matcher, "fetch_batch", lambda *a, **k: {
        "10.123/1": {"doi": "10.123/1", "_ambiguous_openalex_ids": ["W111", "W222"]}})
    result = advance(s3, table)
    assert result["done"] and result["conflicts"] == 1 and result["matched"] == 0
    assert result["errors"] == 0 and not table.updates
    detail = json.loads(s3.objects[result["progress_key"]])["conflict_details"][0]
    assert detail["reason"] == "ambiguous_doi" and detail["openalex_ids"] == ["W111", "W222"]


def test_existing_different_openalex_identity_is_preserved_for_review(monkeypatch):
    table, s3 = MemoryTable([paper(1, openalex_id="W999")]), MemoryS3()
    monkeypatch.setattr(matcher, "fetch_batch", lambda dois, **k: {doi: work(doi) for doi in dois})
    result = advance(s3, table)
    assert result["conflicts"] == 1 and result["matched"] == 0 and not table.updates
    assert table.rows["paper-1"]["openalex_id"] == "W999"
    detail = json.loads(s3.objects[result["progress_key"]])["conflict_details"][0]
    assert detail["existing_openalex_id"] == "W999" and detail["returned_openalex_id"] == "W123"


def test_equivalent_existing_openalex_identity_is_not_a_conflict(monkeypatch):
    table, s3 = MemoryTable([paper(1, openalex_id="https://openalex.org/w123/")]), MemoryS3()
    monkeypatch.setattr(matcher, "fetch_batch", lambda dois, **k: {doi: work(doi) for doi in dois})
    result = advance(s3, table)
    assert result["matched"] == 1 and result["conflicts"] == 0
    assert table.rows["paper-1"]["openalex_id"] == "W123"


def test_mismatching_identity_diagnostic_is_bounded(monkeypatch):
    table, s3 = MemoryTable([paper(1, openalex_id="W123")]), MemoryS3()
    def fetch(dois, **kwargs):
        result = work(dois[0])
        result["id"] = "W" + "9" * 10000
        return {dois[0]: result}
    monkeypatch.setattr(matcher, "fetch_batch", fetch)
    result = advance(s3, table)
    detail = json.loads(s3.objects[result["progress_key"]])["conflict_details"][0]
    assert len(detail["returned_openalex_id"]) == 120
    assert result["conflicts"] == 1 and not table.updates


def test_concurrently_deleted_openalex_identity_is_not_restored(monkeypatch):
    table, s3 = MemoryTable([paper(1, openalex_id="W123")]), MemoryS3()
    def fetch(dois, **kwargs):
        table.rows["paper-1"].pop("openalex_id")
        return {doi: work(doi) for doi in dois}
    monkeypatch.setattr(matcher, "fetch_batch", fetch)
    result = advance(s3, table)
    assert result["conflicts"] == 1 and result["matched"] == 0
    assert "openalex_id" not in table.rows["paper-1"]


def test_ambiguous_identity_diagnostics_are_bounded(monkeypatch):
    table, s3 = MemoryTable([paper(1)]), MemoryS3()
    ids = [str(i) + "x" * 10000 for i in range(30)]
    monkeypatch.setattr(matcher, "fetch_batch", lambda *a, **k: {
        "10.123/1": {"doi": "10.123/1", "_ambiguous_openalex_ids": ids}})
    result = advance(s3, table)
    detail = json.loads(s3.objects[result["progress_key"]])["conflict_details"][0]
    assert detail["identity_count"] == 30 and len(detail["openalex_ids"]) == 20
    assert all(len(value) <= 120 for value in detail["openalex_ids"])


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_biomedical_identifiers_are_filled_with_concurrent_guards(monkeypatch, empty):
    table, s3 = MemoryTable([paper(1, pmid=empty, pmcid=empty)]), MemoryS3()
    def fetch(dois, **kwargs):
        result = work(dois[0])
        result["ids"]["pmcid"] = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC321"
        return {dois[0]: result}
    monkeypatch.setattr(matcher, "fetch_batch", fetch)
    result = advance(s3, table)
    assert result["matched"] == 1 and table.rows["paper-1"]["pmid"] == "123"
    assert table.rows["paper-1"]["pmcid"] == "PMC321"
    assert "#old_pmid = :old_pmid" in table.updates[0]["ConditionExpression"]
    assert "#old_pmcid = :old_pmcid" in table.updates[0]["ConditionExpression"]


@pytest.mark.parametrize("field,new_value", [("openalex_id", "W999"), ("pmid", "999"), ("pmcid", "PMC999")])
def test_concurrent_identifier_change_is_preserved_and_reported(monkeypatch, field, new_value):
    table, s3 = MemoryTable([paper(1, **{field: ""})]), MemoryS3()
    def fetch(dois, **kwargs):
        table.rows["paper-1"][field] = new_value
        result = work(dois[0])
        result["ids"]["pmcid"] = "PMC321"
        return {dois[0]: result}
    monkeypatch.setattr(matcher, "fetch_batch", fetch)
    result = advance(s3, table)
    assert result["conflicts"] == 1 and result["matched"] == 0
    assert table.rows["paper-1"][field] == new_value
    assert "openalex_match_run" not in table.rows["paper-1"]
    detail = json.loads(s3.objects[result["progress_key"]])["conflict_details"][0]
    assert detail["reason"] == "catalogue_changed"
    assert detail["changes"][field] == {"selected": "", "current": new_value}


def test_existing_pmid_conflict_is_a_review_disagreement_without_blocking_match(monkeypatch):
    table, s3 = MemoryTable([paper(1, pmid="999", title="External title", year=2025)]), MemoryS3()
    monkeypatch.setattr(matcher, "fetch_batch", lambda dois, **k: {doi: work(doi) for doi in dois})
    result = advance(s3, table)
    assert result["matched"] == 1 and result["disagreements"] == 1 and result["conflicts"] == 0
    assert table.rows["paper-1"]["pmid"] == "999"
    assert json.loads(s3.objects[result["progress_key"]])["disagreement_details"][0]["field"] == "pmid"


def test_pmid_url_and_bare_identifier_are_equivalent_for_review(monkeypatch):
    table, s3 = MemoryTable([paper(1, pmid="https://pubmed.ncbi.nlm.nih.gov/123/", title="External title", year=2025)]), MemoryS3()
    monkeypatch.setattr(matcher, "fetch_batch", lambda dois, **k: {doi: work(doi) for doi in dois})
    result = advance(s3, table)
    assert result["matched"] == 1 and result["disagreements"] == 0


def test_read_only_plan_aggregates_server_side_without_http_or_writes(monkeypatch):
    table = MemoryTable([paper(i, openalex_status="matched" if i % 2 else "") for i in range(502)])
    monkeypatch.setattr(matcher, "fetch_batch", lambda *a, **k: pytest.fail("Plan must be read only"))
    result = worker.plan_match({"limit": 10}, table=table)
    assert result["catalogued"] == 502 and result["selected"] == 10 and result["skipped_matched"] == 251
    assert result["dry_run"] and len(table.scans) == 2 and not table.updates


def test_client_only_submits_aws_requests(monkeypatch, tmp_path):
    from byeori import runs
    from byeori.aws_store import AwsStore
    from byeori.config import Settings
    settings = Settings(tmp_path, tmp_path / "data", tmp_path / "state", None, "us-east-1", "bucket", "table", "function")
    calls = []
    monkeypatch.setattr(runs, "start_run", lambda settings, key, payload: calls.append((key, payload)) or {"execution": "arn:run"})
    monkeypatch.setattr(AwsStore, "_invoke", lambda self, payload: calls.append(payload) or {"dry_run": True})
    assert matcher.match_openalex(settings, limit=5) == {"execution": "arn:run"}
    assert calls[0] == ("OpenAlexMatchStateMachineArn", {"limit": 5, "refresh": False, "batch": 50})
    assert matcher.match_openalex(settings, dry_run=True)["dry_run"]
    assert calls[1]["action"] == "openalex_match_plan"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("event", [{"batch": 0}, {"batch": 51}, {"limit": -1}, {"run_id": "../elsewhere"}])
def test_invalid_bounds_are_rejected(event):
    with pytest.raises(ValueError):
        worker.handle_match_batch(event, table=MemoryTable([]), s3=MemoryS3(), bucket="bucket")


def test_http_error_retains_retry_headers(monkeypatch):
    exc = urllib.error.HTTPError("https://api.openalex.org", 429, "Limited", {"Retry-After": "120"}, None)
    monkeypatch.setattr(matcher.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(exc))
    matcher.RATE.clear()
    with pytest.raises(matcher.OpenAlexRequestError) as raised:
        matcher.fetch_batch(["10.123/test"], attempts=1)
    assert raised.value.retryable and matcher.RATE["retry_after"] == 120


def test_retry_after_accepts_http_date(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(matcher.time, "time", lambda: 0)
    matcher.RATE.clear()
    matcher._read_rate(SimpleNamespace(headers={"Retry-After": "Thu, 01 Jan 1970 00:02:00 GMT"}))
    assert 120 <= matcher.RATE["retry_after"] <= 121
