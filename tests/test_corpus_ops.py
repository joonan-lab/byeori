"""Corpus reporting selects, aggregates and checks index coverage inside AWS."""
from __future__ import annotations

import io
import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from byeori import cli, corpus_ops, wiki_ops
from byeori.aws_store import AwsStore
from byeori.config import Settings


class PagedTable:
    def __init__(self, rows, page_size=2):
        self.rows, self.page_size, self.requests = rows, page_size, []

    def scan(self, **request):
        self.requests.append(dict(request))
        start = request.get("ExclusiveStartKey", {}).get("position", 0)
        response = {"Items": self.rows[start:start + self.page_size]}
        if start + self.page_size < len(self.rows):
            response["LastEvaluatedKey"] = {"position": start + self.page_size}
        return response


def row(stem, status=None, **fields):
    return {"work_id": stem, "id_kind": "stem", "ingest_status": "fulltext_ready",
            "source_note_status": status, "category": "asd-ndd", **fields}


def test_failure_report_paginates_full_aws_scan_and_counts_all_failures():
    rows = [row("z-queued"), row("c-failed", "source_failed"), row("b-ready", "source_ready"),
            row("a-failed", "source_failed", source_note_problems=["Missing heading"]),
            row("d-ineligible", "source_failed", ingest_status="extraction_failed"),
            row("e-page", "source_ready", page_status="page_failed", page_problems=["Bad link"])]
    table = PagedTable(rows)
    first = corpus_ops.report({"action": "pipeline_failures", "limit": 1}, table=table)
    assert first["papers"] == ["a-failed"] and first["failed"] == 2 and first["next_offset"] == 1
    assert first["by_reason"] == {"Missing heading": 1, "(no problem recorded)": 1}
    assert [r.get("ExclusiveStartKey") for r in table.requests] == [None, {"position": 2}, {"position": 4}]
    condition = table.requests[0]["FilterExpression"].get_expression()
    assert condition["values"][0].name == "id_kind" and condition["values"][1] == "stem"
    second = corpus_ops.report({"action": "pipeline_failures", "limit": 1, "offset": 1, "verbose": True}, table=table)
    assert second["papers"][0]["stem"] == "c-failed" and second["next_offset"] is None
    exhausted = corpus_ops.report({"action": "pipeline_failures", "offset": 100}, table=table)
    assert exhausted["papers"] == [] and exhausted["next_offset"] is None and exhausted["failed"] == 2


@pytest.mark.parametrize("window", [{"offset": -1}, {"offset": True}, {"offset": "1"},
                                    {"limit": 0}, {"limit": 101}, {"limit": 1.5}, {"limit": False}])
def test_report_rejects_invalid_windows_before_reading_aws(window):
    table = PagedTable([])
    with pytest.raises(ValueError):
        corpus_ops.report({"action": "pipeline_failures", **window}, table=table)
    assert table.requests == []


def test_corpus_index_check_distinguishes_ids_types_and_generation_times():
    rows = [row("indexed-old", "source_ready", source_note_at="2026-09-20T10:00:00Z"),
            row("indexed-new", "source_ready", source_note_at="2026-09-20T20:00:01+09:00"),
            row("indexed-equal", "source_ready", source_note_at="2026-09-20T11:00:00"),
            row("paper-only", "source_ready", source_note_at="2026-09-20T10:00:00Z"),
            row("missing", "source_ready", source_note_at="2026-09-20T10:00:00Z"),
            row("unknown-missing", "source_ready"), row("unknown-invalid", "source_ready", source_note_at="not a time"),
            row("unknown-null", "source_ready", source_note_at=None), row("queued"), row("failed", "source_failed")]
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE docs (doc_type TEXT, doc_id TEXT)")
    connection.executemany("INSERT INTO docs VALUES (?, ?)", [
        ("note", "indexed-old"), ("note", "indexed-new"), ("note", "indexed-equal"), ("paper", "paper-only"),
        ("note", "unknown-missing"), ("note", "unknown-invalid"), ("note", "unknown-null")])
    heads = []
    s3 = SimpleNamespace(head_object=lambda **kw: heads.append(kw) or {
        "LastModified": datetime(2026, 9, 20, 11, tzinfo=timezone.utc)})
    result = wiki_ops.dispatch({"action": "corpus_status", "check_index": True}, table=PagedTable(rows),
                               s3=s3, bucket="bucket", index=lambda: (connection, "etag-snapshot"))
    assert result["execution"] == "aws" and result["catalogued"] == 10 and result["ready_notes"] == 8
    assert result["note_statuses"] == {"source_ready": 8, "unattempted": 1, "source_failed": 1}
    assert result["notes_by_category"] == {"asd-ndd": 8}
    coverage = result["index"]
    assert coverage["indexed_notes"] == 6 and coverage["missing_ready_count"] == 2
    assert coverage["missing_ready_sample"] == ["missing", "paper-only"]
    assert coverage["newer_than_index_count"] == 1 and coverage["newer_than_index_sample"] == ["indexed-new"]
    assert coverage["unknown_note_timestamp_count"] == 3 and coverage["etag"] == "etag-snapshot"
    assert heads == [{"Bucket": "bucket", "Key": "index/wiki-index.sqlite3"}]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_corpus_summary_does_not_read_index_unless_requested():
    result = wiki_ops.dispatch({"action": "corpus_status"}, table=PagedTable([row("queued")]), s3=None, bucket="b",
                               index=lambda: pytest.fail("Summary should not read the index"))
    assert result["catalogued"] == 1 and result["ready_notes"] == 0 and "index" not in result


def test_index_report_distinguishes_failed_unattempted_and_uncatalogued_notes():
    rows = [row("ready", "source_ready", source_note_at="2026-09-20T10:00:00Z"),
            row("failed", "source_failed"), row("queued")]
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE docs (doc_type TEXT, doc_id TEXT)")
    connection.executemany("INSERT INTO docs VALUES ('note', ?)", [(stem,) for stem in ("ready", "failed", "queued", "legacy")])
    s3 = SimpleNamespace(head_object=lambda **kwargs: {"LastModified": datetime(2026, 9, 20, 11, tzinfo=timezone.utc)})
    result = corpus_ops.report({"action": "corpus_status", "check_index": True}, table=PagedTable(rows),
                               s3=s3, bucket="b", index=lambda: (connection, "etag"))["index"]
    assert result["missing_ready_count"] == 0 and result["indexed_nonready_count"] == 3
    assert result["indexed_nonready_by_status"] == {"source_failed": 1, "unattempted": 1, "not_catalogued": 1}
    assert result["indexed_nonready_sample"] == [
        {"stem": "failed", "status": "source_failed", "category": "asd-ndd"},
        {"stem": "legacy", "status": "not_catalogued", "category": None},
        {"stem": "queued", "status": "unattempted", "category": "asd-ndd"}]


def test_client_and_cli_corpus_reports_use_only_lambda(tmp_path, monkeypatch, capsys):
    invocations = []
    table = PagedTable([row("a-failed", "source_failed", source_note_problems=["Missing heading"]), row("z-queued")])

    class Lambda:
        def invoke(self, **kwargs):
            event = json.loads(kwargs["Payload"])
            invocations.append(event)
            result = wiki_ops.dispatch(event, table=table, s3=None, bucket="b", index=None)
            return {"Payload": io.BytesIO(json.dumps(result).encode())}

    class Session:
        def client(self, name, **kwargs):
            assert name == "lambda", "Client must not download the catalogue, index or notes"
            return Lambda()

        def resource(self, name):
            pytest.fail("Client must not scan DynamoDB")

    settings = Settings(tmp_path, tmp_path / "data", tmp_path / "state", None, "us-east-1", "b", "table", "function")
    store = AwsStore(settings, session=Session())
    assert store.pipeline_failures(verbose=True, limit=1)["papers"][0]["stem"] == "a-failed"
    assert store.corpus_status()["note_statuses"]["unattempted"] == 1
    assert {"pipeline_failures", "corpus_status"} <= wiki_ops.ACTIONS
    monkeypatch.setattr(cli, "AwsStore", lambda settings: store)
    args = cli.build_parser().parse_args(["aws-pipeline-failures", "--verbose", "--offset", "1"])
    assert args.handler(settings, args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["failed"] == 1 and result["papers"] == [] and result["offset"] == 1
    args = cli.build_parser().parse_args(["aws-corpus-status"])
    assert args.handler(settings, args) == 0
    assert json.loads(capsys.readouterr().out)["catalogued"] == 2
    assert invocations[0] == {"action": "pipeline_failures", "verbose": True, "offset": 0, "limit": 1}
    assert not list(tmp_path.iterdir()), "Read reports must leave no local cloud-data mirror"
