"""Adversarial report sizes preserve totals and allow complete problem-text follow-up."""
import json

import pytest

from byeori import corpus_ops, synthesis_support, wiki_ops
from byeori.failure_reports import bounded_report, MAX_REPORT_BYTES


@pytest.fixture(params=["corpus", "synthesis"])
def reporting(request):
    kind = request.param
    rows = []
    for i in range(400):
        problems = ["😀" * 400 + f"-{i}-{j}" for j in range(8)]
        if kind == "corpus":
            rows.append({"work_id": f"paper-{i:04}", "ingest_status": "fulltext_ready", "category": "asd-ndd",
                         "source_note_status": "source_failed", "source_note_problems": problems})
        else:
            rows.append({"work_id": f"concept#gene-{i:04}", "id_kind": "concept", "synthesis_status": "failed",
                         "problems": problems, "calls": 2, "updated_at": "2026-09-20T00:00:00Z"})

    class Table:
        def scan(self, **kwargs):
            if kwargs.get("ExclusiveStartKey"):
                return {"Items": rows[200:]}
            return {"Items": rows[:200], "LastEvaluatedKey": {"work_id": rows[199]["work_id"]}}

    def report(**event):
        if kind == "corpus":
            return wiki_ops.dispatch({"action": "pipeline_failures", **event}, table=Table(),
                                     s3=None, bucket=None, index=None)
        return synthesis_support.failures(event, table=Table())
    return report, rows, "papers" if kind == "corpus" else "pages", "stem" if kind == "corpus" else "id"


def test_hundreds_of_unicode_reason_groups_cannot_overflow_even_at_limit_one(reporting):
    report, rows, key, _ = reporting
    result = report(limit=1, verbose=True)
    assert result["failed"] == 400 and len(result[key]) == 1 and result["next_offset"] == 1
    assert result["reason_groups_total"] == 3200 and result["reason_groups_omitted"] == 3180
    assert result["reason_occurrences_total"] == 3200 and result["reason_occurrences_omitted"] == 3180
    assert len(result["by_reason"]) == 20 and sum(result["by_reason"].values()) == 20
    assert result["reason_previews_truncated"] == 20
    assert len(json.dumps(result).encode()) < 10_000


def test_hundred_verbose_rows_with_huge_unicode_problem_lists_are_bounded(reporting):
    report, rows, key, _ = reporting
    result = report(limit=100, verbose=True)
    assert len(result[key]) == 100 and result["next_offset"] == 100
    assert len(json.dumps(result).encode()) < MAX_REPORT_BYTES
    for entry in result[key]:
        assert entry["problem_count"] == 8 and len(entry["problems"]) == 3
        assert entry["problems_omitted"] == 5 and entry["problem_previews_truncated"] == 3
    following = report(limit=100, offset=result["next_offset"], verbose=True)
    assert following["offset"] == 100 and following["next_offset"] == 200


def test_omitted_problem_can_be_recovered_in_bounded_slices(reporting):
    report, rows, key, id_key = reporting
    preview = report(limit=1, verbose=True)[key][0]
    first = report(problem_id=preview[id_key], problem_index=7, max_chars=123)
    second = report(problem_id=preview[id_key], problem_index=7, start=first["next_start"], max_chars=8000)
    problem_key = "source_note_problems" if id_key == "stem" else "problems"
    assert first["text"] + second["text"] == rows[0][problem_key][7]
    assert first["problem_count"] == 8 and second["next_start"] is None
    assert len(json.dumps(second).encode()) < 128 * 1024
    for invalid in ({"max_chars": 8001}, {"max_chars": True}, {"start": -1}, {"problem_index": 8}):
        with pytest.raises(ValueError):
            report(problem_id=preview[id_key], **invalid)


def test_response_budget_reduces_page_size_without_skipping_large_metadata():
    rows = [{"id": f"concept#gene-{i}", "kind": "concept", "metadata": "😀" * 500, "problems": []}
            for i in range(100)]
    first = bounded_report(rows, offset=0, limit=100, verbose=True, id_key="id", rows_key="pages")
    assert 1 <= len(first["pages"]) < 100 and first["next_offset"] == len(first["pages"])
    assert len(json.dumps(first).encode()) < MAX_REPORT_BYTES
    following = bounded_report(rows, offset=first["next_offset"], limit=100, verbose=True, id_key="id", rows_key="pages")
    assert following["pages"][0]["id"] == rows[len(first["pages"])]["id"]
