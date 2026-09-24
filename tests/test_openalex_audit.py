from __future__ import annotations

import io
import json
from copy import deepcopy

import pytest

from byeori import openalex_audit as audit


class ReadOnlyTable:
    def __init__(self, rows, page_size=2):
        self.rows = deepcopy(rows)
        self.page_size = page_size
        self.scans = []

    def scan(self, **kwargs):
        self.scans.append(kwargs.copy())
        start = kwargs.get("ExclusiveStartKey", {}).get("offset", 0)
        end = start + min(self.page_size, kwargs["Limit"])
        page = {"Items": deepcopy(self.rows[start:end])}
        if end < len(self.rows):
            page["LastEvaluatedKey"] = {"offset": end}
        return page

    def update_item(self, **kwargs):
        pytest.fail("Audit must never write DynamoDB")

    put_item = delete_item = update_item


class ReadOnlyS3:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.reads = []

    def get_object(self, **kwargs):
        self.reads.append(kwargs)
        self.body = io.BytesIO(json.dumps(self.checkpoint).encode())
        return {"Body": self.body}

    def put_object(self, **kwargs):
        pytest.fail("Audit must never write S3")


def paper(name, **fields):
    return {"work_id": name, "id_kind": "stem", "doi": f"10.1234/{name}", **fields}


def test_paginated_audit_separates_stems_legacy_and_other_without_writes(monkeypatch):
    rows = [paper("ready", openalex_status="matched", openalex_id="W1", source_note_status="source_ready"),
            paper("failed", openalex_status="unmatched", source_note_status="source_failed"),
            paper("nodoi", doi=""), {"work_id": "W40", "source_note_status": "source_ready"},
            {"work_id": "candidate-other", "record": {"title": "Not a stem"}}]
    table = ReadOnlyTable(rows)
    monkeypatch.setattr(audit.matcher, "fetch_batch", lambda *a, **k: pytest.fail("No external fetch"))
    monkeypatch.setattr(audit.matcher, "fetch_one", lambda *a, **k: pytest.fail("No external fetch"))
    result = audit.audit_catalogue(table=table)
    assert result["total_rows"] == 5 and result["scan_pages"] == 3
    assert result["scopes"]["stem"]["rows"] == 3
    assert result["scopes"]["stem"]["source_note_failed"] == 1
    assert result["scopes"]["legacy_openalex"]["rows"] == 1
    assert result["scopes"]["legacy_openalex"]["source_note_ready"] == 1
    assert result["scopes"]["other"]["rows"] == 1
    assert result["scopes"]["stem"]["without_doi"] == 1
    assert result["scopes"]["stem"]["status"]["unattempted"] == 1
    assert table.rows == rows
    assert all(s["ConsistentRead"] and "FilterExpression" not in s for s in table.scans)
    projected = set(table.scans[0]["ExpressionAttributeNames"].values())
    assert {"id_kind", "authors", "pmid_openalex", "openalex_referenced_works"} <= projected
    assert not {"record", "raw", "extraction", "source_note", "source_markdown"} & projected


def test_coverage_false_oa_is_present_and_original_gaps_are_not_overwritten():
    rows = [paper("one", title="Nrxn1α", openalex_title="Nrxn1a", year="2025", openalex_year=2026,
                  authors="Jane Smith, Adam Jones et al.", openalex_authors=["Jane Smith"],
                  journal="", openalex_venue="Nature", type="review", openalex_type="article",
                  pmid="https://pubmed.ncbi.nlm.nih.gov/123/", pmid_openalex="123",
                  pmcid="pmc456", pmcid_openalex="PMC789", is_open_access=False, openalex_is_oa=False,
                  openalex_referenced_works=["W123"], openalex_status="matched", openalex_id="W1"),
            paper("two", openalex_is_oa=False, openalex_status="matched", openalex_id="W2")]
    result = audit.audit_catalogue(table=ReadOnlyTable(rows))
    coverage = result["coverage"]
    assert coverage["oa"] == {"catalogue": 1, "openalex": 2, "catalogue_missing_openalex_present": 1,
                              "both_present": 1, "compared": 1, "incomparable": 0, "review_candidates": 0}
    assert coverage["journal"]["catalogue_missing_openalex_present"] == 1
    assert coverage["references"]["catalogue_missing_openalex_present"] == 1
    assert coverage["title"]["review_candidates"] == 0
    assert coverage["authors"]["review_candidates"] == 0
    assert coverage["pmid"]["review_candidates"] == 0
    assert coverage["pmcid"]["review_candidates"] == 1
    assert coverage["type"]["review_candidates"] == 1
    assert result["year_difference"]["one_year"] == 1
    assert result["matched_without_references"] == 1
    assert result["review_candidate_stem_rows"] == 1
    assert result["coverage_denominator"] == 2


def test_authors_compare_first_surname_without_conflating_truncation_and_name_cap():
    rows = [paper("capped", authors="Smith J et al.", openalex_authors=["Jane Smith"] * 30),
            paper("surname_first", authors="Smith, Jane; Jones, Adam", openalex_authors=["Jane Smith"]),
            paper("disagreement", authors="Jane Smith et al.", openalex_authors=["Adam Jones"]),
            paper("consortium", authors="Genome Consortium", openalex_authors=["Genome Consortium"]),
            paper("malformed", authors={"name": "Jane Smith"}, openalex_authors=["Jane Smith"])]
    result = audit.audit_catalogue(table=ReadOnlyTable(rows))
    assert result["openalex_author_lists_at_cap"] == 1
    assert result["coverage"]["authors"]["compared"] == 3
    assert result["coverage"]["authors"]["incomparable"] == 2
    assert result["coverage"]["authors"]["review_candidates"] == 1
    assert result["samples"]["authors_disagreement"][0]["work_id"] == "disagreement"


def test_identifier_and_duplicate_issues_are_aggregated_across_scopes():
    rows = [paper("first", doi="https://doi.org/10.1234/SHARED", openalex_id="W7"),
            paper("second", doi="doi:10.1234/shared", openalex_id="https://openalex.org/W7"),
            {"work_id": "W7", "doi": "10.1234/shared"},
            paper("missing", doi="bad", openalex_status="matched"),
            paper("invalid", openalex_id="https://wrong.example/W1", openalex_status="matched",
                  openalex_referenced_works=["W1", "https://wrong.example/W2", "broken"])]
    result = audit.audit_catalogue(table=ReadOnlyTable(rows))
    for field in ("doi", "openalex_id"):
        assert result["duplicate_identities"][field] == {
            "groups": 1, "rows_in_groups": 3, "stem_groups": 1, "legacy_groups": 0, "cross_scope_groups": 1}
    assert result["review_candidates"]["matched_without_valid_openalex_id"] == 2
    assert result["review_candidates"]["invalid_openalex_id"] == 1
    assert result["review_candidates"]["invalid_doi"] == 1
    assert result["invalid_reference_entries"] == 2
    assert result["review_candidates"]["invalid_reference_ids"] == 1


def test_checkpoint_is_read_only_and_run_membership_is_distinct_from_error_events():
    table = ReadOnlyTable([paper("matched", openalex_status="matched", openalex_id="W1", openalex_match_run="run-1"),
                          paper("unmatched", openalex_status="unmatched", openalex_match_run="run-1"),
                          paper("prior", openalex_status="matched", openalex_match_run="earlier"),
                          paper("error-without-status")])
    s3 = ReadOnlyS3({"run_id": "run-1", "done": True, "selected": 3, "matched": 1,
                     "unmatched": 1, "errors": 1, "pending": [],
                     "error_details": [{"stem": "error-without-status", "doi": "10.1234/error", "error": "Timeout"}],
                     "conflict_details": [{"stem": f"conflict-{i}", "reason": "Changed"} for i in range(10)]})
    result = audit.audit_catalogue(table=table, s3=s3, bucket="bucket", run_id="run-1")
    assert s3.reads == [{"Bucket": "bucket", "Key": "runs/openalex-match/run-1.json"}]
    assert s3.body.closed
    assert result["checkpoint"]["unaccounted_selected"] == 0
    assert result["checkpoint"]["table_requested_run_matched"] == 1
    assert result["checkpoint"]["table_requested_run_unmatched"] == 1
    assert result["run_membership"]["other_run"]["matched"] == 1
    assert result["run_membership"]["no_run"]["unattempted"] == 1
    assert result["checkpoint"]["errors"] == 1
    assert result["scopes"]["stem"]["status"]["error"] == 0
    assert len(result["samples"]["checkpoint_conflict"]) == 3
    assert result["samples"]["checkpoint_conflict"][0]["reason"] == "Changed"


def test_giant_metadata_and_many_distinct_values_have_bounded_deterministic_response():
    rows = [paper(f"paper-{i:04}", title="한" * 10000 + str(i), openalex_title="語" * 10000 + str(i),
                  journal="한" * 10000, openalex_venue="語" * 10000,
                  year="한" * 10000, openalex_year="語" * 10000,
                  pmid="1" * 10000, pmid_openalex="2" * 10000,
                  pmcid="한" * 10000, pmcid_openalex="語" * 10000,
                  doi="invalid" * 1000, openalex_id="bad" * 1000,
                  authors="Jane Smith et al.", openalex_authors=["Adam Jones"] * 30,
                  openalex_status="custom" + str(i), openalex_match_run="run-" + str(i)) for i in range(25)]
    first = audit.audit_catalogue(table=ReadOnlyTable(rows, 10))
    second = audit.audit_catalogue(table=ReadOnlyTable(list(reversed(rows)), 10))
    assert first == second
    assert len(json.dumps(first, ensure_ascii=True).encode()) < audit.MAX_RESPONSE_BYTES
    assert all(len(samples) <= 3 for samples in first["samples"].values())
    assert first["scopes"]["stem"]["status"]["other"] == 25
    assert first["samples"]["pmid_disagreement"][0]["work_id"] == "paper-0000"


def test_empty_scan_page_with_continuation_is_not_treated_as_end():
    class SparseTable(ReadOnlyTable):
        def scan(self, **kwargs):
            self.scans.append(kwargs)
            if "ExclusiveStartKey" not in kwargs:
                return {"Items": [], "LastEvaluatedKey": {"work_id": "cursor"}}
            assert kwargs["ExclusiveStartKey"] == {"work_id": "cursor"}
            return {"Items": [paper("last")]}
    assert audit.audit_catalogue(table=SparseTable([]))["total_rows"] == 1


@pytest.mark.parametrize("kwargs", [{"sample_limit": -1}, {"sample_limit": 4}, {"sample_limit": True},
                                   {"sample_limit": "3"}, {"run_id": "../secret"}, {"run_id": "valid"}])
def test_invalid_options_fail_before_any_read(kwargs):
    table = ReadOnlyTable([])
    with pytest.raises(ValueError):
        audit.audit_catalogue(table=table, **kwargs)
    assert not table.scans


def test_zero_samples_and_wrong_checkpoint_identity():
    assert audit.audit_catalogue(table=ReadOnlyTable([paper("nodoi", doi="")]), sample_limit=0)["samples"] == {}
    with pytest.raises(ValueError, match="Checkpoint run_id"):
        audit.audit_catalogue(table=ReadOnlyTable([]), s3=ReadOnlyS3({"run_id": "wrong"}), bucket="bucket", run_id="run-1")


def test_invalid_placeholders_are_issues_not_duplicate_doi_identities():
    result = audit.audit_catalogue(table=ReadOnlyTable([
        paper('placeholder-a', doi='unknown'), paper('placeholder-b', doi='UNKNOWN')]))
    assert result['review_candidates']['invalid_doi'] == 2
    assert result['duplicate_identities']['doi']['groups'] == 0
