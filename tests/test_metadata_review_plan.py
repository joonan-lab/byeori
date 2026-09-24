from __future__ import annotations

import io
import json
from copy import deepcopy

import pytest
from botocore.exceptions import ClientError

from byeori import metadata_review_plan as planner


class Table:
    def __init__(self, rows, page_size=3):
        self.rows = deepcopy(rows)
        self.scans = []
        self.page_size = page_size

    def scan(self, **kwargs):
        self.scans.append(deepcopy(kwargs))
        start = kwargs.get("ExclusiveStartKey", {}).get("index", 0)
        end = start + min(self.page_size, kwargs["Limit"])
        fields = set(kwargs["ExpressionAttributeNames"].values())
        page = {"Items": [{k: deepcopy(v) for k, v in row.items() if k in fields} for row in self.rows[start:end]]}
        if end < len(self.rows):
            page["LastEvaluatedKey"] = {"index": end}
        return page

    def update_item(self, **kwargs):
        pytest.fail("Planner must not modify DynamoDB")

    put_item = delete_item = update_item


class S3:
    def __init__(self):
        self.objects = {}
        self.reads = []
        self.writes = []
        self.streams = []

    def get_object(self, **kwargs):
        self.reads.append(kwargs)
        key = kwargs["Key"]
        assert key.startswith("runs/metadata-review/") and key.endswith("/plan.json")
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = io.BytesIO(self.objects[key])
        self.streams.append(body)
        return {"Body": body}

    def put_object(self, **kwargs):
        self.writes.append(kwargs)
        assert kwargs["IfNoneMatch"] == "*"
        if kwargs["Key"] in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[kwargs["Key"]] = kwargs["Body"]


def paper(stem, **overrides):
    return {"work_id": stem, "id_kind": "stem", "doi": f"10.1234/{stem}",
            "title": "Original title", "openalex_title": "Original title", "year": "2020", "openalex_year": 2020,
            "authors": "Alice Smith, John Jones", "openalex_authors": ["Alice Smith", "John Jones"],
            "journal": "Nature", "openalex_venue": "Nature", "pmid": "123", "pmid_openalex": "123",
            "pmcid": "PMC1", "openalex_referenced_works": ["W1"], "openalex_status": "matched",
            "pdf_key": f"papers/{stem}/original.pdf", "pdf_sha256": "a" * 64,
            "tei_key": f"papers/{stem}/grobid.tei.xml", "grobid_sha256": "b" * 64,
            **overrides}


def plan(rows, *, run_id="test-run", s3=None, page_size=3):
    s3 = s3 or S3()
    table = Table(rows, page_size=page_size)
    result = planner.plan_metadata_review({"run_id": run_id}, table=table, s3=s3, bucket="bucket")
    manifest = json.loads(s3.objects[result["plan_key"]])
    return result, manifest, table, s3


def test_prioritized_deduplicated_queue_preserves_metadata_and_ignores_legacy():
    rows = [paper("conflict", pmid_openalex="456", openalex_year=2023, openalex_title="Other title"),
            paper("year-one", openalex_year=2021), paper("nodoi", doi=None), paper("invalid", doi="not-doi"),
            paper("duplicate-a", doi="10.1234/shared", pmcid=""),
            paper("duplicate-b", doi="HTTPS://doi.org/10.1234/SHARED"),
            paper("enrich", pmcid=None, openalex_referenced_works=[], openalex_authors=["Alice Smith"] * 30),
            paper("format", openalex_title="Different title", openalex_authors=["John Jones"], openalex_venue="Cell"),
            paper("exact30", openalex_authors=["Alice Smith"] * 30, openalex_authors_truncated=False),
            paper("W100", id_kind="openalex", doi="10.1234/shared")]
    result, manifest, table, s3 = plan(rows)
    assert result["catalogued_stems"] == 9 and result["planned_papers"] == 7
    assert result["priority_counts"] == {"1": 5, "2": 1, "3": 1}
    assert result["duplicate_doi_groups"] == 1
    assert result["issue_counts"]["duplicate_stem_doi"] == 2
    assert result["issue_counts"]["missing_pmcid"] == 0
    assert result["issue_counts"]["authors_at_cap"] == 1
    items = {item["work_id"]: item for item in manifest["items"]}
    assert items["duplicate-b"]["metadata"]["doi"] == "HTTPS://doi.org/10.1234/SHARED"
    assert items["duplicate-b"]["normalized_doi"] == "10.1234/shared"
    assert items["duplicate-a"]["duplicate_doi_stems"] == ["duplicate-a", "duplicate-b"]
    assert items["conflict"]["issues"] == ["pmid_disagreement", "year_difference_over_one", "title_disagreement"]
    assert items["enrich"]["priority"] == 2 and len(items["enrich"]["issues"]) == 2
    assert items["format"]["priority"] == 3
    assert "openalex_referenced_works" not in items["format"]["metadata"]
    assert items["format"]["metadata"]["openalex_references_count"] == 1
    assert items["format"]["metadata"]["pdf_key"] == "papers/format/original.pdf"
    assert len(items["format"]["input_sha256"]) == 64
    assert table.rows == rows and len(s3.writes) == 1
    projected = set(table.scans[0]["ExpressionAttributeNames"].values())
    assert {"pdf_key", "pdf_sha256", "tei_key", "grobid_sha256", "extracted_pdf_sha256"} <= projected
    assert not {"source_note", "source_key", "extraction", "raw", "record"} & projected


def test_same_run_reuses_immutable_plan_without_rescanning_or_resetting_worker_progress():
    first, manifest, table, s3 = plan([paper("pending", openalex_referenced_works=[])])
    s3.objects["runs/metadata-review/test-run/results.json"] = b'{"completed":true}'
    table.rows = [paper("new", pmid="changed")]
    table.scans.clear()
    repeated = planner.plan_metadata_review({"run_id": "test-run"}, table=table, s3=s3, bucket="bucket")
    assert repeated == {**first, "reused": True}
    assert not table.scans and len(s3.writes) == 1
    assert json.loads(s3.objects[first["plan_key"]]) == manifest
    assert s3.objects["runs/metadata-review/test-run/results.json"] == b'{"completed":true}'
    assert all(body.closed for body in s3.streams)


def test_immutable_create_race_reuses_winner_and_does_not_overwrite():
    _, winner, _, previous = plan([paper("winner", openalex_referenced_works=[])])
    winning_bytes = next(iter(previous.objects.values()))

    class RaceS3(S3):
        def put_object(self, **kwargs):
            self.objects[kwargs["Key"]] = winning_bytes
            super().put_object(**kwargs)

    result, manifest, _, s3 = plan([paper("loser", pmid_openalex="2")], s3=RaceS3())
    assert result["reused"] and manifest == winner
    assert manifest["items"][0]["work_id"] == "winner" and len(s3.writes) == 1


def test_inspection_filters_in_aws_returns_only_bounded_pages_and_summary():
    result, manifest, _, s3 = plan([paper(f"item-{i:02}", openalex_referenced_works=[]) for i in range(17)])
    page = planner.inspect_metadata_review({"run_id": "test-run", "issue": "missing_references", "offset": 4, "limit": 10}, s3=s3, bucket="bucket")
    assert page["total"] == 17 and len(page["items"]) == 10 and page["next_offset"] == 14
    assert page["items"][0]["item_index"] == 4 and page["items"][0]["work_id"] == "item-04"
    assert all(item["issues"] == ["missing_references"] for item in page["items"])
    end = planner.inspect_metadata_review({"run_id": "test-run", "offset": 14, "limit": 10}, s3=s3, bucket="bucket")
    assert len(end["items"]) == 3 and end["next_offset"] is None
    assert planner.metadata_review_summary({"run_id": "test-run"}, s3=s3, bucket="bucket") == {**result, "reused": True}
    assert len(manifest["items"]) == 17 and len(result["samples"]["missing_references"]) == 3


def test_large_unicode_fields_are_complete_only_in_s3_and_response_size_stays_bounded():
    giant = "\U0001f9ec" * 10000
    rows = [paper(f"giant-{i:02}", **{field: giant for field in planner.METADATA_FIELDS}) for i in range(12)]
    for row in rows:
        row.update(title=giant + "A", openalex_title=giant + "B", pmid=giant + "1", pmid_openalex=giant + "2",
                   authors="Alice Smith" + giant, openalex_authors=["John Jones" + giant] * 30)
    result, manifest, _, s3 = plan(rows)
    page = planner.inspect_metadata_review({"run_id": "test-run", "limit": 10}, s3=s3, bucket="bucket")
    assert len(page["items"]) == 10
    assert len(json.dumps(result, ensure_ascii=True).encode()) < planner.MAX_RESPONSE_BYTES
    assert len(json.dumps(page, ensure_ascii=True).encode()) < planner.MAX_RESPONSE_BYTES
    assert len(manifest["items"][0]["metadata"]["title"]) > 10000
    assert len(page["items"][0]["metadata"]["title"]) < 160
    assert all(len(samples) <= 3 for samples in result["samples"].values())


def test_plan_order_fingerprint_and_samples_are_independent_of_scan_order():
    rows = [paper(f"row-{i:02}", pmid_openalex="999") for i in range(12)]
    first, first_manifest, _, _ = plan(rows, page_size=4)
    second, second_manifest, _, _ = plan(list(reversed(rows)), page_size=4)
    assert first_manifest["items"] == second_manifest["items"]
    assert first["samples"] == second["samples"]


def test_absent_artifact_keys_are_reported_without_invented_paths_or_content_reads():
    row = paper("missing", pmid_openalex="999")
    for field in ("pdf_key", "tei_key", "pdf_sha256", "grobid_sha256"):
        del row[field]
    result, manifest, _, s3 = plan([row])
    assert result["missing_pdf_key"] == result["missing_tei_key"] == 1
    assert "pdf_key" not in manifest["items"][0]["metadata"]
    assert "tei_key" not in manifest["items"][0]["metadata"]
    assert all(read["Key"].endswith("/plan.json") for read in s3.reads)


def test_sparse_scan_continuation_and_repeated_rows_do_not_duplicate_plan_items():
    class Sparse(Table):
        def scan(self, **kwargs):
            self.scans.append(kwargs)
            if "ExclusiveStartKey" not in kwargs:
                return {"Items": [], "LastEvaluatedKey": {"index": 1}}
            return {"Items": [paper("same", openalex_referenced_works=[])] * 2}
    s3 = S3()
    result = planner.plan_metadata_review({"run_id": "sparse"}, table=Sparse([]), s3=s3, bucket="bucket")
    assert result["catalogued_stems"] == result["planned_papers"] == 1 and result["scan_pages"] == 2


@pytest.mark.parametrize("run_id", [None, "", "../bad", "a/b", "x" * 101, 123])
def test_invalid_run_ids_fail_before_any_cloud_operation(run_id):
    table, s3 = Table([]), S3()
    with pytest.raises(ValueError):
        planner.plan_metadata_review({"run_id": run_id}, table=table, s3=s3, bucket="bucket")
    assert not table.scans and not s3.reads and not s3.writes


@pytest.mark.parametrize("options", [{"offset": -1}, {"offset": True}, {"offset": "0"},
                                    {"limit": 0}, {"limit": 11}, {"limit": True}, {"issue": "other"}, {"issue": []}])
def test_invalid_inspection_options_fail_before_read(options):
    s3 = S3()
    with pytest.raises(ValueError):
        planner.inspect_metadata_review({"run_id": "test", **options}, s3=s3, bucket="bucket")
    assert not s3.reads


def test_normalization_avoids_false_conflicts_and_empty_pmcid_is_allowed():
    rows = [paper("normalized", title="Nrxn1α", openalex_title="Nrxn1a", pmid="https://pubmed.ncbi.nlm.nih.gov/123/"),
            paper("invalid-doi", doi="bad", pmcid=None), paper("no-doi", doi=None, pmcid=None),
            paper("non-numeric-year", year="Unknown", openalex_year=2020)]
    result, manifest, _, _ = plan(rows)
    assert result["issue_counts"]["pmid_disagreement"] == 0
    assert result["issue_counts"]["title_disagreement"] == 0
    assert result["issue_counts"]["year_difference_over_one"] == 0
    assert result["issue_counts"]["missing_pmcid"] == 0
    assert {item["work_id"] for item in manifest["items"]} == {"invalid-doi", "no-doi"}


@pytest.mark.parametrize("pmcid", [None, ""])
def test_empty_pmcid_does_not_select_an_otherwise_healthy_paper(pmcid):
    summary, manifest, _, _ = plan([paper("optional-pmcid", pmcid=pmcid, pmcid_openalex=None)])
    assert summary["planned_papers"] == 0
    assert summary["issue_counts"]["missing_pmcid"] == 0
    assert manifest["items"] == []


def test_existing_pmcid_disagreement_remains_in_the_review_queue():
    summary, manifest, _, _ = plan([paper("existing-pmcid", pmcid="PMC1", pmcid_openalex="PMC2")])
    assert summary["issue_counts"]["pmcid_disagreement"] == 1
    assert manifest["items"][0]["issues"] == ["pmcid_disagreement"]


def test_unknown_snapshot_version_and_scan_race_are_rejected():
    _, _, _, s3 = plan([])
    key = "runs/metadata-review/test-run/plan.json"
    invalid = json.loads(s3.objects[key])
    invalid["run_id"] = "different"
    s3.objects[key] = json.dumps(invalid).encode()
    with pytest.raises(ValueError, match="identity or schema"):
        planner.metadata_review_summary({"run_id": "test-run"}, s3=s3, bucket="bucket")
    with pytest.raises(ValueError, match="changed during scan"):
        plan([paper("duplicate", pmcid=None), paper("duplicate", pmid="changed")])


def test_repeated_invalid_doi_placeholders_are_not_duplicate_papers():
    rows = [paper('null-a', doi='null'), paper('null-b', doi='NULL'),
            paper('unknown-a', doi='unknown'), paper('unknown-b', doi='unknown')]
    summary, manifest, _, _ = plan(rows)
    assert summary['duplicate_doi_groups'] == 0
    assert summary['issue_counts']['duplicate_stem_doi'] == 0
    assert summary['issue_counts']['doi_invalid'] == 4
    assert all('duplicate_stem_doi' not in item['issues'] for item in manifest['items'])
