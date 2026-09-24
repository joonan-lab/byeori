from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from byeori.aws_store import AwsStore
from byeori.catalog import Catalog
from byeori.config import Settings
from byeori.corpus import annotate, collect, eligible_fulltext, export_report, screen_work, search_corpus


pytestmark = pytest.mark.usefixtures("cloud_catalog")

def settings_for(root: Path) -> Settings:
    return Settings(root, root / "data", root / "state", None, "us-east-1", "bucket", "table", "function")


def work(work_id: str = "W123", **changes):
    return {"work_id": work_id, "doi": "10.1000/example", "title": "De novo mutations in autism",
            "publication_year": 2020, "publication_date": "2020-05-01", "authors": ["Example Author"],
            "type": "article", "topics": ["Autism Spectrum Disorder Research"], "source": "Nature Genetics",
            "is_open_access": True, "oa_license": "cc-by", "openalex_pdf_url": "https://content.openalex.org/works/W123.pdf",
            "grobid_xml_url": "https://content.openalex.org/works/W123.grobid-xml", "cited_by_count": 10, **changes}


def test_screening_date_boundaries_and_false_positive():
    assert screen_work(work(publication_date="2011-09-16"), "2011-09-17", "2026-09-17")[0] is None
    assert screen_work(work(publication_date="2011-09-17"), "2011-09-17", "2026-09-17")[0] == "autism_title"
    assert screen_work(work(publication_date="2026-09-18"), "2011-09-17", "2026-09-17")[0] is None
    unrelated = work(title="Clinical whole-genome sequencing", topics=["Rare disease diagnosis"])
    assert screen_work(unrelated, "2011-09-17", "2026-09-17")[0] is None
    related = work(title="De novo variants in neurodevelopmental disorders")
    assert screen_work(related, "2011-09-17", "2026-09-17")[0] == "related_topic"


def test_oa_does_not_imply_fulltext_eligible():
    assert eligible_fulltext(work())
    assert not eligible_fulltext(work(oa_license=None))
    assert not eligible_fulltext(work(grobid_xml_url=None))
    assert not eligible_fulltext(work(is_open_access=False))


def test_collect_deduplicates_doi_and_resumes_without_network(tmp_path):
    s = settings_for(tmp_path)
    originals = [work(), work("W999", doi="https://doi.org/10.1000/EXAMPLE"),
                 work("W456", doi="10.1000/other", title="Unrelated clinical sequencing", topics=[])]
    with patch("byeori.corpus.QUERIES", ("autism genomics",)), \
         patch.object(AwsStore, "search_openalex", return_value=originals) as api:
        first = collect(s, start="2020-01-01", end="2020-12-31", per_query_limit=3)
        second = collect(s, start="2020-01-01", end="2020-12-31", per_query_limit=3,
                         resume=Path(first["run_dir"]))
        assert api.call_count == 1
    assert first["candidate_count"] == second["candidate_count"] == 1
    assert first["duplicate_occurrences"] == 1
    assert first["excluded_occurrences"] == {"no_autism_title_or_topic": 1}
    assert first["at_limit_queries"] == 1
    assert first["returned_occurrences"] == first["candidate_count"] + first["duplicate_occurrences"] + sum(first["excluded_occurrences"].values())
    with Catalog(s) as catalog:
        assert len(catalog.list_candidates()) == 1
    assert not list((s.data_dir / "wiki" / "papers").glob("*.md"))


@pytest.mark.parametrize("bridge_order", [False, True])
def test_deduplication_when_doi_appears_on_later_hit(tmp_path, bridge_order):
    hits = [work(doi=None), work(), work("W999")]
    if bridge_order:
        hits = [work(doi=None), work("W999"), work()]
    with patch("byeori.corpus.QUERIES", ("autism genomics",)), \
         patch.object(AwsStore, "search_openalex", return_value=hits):
        result = collect(settings_for(tmp_path), start="2020-01-01", end="2020-12-31")
    assert result["candidate_count"] == 1
    assert result["duplicate_occurrences"] == 2
    with Catalog(settings_for(tmp_path)) as catalog:
        assert catalog.list_candidates()[0]["doi"] == "10.1000/example"
    assert search_corpus(settings_for(tmp_path), query="10.1000/example")["total"] == 1


def test_resume_preserves_successful_queries_and_retries_failures(tmp_path):
    s = settings_for(tmp_path)
    with patch("byeori.corpus.QUERIES", ("autism genomics", "autism variants")), \
         patch.object(AwsStore, "search_openalex", side_effect=[[work()], RuntimeError("transient")]):
        first = collect(s, start="2020-01-01", end="2020-12-31")
    assert len(first["errors"]) == 1
    with patch("byeori.corpus.QUERIES", ("autism genomics", "autism variants")), \
         patch.object(AwsStore, "search_openalex", return_value=[work()]) as api:
        second = collect(s, start="2020-01-01", end="2020-12-31", resume=Path(first["run_dir"]))
    assert api.call_count == 1
    assert second["errors"] == []
    assert len(second["queries"]) == 2


def test_metadata_filters_and_sql_literal_query(tmp_path):
    s = settings_for(tmp_path)
    with Catalog(s) as catalog:
        catalog.save_candidate(annotate(work(), "autism_title", ["2020-01"]))
        catalog.save_candidate(annotate(work("W456", doi="10.1000/two", publication_year=2024,
                                             is_open_access=False, title="Whole genome sequencing of autism"), "autism_title", []))
    assert search_corpus(s, query="de novo", tag="De novo", oa_only=True)["total"] == 1
    assert search_corpus(s, query="example author", from_year=2021)["results"][0]["work_id"] == "W456"
    assert search_corpus(s, fulltext_only=True)["total"] == 1
    assert search_corpus(s, query="' OR 1=1 --")["total"] == 0
    with pytest.raises(ValueError):
        search_corpus(s, from_year=2025, to_year=2020)


def test_candidate_file_identity_and_refresh_preserve_asset_status(tmp_path):
    s = settings_for(tmp_path)
    with Catalog(s) as catalog:
        a = catalog.save_candidate(work())
        b = catalog.save_candidate(work("W456", doi="10.1000/different"))
        assert a["stem"] != b["stem"]
        pdf = tmp_path / "input.pdf"
        pdf.write_bytes(b"%PDF-1.7\noriginal")
        catalog.attach_pdf("W123", pdf)
        refreshed = catalog.save_candidate(work(title="Updated title"))
        assert refreshed["stem"] == a["stem"]
        assert refreshed["status"] == "pdf_attached"
    assert not s.data_dir.exists()


class FakeTable:
    def __init__(self):
        self.updates = []
        self.scan_calls = []

    def update_item(self, **request):
        self.updates.append(request)

    def scan(self, **request):
        self.scan_calls.append(request)
        if len(self.scan_calls) == 1:
            return {"Items": [], "ScannedCount": 5, "LastEvaluatedKey": {"work_id": "W5"},
                    "ConsumedCapacity": {"CapacityUnits": 1}}
        assert request["ExclusiveStartKey"] == {"work_id": "W5"}
        return {"Items": [{"work_id": "W123"}], "ScannedCount": 1, "ConsumedCapacity": {"CapacityUnits": .5}}


class FakeSession:
    def __init__(self, table):
        self.table = table

    def resource(self, service):
        assert service == "dynamodb"
        return self

    def Table(self, name):
        assert name == "table"
        return self.table


def test_dynamodb_metadata_refresh_does_not_replace_ingest_fields(tmp_path):
    table = FakeTable()
    with Catalog(settings_for(tmp_path)) as catalog:
        candidate = catalog.save_candidate(annotate(work(raw={"large": "metadata"}), "autism_title", []))
    AwsStore(settings_for(tmp_path), FakeSession(table)).push_candidate(candidate)
    request = table.updates[0]
    assert request["Key"] == {"work_id": "W123"}
    assert "if_not_exists" in request["UpdateExpression"]
    assert "ingest_status" not in request["ExpressionAttributeNames"].values()
    assert "pdf_key" not in request["ExpressionAttributeNames"].values()
    record_key = next(key for key, value in request["ExpressionAttributeNames"].items() if value == "record")
    assert "raw" not in request["ExpressionAttributeValues"][":v" + record_key[2:]]


def test_dynamodb_scan_continues_after_empty_filtered_page(tmp_path):
    table = FakeTable()
    result = AwsStore(settings_for(tmp_path), FakeSession(table)).search_corpus(query="autism", from_year=2020)
    assert result["items"] == [{"work_id": "W123"}]
    assert result["scanned_count"] == 6
    assert result["read_capacity_units"] == 1.5


def test_report_embedded_data_cannot_close_script_element(tmp_path, cloud_catalog):
    s = settings_for(tmp_path)
    with Catalog(s) as catalog:
        catalog.save_candidate(annotate(work(title='Autism </script><script>alert(1)</script> genetics'), "autism_title", []))
    output = export_report(s)
    html = cloud_catalog.objects[output.split("/",3)[3]].decode()
    assert '</script><script>alert(1)</script>' not in html
    assert '\\u003c/script>' in html
    payload = json.loads(cloud_catalog.objects[output.split("/",3)[3]].decode().split('id="data">')[1].split("</script>",1)[0])
    assert payload["papers"][0]["title"].endswith("genetics")


def test_report_date_bounds_follow_cumulative_records_not_latest_run(tmp_path, cloud_catalog):
    s = settings_for(tmp_path)
    with Catalog(s) as catalog:
        catalog.save_candidate(annotate(work(publication_date="2018-05-01", publication_year=2018), "autism_title", []))
    folder = s.state_dir / "corpus" / "autism-genomics"
    run = folder / "runs" / "latest"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"configuration": {"start": "2020-01-01", "end": "2020-12-31"}}))
    (folder / "latest-run.json").write_text(json.dumps({"run_dir": str(run)}))
    output = export_report(s)
    payload = json.loads(cloud_catalog.objects[output.split("/",3)[3]].decode().split('id="data">')[1].split("</script>",1)[0])
    assert payload["catalog_date_range"]["start"] == "2018-05-01"
    assert payload["manifest"]["configuration"]["start"] == "2020-01-01"
