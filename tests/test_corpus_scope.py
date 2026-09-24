"""`corpus-search` reaches every intake, and still refuses to report an unclassified row."""
from __future__ import annotations

from pathlib import Path

import pytest

from byeori.catalog import Catalog
from byeori.config import Settings
from byeori.corpus import CORPUS_ID, annotate, search_corpus
from byeori.journal_policy import apply_journal_policy
from byeori.pipeline import select_candidates


LAB_CORPUS = "lab-shared"


def settings_for(root: Path) -> Settings:
    return Settings(root, root / "data", root / "state", None, "us-east-1", "bucket", "table", "function")


def work(work_id: str, source: str | None, title: str = "Autism de novo variants"):
    return {"work_id": work_id, "doi": f"10.1000/{work_id.lower()}", "title": title,
            "publication_year": 2021, "publication_date": "2021-01-01", "authors": ["A"], "type": "article",
            "topics": [], "source": source, "is_open_access": True, "oa_license": "cc-by",
            "openalex_pdf_url": "https://content.openalex.org/x.pdf",
            "grobid_xml_url": "https://content.openalex.org/x.xml", "cited_by_count": 1}


def lab_item(stem: str, source: str | None, title: str):
    """What `backfill_to_s3_identity.py` leaves on a stem-keyed row: the same record shape under a
    different corpus id, written straight onto the item the uploader made rather than through
    `save_candidate`, which admits only OpenAlex work ids."""
    record = work(stem, source, title)
    record["corpus"] = {"id": LAB_CORPUS, "scope": "lab_shared_pdf", "intake": "to-s3",
                        "tags": ["Unclassified"], "tag_basis": "openalex_topics",
                        "screening": "pdf_uploaded", "fulltext_eligible": True, "fulltext_stored": True}
    record["catalog_search_text"] = f"{title} {record['doi']} {source or ''}".lower()
    record = apply_journal_policy(record)
    return {"work_id": stem, "stem": stem, "source": "to-s3", "ingest_status": "fulltext_ready",
            "doi": record["doi"], "title": title, "publication_year": record["publication_year"],
            "identity_status": "verified", "record": record,
            "search_text": record["catalog_search_text"]}


def populated(tmp_path, cloud):
    settings = settings_for(tmp_path)
    with Catalog(settings) as catalog:
        catalog.save_candidate(annotate(work("W1", "Nature Genetics"), "autism_title", []))
    for item in (lab_item("lab-included", "Nature Communications", "Tunneling nanotubes"),
                 lab_item("lab-excluded", "Cell Reports", "Galanin impairs tumour immunity")):
        cloud.items[item["work_id"]] = item
    return settings


def test_a_paper_from_another_intake_is_no_longer_invisible(tmp_path, cloud_catalog):
    settings = populated(tmp_path, cloud_catalog)
    found = search_corpus(settings, query="tunneling nanotubes")
    assert [result["work_id"] for result in found["results"]] == ["lab-included"]
    assert found["results"][0]["corpus"] == LAB_CORPUS


def test_naming_one_intake_still_searches_only_that_one(tmp_path, cloud_catalog):
    settings = populated(tmp_path, cloud_catalog)
    assert [r["work_id"] for r in search_corpus(settings, corpus=CORPUS_ID)["results"]] == ["W1"]
    assert search_corpus(settings, corpus=CORPUS_ID, query="tunneling nanotubes")["total"] == 0
    assert [r["work_id"] for r in search_corpus(settings, corpus=LAB_CORPUS)["results"]] == ["lab-included"]


def test_the_allowlist_still_applies_across_every_intake(tmp_path, cloud_catalog):
    """Reaching the lab's papers must not become a way around the journal policy."""
    settings = populated(tmp_path, cloud_catalog)
    assert sorted(r["work_id"] for r in search_corpus(settings)["results"]) == ["W1", "lab-included"]
    everything = {r["work_id"]: r for r in search_corpus(settings, journal_policy="all")["results"]}
    assert sorted(everything) == ["W1", "lab-excluded", "lab-included"]
    assert everything["lab-excluded"]["journal_verdict"] == "exclude"


def test_a_row_with_no_corpus_at_all_is_left_out_of_every_search(tmp_path, cloud_catalog):
    """The bare rows a bulk PDF upload writes carry no record; reporting them would mean reporting
    a paper with no journal verdict as though it had passed the policy."""
    settings = populated(tmp_path, cloud_catalog)
    cloud_catalog.items["bare-upload"] = {
        "work_id": "bare-upload", "stem": "bare-upload", "source": "to-s3", "id_kind": "stem",
        "ingest_status": "fulltext_ready", "search_text": "tunneling nanotubes",
    }
    for policy in ("include", "all"):
        for corpus in ("all", LAB_CORPUS):
            found = search_corpus(settings, journal_policy=policy, corpus=corpus)
            assert "bare-upload" not in [result["work_id"] for result in found["results"]]


def test_the_result_names_the_intake_and_how_far_the_paper_got(tmp_path, cloud_catalog):
    settings = populated(tmp_path, cloud_catalog)
    by_id = {r["work_id"]: r for r in search_corpus(settings, journal_policy="all")["results"]}
    assert by_id["W1"]["corpus"] == CORPUS_ID and by_id["W1"]["screening"] == "metadata_candidate"
    assert by_id["lab-included"]["corpus"] == LAB_CORPUS
    assert by_id["lab-included"]["screening"] == "pdf_uploaded"
    assert search_corpus(settings, corpus=LAB_CORPUS)["corpus"] == LAB_CORPUS


def test_the_ingest_selection_stays_on_the_discovery_corpus(tmp_path, cloud_catalog):
    """`select_candidates` picks papers to fetch and extract; an intake whose PDFs are already
    stored has nothing to fetch, and its papers must not be queued for download."""
    settings = populated(tmp_path, cloud_catalog)
    assert [candidate["work_id"] for candidate in select_candidates(settings)] == ["W1"]


def test_reclassification_skips_a_retired_duplicate_instead_of_failing_on_it(tmp_path, cloud_catalog):
    """`save_candidate` refuses a candidate whose DOI belongs to another row, so a retired
    duplicate must be skipped rather than retried into an error on every run."""
    from byeori.corpus import apply_policy_to_catalog
    settings = settings_for(tmp_path)
    with Catalog(settings) as catalog:
        catalog.save_candidate(annotate(work("W1", "Nature Genetics"), "autism_title", []))
        catalog.save_candidate(annotate(work("W2", "medRxiv"), "autism_title", []))
    cloud_catalog.items["W2"]["superseded_by"] = "mendes-2026-characterizing"
    result = apply_policy_to_catalog(settings)
    assert result["superseded_skipped"] == 1 and result["errors"] == []
    assert result["classified"] == 1 and result["aws_published"] == 1
