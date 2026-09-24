from __future__ import annotations

import io
import json
from pathlib import Path

from byeori import openalex_match as matcher
from byeori.openalex_match import normalize_doi, normalize_openalex_id, work_fields


WORK = {
    "id": "https://openalex.org/W4407354549",
    "doi": "https://doi.org/10.1038/s41581-025-00934-5",
    "ids": {"openalex": "https://openalex.org/W4407354549", "pmid": "https://pubmed.ncbi.nlm.nih.gov/40033168",
            "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/pmc11111111"},
    "display_name": "Advances and challenges in kidney fibrosis therapeutics",
    "publication_year": 2025,
    "type": "review",
    "primary_location": {"source": {"display_name": "Nature Reviews Nephrology"}},
    "authorships": [{"author": {"display_name": "Lilia Abbad"}}, {"author": {"display_name": None}}],
    "cited_by_count": 96,
    "referenced_works": ["https://openalex.org/W123", "https://openalex.org/W456"],
    "topics": [{"display_name": "Chronic Kidney Disease and Diabetes"}],
    "open_access": {"is_oa": True},
}


def test_normalize_doi_strips_every_prefix_form() -> None:
    for value in ("10.1038/NCOMMS7404", "https://doi.org/10.1038/ncomms7404",
                  "http://dx.doi.org/10.1038/ncomms7404", "doi:10.1038/ncomms7404", " 10.1038/ncomms7404 "):
        assert normalize_doi(value) == "10.1038/ncomms7404"
    assert normalize_doi(None) is None and normalize_doi("") is None


def test_work_fields_namespaces_everything_and_keeps_the_reference_list() -> None:
    fields = work_fields(WORK)
    assert fields["openalex_id"] == "W4407354549"
    assert fields["openalex_type"] == "review"
    assert fields["openalex_venue"] == "Nature Reviews Nephrology"
    # Kept as a cross-check (user, 2026-09-19), under their own names so the catalogue's stay
    # authoritative. Citation count and OpenAlex topics remain out.
    assert fields["openalex_title"] == "Advances and challenges in kidney fibrosis therapeutics"
    assert fields["openalex_year"] == 2025
    assert not {"openalex_cited_by", "openalex_topics"} & set(fields)
    assert fields["openalex_referenced_works"] == ["W123", "W456"]
    assert fields["openalex_authors"] == ["Lilia Abbad"]          # an author with no name is dropped
    assert fields["pmid_openalex"] == "40033168"
    assert fields["pmcid_openalex"] == "PMC11111111"
    # The catalogue's own title, year, journal and doi came from the PDF and llm-wiki; OpenAlex is
    # a metadata source, not evidence, so nothing it returns may land on those names.
    assert not {"title", "year", "journal", "doi", "authors"} & set(fields)


def test_author_preview_reports_received_count_and_truncation() -> None:
    work = {**WORK, "authorships": [{"author": {"display_name": f"Author {i}"}} for i in range(45)]}
    fields = work_fields(work)
    assert len(fields["openalex_authors"]) == 30
    assert fields["openalex_authors_returned_count"] == 45
    assert fields["openalex_authors_truncated"] is True
    assert work_fields(WORK)["openalex_authors_truncated"] is False


def test_openalex_identity_normalizes_url_and_case_without_fuzzy_matching() -> None:
    assert normalize_openalex_id(" https://openalex.org/w123/ ") == "W123"
    assert normalize_openalex_id("W123") == "W123"
    assert normalize_openalex_id("W12-3") is None
    assert normalize_openalex_id("W124") != normalize_openalex_id("W123")


def test_duplicate_doi_with_distinct_ids_is_ambiguous_independent_of_result_order(monkeypatch) -> None:
    doi = "10.123/example"
    works = [{"id": f"https://openalex.org/{work_id}", "doi": doi} for work_id in ("W111", "W222", "W111")]
    class Response(io.BytesIO):
        headers = {}
    for results in (works, works[::-1]):
        monkeypatch.setattr(matcher.urllib.request, "urlopen",
                            lambda *a, **k: Response(json.dumps({"results": results}).encode()))
        found = matcher.fetch_batch([doi], attempts=1)
        assert found[doi]["_ambiguous_openalex_ids"] == ["W111", "W222"]


def test_duplicate_doi_with_same_normalized_identity_is_not_ambiguous(monkeypatch) -> None:
    doi = "10.123/example"
    works = [{"id": "https://openalex.org/W123", "doi": doi}, {"id": "w123", "doi": doi}]
    class Response(io.BytesIO):
        headers = {}
    monkeypatch.setattr(matcher.urllib.request, "urlopen",
                        lambda *a, **k: Response(json.dumps({"results": works}).encode()))
    assert "_ambiguous_openalex_ids" not in matcher.fetch_batch([doi], attempts=1)[doi]


def test_batch_asks_for_a_full_page_not_one_result_per_doi() -> None:
    """One DOI can match several OpenAlex records, so per-page must not be the DOI count."""
    source = Path(__file__).parents[1].joinpath("src/byeori/openalex_match.py").read_text()
    assert "per-page=200" in source
    assert "f\"&per-page={len(dois)}" not in source


def test_squash_does_not_report_a_spelling_of_a_character_as_a_disagreement() -> None:
    """A cross-check is only useful if it flags real differences, not two ways of writing one."""
    from byeori.openalex_match import _squash
    assert _squash("Disruption of Nrxn1\u03b1 within") == _squash("Disruption of Nrxn1a within")
    assert _squash("Schr\u00f6dinger Bridges") == _squash("Schrodinger Bridges")
    assert _squash("TGF-\u03b2 signalling") == _squash("TGF-b signalling")
    assert _squash("CHD8 targets") != _squash("CHD7 targets")
