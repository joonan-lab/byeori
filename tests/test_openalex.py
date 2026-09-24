from __future__ import annotations

import json

import httpx

from byeori.openalex import OpenAlexClient, normalize_doi, normalize_work_id


def test_normalize_identifiers() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC.1") == "10.1000/abc.1"
    assert normalize_doi("doi:10.1038/test") == "10.1038/test"
    assert normalize_work_id("https://openalex.org/W12345") == "W12345"


def test_normalize_doi_preserves_canonical_suffix_punctuation() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC)") == "10.1000/abc)"
    assert normalize_doi("10.1000/abc)") != normalize_doi("10.1000/abc")
    assert normalize_doi("10.1000/abc.") == "10.1000/abc."
    assert normalize_doi("text containing 10.1000/abc") is None


def test_search_normalizes_work() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == "brain organoid"
        assert request.url.params["filter"] == "is_oa:true,has_content.grobid_xml:true"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "doi": "https://doi.org/10.1000/TEST",
                        "display_name": "A useful paper",
                        "publication_year": 2026,
                        "authorships": [
                            {"author": {"display_name": "Ada Lovelace"}}
                        ],
                        "open_access": {"is_oa": True},
                        "best_oa_location": {"license": "cc-by"},
                        "content_urls": {
                            "pdf": "https://content.openalex.org/works/W123.pdf",
                            "grobid_xml": "https://content.openalex.org/works/W123.grobid.xml",
                        },
                        "primary_location": {
                            "source": {"display_name": "Test Journal"}
                        },
                        "abstract_inverted_index": {"Useful": [0], "result": [1]},
                    }
                ]
            },
        )

    with OpenAlexClient(transport=httpx.MockTransport(handler)) as client:
        result = client.search(
            "brain organoid",
            limit=1,
            oa_only=True,
            fulltext_only=True,
        )[0]
    assert result["work_id"] == "W123"
    assert result["doi"] == "10.1000/test"
    assert result["authors"] == ["Ada Lovelace"]
    assert result["abstract"] == "Useful result"
    assert result["oa_license"] == "cc-by"
    assert result["grobid_xml_url"].endswith("W123.grobid.xml")


def test_work_json_is_serializable() -> None:
    work = {
        "work_id": "W123",
        "doi": "10.1000/test",
        "title": "A paper",
        "authors": ["Ada Lovelace"],
    }
    assert json.loads(json.dumps(work))["work_id"] == "W123"
