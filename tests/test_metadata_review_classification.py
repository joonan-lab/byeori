"""Independent integration regressions for identity proposals and supplements."""

from copy import deepcopy

import pytest

from byeori.metadata_review_worker import classify_row


DOI = "10.1234/example"
TITLE = "A coherent study"


def item(issues, **changes):
    metadata = {"doi": DOI, "title": TITLE, "pmid": "1", "pmid_openalex": "2",
                "pmcid": "PMC1", "pmcid_openalex": "PMC2", "openalex_id": "W2",
                "openalex_authors": [f"Author {index}" for index in range(30)]}
    return {"work_id": "example", "issues": issues, "input_sha256": "a" * 64,
            "metadata": metadata | changes}


def records():
    def record(pmid, doi, title):
        return {"pmid": pmid, "record_type": "article", "dois": [doi], "title": title,
                "publication_types": [], "publication_relationships": [], "identity_errors": []}
    return {"1": record("1", "10.1234/other", "Another study"), "2": record("2", DOI, TITLE)}


def conversion(pmid="2", pmcid="PMC2"):
    return {DOI: {"status": "ok", "requested_doi": DOI, "doi": DOI, "pmid": pmid, "pmcid": pmcid}}


def pdf_evidence(**changes):
    return {"checked": True, "sha256_matches": True, "doi_present": True, "title_present": True,
            "doi_candidates": [DOI], "errors": [], **changes}


def oa_work(*, count=31, doi=DOI, work_id="W2"):
    return {"id": work_id, "doi": doi,
            "authorships": [{"author": {"display_name": f"Author {index}"}} for index in range(count)],
            "referenced_works": ["https://openalex.org/W101", "W102"]}


@pytest.mark.parametrize("own,other", [
    ("https://pubmed.ncbi.nlm.nih.gov/1/", "2"),
    ("1", "https://pubmed.ncbi.nlm.nih.gov/2/"),
    ("PMID:1", "pmid:2"),
])
def test_pmid_fetch_and_assessment_use_the_same_identifier_normalization(own, other):
    result = classify_row(item(["pmid_disagreement"], pmid=own, pmid_openalex=other),
                          records(), {}, None, pdf_evidence())
    assert result["proposals"]["pmid"] == "2"
    assert result["pmid_assessment"]["winner"] == "2"
    assert "candidate_identifier_or_pubmed_record_missing" not in result["pmid_assessment"]["reasons"]
    assert result["original"]["pmid"] == own


def test_pmcid_disagreement_is_assessed_using_matching_ncbi_conversion():
    result = classify_row(item(["pmcid_disagreement"], pmid="2"), records(), conversion(), None, pdf_evidence())
    assert result["decisions"]["pmcid_disagreement"] != "lookup_unavailable"
    assert result["proposals"]["pmcid"] == "PMC2"
    assert result["original"]["pmcid"] == "PMC1"


def test_conflicting_ncbi_authorities_suppress_all_canonical_identifier_proposals():
    result = classify_row(item(["pmid_disagreement", "pmcid_disagreement"]),
                          records(), conversion(pmid="3", pmcid="PMC3"), None, pdf_evidence())
    assert result["decisions"]["pmid_disagreement"] == "authority_conflict"
    assert not {"pmid", "pmcid"} & result["proposals"].keys()
    assert result["decisions"]["pmcid_disagreement"] == "authority_conflict"


def test_missing_pmcid_alone_is_allowed_without_using_other_authorities():
    result = classify_row(item(["missing_pmcid"], pmid="2", pmcid=None),
                          records(), conversion(pmid="3", pmcid="PMC3"), None, pdf_evidence())
    assert "pmcid" not in result["proposals"]
    assert result["decisions"]["missing_pmcid"] == "allowed_empty"
    assert not result["supplements"]
    assert "pmc_authority" not in result


@pytest.mark.parametrize("pmcid", [None, "", " \t"])
@pytest.mark.parametrize("legacy_issue", [False, True])
def test_empty_pmcid_ignores_shared_converter_evidence_even_when_pmid_needs_review(pmcid, legacy_issue):
    issues = ["pmid_disagreement"] + (["missing_pmcid"] if legacy_issue else [])
    target = item(issues, pmcid=pmcid)
    pubmed = records()
    pubmed["2"]["pmcid"] = "PMC2"
    result = classify_row(target, pubmed, conversion(pmid="3", pmcid="PMC3"), None, pdf_evidence())
    assert result["proposals"] == {"pmid": "2"}
    assert result["decisions"]["pmid_disagreement"] == "pdf_confirmed"
    assert result["original"]["pmcid"] == pmcid
    assert not result["supplements"] and "pmc_authority" not in result
    if legacy_issue:
        assert result["decisions"]["missing_pmcid"] == "allowed_empty"


def test_legacy_missing_pmcid_item_does_not_consume_converter_even_if_snapshot_is_inconsistent():
    result = classify_row(item(["missing_pmcid"]), records(), conversion(), None, pdf_evidence())
    assert result["decisions"] == {"missing_pmcid": "allowed_empty"}
    assert result["proposals"] == result["supplements"] == {}
    assert "pmc_authority" not in result


@pytest.mark.parametrize("issues", [
    ["pmcid_disagreement"],
    ["pmid_disagreement"],
    ["pmid_disagreement", "pmcid_disagreement"],
])
def test_same_doi_pmid_with_conflicting_authority_pmcids_holds_all_identifier_proposals(issues):
    pubmed = records()
    pubmed["2"]["pmcid"] = "PMC9"
    target = item(issues, pmcid="PMC9")
    result = classify_row(target, pubmed, conversion(), None, pdf_evidence())
    assert not {"pmid", "pmcid"} & result["proposals"].keys()
    assert all(result["decisions"][issue] == "authority_conflict" for issue in issues)
    assert result["original"]["pmcid"] == target["metadata"]["pmcid"]


def test_ambiguous_pubmed_identity_does_not_enable_a_separate_pmcid_proposal():
    ambiguous_records = records()
    ambiguous_records["1"].update(dois=[DOI], title=TITLE)
    result = classify_row(item(["pmid_disagreement", "missing_pmcid"], pmcid=None),
                          ambiguous_records, conversion(), None, pdf_evidence())
    assert not {"pmid", "pmcid"} & result["proposals"].keys()
    assert result["decisions"]["missing_pmcid"] == "allowed_empty"


@pytest.mark.parametrize("title_present", [False, True])
def test_verified_pdf_with_different_explicit_doi_holds_identifier_proposals(title_present):
    pdf = pdf_evidence(doi_present=False, title_present=title_present, doi_candidates=["10.9999/another"])
    result = classify_row(item(["pmid_disagreement", "missing_pmcid"], pmcid=None),
                          records(), conversion(), None, pdf)
    assert not {"pmid", "pmcid"} & result["proposals"].keys()
    assert result["decisions"]["pmid_disagreement"] not in {"metadata_confirmed", "pdf_confirmed"}
    assert result["decisions"]["missing_pmcid"] == "allowed_empty"


def test_pdf_missing_title_alone_does_not_invent_a_source_identity_conflict():
    result = classify_row(item(["pmid_disagreement"]), records(), {}, None,
                          pdf_evidence(title_present=False))
    assert result["proposals"]["pmid"] == "2"
    assert result["decisions"]["pmid_disagreement"] == "metadata_confirmed"


@pytest.mark.parametrize("count", [0, 10, 30])
def test_fresh_roster_that_does_not_expand_existing_authors_is_not_a_supplement(count):
    result = classify_row(item(["authors_at_cap"]), {}, {}, oa_work(count=count), {"checked": False})
    assert result["decisions"]["authors_at_cap"] == "no_expansion"
    assert "openalex_authors" not in result["supplements"]


def test_larger_identity_matched_author_roster_and_references_are_supplements():
    result = classify_row(item(["authors_at_cap", "missing_references"]), {}, {}, oa_work(), {"checked": False})
    assert len(result["supplements"]["openalex_authors"]) == 31
    assert result["supplements"]["openalex_referenced_works"] == ["W101", "W102"]


@pytest.mark.parametrize("changes", [{"doi": "10.1234/other"}, {"work_id": "W999"}])
def test_supplements_require_both_stored_doi_and_openalex_identity(changes):
    result = classify_row(item(["authors_at_cap", "missing_references"]), {}, {},
                          oa_work(**changes), {"checked": False})
    assert not {"openalex_authors", "openalex_referenced_works"} & result["supplements"].keys()
    assert result["decisions"] == {"authors_at_cap": "identity_conflict", "missing_references": "identity_conflict"}


def test_distinct_journal_parts_are_not_collapsed_as_format_variants():
    result = classify_row(item(["journal_disagreement"], journal="Example Genetics (Part A)",
                               openalex_venue="Example Genetics (Part B)"), {}, {}, None, {"checked": False})
    assert result["decisions"]["journal_disagreement"] == "review_required"


def test_classification_preserves_all_original_inputs_even_with_proposals():
    args = [item(["pmid_disagreement", "missing_pmcid"], pmcid=None), records(), conversion(), None, pdf_evidence()]
    before = deepcopy(args)
    result = classify_row(*args)
    assert args == before
    assert result["original"]["pmid"] == "1" and result["original"]["pmcid"] is None
    assert result["proposals"]["pmid"] == "2"
