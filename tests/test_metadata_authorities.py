from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
from copy import deepcopy
from email.message import Message

import pytest

from byeori import metadata_authorities as authorities


def article(pmid="123", doi="10.1234/example", title="A coherent study title", *, extra="", article_extra="", ids=""):
    return f"""<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>
      <ArticleTitle>{title}</ArticleTitle><Journal><Title>Example Journal</Title><JournalIssue>
      <PubDate><Year>2021</Year><Month>Jan</Month></PubDate></JournalIssue></Journal>
      <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
      {article_extra}</Article>{extra}</MedlineCitation><PubmedData>
      <ArticleIdList><ArticleId IdType="pubmed">{pmid}</ArticleId><ArticleId IdType="doi">{doi}</ArticleId>
      {ids}</ArticleIdList><History><PubMedPubDate PubStatus="received"><Year>2020</Year>
      </PubMedPubDate></History><ReferenceList><Reference><ArticleIdList>
      <ArticleId IdType="doi">10.5555/reference</ArticleId><ArticleId IdType="pubmed">999</ArticleId>
      </ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle>"""


def xml(*articles):
    return "<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>"


def pmc_record(doi="10.1234/example", **updates):
    return {"requested-id": doi, "doi": doi, "pmid": 123, "pmcid": "PMC456", **updates}


def pmc_payload(*records, ids=None, **updates):
    result = {"status": "ok", "records": list(records), **updates}
    if ids is not None:
        result["request"] = {"ids": ids, "idtype": "doi"}
    return result


def test_pubmed_uses_only_current_citation_identifiers_and_mixed_title_text():
    raw = xml(article(title="A <i>gene</i><sup>2</sup> study &#945; &amp; β",
                      doi="HTTPS://DOI.ORG/10.1234/EXAMPLE", ids='<ArticleId IdType="pmc">PMC456</ArticleId>',
                      extra='<CommentsCorrectionsList><CommentsCorrections RefType="CommentIn"><PMID>888</PMID>'
                            '</CommentsCorrections></CommentsCorrectionsList>',
                      article_extra='<ELocationID EIdType="doi">10.1234/example</ELocationID>'
                                    '<ELocationID EIdType="doi" ValidYN="N">10.4444/invalid</ELocationID>'
                                    '<AuthorList><Author><ForeName>José</ForeName><LastName>Müller</LastName></Author>'
                                    '<Author><CollectiveName>Study Consortium</CollectiveName></Author></AuthorList>'))
    records = authorities.parse_pubmed_xml(raw)
    assert set(records) == {"123"}
    record = records["123"]
    assert record["title"] == "A gene2 study α & β"
    assert record["dois"] == ["10.1234/example"]
    assert record["pmcid"] == "PMC456" and record["journal"] == "Example Journal"
    assert record["authors"] == ["José Müller", "Study Consortium"]
    assert record["publication_types"] == ["Journal Article"]
    assert record["publication_relationships"] == ["CommentIn"]
    assert record["identity_errors"] == []


def test_pubmed_dates_preserve_issue_and_electronic_dates_without_using_receipt_year():
    records = authorities.parse_pubmed_xml(xml(article(article_extra=
        '<ArticleDate DateType="Electronic"><Year>2020</Year><Month>12</Month><Day>31</Day></ArticleDate>')))
    record = records["123"]
    assert record["publication_year"] == 2021
    assert record["publication_dates"] == [
        {"kind": "journal", "year": "2021", "month": "Jan"},
        {"kind": "article_electronic", "year": "2020", "month": "12", "day": "31"},
        {"kind": "history_received", "year": "2020"}]
    raw = xml(article()).replace("<Year>2021</Year><Month>Jan</Month>", "<MedlineDate>2021 Jan-Feb</MedlineDate>")
    assert authorities.parse_pubmed_xml(raw)["123"]["publication_year"] == 2021
    raw = xml(article()).replace("<Year>2021</Year><Month>Jan</Month>", "")
    assert authorities.parse_pubmed_xml(raw)["123"]["publication_year"] is None


def test_pubmed_book_is_identified_without_importing_reference_ids():
    raw = """<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>321</PMID>
    <ArticleTitle>Chapter title</ArticleTitle><Book><BookTitle>Book title</BookTitle>
    <PubDate><Year>2020</Year></PubDate></Book><ArticleIdList>
    <ArticleId IdType="doi">10.1234/chapter</ArticleId></ArticleIdList><ReferenceList><Reference>
    <ArticleIdList><ArticleId IdType="doi">10.1234/reference</ArticleId></ArticleIdList>
    </Reference></ReferenceList></BookDocument></PubmedBookArticle></PubmedArticleSet>"""
    result = authorities.parse_pubmed_xml(raw)
    assert result["321"]["record_type"] == "book"
    assert result["321"]["title"] == "Chapter title"
    assert result["321"]["dois"] == ["10.1234/chapter"]
    assert result["321"]["publication_year"] == 2020


def test_pubmed_own_identifier_inconsistency_is_reported():
    raw = xml(article(ids='<ArticleId IdType="pubmed">456</ArticleId>'
                          '<ArticleId IdType="pmc">PMC123</ArticleId><ArticleId IdType="pmc">PMC456</ArticleId>'))
    result = authorities.parse_pubmed_xml(raw)["123"]
    assert result["identity_errors"] == ["own_pubmed_id_mismatch", "multiple_pmcids"]
    assert result["pmcid"] is None


@pytest.mark.parametrize("raw", ["<broken", "<ERROR>Private raw message</ERROR>",
                                   xml(article(), article()), xml(article(pmid="invalid")),
                                   '<!DOCTYPE x [<!ENTITY a "unsafe">]><PubmedArticleSet/>',
                                   '<!DOCTYPE x [<!ENTITY a "unsafe">]><PubmedArticleSet/>'.encode('utf-16')])
def test_pubmed_invalid_responses_fail_compactly(raw):
    with pytest.raises(ValueError) as error:
        authorities.parse_pubmed_xml(raw)
    assert "Private raw message" not in str(error.value)


def test_pubmed_external_dtd_is_not_fetched(monkeypatch):
    monkeypatch.setattr(authorities.urllib.request, "urlopen", lambda *a, **k: pytest.fail("No XML HTTP"))
    result = authorities.parse_pubmed_xml('<!DOCTYPE PubmedArticleSet SYSTEM "https://invalid.example/DTD">' + xml(article()))
    assert list(result) == ["123"]


def test_pmc_explicit_doi_pair_normalizes_and_keeps_missing_pmid_nonfatal():
    raw = pmc_payload(pmc_record(**{"requested-id": "https://doi.org/10.1234/EXAMPLE"}),
                      pmc_record("10.1234/no-pmid", pmid=None), ids=["10.1234/example", "10.1234/no-pmid"])
    result = authorities.parse_pmc_converter(json.dumps(raw).encode())
    assert result["10.1234/example"] == {"requested_doi": "10.1234/example", "doi": "10.1234/example",
                                         "pmid": "123", "pmcid": "PMC456", "status": "ok"}
    assert result["10.1234/no-pmid"]["status"] == "ok"
    assert result["10.1234/no-pmid"]["pmid"] is None


def test_pmc_mismatch_error_and_missing_rows_never_supply_trusted_ids():
    raw = pmc_payload(pmc_record(doi="10.1234/other", **{"requested-id": "10.1234/requested"}),
                      pmc_record("10.1234/error", status="error", errmsg="Private upstream body"),
                      ids=["10.1234/requested", "10.1234/error", "10.1234/absent"])
    result = authorities.parse_pmc_converter(raw)
    assert result["10.1234/requested"]["status"] == "mismatch"
    assert result["10.1234/error"]["error"] == "upstream_error"
    assert result["10.1234/absent"]["error"] == "missing_record"
    assert all(row["pmid"] is None and row["pmcid"] is None for row in result.values())
    assert "Private upstream body" not in json.dumps(result)


def test_pmc_duplicate_differing_records_remain_ambiguous_after_repetition():
    raw = pmc_payload(pmc_record(), pmc_record(pmid=456), pmc_record())
    result = authorities.parse_pmc_converter(raw)["10.1234/example"]
    assert result["status"] == "mismatch" and result["error"] == "ambiguous_records"
    assert result["pmid"] is None and result["pmcid"] is None


@pytest.mark.parametrize("raw", ["not JSON", [], {"records": "bad"},
                                   pmc_payload({"doi": "10.1234/example"}),
                                   pmc_payload(pmc_record(), ids=["10.1234/unrequested"]),
                                   {"request": {"idtype": "pmid"}, "records": []}])
def test_pmc_unsupported_or_unassociated_responses_fail_closed(raw):
    with pytest.raises(ValueError):
        authorities.parse_pmc_converter(raw)


class Opener:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def open(self, request, **kwargs):
        self.calls.append((request, kwargs))
        if self.error:
            raise self.error
        return self.response


def install_opener(monkeypatch, *, data=b"", error=None):
    response = io.BytesIO(data)
    opener = Opener(response=response, error=error)
    monkeypatch.setattr(authorities.urllib.request, "build_opener", lambda *a: opener)
    return opener, response


def test_pubmed_fetch_is_one_fixed_post_with_optional_runtime_email(monkeypatch):
    opener, response = install_opener(monkeypatch, data=xml(article()).encode())
    result = authorities.fetch_pubmed(["123", 123], email="maintainer@example.org")
    assert set(result) == {"123"} and len(opener.calls) == 1 and response.closed
    request, options = opener.calls[0]
    assert request.full_url == authorities.PUBMED_URL and request.get_method() == "POST"
    assert options == {"timeout": 60}
    assert urllib.parse.parse_qs(request.data.decode()) == {
        "db": ["pubmed"], "id": ["123"], "retmode": ["xml"],
        "tool": ["byeori-metadata-review"], "email": ["maintainer@example.org"]}


def test_pmc_fetch_is_one_fixed_get_and_missing_records_are_explicit(monkeypatch):
    opener, response = install_opener(monkeypatch, data=json.dumps(pmc_payload(pmc_record())).encode())
    result = authorities.fetch_pmc_ids(["10.1234/example", "10.1234/absent"])
    assert len(opener.calls) == 1 and response.closed
    request, options = opener.calls[0]
    url = urllib.parse.urlsplit(request.full_url)
    assert url.scheme + "://" + url.netloc + url.path == authorities.PMC_URL
    assert request.get_method() == "GET" and options == {"timeout": 30}
    assert urllib.parse.parse_qs(url.query) == {"tool": ["byeori-metadata-review"],
        "ids": ["10.1234/example,10.1234/absent"], "idtype": ["doi"], "format": ["json"]}
    assert result["10.1234/absent"]["error"] == "missing_record"


@pytest.mark.parametrize("function,values", [(authorities.fetch_pubmed, []),
    (authorities.fetch_pubmed, ["1"] * 201), (authorities.fetch_pubmed, ["https://pubmed.ncbi.nlm.nih.gov/123/"]),
    (authorities.fetch_pubmed, [True]), (authorities.fetch_pubmed, ["1 OR 2"]),
    (authorities.fetch_pmc_ids, ["10.1234/x"] * 201), (authorities.fetch_pmc_ids, ["arxiv:1"]),
    (authorities.fetch_pubmed, "123")])
def test_invalid_batches_do_not_make_requests(monkeypatch, function, values):
    monkeypatch.setattr(authorities, "_request_bytes", lambda *a, **k: pytest.fail("No invalid HTTP request"))
    with pytest.raises(ValueError):
        function(values)


@pytest.mark.parametrize("function,data,identifiers", [
    (authorities.fetch_pubmed, xml(article("999")).encode(), ["123"]),
    (authorities.fetch_pmc_ids, json.dumps(pmc_payload(pmc_record("10.1234/other"))).encode(), ["10.1234/example"])])
def test_fetch_rejects_unrequested_response_identifiers(monkeypatch, function, data, identifiers):
    install_opener(monkeypatch, data=data)
    with pytest.raises(ValueError, match="unrequested"):
        function(identifiers)


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (404, False), (302, False)])
def test_http_errors_preserve_retry_after_without_urls_body_or_retries(monkeypatch, status, retryable):
    headers = Message()
    headers["Retry-After"] = "120"
    error = urllib.error.HTTPError("https://invalid.example/?email=private@example.org", status,
                                  "Private upstream reason", headers, io.BytesIO(b"Private body"))
    opener, _ = install_opener(monkeypatch, error=error)
    with pytest.raises(authorities.AuthorityRequestError) as raised:
        authorities.fetch_pubmed(["123"])
    exc = raised.value
    assert exc.status_code == status and exc.retry_after == "120" and exc.retryable is retryable
    assert len(opener.calls) == 1
    assert all(value not in str(exc) for value in ("invalid.example", "private@", "Private", "123"))


def test_transport_errors_and_response_cap_are_compact_and_single_request(monkeypatch):
    opener, _ = install_opener(monkeypatch, error=urllib.error.URLError("https://private.example/?secret=x"))
    with pytest.raises(authorities.AuthorityRequestError) as raised:
        authorities.fetch_pmc_ids(["10.1234/example"])
    assert raised.value.retryable and "private" not in str(raised.value) and len(opener.calls) == 1
    monkeypatch.setattr(authorities, "MAX_RESPONSE_BYTES", 20)
    opener, response = install_opener(monkeypatch, data=b"x" * 21)
    with pytest.raises(authorities.AuthorityRequestError, match="exceeds") as raised:
        authorities.fetch_pubmed(["123"])
    assert not raised.value.retryable and response.closed and len(opener.calls) == 1


def test_redirects_are_not_followed():
    assert authorities._NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.example/") is None


def evidence_records():
    return authorities.parse_pubmed_xml(xml(article("123", "10.1234/wrong", "Different study title"),
                                             article("456")))


def metadata():
    return {"doi": "10.1234/example", "title": "A coherent study title.", "pmid": "123", "pmid_openalex": "456"}


def test_assessment_proposes_unique_doi_and_title_winner_without_mutation():
    data, records = metadata(), evidence_records()
    before = deepcopy((data, records))
    result = authorities.assess_pmid_pair(data, records)
    assert result["classification"] == "metadata_confirmed"
    assert result["source"] == "openalex" and result["winner"] == result["proposed_pmid"] == "456"
    assert (data, records) == before
    data["pmid"], data["pmid_openalex"] = data["pmid_openalex"], data["pmid"]
    assert authorities.assess_pmid_pair(data, records)["source"] == "catalogue"


def test_assessment_doi_ambiguity_is_not_resolved_by_title():
    records = evidence_records()
    records["123"]["dois"] = ["10.1234/example"]
    result = authorities.assess_pmid_pair(metadata(), records,
                                         {"sha256_matches": True, "doi_present": True, "title_present": True})
    assert result["classification"] == "review_required" and result["proposed_pmid"] is None
    assert result["reasons"] == ["both_candidates_claim_doi"]


@pytest.mark.parametrize("change", [{"record_type": "book"}, {"publication_types": ["Published Erratum"]},
    {"publication_types": ["Retracted Publication"]}, {"publication_relationships": ["ErratumIn"]},
    {"title": "Author correction: A coherent study title"}, {"identity_errors": ["own_pubmed_id_mismatch"]}])
def test_publication_or_identity_flags_stay_in_review(change):
    records = evidence_records()
    records["456"].update(change)
    result = authorities.assess_pmid_pair(metadata(), records)
    assert result["classification"] == "review_required" and result["winner"] is None


@pytest.mark.parametrize("flags,expected", [(None, "metadata_confirmed"),
    ({"sha256_matches": True, "doi_present": True, "title_present": True}, "pdf_confirmed"),
    ({"sha256_matches": False, "doi_present": True, "title_present": True}, "metadata_confirmed"),
    ({"sha256_matches": True, "doi_present": True}, "metadata_confirmed"),
    ({"sha256_matches": 1, "doi_present": True, "title_present": True}, "metadata_confirmed")])
def test_pdf_confirmation_requires_all_literal_true_flags(flags, expected):
    assert authorities.assess_pmid_pair(metadata(), evidence_records(), flags)["classification"] == expected


def test_missing_candidate_or_title_difference_stays_review():
    records = evidence_records()
    del records["123"]
    assert authorities.assess_pmid_pair(metadata(), records)["classification"] == "review_required"
    records = evidence_records()
    records["456"]["title"] = "A coherent study of a different subject"
    assert authorities.assess_pmid_pair(metadata(), records)["reasons"] == ["winning_doi_title_disagreement"]


def test_referenced_doi_cannot_create_a_winner_and_candidate_preview_is_bounded():
    records = authorities.parse_pubmed_xml(xml(article("123", "10.1234/wrong"), article("456", "10.1234/other")))
    data = {**metadata(), "doi": "10.5555/reference"}
    assert authorities.assess_pmid_pair(data, records)["reasons"] == ["no_candidate_claims_doi"]
    records["123"]["title"] = "Long " * 1000
    result = authorities.assess_pmid_pair(metadata(), records)
    assert len(result["candidates"]) == 2 and len(result["candidates"][0]["title"]) == 240
