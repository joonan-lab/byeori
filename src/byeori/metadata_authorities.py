"""Bounded public identity metadata reads and proposals, never canonical writes.

The AWS caller owns pacing, retries and caching. Each fetch makes at most one HTTP
request. PubMed identifiers are read only from the current citation, not references
or CommentsCorrections. PMC absence does not establish an invalid PMID.
"""

from __future__ import annotations

import json
import http.client
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
TOOL_NAME = "byeori-metadata-review"
MAX_IDS = 200
MAX_RESPONSE_BYTES = 20 * 1024 * 1024


class AuthorityRequestError(RuntimeError):
    """Compact retry information without request URLs, identifiers, email or bodies."""

    def __init__(self, provider: str, reason: str, *, status_code: int | None = None,
                 retry_after: str | None = None, retryable: bool = False):
        self.provider = provider
        self.reason = reason
        self.status_code = status_code
        self.retry_after = retry_after[:128] if retry_after else None
        self.retryable = retryable
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"{provider}: {reason}{suffix}")


def normalize_doi(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if re.match(r"^https?://(?:dx\.)?doi\.org/", text, re.I):
        text = urllib.parse.unquote(re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I))
    text = re.sub(r"^doi:\s*", "", text, flags=re.I).strip().lower()
    return text if len(text) <= 512 and re.fullmatch(r"10\.[0-9]{4,9}/\S+", text) else None


def _pmid(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    return str(int(text)) if re.fullmatch(r"[0-9]{1,12}", text) and int(text) > 0 else None


def _pmcid(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if re.fullmatch(r"PMC[1-9][0-9]{0,11}", text) else None


def _raw_bytes(raw: bytes | str) -> bytes:
    if not isinstance(raw, (bytes, str)):
        raise ValueError("Response must be bytes or text")
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("Authority response exceeds 20 MiB")
    return data


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _date(element: ET.Element, kind: str) -> dict[str, str]:
    result = {"kind": kind}
    for name in ("Year", "Month", "Day", "Season", "MedlineDate"):
        value = _text(element.find(name))
        if value:
            result["medline_date" if name == "MedlineDate" else name.lower()] = value
    return result


def parse_pubmed_xml(raw: bytes | str) -> dict[str, dict[str, Any]]:
    """Parse current-citation IDs only; an external DTD is never retrieved.

    XML's built-in/numeric entities and inline title elements are supported. Custom
    entity declarations are rejected before parsing. Publication history dates are
    kept separately; receipt/acceptance dates never become the publication year.
    """
    data = _raw_bytes(raw)
    if b"\x00" in data or b"<!ENTITY" in data.upper():
        raise ValueError("Custom XML entities are not accepted")
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raise ValueError("Invalid PubMed XML") from None
    if root.tag == "PubmedArticleSet":
        articles = list(root)
    elif root.tag in ("PubmedArticle", "PubmedBookArticle"):
        articles = [root]
    else:
        raise ValueError("Unexpected PubMed response root")
    if len(articles) > MAX_IDS:
        raise ValueError("PubMed response exceeds 200 records")
    records: dict[str, dict[str, Any]] = {}
    for article in articles:
        if article.tag not in ("PubmedArticle", "PubmedBookArticle"):
            raise ValueError("Unexpected PubMed response element")
        is_book = article.tag == "PubmedBookArticle"
        document = article.find("BookDocument" if is_book else "MedlineCitation")
        if document is None:
            raise ValueError("PubMed citation document is missing")
        pmid = _pmid(_text(document.find("PMID")))
        if pmid is None or pmid in records:
            raise ValueError("Invalid or duplicate PubMed record identity")
        content = document if is_book else document.find("Article")
        if content is None:
            raise ValueError("PubMed article metadata is missing")
        identifiers = list(article.findall("PubmedBookData/ArticleIdList/ArticleId" if is_book
                                           else "PubmedData/ArticleIdList/ArticleId"))
        if is_book:
            identifiers += list(document.findall("ArticleIdList/ArticleId"))
        own_dois = [_text(item) for item in identifiers if item.get("IdType", "").lower() == "doi"]
        own_dois += [_text(item) for item in content.findall("ELocationID")
                     if item.get("EIdType", "").lower() == "doi" and item.get("ValidYN", "Y") != "N"]
        pmcids = sorted({_pmcid(_text(item)) for item in identifiers
                         if item.get("IdType", "").lower() == "pmc"} - {None})
        own_pmids = {_pmid(_text(item)) for item in identifiers
                     if item.get("IdType", "").lower() == "pubmed"}
        identity_errors = []
        if own_pmids - {pmid}:
            identity_errors.append("own_pubmed_id_mismatch")
        if len(pmcids) > 1:
            identity_errors.append("multiple_pmcids")
        dates = [_date(item, "journal") for item in content.findall("Journal/JournalIssue/PubDate")]
        if is_book:
            dates += [_date(item, "book") for item in content.findall("Book/PubDate")]
        dates += [_date(item, "article_" + item.get("DateType", "unknown").lower())
                  for item in content.findall("ArticleDate")]
        publication_year = None
        for date in dates:
            match = re.search(r"\b[12][0-9]{3}\b", date.get("year") or date.get("medline_date", ""))
            if match:
                publication_year = int(match.group())
                break
        history = article.findall("PubmedBookData/History/PubMedPubDate" if is_book
                                  else "PubmedData/History/PubMedPubDate")
        dates += [_date(item, "history_" + item.get("PubStatus", "unknown")) for item in history]
        authors = []
        for author in content.findall("AuthorList/Author"):
            name = _text(author.find("CollectiveName"))
            if not name:
                name = " ".join(filter(None, [_text(author.find("ForeName")) or _text(author.find("Initials")),
                                               _text(author.find("LastName")), _text(author.find("Suffix"))]))
            if name:
                authors.append(name)
        records[pmid] = {
            "pmid": pmid, "record_type": "book" if is_book else "article",
            "title": _text(content.find("ArticleTitle")) or (_text(content.find("Book/BookTitle")) if is_book else ""),
            "dois": sorted({normalize_doi(doi) for doi in own_dois} - {None}),
            "pmcid": pmcids[0] if len(pmcids) == 1 else None,
            "journal": _text(content.find("Journal/Title")),
            "publication_year": publication_year, "publication_dates": dates,
            "publication_types": [_text(item) for item in content.findall("PublicationTypeList/PublicationType")],
            "publication_relationships": sorted({item.get("RefType", "unknown") for item in
                                                   document.findall("CommentsCorrectionsList/CommentsCorrections")}),
            "authors": authors, "identity_errors": identity_errors,
        }
    return records


def parse_pmc_converter(raw: bytes | str | dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return explicit requested-DOI associations; absence/error is never PMID invalidity."""
    try:
        payload = json.loads(_raw_bytes(json.dumps(raw) if isinstance(raw, dict) else raw))
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("Invalid PMC converter JSON") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("records", []), list):
        raise ValueError("Unexpected PMC converter response")
    request = payload.get("request") or {}
    if not isinstance(request, dict) or request.get("idtype", "doi") != "doi":
        raise ValueError("PMC converter response must use DOI requests")
    requested_values = request.get("ids", [])
    if isinstance(requested_values, str):
        requested_values = requested_values.split(",")
    if not isinstance(requested_values, list) or len(requested_values) > MAX_IDS:
        raise ValueError("Invalid PMC requested-ID list")
    requested = {normalize_doi(value) for value in requested_values}
    if None in requested or len(payload.get("records", [])) > MAX_IDS:
        raise ValueError("Invalid or oversized PMC response identities")
    records: dict[str, dict[str, Any]] = {}
    for entry in payload.get("records", []):
        if not isinstance(entry, dict):
            raise ValueError("Invalid PMC record")
        requested_doi = normalize_doi(entry.get("requested-id"))
        if requested_doi is None:
            raise ValueError("PMC record lacks an explicit requested DOI")
        if requested and requested_doi not in requested:
            raise ValueError("PMC response contains an unrequested DOI")
        doi = normalize_doi(entry.get("doi"))
        record = {"requested_doi": requested_doi, "doi": doi, "pmid": None, "pmcid": None, "status": "error"}
        if payload.get("status", "ok") != "ok" or entry.get("status", "ok") != "ok" or entry.get("errmsg"):
            record["error"] = "upstream_error"
        elif doi != requested_doi:
            record.update(status="mismatch", error="returned_doi_mismatch")
        elif not _pmcid(entry.get("pmcid")) or (entry.get("pmid") is not None and not _pmid(entry["pmid"])):
            record.update(status="mismatch", error="invalid_returned_identifier")
        else:
            record.update(status="ok", pmid=_pmid(entry.get("pmid")), pmcid=_pmcid(entry.get("pmcid")))
        if requested_doi in records and records[requested_doi] != record:
            record.update(status="mismatch", error="ambiguous_records", pmid=None, pmcid=None)
        records[requested_doi] = record
    for doi in requested - records.keys():
        records[doi] = {"requested_doi": doi, "doi": None, "pmid": None, "pmcid": None,
                        "status": "error", "error": "missing_record"}
    return records


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request_bytes(provider: str, request: urllib.request.Request, *, timeout: int) -> bytes:
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise AuthorityRequestError(provider, "Response exceeds 20 MiB")
            return data
    except urllib.error.HTTPError as exc:
        status = exc.code
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        exc.close()
        raise AuthorityRequestError(provider, "HTTP request failed", status_code=status,
                                    retry_after=retry_after, retryable=status in (408, 429) or status >= 500) from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        raise AuthorityRequestError(provider, "Transport request failed", retryable=True) from None


def _parameters(values: Any, *, kind: str, email: str | None) -> tuple[list[str], dict[str, str]]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_IDS:
        raise ValueError("Provide 1..200 identifiers in one list or tuple")
    normalizer = _pmid if kind == "pmid" else normalize_doi
    identities = [normalizer(value) for value in values]
    if any(value is None for value in identities):
        raise ValueError(f"Invalid {kind.upper()} identifier")
    identities = list(dict.fromkeys(identities))
    params = {"tool": TOOL_NAME}
    if email is not None:
        if not isinstance(email, str) or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254:
            raise ValueError("Invalid maintainer email")
        params["email"] = email
    return identities, params


def fetch_pubmed(pmids: list[str], *, email: str | None = None) -> dict[str, dict[str, Any]]:
    identities, params = _parameters(pmids, kind="pmid", email=email)
    params.update(db="pubmed", id=",".join(identities), retmode="xml")
    request = urllib.request.Request(PUBMED_URL, data=urllib.parse.urlencode(params).encode("ascii"),
                                     headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    records = parse_pubmed_xml(_request_bytes("PubMed", request, timeout=60))
    if records.keys() - set(identities):
        raise ValueError("PubMed returned an unrequested PMID")
    return records


def fetch_pmc_ids(dois: list[str], *, email: str | None = None) -> dict[str, dict[str, Any]]:
    identities, params = _parameters(dois, kind="doi", email=email)
    params.update(ids=",".join(identities), idtype="doi", format="json")
    request = urllib.request.Request(PMC_URL + "?" + urllib.parse.urlencode(params), method="GET")
    records = parse_pmc_converter(_request_bytes("PMC", request, timeout=30))
    if records.keys() - set(identities):
        raise ValueError("PMC returned an unrequested DOI")
    for doi in set(identities) - records.keys():
        records[doi] = {"requested_doi": doi, "doi": None, "pmid": None, "pmcid": None,
                        "status": "error", "error": "missing_record"}
    return records


def _title_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text
                             if not unicodedata.combining(char)).split())


def _publication_exclusions(record: dict[str, Any]) -> list[str]:
    reasons = []
    if record.get("record_type") != "article":
        reasons.append("not_journal_article")
    types = " ".join(str(value).lower() for value in record.get("publication_types", []))
    if re.search(r"erratum|correction|corrected|retract|expression of concern|duplicate publication|book", types):
        reasons.append("publication_type_requires_review")
    if re.match(r"^(?:\[\s*)?(?:author correction|publisher correction|correction|erratum|corrigendum|retraction|withdrawal|expression of concern)\b",
                str(record.get("title", "")), re.I):
        reasons.append("notice_title_requires_review")
    if record.get("publication_relationships"):
        reasons.append("publication_relationship_requires_review")
    if record.get("identity_errors"):
        reasons.append("record_identity_error")
    return reasons


def assess_pmid_pair(metadata: dict[str, Any], pubmed_records: dict[str, dict[str, Any]],
                     pdf_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Propose a unique PMID using own DOI and coherent title, without mutating inputs.

    Both candidates claiming the DOI stay ambiguous regardless of title. Missing
    records and publication relationships remain review cases. PDF confirmation
    only strengthens a metadata proposal when all three worker-supplied flags are
    literal True; it cannot override contradictory or insufficient metadata.
    """
    doi = normalize_doi(metadata.get("doi"))
    title = " ".join(_title_key(metadata.get("title")).split())
    result: dict[str, Any] = {"classification": "review_required", "source": None, "winner": None,
                              "proposed_pmid": None, "reasons": [], "candidates": []}
    if not doi or not title:
        result["reasons"] = ["missing_usable_catalogue_doi_or_title"]
        return result
    seen = set()
    missing = False
    for source, field in (("catalogue", "pmid"), ("openalex", "pmid_openalex")):
        pmid = _pmid(metadata.get(field))
        if pmid is None:
            missing = True
            continue
        if pmid in seen:
            continue
        seen.add(pmid)
        record = pubmed_records.get(pmid)
        if not isinstance(record, dict) or record.get("pmid") != pmid:
            missing = True
            result["candidates"].append({"source": source, "pmid": pmid, "available": False})
            continue
        own_dois = {normalize_doi(value) for value in record.get("dois", [])}
        result["candidates"].append({"source": source, "pmid": pmid, "available": True,
                                      "doi_match": doi in own_dois,
                                      "title_coherent": title == " ".join(_title_key(record.get("title")).split()),
                                      "title": str(record.get("title") or "")[:240],
                                      "exclusions": _publication_exclusions(record)})
    if missing:
        result["reasons"] = ["candidate_identifier_or_pubmed_record_missing"]
        return result
    candidates = result["candidates"]
    if any(candidate["exclusions"] for candidate in candidates):
        result["reasons"] = ["publication_or_identity_requires_review"]
        return result
    matching = [candidate for candidate in candidates if candidate["doi_match"]]
    if len(matching) != 1:
        result["reasons"] = ["both_candidates_claim_doi" if len(matching) > 1 else "no_candidate_claims_doi"]
        return result
    winner = matching[0]
    if not winner["title_coherent"]:
        result["reasons"] = ["winning_doi_title_disagreement"]
        return result
    pdf_confirmed = isinstance(pdf_evidence, dict) and all(
        pdf_evidence.get(name) is True for name in ("sha256_matches", "doi_present", "title_present"))
    result.update(classification="pdf_confirmed" if pdf_confirmed else "metadata_confirmed",
                  source=winner["source"], winner=winner["pmid"], proposed_pmid=winner["pmid"],
                  reasons=["unique_own_doi_and_coherent_title"])
    return result
