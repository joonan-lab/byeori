from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

try:
    import httpx
except ModuleNotFoundError:
    # The Lambda runtime has no httpx. The ingest function imports this module, through identity,
    # for the pure normalizers; only OpenAlexClient needs httpx, and the Lambda never builds one.
    # Without this, every resolve_identity call failed on import (2026-09-23).
    httpx = None


OPENALEX_API = "https://api.openalex.org"
DOI_PATTERN = re.compile(r"(?:https?://(?:dx\.)?doi\.org/|doi:)?(10\.\d{4,9}/\S+)", re.I)
WORK_ID_PATTERN = re.compile(r"(?:https?://openalex\.org/)?(W\d+)$", re.I)


class OpenAlexError(RuntimeError):
    pass


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    match = DOI_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    return match.group(1).lower()


def normalize_work_id(value: str | None) -> str | None:
    if not value:
        return None
    match = WORK_ID_PATTERN.fullmatch(value.strip())
    return match.group(1).upper() if match else None


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    if not inverted_index:
        return None
    positioned: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        positioned.extend((position, token) for position in positions)
    return " ".join(token for _, token in sorted(positioned))


def eligible_fulltext(work: dict[str, Any]) -> bool:
    """An open-access work with a PDF and GROBID text OpenAlex hosts under a CC or public-domain licence.

    Here rather than in `corpus` so the ingest Lambda can ask without importing the client store.
    """
    license_name = work.get("oa_license") or ""
    return bool(work.get("is_open_access") and work.get("openalex_pdf_url")
                and work.get("grobid_xml_url")
                and (license_name.startswith("cc-") or license_name in {"cc0", "public-domain"}))


def normalize_work(work: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)

    best_oa = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    content_urls = work.get("content_urls") or {}
    source = primary.get("source") or {}
    topics = [
        topic.get("display_name")
        for topic in (work.get("topics") or [])
        if topic.get("display_name")
    ]
    work_id = normalize_work_id(work.get("id"))
    return {
        "work_id": work_id,
        "doi": normalize_doi(work.get("doi")),
        "title": work.get("display_name") or work.get("title"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "authors": authors,
        "source": source.get("display_name"),
        # Kept since 2026-09-22 so the journal policy can match on an identifier rather than a
        # title. OpenAlex registers JAMA Pediatrics under its former name, and titles arrive as
        # "Science (New York, N.Y.)" or "Nature reviews. Genetics"; an ISSN has none of that drift.
        # The publisher is what makes a house-wide refusal (MDPI, Frontiers) possible at all.
        "source_id": source.get("id"),
        "source_issn": [issn for issn in (source.get("issn") or []) if issn],
        "source_publisher": source.get("host_organization_name"),
        "cited_by_count": work.get("cited_by_count", 0),
        "is_open_access": bool((work.get("open_access") or {}).get("is_oa")),
        "oa_license": best_oa.get("license"),
        "landing_page_url": best_oa.get("landing_page_url") or primary.get("landing_page_url"),
        "pdf_url": best_oa.get("pdf_url"),
        "openalex_pdf_url": content_urls.get("pdf"),
        "grobid_xml_url": content_urls.get("grobid_xml"),
        "topics": topics,
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "referenced_works": [
            normalize_work_id(item) for item in (work.get("referenced_works") or [])
            if normalize_work_id(item)
        ],
        "raw": work,
    }


class OpenAlexClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = OPENALEX_API,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "byeori/0.1"},
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OpenAlexClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_params = dict(params or {})
        if self.api_key:
            request_params["api_key"] = self.api_key
        try:
            response = self.client.get(path, params=request_params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else "unavailable"
            # HTTP exception strings can include the API key from the request URL.
            raise OpenAlexError(f"OpenAlex request failed ({type(exc).__name__}, status {status})") from None
        payload = response.json()
        if not isinstance(payload, dict):
            raise OpenAlexError("OpenAlex returned a non-object JSON response")
        return payload

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        from_year: int | None = None,
        to_year: int | None = None,
        oa_only: bool = False,
        fulltext_only: bool = False,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        filters = []
        if from_year is not None:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year is not None:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if oa_only:
            filters.append("is_oa:true")
        if fulltext_only:
            filters.append("has_content.grobid_xml:true")
        params: dict[str, Any] = {"search": query.strip(), "per_page": limit}
        if filters:
            params["filter"] = ",".join(filters)
        payload = self._get("/works", params)
        return [normalize_work(work) for work in payload.get("results") or []]

    def get_work(self, identifier: str) -> dict[str, Any]:
        work_id = normalize_work_id(identifier)
        if work_id:
            return normalize_work(self._get(f"/works/{work_id}"))
        doi = normalize_doi(identifier)
        if not doi:
            raise ValueError("identifier must be an OpenAlex work ID or DOI")
        payload = self._get("/works", {"filter": f"doi:https://doi.org/{quote(doi, safe='/')}"})
        results = payload.get("results") or []
        if not results:
            raise OpenAlexError(f"OpenAlex work not found for DOI {doi}")
        return normalize_work(results[0])
